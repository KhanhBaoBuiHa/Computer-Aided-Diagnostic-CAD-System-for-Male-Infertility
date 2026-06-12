"""
inference.py
------------
Production-ready inference script cho CAD Sperm Analysis System.
Hỗ trợ cả PyTorch (.pth) và ONNX Runtime (.onnx).

Usage:
    # Single image, ONNX backend
    python inference.py --image path/to/sperm.png \
                        --binary_onnx  onnx_models/binary_resnet50.onnx \
                        --multi_onnx   onnx_models/multiclass_efficientnet_b0.onnx

    # Single image, PyTorch backend
    python inference.py --image path/to/sperm.png \
                        --binary_pth   binary_best.pth \
                        --multi_pth    multiclass_best.pth \
                        --backend pytorch

    # Batch benchmark trên thư mục ảnh
    python inference.py --image_dir path/to/test_images/ \
                        --binary_onnx  onnx_models/binary_resnet50.onnx \
                        --multi_onnx   onnx_models/multiclass_efficientnet_b0.onnx \
                        --benchmark
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ======================================================
# CONSTANTS
# ======================================================

IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BINARY_LABELS    = {0: "Non-sperm", 1: "Sperm"}
HALO_LABELS      = {0: "Large halo", 1: "Medium halo", 2: "Small halo", 3: "Without halo"}
IMG_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ======================================================
# PREPROCESSING (mirror từ main.ipynb GD1)
# ======================================================

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Đọc và tiền xử lý ảnh giống pipeline GD1:
      Đọc → Resize → Bilateral denoise → CLAHE → ImageNet normalize
    
    Returns:
        np.ndarray shape (1, 3, 224, 224) float32 — sẵn sàng cho inference
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (h > IMAGE_SIZE[0] or w > IMAGE_SIZE[1]) else cv2.INTER_LINEAR
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=interp)

    # Bilateral denoise (giảm noise giữ cạnh)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE trên kênh L (LAB space)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ImageNet normalize
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW → batch
    img = img.transpose(2, 0, 1)          # (3, 224, 224)
    img = np.expand_dims(img, axis=0)     # (1, 3, 224, 224)
    return img.astype(np.float32)


# ======================================================
# MODEL LOADERS
# ======================================================

def build_binary_model():
    model = models.resnet50(weights=None)
    in_f  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(256, 1),
    )
    return model


def build_multiclass_model(num_classes=4):
    model   = models.efficientnet_b0(weights=None)
    in_f    = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(256, num_classes),
    )
    return model


def load_pytorch_models(binary_pth: str, multi_pth: str, device):
    """Load cả 2 PyTorch model từ checkpoint."""
    def _load(model, path):
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model

    model_A = _load(build_binary_model(), binary_pth).to(device)
    model_B = _load(build_multiclass_model(), multi_pth).to(device)
    print(f"✅ PyTorch models loaded")
    return model_A, model_B


def load_onnx_sessions(binary_onnx: str, multi_onnx: str):
    """Load cả 2 ONNX InferenceSession."""
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime chưa cài. Chạy: pip install onnxruntime")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_A = ort.InferenceSession(binary_onnx,   providers=providers)
    sess_B = ort.InferenceSession(multi_onnx,    providers=providers)
    used_ep = sess_A.get_providers()[0]
    print(f"✅ ONNX sessions loaded  [EP: {used_ep}]")
    return sess_A, sess_B


# ======================================================
# INFERENCE FUNCTIONS
# ======================================================

def run_pytorch(model_A, model_B, img_array: np.ndarray, device):
    """
    Chạy cả pipeline binary → multiclass với PyTorch.
    
    Returns:
        dict với binary_label, binary_conf, halo_label, halo_conf, latency_ms
    """
    tensor = torch.from_numpy(img_array).to(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        # Task A — Binary
        out_A   = model_A(tensor)
        prob_A  = torch.sigmoid(out_A.squeeze()).item()
        pred_A  = 1 if prob_A >= 0.5 else 0

        # Task B — Multiclass (chỉ chạy nếu là Sperm)
        out_B    = model_B(tensor)
        probs_B  = torch.softmax(out_B.squeeze(), dim=0).cpu().numpy()
        pred_B   = int(probs_B.argmax())

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "binary_label": BINARY_LABELS[pred_A],
        "binary_conf" : round(prob_A if pred_A == 1 else 1 - prob_A, 4),
        "halo_label"  : HALO_LABELS[pred_B] if pred_A == 1 else "N/A",
        "halo_conf"   : round(float(probs_B[pred_B]), 4) if pred_A == 1 else None,
        "latency_ms"  : round(latency_ms, 2),
        "backend"     : "PyTorch",
    }


def run_onnx(sess_A, sess_B, img_array: np.ndarray):
    """
    Chạy cả pipeline binary → multiclass với ONNX Runtime.
    
    Returns:
        dict với binary_label, binary_conf, halo_label, halo_conf, latency_ms
    """
    in_A = sess_A.get_inputs()[0].name
    in_B = sess_B.get_inputs()[0].name

    t0 = time.perf_counter()

    # Task A — Binary
    out_A  = sess_A.run(None, {in_A: img_array})[0]      # (1, 1)
    prob_A = 1 / (1 + np.exp(-out_A.squeeze()))           # sigmoid
    pred_A = 1 if prob_A >= 0.5 else 0

    # Task B — Multiclass
    out_B   = sess_B.run(None, {in_B: img_array})[0]     # (1, 4)
    exp_B   = np.exp(out_B.squeeze() - out_B.max())
    probs_B = exp_B / exp_B.sum()                         # softmax
    pred_B  = int(probs_B.argmax())

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "binary_label": BINARY_LABELS[pred_A],
        "binary_conf" : round(float(prob_A if pred_A == 1 else 1 - prob_A), 4),
        "halo_label"  : HALO_LABELS[pred_B] if pred_A == 1 else "N/A",
        "halo_conf"   : round(float(probs_B[pred_B]), 4) if pred_A == 1 else None,
        "latency_ms"  : round(latency_ms, 2),
        "backend"     : "ONNX Runtime",
    }


# ======================================================
# DISPLAY
# ======================================================

def print_result(result: dict, image_path: str):
    """In kết quả inference đẹp ra console."""
    print(f"\n{'─'*50}")
    print(f"  Image   : {Path(image_path).name}")
    print(f"  Backend : {result['backend']}")
    print(f"{'─'*50}")
    print(f"  [Task A] {'Sperm' if result['binary_label'] == 'Sperm' else 'Non-sperm':12s}  "
          f"confidence: {result['binary_conf']:.4f}")
    if result["halo_label"] != "N/A":
        print(f"  [Task B] {result['halo_label']:14s}  "
              f"confidence: {result['halo_conf']:.4f}")
    else:
        print(f"  [Task B] Skipped (Non-sperm detected)")
    print(f"  Latency : {result['latency_ms']:.2f} ms")
    print(f"{'─'*50}")


# ======================================================
# BATCH BENCHMARK
# ======================================================

def batch_benchmark(image_dir: str, run_fn, n_warmup: int = 5):
    """
    Chạy inference trên toàn bộ ảnh trong thư mục, báo cáo latency.
    
    Args:
        image_dir : thư mục chứa ảnh test
        run_fn    : hàm (img_array) → result dict
        n_warmup  : số ảnh đầu dùng warm-up, không tính vào stats
    """
    paths = [
        p for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not paths:
        print(f"⚠️  Không tìm thấy ảnh trong: {image_dir}")
        return

    print(f"\n🔥 Batch benchmark: {len(paths)} ảnh  (warm-up: {n_warmup})")
    latencies = []
    errors    = 0

    for i, path in enumerate(paths):
        try:
            img = preprocess_image(path)
            result = run_fn(img)
            if i >= n_warmup:
                latencies.append(result["latency_ms"])
        except Exception as e:
            errors += 1
            print(f"  ⚠️  Lỗi {path.name}: {e}")

    if not latencies:
        print("  Không có kết quả hợp lệ.")
        return

    latencies = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"  BATCH RESULTS ({len(latencies)} ảnh, bỏ {n_warmup} warm-up)")
    print(f"{'='*50}")
    print(f"  Mean latency : {latencies.mean():.2f} ms")
    print(f"  Std          : {latencies.std():.2f} ms")
    print(f"  Min          : {latencies.min():.2f} ms")
    print(f"  Max          : {latencies.max():.2f} ms")
    print(f"  Throughput   : {1000/latencies.mean():.1f} FPS")
    if errors:
        print(f"  Errors       : {errors}")
    print()


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="CAD Sperm — Inference Pipeline")
    parser.add_argument("--image",       help="Path tới một ảnh")
    parser.add_argument("--image_dir",   help="Thư mục ảnh để batch benchmark")
    parser.add_argument("--backend",     default="onnx", choices=["onnx", "pytorch"],
                        help="Backend inference (default: onnx)")
    # ONNX paths
    parser.add_argument("--binary_onnx", default="onnx_models/binary_resnet50.onnx")
    parser.add_argument("--multi_onnx",  default="onnx_models/multiclass_efficientnet_b0.onnx")
    # PyTorch paths
    parser.add_argument("--binary_pth",  default="binary_best.pth")
    parser.add_argument("--multi_pth",   default="multiclass_best.pth")
    parser.add_argument("--benchmark",   action="store_true",
                        help="Bật chế độ batch benchmark")
    args = parser.parse_args()

    # --- Load models ---
    if args.backend == "onnx":
        sess_A, sess_B = load_onnx_sessions(args.binary_onnx, args.multi_onnx)
        run_fn = lambda img: run_onnx(sess_A, sess_B, img)
    else:
        device = torch.device("cpu")
        model_A, model_B = load_pytorch_models(args.binary_pth, args.multi_pth, device)
        run_fn = lambda img: run_pytorch(model_A, model_B, img, device)

    # --- Single image ---
    if args.image:
        img    = preprocess_image(args.image)
        result = run_fn(img)
        print_result(result, args.image)

    # --- Batch benchmark ---
    if args.image_dir or args.benchmark:
        target = args.image_dir or "."
        batch_benchmark(target, run_fn)

    if not args.image and not args.image_dir and not args.benchmark:
        parser.print_help()


if __name__ == "__main__":
    main()
