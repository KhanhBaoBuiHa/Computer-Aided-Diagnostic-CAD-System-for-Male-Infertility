"""
export_onnx.py
--------------
Export model_A (ResNet50 binary) và model_B (EfficientNet-B0 multiclass)
sang định dạng ONNX để benchmark inference và deploy.

Usage:
    python export_onnx.py \
        --binary_pth  path/to/binary_best.pth \
        --multi_pth   path/to/multiclass_best.pth \
        --output_dir  ./onnx_models
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnx / onnxruntime chưa cài. Chạy: pip install onnx onnxruntime")


# ======================================================
# MODEL DEFINITIONS (mirror từ main.ipynb)
# ======================================================

def build_binary_model(pretrained=False, freeze_backbone=False):
    """ResNet50 + custom head → 1 output (binary)."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model


def build_multiclass_model(num_classes=4, pretrained=False, freeze_backbone=False):
    """EfficientNet-B0 + custom head → 4 outputs (halo classification)."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ======================================================
# EXPORT FUNCTION
# ======================================================

def export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224),
                   opset_version=18, dynamic_batch=True):
    """
    Export PyTorch model → ONNX.

    Args:
        model        : PyTorch model (đã load weights, đã eval())
        onnx_path    : đường dẫn lưu file .onnx
        input_shape  : shape của dummy input
        opset_version: ONNX opset (17 tương thích rộng)
        dynamic_batch: cho phép batch size động khi inference

    Returns:
        onnx_path nếu thành công, None nếu lỗi
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"  Đang export → {onnx_path} ...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,  # opset 18 = min version supported by current torch.onnx dynamo path
            do_constant_folding=True,       # fold constants → nhỏ hơn, nhanh hơn
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        print(f"  ✅ Export thành công: {onnx_path}")
    except Exception as e:
        print(f"  ❌ Export thất bại: {e}")
        return None

    # Validate ONNX graph
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✅ ONNX graph hợp lệ")
        except Exception as e:
            print(f"  ⚠️  ONNX validation warning: {e}")

    return onnx_path


# ======================================================
# BENCHMARK FUNCTION
# ======================================================

def benchmark_latency(model_or_session, input_shape=(1, 3, 224, 224),
                       n_warmup=20, n_runs=200, backend="pytorch"):
    """
    Đo latency trung bình (ms) của model trên CPU.

    Args:
        model_or_session : torch.nn.Module hoặc ort.InferenceSession
        n_warmup         : số lần chạy warm-up (không tính)
        n_runs           : số lần chạy tính trung bình
        backend          : "pytorch" hoặc "onnx"

    Returns:
        dict: mean_ms, std_ms, min_ms, max_ms, throughput_fps
    """
    dummy = np.random.randn(*input_shape).astype(np.float32)

    if backend == "pytorch":
        model_or_session.eval()
        tensor = torch.from_numpy(dummy)
        # Warm-up
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model_or_session(tensor)
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model_or_session(tensor)
                latencies.append((time.perf_counter() - t0) * 1000)

    elif backend == "onnx":
        input_name = model_or_session.get_inputs()[0].name
        # Warm-up
        for _ in range(n_warmup):
            _ = model_or_session.run(None, {input_name: dummy})
        # Benchmark
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model_or_session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

    else:
        raise ValueError(f"backend phải là 'pytorch' hoặc 'onnx', nhận: {backend}")

    latencies = np.array(latencies)
    mean_ms   = latencies.mean()
    return {
        "mean_ms"       : round(mean_ms, 2),
        "std_ms"        : round(latencies.std(), 2),
        "min_ms"        : round(latencies.min(), 2),
        "max_ms"        : round(latencies.max(), 2),
        "throughput_fps": round(1000 / mean_ms, 1),
    }


def print_benchmark_table(results: dict, model_name: str):
    """In bảng benchmark đẹp ra console."""
    print(f"\n{'='*55}")
    print(f"  BENCHMARK — {model_name}")
    print(f"{'='*55}")
    header = f"{'Backend':<18} {'Mean(ms)':>9} {'Std(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'FPS':>8}"
    print(header)
    print("-" * 55)
    for backend, stats in results.items():
        row = (f"{backend:<18} "
               f"{stats['mean_ms']:>9} "
               f"{stats['std_ms']:>8} "
               f"{stats['min_ms']:>8} "
               f"{stats['max_ms']:>8} "
               f"{stats['throughput_fps']:>8}")
        print(row)

    # Tính speedup nếu có cả 2 backend
    if "PyTorch (CPU)" in results and "ONNX Runtime (CPU)" in results:
        pt_mean   = results["PyTorch (CPU)"]["mean_ms"]
        onnx_mean = results["ONNX Runtime (CPU)"]["mean_ms"]
        speedup   = pt_mean / onnx_mean
        tag       = "✅ nhanh hơn" if speedup > 1 else "⚠️  chậm hơn"
        print(f"\n  ONNX vs PyTorch speedup: {speedup:.2f}x {tag}")
    print()


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Export & benchmark CAD models → ONNX")
    parser.add_argument("--binary_pth",  default="binary_best.pth",
                        help="Path tới checkpoint binary model (ResNet50)")
    parser.add_argument("--multi_pth",   default="multiclass_best.pth",
                        help="Path tới checkpoint multiclass model (EfficientNet-B0)")
    parser.add_argument("--output_dir",  default="onnx_models",
                        help="Thư mục lưu file ONNX")
    parser.add_argument("--n_runs",      type=int, default=200,
                        help="Số lần chạy benchmark (default: 200)")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="Bỏ qua bước benchmark (chỉ export)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")   # benchmark trên CPU để fair comparison

    configs = [
        {
            "name"      : "Binary (ResNet50)",
            "pth_path"  : args.binary_pth,
            "onnx_name" : "binary_resnet50.onnx",
            "builder"   : lambda: build_binary_model(pretrained=False),
        },
        {
            "name"      : "Multiclass (EfficientNet-B0)",
            "pth_path"  : args.multi_pth,
            "onnx_name" : "multiclass_efficientnet_b0.onnx",
            "builder"   : lambda: build_multiclass_model(num_classes=4, pretrained=False),
        },
    ]

    all_results = {}

    for cfg in configs:
        print(f"\n{'#'*60}")
        print(f"  {cfg['name']}")
        print(f"{'#'*60}")

        # --- Load PyTorch model ---
        if not os.path.exists(cfg["pth_path"]):
            print(f"  ⚠️  Không tìm thấy checkpoint: {cfg['pth_path']} → bỏ qua")
            continue

        model = cfg["builder"]()
        state = torch.load(cfg["pth_path"], map_location=device, weights_only=False)
        # Hỗ trợ cả 2 format lưu (dict có key 'model_state_dict' hoặc raw state_dict)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        print(f"  ✅ Loaded PyTorch checkpoint")

        # --- Export ONNX ---
        onnx_path = os.path.join(args.output_dir, cfg["onnx_name"])
        result    = export_to_onnx(model, onnx_path)
        if result is None:
            continue

        # File size
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  📦 ONNX file size: {size_mb:.1f} MB")

        if args.skip_benchmark:
            continue

        if not ONNX_AVAILABLE:
            print("  ⚠️  Bỏ qua benchmark vì onnxruntime chưa cài")
            continue

        # --- Benchmark ---
        print(f"\n  🔥 Đang benchmark ({args.n_runs} runs, CPU only) ...")

        pt_stats = benchmark_latency(model, n_runs=args.n_runs, backend="pytorch")

        ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        onnx_stats = benchmark_latency(ort_session, n_runs=args.n_runs, backend="onnx")

        results = {
            "PyTorch (CPU)":      pt_stats,
            "ONNX Runtime (CPU)": onnx_stats,
        }
        print_benchmark_table(results, cfg["name"])
        all_results[cfg["name"]] = results

    # --- Summary ---
    if all_results:
        print(f"\n{'='*60}")
        print("  SUMMARY — ONNX Speedup")
        print(f"{'='*60}")
        for name, res in all_results.items():
            if "PyTorch (CPU)" in res and "ONNX Runtime (CPU)" in res:
                pt   = res["PyTorch (CPU)"]["mean_ms"]
                onnx = res["ONNX Runtime (CPU)"]["mean_ms"]
                print(f"  {name:<35} {pt:.1f}ms → {onnx:.1f}ms  "
                      f"({pt/onnx:.2f}x speedup)")
        print()
        print("  📁 ONNX models saved to:", args.output_dir)


if __name__ == "__main__":
    main()
