import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from inference import (
    build_binary_model, build_multiclass_model,
    preprocess_image, IMAGE_SIZE,
    BINARY_LABELS, HALO_LABELS,
    IMAGENET_MEAN, IMAGENET_STD,
)

# ============================================================
# Load PyTorch models 1 lần (dùng riêng cho Grad-CAM)
# ONNX không support backward pass nên phải dùng PyTorch
# ============================================================

_device = torch.device("cpu")
_pt_model_A = None
_pt_model_B = None

def load_pytorch_for_gradcam(binary_pth: str, multi_pth: str):
    """Gọi 1 lần khi server start, lưu vào module-level variables."""
    global _pt_model_A, _pt_model_B

    def _load(model, path):
        import os
        if not os.path.exists(path):
            return None
        state = torch.load(path, map_location=_device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model

    _pt_model_A = _load(build_binary_model(), binary_pth)
    _pt_model_B = _load(build_multiclass_model(), multi_pth)
    print("✅ PyTorch models loaded for Grad-CAM")


# ============================================================
# Grad-CAM core
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation dùng forward/backward hooks.
    
    target_layer: layer muốn visualize
      - ResNet50:       model.layer4[-1]
      - EfficientNet-B0: model.features[-1]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            img_tensor: (1, 3, H, W) float32
            class_idx:  class muốn visualize (None = argmax)
        Returns:
            heatmap: np.ndarray (H, W) float32, range [0, 1]
        """
        self.model.zero_grad()
        output = self.model(img_tensor)

        # Xác định class target
        if output.shape[-1] == 1:
            # Binary: sigmoid
            score = torch.sigmoid(output.squeeze())
            score.backward()
        else:
            # Multiclass: softmax
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]
            score.backward()

        # Global Average Pooling trên gradients
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weighted sum of activation maps
        activations = self.activations[0]  # (C, H, W)
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).numpy()  # (H, W)
        heatmap = np.maximum(heatmap, 0)           # ReLU

        # Normalize về [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


def heatmap_to_overlay(original_img_bytes: bytes, heatmap: np.ndarray) -> str:
    """
    Overlay Grad-CAM heatmap lên ảnh gốc.
    
    Returns:
        base64-encoded PNG string (dùng trực tiếp trong <img src="data:image/png;base64,...">)
    """
    # Decode ảnh gốc
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    # Resize heatmap về kích thước ảnh
    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Áp colormap JET
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay: 60% ảnh gốc + 40% heatmap
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # Encode sang PNG → base64
    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64


# ============================================================
# Hàm chính — gọi từ /explain endpoint trong app.py
# ============================================================

def run_gradcam_explain(img_bytes: bytes):
    """
    Chạy Grad-CAM cho cả Task A và Task B.
    
    Returns dict:
    {
        "binary_label": ...,
        "binary_conf": ...,
        "halo_label": ...,
        "halo_conf": ...,
        "gradcam_task_a": "<base64 PNG>",   # heatmap overlay Task A
        "gradcam_task_b": "<base64 PNG>",   # heatmap overlay Task B
    }
    """
    if _pt_model_A is None or _pt_model_B is None:
        raise RuntimeError("PyTorch models chưa load. Gọi load_pytorch_for_gradcam() trước.")

    # Preprocess
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = cv2.resize(img_cv, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_cv = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_np = img_cv.astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    img_tensor.requires_grad_(False)

    # ── Task A — ResNet50, hook tại layer4[-1] ──
    gcam_A = GradCAM(_pt_model_A, _pt_model_A.layer4[-1])
    t_A = img_tensor.clone().requires_grad_(True)
    heatmap_A = gcam_A.generate(t_A)
    
    with torch.no_grad():
        out_A = _pt_model_A(img_tensor)
        prob_A = torch.sigmoid(out_A.squeeze()).item()
        pred_A = 1 if prob_A >= 0.5 else 0

    overlay_A = heatmap_to_overlay(img_bytes, heatmap_A)

    # ── Task B — EfficientNet-B0, hook tại features[-1] ──
    gcam_B = GradCAM(_pt_model_B, _pt_model_B.features[-1])
    t_B = img_tensor.clone().requires_grad_(True)
    heatmap_B = gcam_B.generate(t_B)

    with torch.no_grad():
        out_B = _pt_model_B(img_tensor)
        probs_B = torch.softmax(out_B.squeeze(), dim=0).numpy()
        pred_B = int(probs_B.argmax())

    overlay_B = heatmap_to_overlay(img_bytes, heatmap_B)

    return {
        "binary_label":    BINARY_LABELS[pred_A],
        "binary_conf":     round(float(prob_A if pred_A == 1 else 1 - prob_A), 4),
        "halo_label":      HALO_LABELS[pred_B] if pred_A == 1 else "N/A",
        "halo_conf":       round(float(probs_B[pred_B]), 4) if pred_A == 1 else None,
        "gradcam_task_a":  overlay_A,   # base64 PNG
        "gradcam_task_b":  overlay_B,   # base64 PNG
        "note": "Images are base64-encoded PNG. Use: <img src='data:image/png;base64,{gradcam_task_a}'>"
    }

