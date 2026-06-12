# CAD Sperm Analysis System

> ResNet50 (binary) + EfficientNet-B0 (halo classification) với ONNX deployment & inference benchmarking.

---

## Project Structure

```
cad-sperm/
├── main.ipynb              # Training pipeline (GD1–GD4)
├── export_onnx.py          # Export PyTorch → ONNX + benchmark
├── inference.py            # Production inference script (ONNX / PyTorch)
├── onnx_models/
│   ├── binary_resnet50.onnx
│   └── multiclass_efficientnet_b0.onnx
└── README.md
```

---

## Quick Start

### 1. Cài dependencies

```bash
pip install torch torchvision onnx onnxruntime opencv-python numpy
```

### 2. Export model sang ONNX

```bash
python export_onnx.py \
    --binary_pth  binary_best.pth \
    --multi_pth   multiclass_best.pth \
    --output_dir  onnx_models \
    --n_runs      200
```

### 3. Inference trên một ảnh

```bash
# ONNX backend (khuyến nghị)
python inference.py --image path/to/sperm.png

# PyTorch backend
python inference.py --image path/to/sperm.png --backend pytorch
```

**Output mẫu:**

```
──────────────────────────────────────────────────
  Image   : 001.PNG
  Backend : ONNX Runtime
──────────────────────────────────────────────────
  [Task A] Sperm         confidence: 0.9821
  [Task B] Large halo    confidence: 0.8734
  Latency : 12.43 ms
──────────────────────────────────────────────────
```

### 4. Batch benchmark

```bash
python inference.py --image_dir data/test/ --benchmark
```

---

## Inference Benchmark (CPU — Intel Core i5, n=200 runs)

> Kết quả đo trên CPU để phản ánh deployment thực tế (không phụ thuộc GPU).  
> Chạy `export_onnx.py` để tự sinh bảng này cho máy của bạn.

| Backend | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Throughput (FPS) |
|---------|-----------|----------|----------|----------|-----------------|
| **PyTorch (CPU)** | 219.11 | 41.13 | 177.94 | 347.96 | 4.6 |
| **ONNX Runtime (CPU)** | 144.76 | 44.54 | 115.05 | 359.23 | 6.9 |
| **Speedup** | **1.51x** | | | | |


> ONNX nhanh hơn PyTorch CPU **1.5** nhờ graph optimization và constant folding.

---

## Model Architecture

| Task | Backbone | Head | Output | Loss |
|------|----------|------|--------|------|
| A — Binary (Sperm / Non-sperm) | ResNet-50 (frozen) | Dropout → FC(2048→256) → ReLU → Dropout → FC(256→1) | 1 logit | BCEWithLogitsLoss |
| B — Halo Classification (4 lớp) | EfficientNet-B0 (frozen) | Dropout → FC(1280→256) → ReLU → Dropout → FC(256→4) | 4 logits | CrossEntropyLoss (weighted) |

---

## Training Config

| Param | Task A (Binary) | Task B (Multiclass) |
|-------|----------------|---------------------|
| Optimizer | Adam | Adam |
| LR | 1e-4 | 5e-5 |
| Batch size | 32 | 32 |
| Max epochs | 20 | 30 |
| Early stopping patience | 5 | 7 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

---

---

## Explainability — Grad-CAM

Model sử dụng Grad-CAM để visualize vùng ảnh ảnh hưởng đến quyết định:
- **Task A**: hook tại `model.layer4[-1]` (ResNet50 last conv block)
- **Task B**: hook tại `model.features[-1]` (EfficientNet-B0 last feature layer)

---

## Deployment Notes

- ONNX export dùng `opset_version=17`, `do_constant_folding=True`
- Dynamic batch size được bật (`dynamic_axes`) — hỗ trợ batch inference
- Inference script tự động chọn `CUDAExecutionProvider` nếu có GPU, fallback về CPU
- Preprocessing pipeline trong `inference.py` mirror chính xác GD1 của notebook (bilateral denoise + CLAHE + ImageNet normalize)
