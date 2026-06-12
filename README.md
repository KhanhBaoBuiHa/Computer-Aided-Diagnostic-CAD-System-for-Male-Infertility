# CAD Sperm Analysis System

> ResNet50 (binary) + EfficientNet-B0 (halo classification) với ONNX deployment, inference benchmarking, và FastAPI serving.

---

## Project Structure

```
tinh-trung/
├── main.ipynb              # Training pipeline (GD1–GD4)
├── export_onnx.py          # Export PyTorch → ONNX + benchmark
├── inference.py            # Inference helpers (ONNX / PyTorch)
├── app.py                  # FastAPI serving layer
├── checkpoints/
│   ├── binary_best.pth
│   └── multiclass_best.pth
├── onnx_models/
│   ├── binary_resnet50.onnx
│   └── multiclass_efficientnet_b0.onnx
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Export model sang ONNX

```bash
python export_onnx.py \
    --binary_pth  checkpoints/binary_best.pth \
    --multi_pth   checkpoints/multiclass_best.pth \
    --output_dir  onnx_models
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

### 5. Chạy API server

```bash
uvicorn app:app --reload
```

Mở `http://127.0.0.1:8000/docs` để test endpoint `/predict` qua Swagger UI.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sperm.png"
```

---

## Model Architecture

| Task | Backbone | Head | Output | Loss |
|------|----------|------|--------|------|
| A — Binary (Sperm / Non-sperm) | ResNet-50 | Dropout → FC(2048→256) → ReLU → Dropout → FC(256→1) | 1 logit | BCEWithLogitsLoss |
| B — Halo Classification (4 lớp) | EfficientNet-B0 | Dropout → FC(1280→256) → ReLU → Dropout → FC(256→4) | 4 logits | CrossEntropyLoss |

---

## Training Config

| Param | Task A (Binary) | Task B (Multiclass) |
|-------|----------------|---------------------|
| Optimizer | Adam | Adam |
| LR | 1e-4 | 5e-5 |
| Batch size | 32 | 32 |
| Max epochs | 20 | 30 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

---

## Results vs Baseline

| Metric | Our Model | Majority Vote | Random |
|--------|-----------|--------------|--------|
| **Task A — Accuracy** | **0.94** | – | – |
| **Task B — Accuracy** | **0.5574** | – | – |
| **Task B — Macro F1** | **0.5604** | – | – |

### Task B — Per-class (Halo Grading)

| Class | Precision | Recall |
|-------|-----------|--------|
| Large | 0.667 | 0.625 |
| Medium | 0.364 | 0.500 |
| Small | 0.500 | 0.357 |
| No Halo | 0.786 | 0.733 |

> Task B đạt accuracy 55.74%, thấp hơn nhiều so với Task A (94%). Nguyên nhân chủ yếu là ranh giới mờ giữa các lớp halo (Large/Medium/Small), và backbone freeze nên chưa học được đặc trưng tinh tế. Hướng cải thiện: unfreeze thêm các layer cuối backbone, dùng weighted loss / oversampling cho các lớp ít mẫu.

---

## Inference Benchmark (CPU, n=200 runs)

| Backend | Binary (ResNet50) | Multiclass (EfficientNet-B0) |
|---------|-------------------|-------------------------------|
| PyTorch (CPU) | 87.4 ms | 34.4 ms |
| ONNX Runtime (CPU) | 36.8 ms | 8.7 ms |
| **Speedup** | **2.37x** | **3.97x** |

ONNX export dùng `opset_version=17` (yêu cầu package `onnxscript`), `do_constant_folding=True`, dynamic batch size.

---

## Explainability — Grad-CAM

Model sử dụng Grad-CAM để visualize vùng ảnh ảnh hưởng đến quyết định:
- **Task A**: hook tại `model.layer4[-1]` (ResNet50 last conv block)
- **Task B**: hook tại `model.features[-1]` (EfficientNet-B0 last feature layer)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Kiểm tra trạng thái server và model đã load |
| POST | `/predict` | Upload ảnh (png/jpg/bmp) → trả về kết quả binary + halo classification |

**Response mẫu `/predict`:**
```json
{
  "binary_label": "Sperm",
  "binary_conf": 0.9821,
  "halo_label": "Large halo",
  "halo_conf": 0.8734,
  "latency_ms": 12.43,
  "backend": "ONNX Runtime"
}
```

---

## Deployment Notes

- Preprocessing pipeline trong `inference.py`/`app.py` mirror chính xác GD1 của notebook (bilateral denoise + CLAHE + ImageNet normalize)
- Inference script tự động chọn `CUDAExecutionProvider` nếu có GPU, fallback về `CPUExecutionProvider`
- Models được load 1 lần khi server start (qua `lifespan`), không load lại mỗi request
