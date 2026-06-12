"""
app.py - FastAPI serving layer cho CAD Sperm Analysis System
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import numpy as np
import cv2

from inference import (
    load_onnx_sessions, run_onnx, preprocess_image,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
)

# Load models 1 lần khi server start, không load lại mỗi request
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    sess_A, sess_B = load_onnx_sessions(
        "onnx_models/binary_resnet50.onnx",
        "onnx_models/multiclass_efficientnet_b0.onnx"
    )
    sessions["A"], sessions["B"] = sess_A, sess_B
    yield
    sessions.clear()

app = FastAPI(title="CAD Sperm Analysis API", lifespan=lifespan)


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Giống preprocess_image nhưng nhận bytes thay vì path."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không decode được ảnh")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0).astype(np.float32)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": "A" in sessions}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/bmp"):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    try:
        img_bytes = await file.read()
        img_array = preprocess_bytes(img_bytes)
        result = run_onnx(sessions["A"], sessions["B"], img_array)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))