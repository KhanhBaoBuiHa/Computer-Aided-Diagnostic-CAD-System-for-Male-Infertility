"""
app.py - FastAPI serving layer cho CAD Sperm Analysis System
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import numpy as np
import cv2
import os
import google.generativeai as genai

from inference import (
    load_onnx_sessions, run_onnx, preprocess_image,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
)

from gradcam_explain import load_pytorch_for_gradcam, run_gradcam_explain

# Load models 1 lần khi server start, không load lại mỗi request
sessions = {}

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

@asynccontextmanager
async def lifespan(app: FastAPI):
    sess_A, sess_B = load_onnx_sessions(
        "onnx_models/binary_resnet50.onnx",
        "onnx_models/multiclass_efficientnet_b0.onnx"
    )
    sessions["A"], sessions["B"] = sess_A, sess_B

    load_pytorch_for_gradcam(
        "checkpoints/binary_best.pth",
        "checkpoints/multiclass_best.pth"
    )
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
    
# Log prediction + confidence vào CSV hoặc SQLite mỗi lần gọi API
from datetime import datetime
import time
import sqlite3

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            binary_label TEXT,
            binary_conf REAL,
            halo_label TEXT,
            halo_conf REAL,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, binary_label, binary_conf, halo_label, halo_conf, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            result["binary_label"],
            result["binary_conf"],
            result["halo_label"],
            result.get("halo_conf"),
            result["latency_ms"],
        )
    )
    conn.commit()
    conn.close()
    
@app.get("/stats")
async def stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    cur.execute("SELECT binary_label, COUNT(*) FROM predictions GROUP BY binary_label")
    binary_dist = dict(cur.fetchall())

    cur.execute("SELECT halo_label, COUNT(*) FROM predictions WHERE halo_label != 'N/A' GROUP BY halo_label")
    halo_dist = dict(cur.fetchall())

    cur.execute("SELECT AVG(latency_ms), AVG(binary_conf) FROM predictions")
    avg_latency, avg_conf = cur.fetchone()

    conn.close()
    return {
        "total_predictions": total,
        "binary_distribution": binary_dist,
        "halo_distribution": halo_dist,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
        "avg_binary_confidence": round(avg_conf, 4) if avg_conf else None,
    }

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """
    Upload ảnh → Grad-CAM heatmap overlay cho cả Task A và Task B.
    Response chứa base64-encoded PNG cho từng task.
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/bmp"):
        raise HTTPException(400, f"Unsupported: {file.content_type}")
    try:
        img_bytes = await file.read()
        return run_gradcam_explain(img_bytes)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    """
    Upload ảnh → chạy ONNX inference → Gemini API generate clinical summary.
    
    Response bao gồm:
    - Kết quả model (binary + halo classification)
    - Clinical summary do LLM generate (tiếng Anh, tone bác sĩ)
    - Latency breakdown (inference vs LLM)
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/bmp"):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
 
    try:
        # Step 1: ONNX inference (tái dụng logic từ /predict)
        img_bytes = await file.read()
        img_array = preprocess_bytes(img_bytes)
        result = run_onnx(sessions["A"], sessions["B"], img_array)
 
        # Step 2: Build prompt cho Gemini
        binary_label  = result["binary_label"]
        binary_conf   = result["binary_conf"]
        halo_label    = result.get("halo_label", "N/A")
        halo_conf     = result.get("halo_conf", 0.0)
        latency_ms    = result["latency_ms"]
 
        prompt = f"""You are a clinical assistant supporting andrologists in sperm morphology screening.
 
A CAD system analyzed a microscopy image and returned the following results:
- Task A (Binary Detection): {binary_label} — confidence {binary_conf:.1%}
- Task B (Halo Morphology):  {halo_label} — confidence {halo_conf:.1%}
- Inference latency: {latency_ms:.1f} ms
 
Generate a concise clinical summary (3–4 sentences) that:
1. States the detection outcome and confidence level
2. Interprets the halo morphology class in clinical context (DNA fragmentation risk)
3. Notes any low-confidence findings that may require manual review
4. Ends with a recommended next step for the clinician
 
Use professional medical language. Do NOT make a definitive diagnosis."""
 
        # Step 3: Gọi gemini API
        llm_start = time.time()
        response = gemini_model.generate_content(prompt)

        llm_latency_ms = (time.time() - llm_start) * 1000
        clinical_summary = response.text
 
        # Step 4: Log vào DB (tái dụng hàm có sẵn)
        log_prediction(result)
 
        return {
            # Model results
            "binary_label":       binary_label,
            "binary_conf":        binary_conf,
            "halo_label":         halo_label,
            "halo_conf":          halo_conf,
            "inference_ms":       latency_ms,
            # LLM output
            "clinical_summary":   clinical_summary,
            "llm_latency_ms":     round(llm_latency_ms, 1),
            "llm_model":          "gemini-2.5-flash",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(502, f"Gemini API error: {str(e)}")

    """
    except Exception as e:
        raise HTTPException(502, f"Gemini API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, str(e))
    """
