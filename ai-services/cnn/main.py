"""API klasifikasi kondisi fisik smartphone dari foto (port 8002).

Membungkus predict.py milik Mahasiswa 2: RAW_CLASSES, LABEL_ID, load_model, dan
calculate_condition_score dipakai langsung dari predict.py, jadi jika rumus skor di sana
diubah, API ikut berubah tanpa mengubah file ini.

Jalankan:  uvicorn main:app --reload --port 8002
Dokumentasi interaktif: http://127.0.0.1:8002/docs
"""
import io
import os
from datetime import datetime

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import predict as P          # kode teman Anda (tidak diubah)

MAX_UPLOAD_MB = 8
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIN_SIDE_PX = 200
THRESHOLD = 0.5              # sama dengan predict.py
UNCERTAIN_MARGIN = 0.10
KEYS = {"Multi fissure": "multi_fissure", "fissure": "fissure", "impact": "impact"}
SCOPE_NOTE = ("Model dilatih pada foto retak/benturan layar. Goresan halus, bodi belakang, "
              "bingkai, baterai, dan kerusakan internal tidak dinilai.")

app = FastAPI(title="API Klasifikasi Kondisi Fisik Smartphone", version="cnn-api-1.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, LOAD_ERROR, MODEL_VERSION = None, None, None
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

try:
    if os.path.exists(P.MODEL_PATH):
        MODEL = P.load_model(DEVICE)
        MODEL_VERSION = "cnn-" + datetime.fromtimestamp(os.path.getmtime(P.MODEL_PATH)).strftime("%Y%m%d%H%M")
    else:
        LOAD_ERROR = f"File model tidak ditemukan: {P.MODEL_PATH}"
except Exception as e:                      # checkpoint rusak / arsitektur tidak cocok
    LOAD_ERROR = str(e)

MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _score_cliff() -> float:
    """Selisih skor antara probabilitas 0,49 dan 0,50 pada satu kelas (deteksi lompatan di ambang)."""
    c = P.RAW_CLASSES[0]
    zero = {k: 0.0 for k in P.RAW_CLASSES}
    a = P.calculate_condition_score({**zero, c: THRESHOLD - 0.01}, [])[0]
    b = P.calculate_condition_score({**zero, c: THRESHOLD}, [c])[0]
    return abs(a - b)


@app.get("/health")
def health():
    if MODEL is None:
        return {"status": "ok", "model_loaded": False, "error": LOAD_ERROR}
    warnings = ["Kualitas model belum divalidasi (lihat evaluate.py).", SCOPE_NOTE]
    cliff = _score_cliff()
    if cliff > 5:
        warnings.append(f"Rumus skor di predict.py melompat {cliff:.1f} poin di ambang {THRESHOLD}; "
                        "skor bisa berubah drastis untuk perubahan probabilitas yang kecil.")
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION, "device": str(DEVICE),
            "classes": [KEYS[c] for c in P.RAW_CLASSES], "peringatan_model": warnings}


@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, f"Model belum tersedia. {LOAD_ERROR or ''}".strip())
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(422, f"Tipe file '{file.content_type}' tidak didukung. Gunakan JPEG, PNG, atau WebP "
                                 "(foto HEIC dari iPhone: ekspor dulu ke JPEG).")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Ukuran file melebihi {MAX_UPLOAD_MB} MB.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, "File bukan gambar yang valid atau rusak.")

    with torch.inference_mode():
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(MODEL(x)).squeeze(0).cpu().numpy()
    probs_dict = {c: float(probs[i]) for i, c in enumerate(P.RAW_CLASSES)}
    detected = [c for c, p in probs_dict.items() if p >= THRESHOLD]
    score, category = P.calculate_condition_score(probs_dict, detected)

    warnings = []
    w, h = img.size
    if min(w, h) < MIN_SIDE_PX:
        warnings.append(f"Resolusi foto rendah ({w}x{h}px); hasil kurang andal.")
    ragu = [KEYS[c] for c, p in probs_dict.items() if abs(p - THRESHOLD) < UNCERTAIN_MARGIN]
    if ragu:
        warnings.append(f"Model ragu pada: {ragu} (probabilitas dekat ambang {THRESHOLD}); "
                        "skor bisa berubah drastis. Coba foto yang lebih jelas.")
    return {
        "skor_kondisi": score,                       # 0-100, 100 = terbaik -> masuk ke Fuzzy
        "kategori_kualitas": category,
        "probabilitas": {KEYS[c]: round(p, 4) for c, p in probs_dict.items()},
        "kerusakan_terdeteksi": [P.LABEL_ID[c] for c in detected],
        "peringatan": warnings,
        "cakupan": SCOPE_NOTE,
        "model_version": MODEL_VERSION,
    }
