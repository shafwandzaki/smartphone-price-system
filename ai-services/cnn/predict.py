import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# Urutan kelas asli sesuai _classes.csv (jangan diubah urutannya)
RAW_CLASSES = ["Multi fissure", "fissure", "impact"]

# Kamus terjemahan ke Bahasa Indonesia untuk tampilan output
LABEL_ID = {
    "Multi fissure": "Retak Parah",
    "fissure": "Retak Halus",
    "impact": "Penyok / Benturan"
}

# Bobot penalti maksimal untuk masing-masing jenis kerusakan
PENALTY_WEIGHTS = {
    "Multi fissure": 55.0,
    "fissure": 30.0,
    "impact": 20.0
}

def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(RAW_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def calculate_condition_score(probs_dict, detected_raw_labels):
    """
    Menghitung skor kondisi fisik (0 - 100) dari probabilitas Sigmoid
    dan menentukan kategori kualitas perangkat.
    """
    total_penalty = 0.0
    for raw_label, prob in probs_dict.items():
        # Penalti penuh jika terdeteksi (>= 0.5), penalti mikro (0.15) jika di bawah threshold
        factor = 1.0 if raw_label in detected_raw_labels else 0.15
        total_penalty += prob * PENALTY_WEIGHTS[raw_label] * factor

    score = max(0.0, min(100.0, 100.0 - total_penalty))
    score = round(score, 2)

    if score >= 90.0:
        category = "Sangat Baik"
    elif score >= 75.0:
        category = "Baik"
    elif score >= 50.0:
        category = "Sedang"
    else:
        category = "Rusak Berat"

    return score, category

def predict_image(image_path):
    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    probs_dict = {RAW_CLASSES[i]: float(probs[i]) for i in range(len(RAW_CLASSES))}
    detected_raw = [label for label, p in probs_dict.items() if p >= 0.5]
    detected_id = [LABEL_ID[label] for label in detected_raw]
    score, category = calculate_condition_score(probs_dict, detected_raw)

    print("\n" + "=" * 58)
    print(f"   HASIL ANALISIS KONDISI FISIK: {os.path.basename(image_path)}")
    print("=" * 58)
    for raw_label, p in probs_dict.items():
        nama_id = LABEL_ID[raw_label]
        status = "[TERDETEKSI]" if p >= 0.5 else "[TIDAK TERDETEKSI]"
        print(f" - {nama_id:<22} : {p*100:6.2f}%  {status}")
    print("-" * 58)
    print(f"KERUSAKAN TERDETEKSI  : {', '.join(detected_id) if detected_id else 'Tidak Ada (Mulus)'}")
    print(f"SKOR KONDISI FISIK    : {score} / 100")
    print(f"KATEGORI KUALITAS     : {category.upper()}")
    print("=" * 58 + "\n")

    return {
        "detail_probabilitas": {LABEL_ID[k]: round(v * 100, 2) for k, v in probs_dict.items()},
        "kerusakan_terdeteksi": detected_id,
        "skor_kondisi": score,
        "kategori_kualitas": category
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python predict.py <path_gambar>")
        sys.exit(1)
    predict_image(sys.argv[1])