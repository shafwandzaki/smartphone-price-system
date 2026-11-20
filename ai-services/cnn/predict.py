import os
import sys
import random
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

RAW_CLASSES = ["Multi fissure", "fissure", "impact"]

LABEL_ID = {
    "Multi fissure": "Retak Parah / Ganda",
    "fissure": "Retak Halus / Tunggal",
    "impact": "Penyok / Benturan"
}

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
    total_penalty = 0.0
    for raw_label, prob in probs_dict.items():
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

def get_ground_truth(image_path):
    """Mencari label asli jika foto berasal dari folder dataset (train/valid/test)."""
    parent_dir = os.path.dirname(image_path)
    csv_path = os.path.join(parent_dir, "_classes.csv")
    fname = os.path.basename(image_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        row = df[df["filename"].str.strip() == fname]
        if not row.empty:
            sample = row.iloc[0]
            true_labels = [LABEL_ID[col] for col in RAW_CLASSES if sample[col] == 1]
            return ", ".join(true_labels) if true_labels else "Tidak Ada (Mulus)"
    return None

def select_via_explorer():
    """Membuka jendela File Explorer Windows untuk memilih gambar."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_dir = os.path.join(BASE_DIR, "dataset", "test")
    if not os.path.exists(initial_dir):
        initial_dir = BASE_DIR

    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Pilih Gambar Smartphone untuk Dianalisis",
        filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.webp"), ("Semua File", "*.*")]
    )
    root.destroy()
    return file_path

def select_from_test_list():
    """Menampilkan daftar bernomor dari folder dataset/test di terminal."""
    test_dir = os.path.join(BASE_DIR, "dataset", "test")
    images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not images:
        print("Tidak ada gambar di dalam folder dataset/test.")
        return None

    print("\n--- DAFTAR GAMBAR DI DATASET/TEST ---")
    for idx, fname in enumerate(images, start=1):
        full_p = os.path.join(test_dir, fname)
        gt = get_ground_truth(full_p)
        gt_info = f" | Asli: {gt}" if gt else ""
        print(f" [{idx:2d}] {fname}{gt_info}")

    while True:
        choice = input(f"\nMasukkan nomor gambar (1 - {len(images)}) atau '0' untuk batal: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            return os.path.join(test_dir, images[int(choice) - 1])
        print("Input tidak valid. Masukkan angka yang sesuai.")

def predict_image(image_path):
    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    ground_truth = get_ground_truth(image_path)
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
    if ground_truth is not None:
        print(f"LABEL ASLI (DATASET)  : {ground_truth}")
        print("-" * 58)
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
    # Jika argumen file langsung diberikan lewat CLI (contoh: python predict.py dataset/test/foto.jpg)
    if len(sys.argv) >= 2:
        predict_image(sys.argv[1])
    else:
        while True:
            print("\n=== MENU DETEKSI KONDISI FISIK SMARTPHONE ===")
            print("1. Pilih gambar lewat File Explorer (Windows)")
            print("2. Pilih gambar dari daftar folder dataset/test")
            print("3. Keluar")
            pilihan = input("Pilih menu (1/2/3): ").strip()

            if pilihan == "1":
                selected_file = select_via_explorer()
                if selected_file:
                    predict_image(selected_file)
                else:
                    print("Pemilihan gambar dibatalkan.")
            elif pilihan == "2":
                selected_file = select_from_test_list()
                if selected_file:
                    predict_image(selected_file)
            elif pilihan == "3":
                print("Keluar dari program.")
                break
            else:
                print("Pilihan tidak valid, silakan masukkan 1, 2, atau 3.")