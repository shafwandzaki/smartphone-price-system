import os
import sys
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 1. Konfigurasi Device & Label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Multi fissure', 'fissure', 'impact']
NUM_CLASSES = len(CLASSES)
THRESHOLD = 0.5

# 2. Re-build Arsitektur ResNet-18 & Load Bobot Terbaik
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

model_path = os.path.join("models", "best_model.pth")
if not os.path.exists(model_path):
    print(f"[ERROR] File '{model_path}' tidak ditemukan! Jalankan train.py terlebih dahulu.")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 3. Transformasi Citra
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def determine_severity(detected_dict):
    """
    Menentukan tingkat kerusakan berdasarkan label yang terdeteksi dan skor probabilitas.
    """
    num_detected = len(detected_dict)
    
    if num_detected == 0:
        return "Sangat Baik"
    
    prob_multi = detected_dict.get('Multi fissure', 0.0)
    prob_fissure = detected_dict.get('fissure', 0.0)
    prob_impact = detected_dict.get('impact', 0.0)
    
    # Kondisi Rusak Berat
    if num_detected >= 3 or (prob_multi >= 0.8 and prob_impact >= 0.8):
        return "Rusak Berat"
    
    # Kondisi Rusak Ringan
    if num_detected == 2:
        return "Rusak Ringan"
    
    # Kondisi Sedang vs Baik
    if num_detected == 1:
        if 'fissure' in detected_dict and prob_fissure < 0.8:
            return "Baik"
        else:
            return "Sedang"
            
    return "Sedang"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File gambar '{image_path}' tidak ditemukan!")
        return

    # Load dan Preprocess Gambar
    image = Image.open(image_path).convert('RGB')
    input_tensor = predict_transform(image).unsqueeze(0).to(device)

    # Inferensi Prediksi
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    detected_dict = {}
    print("\n" + "="*50)
    print(f"   HASIL PREDIKSI UNSEEN IMAGE: {os.path.basename(image_path)}")
    print("="*50)
    
    for i, class_name in enumerate(CLASSES):
        prob = float(probs[i])
        status = "TERDETEKSI" if prob >= THRESHOLD else "TIDAK TERDETEKSI"
        print(f" - {class_name:<15} : {prob*100:6.2f}%  [{status}]")
        if prob >= THRESHOLD:
            detected_dict[class_name] = prob

    severity = determine_severity(detected_dict)

    print("-" * 50)
    if detected_dict:
        defects_str = ', '.join(detected_dict.keys())
        print(f"CACAT TERDETEKSI  : {defects_str}")
    else:
        print("CACAT TERDETEKSI  : Tidak ada (Kondisi Fisik Mulus)")
        
    print(f"TINGKAT KERUSAKAN : {severity.upper()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "hp_bekas_test.jpg" 
        
    predict_image(img_path)