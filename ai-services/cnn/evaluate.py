import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score

# 1. Konfigurasi Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Multi fissure', 'fissure', 'impact']  # Sesuai label Roboflow kamu
NUM_CLASSES = len(CLASSES)

# 2. Re-build Arsitektur ResNet-18 Multi-Label
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

# Load Bobot Terbaik dari Pelatihan
model_path = os.path.join("models", "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File {model_path} tidak ditemukan! Pastikan train.py sudah pernah dirun.")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 3. Transformasi Data Test (Harus persis sama dengan Validation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Custom Dataset Loader untuk Data Test
class DefectTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        # Strip spasi pada nama kolom untuk menghindari mismatch
        self.df.columns = [c.strip() for c in self.df.columns]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        labels = row[CLASSES].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)

# Path Dataset Test
test_csv = os.path.join("dataset", "test", "_classes.csv")
test_dir = os.path.join("dataset", "test")

test_dataset = DefectTestDataset(test_csv, test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Eksekusi Evaluasi
all_preds = []
all_targets = []

print(f"Menguji {len(test_dataset)} citra data uji (test set)...")

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        # Sigmoid activation + Threshold 0.5 untuk Multi-Label
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int().cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(targets.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# 6. Cetak Hasil Evaluasi & Metrik
print("\n" + "="*50)
print("             HASIL EVALUASI MODEL (DATA TEST)            ")
print("="*50)

# Classification Report Per Kelas
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(all_targets, all_preds, target_names=CLASSES, zero_division=0))

# Subset Accuracy (Akurasi Persis Sama Semua Label)
exact_match_acc = accuracy_score(all_targets, all_preds)
print(f"Subset Accuracy (Exact Match): {exact_match_acc * 100:.2f}%")

# Multilabel Confusion Matrix per Kelas
print("\n--- CONFUSION MATRIX PER KELAS ---")
mcm = multilabel_confusion_matrix(all_targets, all_preds)
for i, class_name in enumerate(CLASSES):
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\nKelas: [{class_name}]")
    print(f"  True Negative (TN) : {tn} | False Positive (FP): {fp}")
    print(f"  False Negative (FN): {fn} | True Positive (TP) : {tp}")

print("\n" + "="*50)