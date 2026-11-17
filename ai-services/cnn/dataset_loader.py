import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 1. Transformasi Data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Base Directory (Mendapatkan path absolut folder 'cnn')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PhoneDefectDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Konversi ke path absolut agar aman dari mana saja script dipanggil
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(BASE_DIR, csv_file)
            
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file tidak ditemukan di: {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        self.df.columns = [c.strip() for c in self.df.columns]
        
        if not os.path.isabs(img_dir):   # <--- TAMBAHKAN .path DI SINI JUGA
            self.img_dir = os.path.join(BASE_DIR, img_dir)
        else:
            self.img_dir = img_dir
            
        self.transform = transform
        self.classes = ['Multi fissure', 'fissure', 'impact']
        self.labels = self.df[self.classes].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Bersihkan nama file dari sisa path/backslash Windows
        raw_filename = str(row['filename']).strip()
        clean_filename = os.path.basename(raw_filename)
        
        img_path = os.path.join(self.img_dir, clean_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {img_path}")

        image = Image.open(img_path).convert('RGB')
        labels = row[self.classes].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)