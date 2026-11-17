import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset_loader import PhoneDefectDataset, train_transforms, val_transforms

def main():
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Device: {device}")

    # 2. Path & Dataset Setup
    train_csv = os.path.join("dataset", "train", "_classes.csv")
    train_img_dir = os.path.join("dataset", "train")
    val_csv = os.path.join("dataset", "valid", "_classes.csv")
    val_img_dir = os.path.join("dataset", "valid")

    train_dataset = PhoneDefectDataset(csv_file=train_csv, img_dir=train_img_dir, transform=train_transforms)
    val_dataset = PhoneDefectDataset(csv_file=val_csv, img_dir=val_img_dir, transform=val_transforms)

    # 3. DataLoader Setup
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 4. Model Setup (ResNet-18 Multi-Label)
    CLASSES = train_dataset.classes
    NUM_CLASSES = len(CLASSES)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    os.makedirs("models", exist_ok=True)
    best_val_loss = float('inf')

    print("Memulai Pelatihan Model CNN...")
    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch [{epoch}/20] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join("models", "best_model.pth"))
            print("  --> Model terbaik tersimpan di 'models/best_model.pth'")

    print("Pelatihan Selesai!")

if __name__ == '__main__':
    main()