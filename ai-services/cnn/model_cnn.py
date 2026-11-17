import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, pretrained=True):
    # Menggunakan ResNet-18 Pre-trained (Standar & Cepat)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Ganti layer Fully Connected (fc) terakhir
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
        # Catatan: Tidak perlu Sigmoid() di sini jika menggunakan nn.BCEWithLogitsLoss
    )
    
    return model

if __name__ == "__main__":
    model = get_model(num_classes=3)
    x = torch.randn(16, 3, 224, 224) # Dummy input 1 batch
    output = model(x)
    print(f"Bentuk Output Model: {output.shape}") # Output: torch.Size([16, 3])