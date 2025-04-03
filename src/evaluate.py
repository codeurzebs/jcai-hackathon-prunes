import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

# 🔧 Configurations
DATA_DIR = "data/processed"
MODEL_PATH = "models/plum_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 32

# 📦 Transformations (comme à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# 📂 Chargement des données de validation
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 🔢 Classes
class_names = val_dataset.classes
num_classes = len(class_names)

# 🧠 Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🔍 Évaluation
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 📈 Rapport de classification
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n📊 Résultats sur l'ensemble de validation :\n")
print(report)
