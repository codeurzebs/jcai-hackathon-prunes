# ===============================================================
# JCIA HACKATHON 2025 - TRI INTELLIGENT DES PRUNES AFRICAINES 🍑
# Auteurs : ZEBS HAUPUR & TIOJIO ROMAIN
# Description : Projet d’IA pour la classification automatique
#               des prunes selon leur qualité visuelle.
# Technologies : PyTorch | Streamlit | Azure ML | Python
# GitHub : https://github.com/NGcodeX/jcai-hackathon-prunes
# Tel: +237692077005
# Communauté: NGcodeX
# ===============================================================

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

#  Configuration
DATA_DIR = "data/processed"
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "models/plum_model.pth"

# 🖥 Device (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Utilisation de : {device}")

#  Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

#  Chargement des données
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

#  Nombre de classes
num_classes = len(train_dataset.classes)
print(f"📊 Classes détectées : {train_dataset.classes} ({num_classes})")

#  Modèle simple (ResNet18 pré-entraîné)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

#  Fonction de perte & Optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#  Entraînement
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"📈 Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {running_loss:.4f}")

#  Sauvegarde du modèle
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Modèle sauvegardé dans {MODEL_PATH}")


# ===============================================================
# JCIA HACKATHON 2025 - TRI INTELLIGENT DES PRUNES AFRICAINES 🍑
# Auteurs : ZEBS HAUPUR & TIOJIO ROMAIN
# Description : Projet d’IA pour la classification automatique
#               des prunes selon leur qualité visuelle.
# Technologies : PyTorch | Streamlit | Azure ML | Python
# GitHub : https://github.com/NGcodeX/jcai-hackathon-prunes
# Tel: +237692077005
# Communauté: NGcodeX
# ===============================================================
