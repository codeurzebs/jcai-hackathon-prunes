import torch
from torch import nn, optim
from src.model import PlumClassifier
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres
data_dir = "data/processed"
num_classes = 6
epochs = 10
lr = 0.001

# Load data
train_loader, val_loader, classes = get_dataloaders(data_dir)
model = PlumClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Entraînement
for epoch in range(epochs):
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

    print(f"[{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# Sauvegarder le modèle
torch.save(model.state_dict(), "models/plum_model.pth")
