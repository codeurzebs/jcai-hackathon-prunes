import torch
from torchvision import models, transforms
from PIL import Image

# Liste des classes
CLASS_NAMES = ['bruised', 'cracked', 'rotten', 'spotted', 'unaffected', 'unripe']
MODEL_PATH = "models/plum_model.pth"  # relatif à la racine du projet

# Chargement du modèle
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# Prédiction d'une image
def predict_image(model, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Ajouter batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]
