import os
import torch
from torchvision import transforms, models
from PIL import Image

# 🔧 Configuration
MODEL_PATH = "models/plum_model.pth"
IMG_SIZE = 224
CLASS_NAMES = ['bruised', 'cracked', 'rotten', 'spotted', 'unaffected', 'unripe']

# 📦 Préparation de l'image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # ajout d'une dimension batch

# 🧠 Charger le modèle
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# 🔮 Prédiction
def predict(image_path):
    image_tensor = load_image(image_path)
    model, device = load_model()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]

    print(f"🖼️ Image : {os.path.basename(image_path)}")
    print(f"🔮 Prédiction : {predicted_class}")

# 🏁 Exemple d'utilisation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Chemin de l'image à prédire")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("❌ Le fichier image n'existe pas.")
    else:
        predict(args.image)
