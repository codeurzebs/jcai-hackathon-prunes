import torch
from PIL import Image
from torchvision import transforms
from src.model import PlumClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(image_path, model_path="models/plum_model.pth", class_names=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    model = PlumClassifier(num_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred] if class_names else str(pred)

# Exemple d’utilisation
if __name__ == "__main__":
    prediction = predict("data/sample.jpg", class_names=["Bonne", "Non mûre", "Tachetée", "Fissurée", "Meurtrie", "Pourrie"])
    print("Classe prédite :", prediction)
