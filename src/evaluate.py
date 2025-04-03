import torch
from sklearn.metrics import classification_report
from src.model import PlumClassifier
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data/processed"
model = PlumClassifier(num_classes=6)
model.load_state_dict(torch.load("models/plum_model.pth"))
model.to(device)
model.eval()

_, val_loader, classes = get_dataloaders(data_dir)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=classes))
