from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset.classes
