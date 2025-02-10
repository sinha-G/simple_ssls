import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_imagenette_loaders(batch_size=128, num_workers=0):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((160, 160), scale = (0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        normalize,
    ])

    imagenette_train = datasets.Imagenette('./data', split="train", size = "160px", download=True, transform=train_transform)
    imagenette_test = datasets.Imagenette('./data', split="val", size = "160px", download=True, transform=val_transform)

    train_loader = DataLoader(imagenette_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(imagenette_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader