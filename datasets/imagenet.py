import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_imagenet_loaders(batch_size=128, num_workers=0):
    """
    Creates data loaders for ImageNet dataset.
    Assumes dataset is in ./data/imagenet directory
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((192, 192), scale = (0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageNet(
        'data/imagenet',
        split='train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageNet(
        'data/imagenet', 
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True
    )

    return train_loader, val_loader