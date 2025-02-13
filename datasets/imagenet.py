import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import autoaugment, RandomErasing
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
        transforms.RandomResizedCrop((196, 196), scale = (0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        autoaugment.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.1
        ),
        transforms.ToTensor(),
        normalize,
        RandomErasing(p=0.1, value='random')
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((196, 196)),
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