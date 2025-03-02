import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import autoaugment, RandomErasing
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

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
        # Spatial transforms first (on PIL Image)
        transforms.RandomResizedCrop(
            (192, 192), 
            scale=(0.25, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        
        # Color augmentations (on PIL Image)
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.1
        ),
        # RandAugment before tensor conversion
        # autoaugment.RandAugment(num_ops=2, magnitude=7),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize
        normalize,
        
        # Random erasing (works on tensor)
        RandomErasing(p=0.1, value='random')
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((192, 192)),
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
        prefetch_factor=2,  # Prefetch 2 batches per worker
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