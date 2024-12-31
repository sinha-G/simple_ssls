# datasets/mnist.py
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
