# datasets/__init__.py

from .mnist import get_mnist_loaders
from .cifar10 import get_cifar10_loaders
from .dataset_utils import split_dataset

__all__ = ['get_mnist_loaders', 'get_cifar10_loaders', 'split_dataset']
