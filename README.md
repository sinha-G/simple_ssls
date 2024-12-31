# Semi-Supervised Image Classification

A PyTorch implementation of various semi-supervised learning approaches for image classification, including:
- K-means clustering with consistency regularization
- SimCLR contrastive learning
- Standard supervised learning

## Project Structure

```
├── datasets/
│   ├── mnist.py                          # MNIST dataset loader
│   ├── cifar10.py                        # CIFAR10 dataset loader
│   └── dataset_utils.py                  # Dataset splitting utilities
├── models/
│   └── cnn.py                            # CNN model architecture
├── trainers/
│   ├── semi_supervised_base.py           # Base trainer class
│   ├── kmeans_consistency_trainer.py     # K-means with consistency
│   └── simclr_trainer.py                 # SimCLR implementation
├── checkpoints/                          # Saved model checkpoints
└── main.py                               # Training entry point
```

## Installation

```bash
pip install torch torchvision numpy scikit-learn tqdm optuna matplotlib
```

## Model Architecture

The CNN architecture consists of:

2 convolutional layers with ReLU activation and max pooling
Bottleneck linear layer (128 dimensions)
Optional dropout layer
Classification layer (10 classes)
Optional projection head for contrastive learning

## Training Approaches

K-means Consistency Training:
Combines supervised learning with:
 - K-means clustering loss on unlabeled data
 - Consistency regularization through data augmentation
SimCLR Training:
 - Contrastive learning on augmented image pairs
 - Uses NT-Xent loss and projection head
 - Supports fine-tuning on labeled data

## Datasets
The code currently supports:
 - MNIST (default)
 - CIFAR10
Data is automatically downloaded and split into labeled/unlabeled sets.

## Development
 - Python 3.6+
 - PyTorch 1.7+
CUDA support recommended

## Usage
```python
from models.cnn import CNN
from trainers.kmeans import KMeansTrainer
from datasets.mnist import get_loaders

train_loader, _ = get_mnist_loaders()
model = CNN()
trainer = KMeansTrainer(model)
trainer.train(train_loader)
```
