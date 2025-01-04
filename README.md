# Semi-Supervised Image Classification

The purpose of this repository is to develop a modular framework for benchmarking and comparing common techniques and architectures for semi-supervised image classification. Key features are: multiple SSL approaches in one codebase, standardized evaluations, configurable architecture and hyperparameters, and detailed logging and visualizations.
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
└── main.py                               # Training entry point
```

## Model Architecture

The CNN architecture consists of:

- 2 convolutional layers with ReLU activation and max pooling
- Linear layer
- Optional dropout layer
- Classification head
- Optional projection head for contrastive learning

The ViT architecture allows for a variable patch size, input size (C, H, W), hidden dimension, number of attention heads, number of ViT blocks, and output dimension.

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

## Installation and Usage

```bash
pip install torch torchvision numpy scikit-learn tqdm optuna matplotlib
```

```python
from models.cnn import CNN
from trainers.kmeans import SimCLRTrainer
from datasets.mnist import get_mnist_loaders

train_loader, _ = get_mnist_loaders()
model = CNN()
trainer = SimCLRTrainer(model)
trainer.train(train_loader)
```
