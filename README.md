# Semi-Supervised Image Classification

The purpose of this repository is to develop a modular framework for benchmarking and comparing common techniques and architectures for semi-supervised image classification. Key features are: multiple SSL approaches in one codebase, standardized evaluations, configurable architecture and hyperparameters, and detailed logging and visualizations.
- K-means clustering with consistency regularization
- SimCLR contrastive learning
- DINO self-supervised learning
- Standard supervised learning
- ViT implementation

## Project Structure

```
├── datasets/
│   ├── mnist.py                          # MNIST dataset loader
│   ├── cifar10.py                        # CIFAR10 dataset loader
│   ├── imagenet.py                       # Imagenet dataset loader
│   ├── imagenette.py                     # Imagenette dataset loader
│   └── dataset_utils.py                  # Dataset splitting utilities
├── models/
|   ├── vit.py                            # ViT Implementation
│   └── cnn.py                            # CNN model architecture
├── trainers/
│   ├── semi_supervised_base.py           # Base trainer class
│   ├── kmeans_consistency_trainer.py     # K-means with consistency
│   └── simclr_trainer.py                 # SimCLR implementation
└── main.py                               # Training entry point
```

## Model Architecture

### CNN Architecture

- 2 convolutional layers with ReLU activation and max pooling
- Linear layer
- Optional dropout layer
- Classification head
- Optional projection head for contrastive learning

### Vision Transformer (ViT)

- Configurable patch size and number of patches
- Multi-head self-attention
- Learnable class token for classification
- Positional embeddings
- Optional projection head for self-supervised learning
- Weight initialization matching the original ViT paper

## Training Approaches

### K-means Consistency Training

- Combines supervised learning with:
  - K-means clustering loss on unlabeled data
  - Consistency regularization through data augmentation

### SimCLR Training

- Contrastive learning on augmented image pairs
- Uses NT-Xent loss and projection head
- Supports fine-tuning on labeled data

### DINO Training

- Self-distillation with no labels
- Teacher-student architecture
- Momentum encoder
- Centering and sharpening of the teacher outputs
- Supports fine-tuning with label smoothing and weight decay

## Visualization Tools

- Attention map visualization for transformers
- Class activation mapping (CAM)
- t-SNE visualization of class embeddings across layers

## Datasets

The code currently supports:
- MNIST
- CIFAR10
- ImageNet
- ImageNette

Data is automatically downloaded and split into labeled/unlabeled sets.

## Development
 - Python 3.6+
 - PyTorch 1.7+
 - CUDA recommended

## Installation and Usage

```bash
pip install torch torchvision numpy scikit-learn tqdm optuna matplotlib
```

```python
from models.vit import ViT
from trainers.dino_trainer import DINOTrainer
from datasets.imagenet import get_imagenet_loaders

# Vision Transformer with DINO
train_loader, test_loader = get_imagenet_loaders(batch_size=64)
model = ViT(chw=(3, 192, 192), n_patches=12, n_blocks=12, hidden_d=384, n_heads=6, num_classes=1000)
trainer = DINOTrainer(model)

# Self-supervised pre-training
trainer.train(train_loader, epochs=100)

# Supervised fine-tuning with optimal settings
trainer.finetune(train_loader, test_loader, epochs=100)

# Visualize attention maps
model.visualize_attention(images, layer_idx=-1, head_idx=0)

# Visualize class activation maps
model.visualize_cam(images, target_class=None)
```
