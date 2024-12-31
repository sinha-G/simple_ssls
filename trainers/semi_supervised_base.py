# semi_supervised_base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, use_dropout=False, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bottleneck = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.bottleneck(x)
        if self.training:
            z = self.dropout(z)
        logits = self.classifier(z)
        return logits, z

class SemiSupervisedTrainer(ABC):
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError("Train method not implemented")
