# semi_supervised_base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class SemiSupervisedTrainer(ABC):
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError("Train method not implemented")
