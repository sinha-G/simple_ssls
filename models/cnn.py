# models/cnn.py
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, use_dropout=False, dropout_rate=0.5, use_projection_head=False):
        super(CNN, self).__init__()
        self.use_projection_head = use_projection_head
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bottleneck = nn.LazyLinear(128)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(128, 10)

        if self.use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(f"[DEBUG]: Shape before bottleneck: {x.shape}")  # Debug print
        z = self.bottleneck(x)
        if self.training:
            z = self.dropout(z)
        logits = self.classifier(z)
        
        if self.use_projection_head:
            projection = self.projection_head(z)
            return logits, z, projection
        
        return logits, z