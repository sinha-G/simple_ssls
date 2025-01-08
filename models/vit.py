import numpy as np
import math

from tqdm import tqdm

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_mnist_loaders, get_cifar10_loaders

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        
        self.W_q = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, sequences):
        # sequences shape: (batch_size, seq_len, hidden_dim)
        device = sequences.device
        batch_size, seq_len, _ = sequences.shape
        
        # Split hidden dim into heads
        sequences = sequences.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Initialize output tensor
        attention_output = torch.zeros_like(sequences).to(device)
        
        # Process each attention head
        for head in range(self.n_heads):
            head_input = sequences[:, :, head, :]  # (batch_size, seq_len, d_head)
            
            q = self.W_q[head](head_input)  # (batch_size, seq_len, d_head)
            k = self.W_k[head](head_input)
            v = self.W_v[head](head_input)
            
            # Compute attention scores
            attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_head))
            attention_output[:, :, head, :] = torch.bmm(attention, v)
            
        # Reshape back to original dimensions
        return attention_output.reshape(batch_size, seq_len, -1)

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * mlp_ratio),
            nn.ReLU(),
            nn.Linear(hidden_d * mlp_ratio, hidden_d)
        )

    def forward(self, x):
        out = x + self.msa(self.norm1(x))
        return out + self.mlp(self.norm2(out))

class ViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, num_classes):
        super(ViT, self).__init__()
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.num_classes = num_classes

        if chw[1] != chw[2]:
            raise ValueError('Input images must be square.')
        if chw[1] % n_patches != 0:
            raise ValueError('Input image dimensions must be divisible by the number of patches.')
 
        # Embedding Layer
        self.patch_size = chw[1] // self.n_patches
        self.input_d = int((self.patch_size ** 2) * chw[0])
        self.embedding = nn.Linear(self.input_d, self.hidden_d) 

        # Learnable [CLS] token
        self.class_token = nn.Parameter(torch.randn(1, self.hidden_d))

        # Positional Embeddings
        self.pos_embed = self.get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)
        self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=False)
        # self.pos_embed.requires_grad = False

        # Transformer Encoder
        self.blocks = nn.ModuleList([ViTBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)])

        # MLP Head for classification
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, num_classes),
            nn.Softmax(dim = -1)
        )

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        if h != w:
            raise ValueError('Input images must be square.')
        if h % n_patches != 0:
            raise ValueError('Input image dimensions must be divisible by the number of patches.')
        
        patches = torch.zeros(n, n_patches ** 2, h * w * c  // n_patches ** 2)
        patch_size = h // n_patches
        
        # Reshape to: (batch_size, n_patches², patch_size² * channels)
        patches = images.reshape(
            n, c, 
            n_patches, patch_size, 
            n_patches, patch_size
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(n, n_patches * n_patches, -1)
        
        return patches

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result

    def forward(self, images):
        # Get device from images
        device = images.device
        
        n, c, h, w = images.shape
        patches = self.patchify(images, self.n_patches)
        
        # Move patches to correct device
        patches = patches.to(device)
        embeddings = self.embedding(patches)
        
        # Add class token and move to correct device
        batch_class_tokens = self.class_token.expand(n, 1, -1).to(device)
        embeddings = torch.cat([batch_class_tokens, embeddings], dim=1)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embed.to(device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            embeddings = block(embeddings)
            
        # Get class token output and classify
        class_token_output = embeddings[:, 0]
        return self.mlp(class_token_output)

def main():
    # This function is just for testing the above implementation of the Vision Transformer.
    # It is not a part of the implementation of the Vision Transformer.
    
    # Loading data
    train_loader, test_loader = get_mnist_loaders(batch_size=1024)
    # train_loader, test_loader = get_cifar10_loaders(batch_size=1024)

    # Model initializaiton and training options
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViT((1, 28, 28), n_patches=7, n_blocks=6, hidden_d=32, n_heads=8, out_d=10).to(device)
    # Model initialization and training options
    epochs = 100
    lr = 1e-2
    lr_scheduler_patience = 5  # Shorter patience for LR scheduling
    early_stopping_patience = 15  # Longer patience for early stopping
    min_lr = 1e-5

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=lr_scheduler_patience,  # Use shorter patience here
        min_lr=min_lr
    )
    criterion = CrossEntropyLoss()

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    # Training
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.shape[0]

        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        val_accuracy = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered! Best validation accuracy: {best_val_accuracy:.4f}')
            model.load_state_dict(best_model_state)
            break
        
        if optimizer.param_groups[0]['lr'] <= min_lr:
            print('Learning rate too small, stopping training')
            break

if __name__ == '__main__':
    main()
