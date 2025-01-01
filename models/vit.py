import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

class MSA(nn.Module):
    def __init__(self, d, n_heads = 2):
        self.d = d
        self.n_heads = n_heads

        if d % n_heads != 0:
            raise ValueError('Hidden dimension must be divisible by the number of heads.') 

        self.d_head = d // n_heads
        self.W_q = [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        self.W_k = [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        self.W_v = [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # sequences starts at:      (N, sequence length, token dimension)
        # We go to:                 (N, sequence length, n_heads, token dimension // n_heads)
        # Then concatenate back to  (N, sequence length, token dimension)
        result = []
        for s in sequences:
            s_result = []
            for head in range(self.n_heads):
                s = s[:, head * self.d_head: (head + 1) * self.d_head]  # (N, sequence length, token dimension // n_heads)

                q, k, v = self.W_q[head](s), self.W_k[head](s), self.W_v[head](s)
                attention = self.softmax(q @ k.T / self.d_head ** 0.5)
                s_result.append(attention @ v)
            result.append(torch.hstack(s_result))
        return torch.cat([torch.unsequeeze(r, dim=0) for r in result])

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
    def __init__(self):
        super(ViT, self).__init__()
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.out_d = out_d

        if chw[1] != chw[2]:
            raise ValueError('Input images must be square.')
        if chw[1] % n_patches != 0:
            raise ValueError('Input image dimensions must be divisible by the number of patches.')
 
        # Embedding Layer
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.input_d = int(chw[0] * self.patch_size)
        self.embedding = nn.Linear(self.input_d, self.hidden_d) 

        # Learnable [CLS] token
        self.class_token = nn.Parameter(torch.randn(1, self.hidden_d))

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False

        # Transformer Encoder
        self.blocks = nn.ModuleList([VitBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)])

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        if h != w:
            raise ValueError('Input images must be square.')
        if h % n_patches != 0:
            raise ValueError('Input image dimensions must be divisible by the number of patches.')
        
        patches = torch.zeros(n, n_patches ** 2, h * w * c  // n_patches ** 2)
        patch_size = h // n_patches
        
        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()

        return patches

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
        

    def forward(self, images):
        patches = self.patchify(images)
        embeddings = self.embedding(patches)

        # Add class token
        embeddings = torch.stack([torch.vstack((self.class_token, embeddings[i])) for i in range(len(embeddings))])

        # Add positional embeddings
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        embeddings = embeddings + pos_embed

        # Transformer Encoder
        for block in self.blocks:
            embeddings = block(embeddings)

        out = embeddings[:, 0]  # [CLS] token

        return embeddings   # (N, hidden_d)

def main():
    # This function is just for testing the above implementation of the Vision Transformer.
    # It is not a part of the implementation of the Vision Transformer.
    
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets ', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets ', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Model initializaiton and training options
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionTransformer((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    epochs = 5
    lr = 5e-3
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

    # Testing
    with torch.no_grad():
        model.eval()
        correct, total = 0
        test_loss = 0.0
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f'Test Loss: {test_loss:.4f}, Accuracy: {correct / total:.4f}')
