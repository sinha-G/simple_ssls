import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

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
            attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / self.d_head ** 0.5)
            attention_output[:, :, head, :] = torch.bmm(attention, v)
            
        # Reshape back to original dimensions
        return attention_output.reshape(batch_size, seq_len, -1)

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_d * mlp_ratio, hidden_d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.dropout(self.msa(self.norm1(x)))
        return out + self.dropout(self.mlp(self.norm2(out)))

class ViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, num_classes, 
                 use_projection_head=False, dropout=0.1):
        super(ViT, self).__init__()
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.use_projection_head = use_projection_head

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
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            ViTBlock(self.hidden_d, self.n_heads, dropout=dropout) for _ in range(self.n_blocks)
        ])

        # MLP Head for classification
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, num_classes)
        )

        if use_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_d, hidden_d)
            )

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        if h != w:
            raise ValueError(f'Input images must be square. Got: {h}x{w}')
        if h % n_patches != 0:
            raise ValueError(f'Input image dimensions must be divisible by the number of patches. Got: {h}x{w} and {n_patches} patches.')
        
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

        # Patchify images
        patches = self.patchify(images, self.n_patches)
        patches = patches.to(device)

        # Embed patches 
        embeddings = self.embedding(patches)
        
        # Add class token and move to correct device
        batch_class_tokens = self.class_token.expand(n, 1, -1).to(device)
        embeddings = torch.cat([batch_class_tokens, embeddings], dim=1)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embed.to(device)

        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        for block in self.blocks:
            embeddings = block(embeddings)
            
        # Get class token output
        class_token_output = embeddings[:, 0]

        # Map class token to logits
        logits = self.mlp(class_token_output)
        
        # Optional projection head for SimCLR
        if self.use_projection_head:
            projection = self.projection(class_token_output)
            return logits, class_token_output, projection

        return logits, class_token_output
