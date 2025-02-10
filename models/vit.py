import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import json

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        if  d % n_heads != 0:
            raise ValueError(f"Embedding dimension {d} must be divisible by the number of heads {n_heads}.")

        # Single linear layers for all heads combined.
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        
        self.out_proj = nn.Linear(d, d)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.attention_weights = None

    def forward(self, sequences):
        # sequences shape: (batch_size, seq_len, d)
        batch_size, seq_len, _ = sequences.shape
        
        # Compute Q, K, V for all heads at once.
        q = self.q_proj(sequences)  # (batch_size, seq_len, d)
        k = self.k_proj(sequences)
        v = self.v_proj(sequences)
        
        # Reshape to (batch_size, seq_len, n_heads, d_head) and then transpose to (batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute scaled dot-product attention scores.
        # scores shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.softmax(scores)
        self.attention_weights = attn.detach()  # save for inspection if needed
        
        # Multiply the attention weights by the values.
        # out shape: (batch_size, n_heads, seq_len, d_head)
        out = torch.matmul(attn, v)
        
        # Transpose and reshape back to (batch_size, seq_len, d)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)
        
        out = self.out_proj(out)
        return out

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
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, num_classes, use_projection_head=False, dropout=0.1):
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

    def patchify(self, images):
        n, c, h, w = images.shape
        n_patches = self.n_patches

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

        # print(f"Debug - Final patches shape: {patches.shape}")  # Debug print
        
        return patches

    def get_positional_embeddings(self, sequence_length, d):
        # Create a tensor of positions (shape: [sequence_length, 1])
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        
        # Compute the div_term for even indices (shape: [d/2])
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        
        # Initialize the positional embeddings tensor (shape: [sequence_length, d])
        pe = torch.zeros(sequence_length, d)
        
        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(positions * div_term)
        
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        return pe


    def forward(self, images):
        # Get device from images
        device = images.device
        
        n, c, h, w = images.shape

        # Patchify images
        # print(f"Input shape: {images.shape}")  # Debug
        patches = self.patchify(images).to(device)
        # print(f"Patch shape: {patches.shape}")  # Debug
        
        # Embed patches
        embeddings = self.embedding(patches)
        # print(f"Embedding shape: {embeddings.shape}")  # Debug
        
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
    
    def visualize_attention(self, images, layer_idx=0, head_idx=0, save_path=None, alpha=0.6, ax=None):
        """Visualize attention maps overlaid on images
        Args:
            images: Input images (B, C, H, W)
            layer_idx: Which transformer layer to visualize
            head_idx: Which attention head to visualize
            save_path: Optional path to save visualization
            alpha: Transparency of attention overlay (0-1)
        """
        B, C, H, W = images.shape
        P = self.n_patches
        
        # Get attention weights from specified layer/head
        self.eval()
        with torch.no_grad():
            _ = self(images)  # Forward pass to get attention weights
            attn_weights = self.blocks[layer_idx].msa.attention_weights
            
        # Remove CLS token attention and reshape
        attn_weights = attn_weights[:, head_idx, 1:, 1:]  # (B, P*P, P*P)
        attn_weights = attn_weights.reshape(B, P, P, P, P)  # (B, P, P, P, P)
        
        # Use provided axis or current axis
        if ax is None:
            ax = plt.gca()
        
        # Display original image
        img = images[0].permute(1, 2, 0).cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = img.clip(0, 1)
        ax.imshow(img)
        
        # Calculate attention heatmap
        avg_attn = attn_weights[0].mean(dim=(2, 3))  # (P, P)
        
        # Resize attention map to match image dimensions
        attention_resized = torch.nn.functional.interpolate(
            avg_attn.unsqueeze(0).unsqueeze(0), 
            size=(H, W), 
            mode='bilinear'
        ).squeeze().cpu()
        
        # Create overlay
        im = ax.imshow(attention_resized, cmap='viridis', alpha=alpha)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Add grid lines for patches
        for i in range(P):
            ax.axhline(y=i*(H//P), color='white', alpha=0.3)
            ax.axvline(x=i*(W//P), color='white', alpha=0.3)
        
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')