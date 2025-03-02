import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import math
import json

from sklearn.manifold import TSNE
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

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_weights = None
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, sequences):
        batch_size, seq_len, _ = sequences.shape
        q = self.q_proj(sequences).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(sequences).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(sequences).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.d_head)
        attn = self.softmax(scores)
        self.attention_weights = attn.detach()
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)
        return self.out_proj(out)

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_d * mlp_ratio, hidden_d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.msa(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

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
 
        self.patch_size = chw[1] // self.n_patches
        self.input_d = (self.patch_size ** 2) * chw[0]
        self.embedding = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.randn(1, self.hidden_d))
        self.register_buffer('pos_embed', self.get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ViTBlock(self.hidden_d, self.n_heads, dropout=dropout) for _ in range(self.n_blocks)])
        self.norm = nn.LayerNorm(self.hidden_d)
        self.mlp = nn.Linear(self.hidden_d, num_classes)

        if use_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(hidden_d, hidden_d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_d, hidden_d)
            )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        # Initialize class token
        nn.init.trunc_normal_(self.class_token, std=0.02)
        # Initialize position embedding
        nn.init.zeros_(self.pos_embed)
        
        # Initialize MLP heads
        if hasattr(self, 'mlp'):
            nn.init.xavier_uniform_(self.mlp.weight)
            nn.init.zeros_(self.mlp.bias)
        
        if hasattr(self, 'projection'):
            for layer in self.projection:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def patchify(self, images):
        return F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

    def get_positional_embeddings(self, sequence_length, d):
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(sequence_length, d)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe


    def forward(self, images):
        # Get device from images
        device = images.device
        n, c, h, w = images.shape

        # Patchify images
        patches = self.patchify(images).to(device)
        
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

        # Apply final layer norm
        embeddings = self.norm(embeddings)
            
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
        patch_size = H // P
        
        # Get attention weights from specified layer/head
        self.eval()
        with torch.no_grad():
            _ = self(images)
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
        
        # Create a full-size attention map with blocked patches
        attention_map = torch.zeros((H, W))
        for i in range(P):
            for j in range(P):
                attention_map[i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size] = avg_attn[i, j]
        
        # Create overlay
        im = ax.imshow(attention_map.cpu(), cmap='viridis', alpha=alpha)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Add grid lines for patches
        for i in range(P):
            ax.axhline(y=i*patch_size, color='white', alpha=0.3)
            ax.axvline(x=i*patch_size, color='white', alpha=0.3)
        
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

    def visualize_class_separation(self, class_indices, k=10, dataloader=None, perplexity=5, random_state=42, save_path=None):
        """
        Visualize how two classes are separated in embedding space across transformer layers using animation.
        
        Args:
            class_idx1 (int): First class index (0-999)
            class_idx2 (int): Second class index (0-999)
            k (int): Number of samples per class
            dataloader: DataLoader containing the dataset
            perplexity (int): t-SNE perplexity parameter
            random_state (int): Random seed for reproducibility
            save_path (str): Optional path to save the animation (should end in .gif)
        """
        
        if dataloader is None:
            raise ValueError("DataLoader must be provided")
        
        # Set model to eval mode
        self.eval()
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(class_indices) > len(colors):
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(class_indices)))
        
        # Collect k samples for each class
        all_samples = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                for idx, class_idx in enumerate(class_indices):
                    mask = labels == class_idx
                    if mask.any():
                        class_samples = images[mask][:k]
                        all_samples.extend(class_samples)
                        all_labels.extend([idx] * len(class_samples))
                    
                if all(sum(1 for label in all_labels if label == idx) >= k for idx in range(len(class_indices))):
                    break
        
        # Trim to exactly k samples per class if we have more
        samples_by_class = {idx: [] for idx in range(len(class_indices))}
        labels_by_class = {idx: [] for idx in range(len(class_indices))}
        
        for sample, label in zip(all_samples, all_labels):
            if len(samples_by_class[label]) < k:
                samples_by_class[label].append(sample)
                labels_by_class[label].append(label)
        
        all_samples = []
        all_labels = []
        for idx in range(len(class_indices)):
            all_samples.extend(samples_by_class[idx])
            all_labels.extend(labels_by_class[idx])
        
        # Convert to tensors
        all_samples = torch.stack(all_samples).cuda()
        all_labels = torch.tensor(all_labels).cuda()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Store transformer layer embeddings
        all_embeddings = []
        
        with torch.no_grad():
            # Get patch embeddings
            patches = self.patchify(all_samples)
            embeddings = self.embedding(patches)
            batch_class_tokens = self.class_token.expand(len(all_samples), 1, -1)
            embeddings = torch.cat([batch_class_tokens, embeddings], dim=1)
            embeddings = embeddings + self.pos_embed
            embeddings = self.dropout(embeddings)
            
            # Process each transformer block
            for block in self.blocks:
                embeddings = block(embeddings)
                cls_embedding = embeddings[:, 0].cpu().numpy()
                all_embeddings.append(cls_embedding)
        
        # Initialize t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, learning_rate="auto", init="pca")
        
        tsne_results = []
        for emb in all_embeddings:
            tsne_result = tsne.fit_transform(emb)
            tsne_results.append(tsne_result)
        
        # Find global min/max for consistent axis scaling
        all_tsne = np.vstack(tsne_results)
        x_min, x_max = all_tsne[:, 0].min(), all_tsne[:, 0].max()
        y_min, y_max = all_tsne[:, 1].min(), all_tsne[:, 1].max()
        
        # Add padding to limits
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        def animate(frame):
            ax.clear()
            tsne_result = tsne_results[frame]
            
            for idx in range(len(class_indices)):
                mask = np.array(all_labels.cpu()) == idx
                ax.scatter(
                    tsne_result[mask, 0], 
                    tsne_result[mask, 1], 
                    c=[colors[idx]], 
                    label=f'Class {class_indices[idx]}'
                )
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()
            ax.set_title(f'Layer {frame + 1}')
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(all_embeddings), 
            interval=1000, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        plt.close()
        
        return anim

    def visualize_cam(self, images, target_class=None, layer_idx=-1, head_idx=None, save_path=None, alpha=0.6, ax=None):
        """Visualize class activation mapping for Vision Transformer
        
        Args:
            images: Input images (B, C, H, W)
            target_class: Target class index. If None, uses predicted class
            layer_idx: Which transformer layer to visualize (-1 for last layer)
            head_idx: Which attention head to visualize, None for average of all heads
            save_path: Optional path to save visualization
            alpha: Transparency of attention overlay (0-1)
            ax: Optional matplotlib axis to plot on
        """
        B, C, H, W = images.shape
        P = self.n_patches
        patch_size = H // P
        
        # Ensure layer_idx is valid
        if layer_idx < 0:
            layer_idx = len(self.blocks) + layer_idx
        
        # Forward pass to get class predictions and attention
        self.eval()
        with torch.no_grad():
            logits, _ = self(images)
            
            # Get class to visualize (predicted or specified)
            if target_class is None:
                target_class = logits.argmax(dim=1)[0].item()
            
            # Get attention weights from specified layer
            attn_weights = self.blocks[layer_idx].msa.attention_weights
            
            # Get attention from CLS token to other tokens
            # (B, num_heads, 1+P*P, 1+P*P) -> (B, num_heads, 1, P*P)
            cls_attn = attn_weights[:, :, 0, 1:]
            
            # Average across heads if not specified
            if head_idx is None:
                # Average over all heads
                cls_attn = cls_attn.mean(dim=1)  # (B, P*P)
            else:
                # Use specified head
                cls_attn = cls_attn[:, head_idx, :]  # (B, P*P)
            
            # Get final layer weights for target class
            # These weights connect the embeddings to the class prediction
            cam_weights = self.mlp.weight[target_class].detach()  # (hidden_d)
            
            # Get patch embeddings before the projection to class logits
            patch_tokens = self._forward_features(images)[:, 1:, :]  # (B, P*P, hidden_d)
            
            # Compute activation scores
            patch_cam = torch.matmul(patch_tokens, cam_weights)  # (B, P*P)
            
            # Combine with attention weights
            cam = cls_attn.squeeze() * patch_cam.squeeze()  # (P*P)
            
            # Reshape to patch grid
            cam = cam.reshape(P, P)  # (P, P)
            
        # Use provided axis or current axis
        if ax is None:
            ax = plt.gca()
        
        # Display original image
        img = images[0].permute(1, 2, 0).cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = img.clip(0, 1)
        ax.imshow(img)
        
        # Create full-resolution CAM
        attention_map = torch.zeros((H, W), device=cam.device)
        for i in range(P):
            for j in range(P):
                attention_map[i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size] = cam[i, j]
        
        # Create overlay with same style as visualize_attention
        im = ax.imshow(attention_map.cpu(), cmap='viridis', alpha=alpha)
        plt.colorbar(im, ax=ax, label='CAM Activation')
        
        # Add grid lines for patches
        for i in range(P):
            ax.axhline(y=i*patch_size, color='white', alpha=0.3)
            ax.axvline(x=i*patch_size, color='white', alpha=0.3)
        
        # Add class information
        class_name = f"Class {target_class}"
        ax.set_title(f"CAM for {class_name} (Layer {layer_idx})")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return ax

    # Helper method to extract features
    def _forward_features(self, images):
        """Extract patch features before classification head"""
        # Get device from images
        device = images.device
        n, c, h, w = images.shape

        # Patchify images
        patches = self.patchify(images).to(device)
        
        # Embed patches
        embeddings = self.embedding(patches)
        
        # Add class token and positional embeddings
        batch_class_tokens = self.class_token.expand(n, 1, -1).to(device)
        embeddings = torch.cat([batch_class_tokens, embeddings], dim=1)
        embeddings = embeddings + self.pos_embed.to(device)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        for block in self.blocks:
            embeddings = block(embeddings)
        
        # Apply final layer normalization
        embeddings = self.norm(embeddings)
        
        return embeddings
