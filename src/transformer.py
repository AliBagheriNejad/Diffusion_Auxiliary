import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

import src.models as models

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for timesteps in the diffusion process"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(attn_output)

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class SignalTransformer(models.BaseModel):
    """Transformer-based denoising model for 2-channel signals"""
    def __init__(self, seq_len=1024, d_model=256, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super(SignalTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection for 2 channels
        self.input_projection = nn.Linear(2, d_model)
        
        # Positional encoding for sequence dimension
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t, is_ae=False):
        # x shape: (batch_size, 2, seq_len)
        batch_size = x.shape[0]
        
        # Transpose to (batch_size, seq_len, 2) for transformer
        x = x.transpose(1, 2)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Process time embedding
        t_emb = self.time_embedding(t)  # (batch_size, d_model)
        t_emb = self.time_mlp(t_emb)    # (batch_size, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Add time embedding to input
        if is_ae:
            pass
        else:
            x = x + t_emb
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Project back to 2 channels
        x = self.output_projection(x)  # (batch_size, seq_len, 2)
        
        # Transpose back to (batch_size, 2, seq_len)
        x = x.transpose(1, 2)
        
        return x