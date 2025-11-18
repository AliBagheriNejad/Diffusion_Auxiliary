import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Time Embedding ---
def time_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the embedding.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# --- 2. Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # A 1x1 convolution to match dimensions if needed
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, time_emb):
        # Residual path
        h = self.act(self.conv1(x))
        
        # Time embedding is added to the feature map
        time_emb = self.act(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None] # Broadcast to (B, C, H, W)
        
        h = self.act(self.conv2(h))
        
        # Add the residual connection
        return h + self.residual_conv(x)

# --- 3. Attention Block ---
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1) # Split into Q, K, V
        
        # Scaled dot-product attention
        scale = 1. / math.sqrt(C)
        attn = torch.matmul(q.transpose(-2, -1), k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.matmul(v, attn.transpose(-2, -1))
        h = h.reshape(B, C, H, W)
        
        # Output projection and residual connection
        return x + self.out(h)

# --- 4. Down and Up Blocks ---
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb):
        h = self.res_block(x, time_emb)
        h = self.attn(h)
        return self.downsample(h), h # Return downsampled and the feature map for skip connection

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip_connection, time_emb):
        x = self.upsample(x)
        # Concatenate with skip connection from encoder
        x = torch.cat([x, skip_connection], dim=1)
        h = self.res_block(x, time_emb)
        h = self.attn(h)
        return h

# --- 5. Main U-Net Model ---
class AttentionResidualUNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial projection
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.down1 = Down(base_channels, base_channels * 2, time_emb_dim, has_attn=False)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_emb_dim, has_attn=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, time_emb_dim, has_attn=True)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.bottleneck_attn = AttentionBlock(base_channels * 8)

        # Decoder (Upsampling path)
        self.up1 = Up(base_channels * 8, base_channels * 4, time_emb_dim, has_attn=True)
        self.up2 = Up(base_channels * 4, base_channels * 2, time_emb_dim, has_attn=False)
        self.up3 = Up(base_channels * 2, base_channels, time_emb_dim, has_attn=False)

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x, t):
        # 1. Time embedding
        t_emb = time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # 2. Initial projection
        h = self.init_conv(x)
        
        # 3. Encoder
        h, skip1 = self.down1(h, t_emb)
        h, skip2 = self.down2(h, t_emb)
        h, skip3 = self.down3(h, t_emb)

        # 4. Bottleneck
        h = self.bottleneck(h, t_emb)
        h = self.bottleneck_attn(h)

        # 5. Decoder
        h = self.up1(h, skip3, t_emb)
        h = self.up2(h, skip2, t_emb)
        h = self.up3(h, skip1, t_emb)

        # 6. Final output
        return self.final_conv(h)

# --- How to Use ---
if __name__ == '__main__':
    import math

    # Model parameters
    IMG_SIZE = 64
    IMG_CHANNELS = 3
    BATCH_SIZE = 8
    TIMESTEPS = 1000

    # Instantiate the model
    model = AttentionResidualUNet(image_channels=IMG_CHANNELS, base_channels=64)
    print("Model instantiated successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy inputs
    # A noisy image
    noisy_image = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    # A random timestep for each image in the batch
    timesteps = torch.randint(0, TIMESTEPS, (BATCH_SIZE,))

    # Forward pass
    # The model predicts the noise added to the image
    predicted_noise = model(noisy_image, timesteps)

    # Check the output shape
    print(f"\nInput shape: {noisy_image.shape}")
    print(f"Timestep shape: {timesteps.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")

    assert noisy_image.shape == predicted_noise.shape
    print("\nSuccess! The output shape matches the input shape.")
