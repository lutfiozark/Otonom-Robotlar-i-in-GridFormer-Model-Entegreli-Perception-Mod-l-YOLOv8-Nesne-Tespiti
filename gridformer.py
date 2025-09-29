import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GridFormer(nn.Module):
    """
    GridFormer: Transformer-based Image Restoration for Weather-Degraded Scenes
    Simplified implementation for robot navigation applications
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 dim: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 patch_size: int = 8):
        super(GridFormer, self).__init__()

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.ConvTranspose2d(
                dim, dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, dim, 64, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Restored image [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Patch embedding
        patches = self.patch_embed(x)  # [B, dim, H//patch_size, W//patch_size]

        # Add positional encoding
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        pos_enc = F.interpolate(self.pos_encoding, size=(
            h_patches, w_patches), mode='bilinear')
        patches = patches + pos_enc

        # Reshape for transformer: [B, dim, H*W] -> [B, H*W, dim]
        patches_flat = patches.flatten(2).transpose(1, 2)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            patches_flat = block(patches_flat)

        # Reshape back: [B, H*W, dim] -> [B, dim, H, W]
        patches = patches_flat.transpose(1, 2).reshape(
            B, self.dim, h_patches, w_patches)

        # Feature refinement
        refined = self.refine_conv(patches) + patches

        # Output projection
        output = self.output_proj(refined)

        # Ensure output matches input size
        if output.shape != x.shape:
            output = F.interpolate(output, size=(
                H, W), mode='bilinear', align_corners=False)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and MLP"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# Alias for backward compatibility with training scripts
GridFormerModel = GridFormer


def create_gridformer_model(pretrained=False, **kwargs):
    """Factory function to create GridFormer model."""
    model = GridFormer(**kwargs)

    if pretrained:
        # Load pretrained weights if available
        try:
            checkpoint = torch.load(
                'models/gridformer_pretrained.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded pretrained GridFormer weights")
        except FileNotFoundError:
            print("⚠️  No pretrained weights found, using random initialization")

    return model


if __name__ == "__main__":
    # Test the model
    model = create_gridformer_model()
    model.eval()

    # Test input
    test_input = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = model(test_input)

    print(f"✅ GridFormer test successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(
        f"   Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
