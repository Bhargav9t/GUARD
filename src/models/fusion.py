# ==============================================================================
# File: src/models/fusion.py
# Description: Multi-Modal Spatial-Frequency Fusion Mechanisms for GUARD.
#
# Methodology:
#   Combines spatial representations (1792-d from EfficientNet-B4) and frequency
#   representations (512-d from 2D FFT/DCT Residual CNN).
#   Provides:
#     1. Gated Residual Concatenation Fusion: Dynamically balances spatial and
#        spectral features using gating attention while preserving dedicated,
#        unimpeded gradient flow channels for both streams through residual concatenation.
#     2. Cross-Attention Fusion: Bi-directional multi-head attention where
#        spatial features query frequency artifacts and vice versa.
# ==============================================================================

# Import typing primitives
from typing import Optional, Tuple, Dict, Any

# Import PyTorch core neural network and tensor libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project configurations
from configs.config import FusionConfig, default_config


class GatedFusion(nn.Module):
    """
    Dynamic Gated Residual Concatenation Fusion Block.
    
    Mathematical Formulation:
      h_s = LayerNorm(GELU(W_s * f_spatial))      # (B, 1024)
      h_f = LayerNorm(GELU(W_f * f_frequency))    # (B, 1024)
      joint = [h_s; h_f]                          # (B, 2048)
      gate = sigmoid(W_g * joint + b_g)           # (B, 1024)
      gated_joint = [gate * h_s; (1 - gate) * h_f] + joint  # (B, 2048)
      fused = out_proj(gated_joint)               # (B, 1024)
    
    Guarantees:
      - Both modalities are concatenated and projected so both signals always contribute.
      - The residual addition (+ joint) guarantees strictly non-zero gradient flow
        d(fused)/d(h_s) >= 1.0 and d(fused)/d(h_f) >= 1.0, preventing gradient starvation.
      - Dedicated forward_spatial_stream and forward_frequency_stream methods enable
        accurate individual stream logit analysis with zero shape mismatch.
    """

    def __init__(
        self,
        spatial_dim: int = 1792,
        frequency_dim: int = 512,
        hidden_dim: int = 1024,
        dropout_rate: float = 0.30
    ):
        """
        Initializes multi-modal projection, gating, and output layers.
        
        Parameters:
            spatial_dim (int): Input feature size from SpatialStream (1792-d).
            frequency_dim (int): Input feature size from FrequencyStream (512-d).
            hidden_dim (int): Target joint representation dimension (1024-d).
            dropout_rate (float): Regularization dropout probability.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim
        self.hidden_dim = hidden_dim

        # 1. Project spatial features (1792-d) to joint hidden dimension (1024-d)
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. Project frequency features (512-d) to joint hidden dimension (1024-d)
        self.proj_frequency = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3. Dynamic Gating Generator
        # Takes concatenated projections [h_s; h_f] of size (2 * hidden_dim) and
        # computes element-wise gating coefficients g in [0, 1]
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Initialize gating bias to 0 so initial gate is approximately 0.5 (balanced)
        nn.init.constant_(self.gate_fc[-2].bias, 0.0)

        # 4. Joint projection and refinement layer
        # Projects concatenated multi-modal representation (2048-d) back to hidden_dim (1024-d)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_frequency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses spatial and frequency feature vectors.

        Parameters:
            f_spatial (torch.Tensor): Spatial feature embedding (B, 1792).
            f_frequency (torch.Tensor): Frequency feature embedding (B, 512).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - fused_features: Joint multi-modal representation (B, 1024).
                - gate: Gating attention coefficients (B, 1024) for telemetry.
        """
        # Step 1: Project both modalities to common hidden space
        h_s = self.proj_spatial(f_spatial)      # (B, 1024)
        h_f = self.proj_frequency(f_frequency)  # (B, 1024)

        # Step 2: Form joint concatenated multi-modal representation
        joint = torch.cat([h_s, h_f], dim=1)    # (B, 2048)

        # Step 3: Compute dynamic gating weights in [0.0, 1.0]
        gate = self.gate_fc(joint)              # (B, 1024)

        # Step 4: Modulate both streams and apply residual concatenation
        # [gate * h_s; (1 - gate) * h_f] + joint ensures both streams receive continuous gradients
        gated_s = gate * h_s
        gated_f = (1.0 - gate) * h_f
        gated_joint = torch.cat([gated_s, gated_f], dim=1) + joint  # (B, 2048)

        # Step 5: Final projection to output fused representation
        fused = self.out_proj(gated_joint)      # (B, 1024)

        return fused, gate

    def forward_spatial_stream(self, f_spatial: torch.Tensor) -> torch.Tensor:
        """
        Evaluates spatial features through the fusion projection in isolation
        for independent stream metric breakdown.

        Parameters:
            f_spatial (torch.Tensor): Spatial features (B, 1792).

        Returns:
            torch.Tensor: Projected spatial representation of shape (B, 1024).
        """
        h_s = self.proj_spatial(f_spatial)
        # Pad frequency channels with neutral zeros to match 2048 input dimension
        zero_f = torch.zeros_like(h_s)
        # Scaled by 2 to account for isolated single-branch evaluation
        stream_input = torch.cat([h_s * 2.0, zero_f], dim=1)
        return self.out_proj(stream_input)

    def forward_frequency_stream(self, f_frequency: torch.Tensor) -> torch.Tensor:
        """
        Evaluates frequency features through the fusion projection in isolation
        for independent stream metric breakdown.

        Parameters:
            f_frequency (torch.Tensor): Frequency features (B, 512).

        Returns:
            torch.Tensor: Projected frequency representation of shape (B, 1024).
        """
        h_f = self.proj_frequency(f_frequency)
        # Pad spatial channels with neutral zeros to match 2048 input dimension
        zero_s = torch.zeros_like(h_f)
        # Scaled by 2 to account for isolated single-branch evaluation
        stream_input = torch.cat([zero_s, h_f * 2.0], dim=1)
        return self.out_proj(stream_input)


class CrossAttentionFusion(nn.Module):
    """
    Bi-directional Multi-Head Cross-Attention Fusion.
    Allows spatial tokens to attend to frequency spectral cues, and vice-versa,
    enriching spatial representations with Fourier context before classification.
    """

    def __init__(
        self,
        spatial_dim: int = 1792,
        frequency_dim: int = 512,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        dropout_rate: float = 0.30
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project modalities to common sequence dimension
        self.proj_spatial = nn.Linear(spatial_dim, hidden_dim)
        self.proj_frequency = nn.Linear(frequency_dim, hidden_dim)

        # Multi-Head Cross-Attention (Spatial queries Frequency)
        self.s_to_f_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Multi-Head Cross-Attention (Frequency queries Spatial)
        self.f_to_s_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Output projection
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_frequency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            f_spatial (torch.Tensor): Spatial features (B, 1792).
            f_frequency (torch.Tensor): Frequency features (B, 512).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (fused_features, attention_weights)
        """
        # Reshape to 1-token sequences: (B, 1, hidden_dim)
        q_s = self.proj_spatial(f_spatial).unsqueeze(1)
        q_f = self.proj_frequency(f_frequency).unsqueeze(1)

        # Spatial attends to Frequency
        attn_s, s_weights = self.s_to_f_attn(query=q_s, key=q_f, value=q_f)

        # Frequency attends to Spatial
        attn_f, _ = self.f_to_s_attn(query=q_f, key=q_s, value=q_s)

        # Squeeze sequence dimension
        attn_s = attn_s.squeeze(1)
        attn_f = attn_f.squeeze(1)

        # Concatenate attended features
        cat_features = torch.cat([attn_s, attn_f], dim=-1)
        fused = self.out_proj(self.norm(cat_features))

        return fused, s_weights

    def forward_spatial_stream(self, f_spatial: torch.Tensor) -> torch.Tensor:
        q_s = self.proj_spatial(f_spatial)
        zero_f = torch.zeros_like(q_s)
        return self.out_proj(self.norm(torch.cat([q_s, zero_f], dim=-1)))

    def forward_frequency_stream(self, f_frequency: torch.Tensor) -> torch.Tensor:
        q_f = self.proj_frequency(f_frequency)
        zero_s = torch.zeros_like(q_f)
        return self.out_proj(self.norm(torch.cat([zero_s, q_f], dim=-1)))


class FusionLayer(nn.Module):
    """
    Factory / Dispatcher module instantiating the selected fusion mechanism
    (gated or cross_attention) according to configuration.
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        spatial_dim: int = 1792,
        frequency_dim: int = 512
    ):
        super().__init__()
        self.config = config or default_config.fusion
        self.fusion_type = self.config.fusion_type.lower()
        self.hidden_dim = self.config.hidden_dim

        if self.fusion_type == "cross_attention":
            self.fusion_impl = CrossAttentionFusion(
                spatial_dim=spatial_dim,
                frequency_dim=frequency_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate
            )
        else:
            # Default to Gated Residual Concatenation Fusion
            self.fusion_impl = GatedFusion(
                spatial_dim=spatial_dim,
                frequency_dim=frequency_dim,
                hidden_dim=self.config.hidden_dim,
                dropout_rate=self.config.dropout_rate
            )

    @property
    def out_features(self) -> int:
        """
        Returns output feature dimension (1024-d).
        """
        return self.hidden_dim

    def forward(
        self,
        f_spatial: torch.Tensor,
        f_frequency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatches multi-modal fusion.
        """
        return self.fusion_impl(f_spatial, f_frequency)

    def forward_spatial_stream(self, f_spatial: torch.Tensor) -> torch.Tensor:
        """
        Dispatches spatial-only evaluation through fusion output projection.
        """
        return self.fusion_impl.forward_spatial_stream(f_spatial)

    def forward_frequency_stream(self, f_frequency: torch.Tensor) -> torch.Tensor:
        """
        Dispatches frequency-only evaluation through fusion output projection.
        """
        return self.fusion_impl.forward_frequency_stream(f_frequency)
