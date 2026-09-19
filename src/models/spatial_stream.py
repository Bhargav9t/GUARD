# ==============================================================================
# File: src/models/spatial_stream.py
# Description: RGB Spatial Feature Extraction Stream for GUARD.
#
# Architecture & Methodology:
#   - Utilizes EfficientNet-B4 from PyTorch Image Models (`timm`).
#   - EfficientNet-B4 features inverted residual blocks with Squeeze-and-Excitation
#     attention, capturing pixel-level blending boundaries, warping artifacts,
#     and local texture inconsistencies in RGB space.
#   - Provides both 1D global pooled feature embeddings (1792-d) for fusion
#     and 2D convolutional feature maps for spatial heatmaps / Grad-CAM.
# ==============================================================================

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import timm

from configs.config import SpatialStreamConfig, default_config
from src.utils.logger import app_logger


class ImageNetNormalize(nn.Module):
    """
    In-graph normalization module that scales RGB inputs from [0.0, 1.0]
    using standard ImageNet mean and standard deviation:
        x_norm = (x - mean) / std
    Placing normalization directly inside the PyTorch module graph ensures
    identical preprocessing across training, evaluation, and production deployment.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initializes normalization buffers as non-trainable registered buffers.

        Parameters:
            mean (Tuple[float, float, float]): Per-channel ImageNet mean.
            std (Tuple[float, float, float]): Per-channel ImageNet std.
        """
        super().__init__()
        # Register mean and std as persistent non-trainable buffers
        # Reshaped to (1, 3, 1, 1) for automated broadcasting across batch, height, and width
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies broadcasting normalization.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W) with values in [0.0, 1.0].

        Returns:
            torch.Tensor: Normalized tensor with zero-mean unit-variance distribution.
        """
        return (x - self.mean) / self.std


class SpatialStream(nn.Module):
    """
    Spatial Stream network leveraging EfficientNet-B4 backbone.
    Extracts dense representations of color distribution, facial symmetry,
    and generative synthesis artifacts across spatial coordinates.
    """

    def __init__(self, config: Optional[SpatialStreamConfig] = None):
        """
        Initializes the Spatial Stream backbone and normalization layers.

        Parameters:
            config (Optional[SpatialStreamConfig]): Configuration for spatial stream.
                If None, defaults to default_config.spatial.
        """
        super().__init__()

        # 1. Load configuration settings
        self.config = config or default_config.spatial
        self.backbone_name = self.config.backbone_name
        self.feature_dim = self.config.feature_dim
        self.dropout_rate = self.config.dropout_rate

        # 2. Register normalization module
        self.normalize = ImageNetNormalize(mean=self.config.mean, std=self.config.std)

        # 3. Instantiate EfficientNet-B4 via timm
        # Setting num_classes=0 removes the final linear classification layer,
        # retaining global average pooling to yield a 1D feature embedding vector.
        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.config.pretrained,
                num_classes=0,
                drop_rate=self.dropout_rate,
            )
            app_logger.info(
                f"SpatialStream initialized: Backbone={self.backbone_name}, "
                f"Pretrained={self.config.pretrained}, OutputDim={self.feature_dim}"
            )
        except Exception as e:
            # Fallback for offline environments where downloading weights might time out
            app_logger.warning(
                f"Could not download pretrained weights for {self.backbone_name} ({e}). "
                f"Initializing backbone with random weights."
            )
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=False,
                num_classes=0,
                drop_rate=self.dropout_rate,
            )

        # 4. Feature projection layer with Layer Normalization and Dropout
        # Refines spatial features before feeding into the cross-modal fusion stage
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
        )

    @property
    def out_features(self) -> int:
        """
        Returns the output dimensionality of the spatial feature vector.
        """
        return self.feature_dim

    def forward_features(self, x: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """
        Extracts 2D unpooled convolutional feature maps from the backbone.
        Useful for spatial attention visualization, heatmaps, or Grad-CAM.

        Parameters:
            x (torch.Tensor): Input batch of images (B, 3, H, W) in [0.0, 1.0].
            apply_norm (bool): Whether to apply ImageNet normalization. Default is True.

        Returns:
            torch.Tensor: Feature maps of shape (B, C, H', W'), where C = feature_dim.
        """
        # Apply normalization if requested
        if apply_norm:
            x = self.normalize(x)

        # Extract 2D convolutional features from backbone prior to global pooling
        feature_maps = self.backbone.forward_features(x)
        return feature_maps

    def forward(self, x: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """
        Forward pass producing a 1D spatial feature embedding vector for each image.

        Parameters:
            x (torch.Tensor): Input batch of images (B, 3, H, W) in [0.0, 1.0].
            apply_norm (bool): Whether to apply ImageNet normalization. Default is True.

        Returns:
            torch.Tensor: 1D spatial feature vector of shape (B, 1792).
        """
        # Step 1: Normalize input tensor
        if apply_norm:
            x = self.normalize(x)

        # Step 2: Extract pooled feature embedding from EfficientNet-B4 backbone
        # Output shape from backbone with num_classes=0 is (B, 1792)
        raw_features = self.backbone(x)

        # Step 3: Pass through projection block (Linear + LayerNorm + GELU + Dropout)
        projected_features = self.proj(raw_features)

        return projected_features

    def freeze_backbone(self) -> None:
        """
        Freezes backbone convolutional weights during initial training stages
        to allow fusion and classification heads to stabilize before full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        app_logger.info("SpatialStream: Backbone weights frozen.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreezes all backbone parameters for full end-to-end backpropagation.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        app_logger.info("SpatialStream: Backbone weights unfrozen.")
