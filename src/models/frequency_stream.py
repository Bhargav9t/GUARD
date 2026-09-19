# ==============================================================================
# File: src/models/frequency_stream.py
# Description: Differentiable 2D FFT & 2D DCT Dual-Spectral Frequency Stream for GUARD.
#
# Forensic Rationale:
#   Generative models (GANs, Diffusion models, VAEs) introduce characteristic
#   periodic spectral anomalies due to transposed convolution upsampling,
#   checkerboard grid patterns, and high-frequency spectral roll-off anomalies.
#   This stream applies both 2D Fast Fourier Transform (FFT) and 2D Discrete
#   Cosine Transform (DCT-II) followed by a deep residual frequency CNN to
#   extract robust, domain-invariant frequency representations.
# ==============================================================================

# Import typing primitives
from typing import Optional, Tuple, List
import math

# Import PyTorch core neural network and tensor libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import project configurations and application logger
from configs.config import FrequencyStreamConfig, default_config
from src.utils.logger import app_logger


def compute_2d_dct(x: torch.Tensor) -> torch.Tensor:
    """
    Computes a fully differentiable 2D Discrete Cosine Transform (DCT-II)
    using PyTorch Fast Fourier Transform (FFT) operations.
    
    Parameters:
        x (torch.Tensor): Input spatial tensor of shape (B, C, H, W).
        
    Returns:
        torch.Tensor: 2D DCT-II spectral energy representation (B, C, H, W).
    """
    # Extract spatial height and width
    n_h, n_w = x.shape[-2], x.shape[-1]

    # Step 1: Compute 1D DCT across horizontal width dimension
    # Symmetrically reflect input along horizontal axis to double length: (B, C, H, 2*W)
    x_mirrored_w = torch.cat([x, x.flip(-1)], dim=-1)
    # Compute 1D FFT over the mirrored horizontal dimension
    fft_w = torch.fft.fft(x_mirrored_w, dim=-1)[..., :n_w]
    # Compute modulation phase weights exp(-i * pi * k / (2 * W))
    k_w = torch.arange(n_w, device=x.device, dtype=torch.float32)
    phase_w = torch.exp(-1j * math.pi * k_w / (2.0 * n_w))
    # Multiply and extract real component to yield 1D DCT coefficients
    dct_w = (fft_w * phase_w).real

    # Step 2: Compute 1D DCT across vertical height dimension
    # Symmetrically reflect input along vertical axis: (B, C, 2*H, W)
    x_mirrored_h = torch.cat([dct_w, dct_w.flip(-2)], dim=-2)
    # Compute 1D FFT over mirrored vertical dimension and truncate to H
    fft_h = torch.fft.fft(x_mirrored_h, dim=-2)[..., :n_h, :]
    # Compute modulation phase weights exp(-i * pi * k / (2 * H))
    k_h = torch.arange(n_h, device=x.device, dtype=torch.float32).view(-1, 1)
    phase_h = torch.exp(-1j * math.pi * k_h / (2.0 * n_h))
    # Multiply and extract real component to yield final 2D DCT matrix
    dct_2d = (fft_h * phase_h).real

    return dct_2d


class FourierTransform2D(nn.Module):
    """
    Differentiable in-graph 2D Fast Fourier Transform (FFT) & 2D DCT spectral layer.
    Transforms spatial RGB images into centralized log-magnitude spectral representations
    with numerical stability and gradient preservation.
    """

    def __init__(self, eps: float = 1e-8, channel_mode: str = "rgb"):
        """
        Parameters:
            eps (float): Small constant to prevent division by zero during normalization.
            channel_mode (str): 'rgb' preserves 3-channel spectral maps;
                                'grayscale' converts RGB to 1-channel luminance spectrum.
        """
        super().__init__()
        self.eps = eps
        self.channel_mode = channel_mode.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes 2D FFT and 2D DCT, centralizes DC frequency component, and
        extracts normalized log-magnitude representations.

        Parameters:
            x (torch.Tensor): Input spatial image tensor of shape (B, 3, H, W) with values in [0.0, 1.0].

        Returns:
            torch.Tensor: Normalized log-magnitude spectrum of shape (B, C, H, W),
                          where C=3 for 'rgb' mode and C=1 for 'grayscale' mode.
        """
        # 1. Convert to grayscale luminance if requested
        if self.channel_mode == "grayscale" and x.shape[1] == 3:
            # ITU-R 601-2 standard luminance weights (0.299 R + 0.587 G + 0.114 B)
            weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
            x_in = torch.sum(x * weights, dim=1, keepdim=True)
        else:
            x_in = x

        # 2. Compute 2D Fast Fourier Transform over spatial dimensions (H, W) with orthonormal scaling
        fft_complex = torch.fft.fft2(x_in, dim=(-2, -1), norm="ortho")

        # 3. Shift zero-frequency (DC) component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        # 4. Compute absolute magnitude spectrum: sqrt(real^2 + imag^2)
        magnitude = torch.abs(fft_shifted)

        # 5. Apply numerically stable log dynamic range compression: log(1 + magnitude)
        # Using log1p avoids log(0) singularities and large negative values
        log_fft = torch.log1p(magnitude)

        # 6. Compute 2D Discrete Cosine Transform for cardinal grid artifact detection
        raw_dct = compute_2d_dct(x_in)
        log_dct = torch.log1p(torch.abs(raw_dct))

        # 7. Stable per-sample min-max normalization
        b, c, h, w = log_fft.shape
        flat_fft = log_fft.reshape(b, c, -1)
        min_fft = flat_fft.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        max_fft = flat_fft.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        norm_fft = (log_fft - min_fft) / (max_fft - min_fft + self.eps)

        flat_dct = log_dct.reshape(b, c, -1)
        min_dct = flat_dct.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        max_dct = flat_dct.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        norm_dct = (log_dct - min_dct) / (max_dct - min_dct + self.eps)

        # 8. Blend 2D FFT (radial frequency energy) and 2D DCT (cardinal compression noise)
        # Weighted hybrid preserves exact channel dimensions (B, C, H, W)
        hybrid_spectrum = 0.60 * norm_fft + 0.40 * norm_dct

        return hybrid_spectrum


class ResidualFreqBlock(nn.Module):
    """
    Residual convolutional building block for the frequency spectrum CNN.
    Includes skip connection, Batch Normalization, and GELU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        # 1x1 projection shortcut if channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # Spatial downsampling via MaxPool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip connection and 2x spatial pooling.
        """
        residual = self.shortcut(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        out = self.pool(out)

        return out


class FrequencyCNN(nn.Module):
    """
    Lightweight Deep Convolutional Network tailored to extract spectral features
    from 2D Fourier and DCT magnitude maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        feature_dim: int = 512,
        dropout_rate: float = 0.30
    ):
        """
        Parameters:
            in_channels (int): Channels of input magnitude spectrum (3 for RGB, 1 for Grayscale).
            conv_channels (Optional[List[int]]): Progression of channel dimensions. Default: [32, 64, 128, 256].
            feature_dim (int): Output feature vector embedding dimension. Default is 512.
            dropout_rate (float): Dropout probability for regularization. Default is 0.30.
        """
        super().__init__()
        channels = conv_channels or [32, 64, 128, 256]

        # Construct convolutional blocks
        blocks = []
        curr_in = in_channels
        for c in channels:
            blocks.append(ResidualFreqBlock(curr_in, c))
            curr_in = c

        self.backbone = nn.Sequential(*blocks)

        # Global Average Pooling reduces spatial dimensions to (B, C, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection to target feature dimension
        last_conv_dim = channels[-1]
        self.proj = nn.Sequential(
            nn.Linear(last_conv_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Magnitude spectrum tensor (B, C, H, W).

        Returns:
            torch.Tensor: Frequency feature vector of shape (B, feature_dim).
        """
        # Pass through residual convolutional hierarchy
        feat_map = self.backbone(x)

        # Global average pool and flatten to 1D vector
        pooled = self.global_pool(feat_map).flatten(1)

        # Project to target feature space
        out = self.proj(pooled)

        return out


class FrequencyStream(nn.Module):
    """
    Complete Frequency Feature Extraction Stream for GUARD.
    Combines:
      1. Differential 2D Fast Fourier Transform (FFT) & 2D DCT spectral preprocessing.
      2. Lightweight Residual Frequency CNN.
    """

    def __init__(self, config: Optional[FrequencyStreamConfig] = None):
        """
        Initializes the Frequency Stream with configuration parameters.

        Parameters:
            config (Optional[FrequencyStreamConfig]): Configuration dataclass.
                If None, defaults to default_config.frequency.
        """
        super().__init__()
        self.config = config or default_config.frequency

        # 1. Instantiate 2D Fast Fourier Transform and 2D DCT module
        self.fft_module = FourierTransform2D(
            eps=self.config.eps,
            channel_mode=self.config.channel_mode
        )

        # 2. Determine input channels to CNN (3 if RGB, 1 if Grayscale)
        in_channels = 3 if self.config.channel_mode == "rgb" else 1

        # 3. Instantiate Lightweight Frequency CNN
        self.cnn = FrequencyCNN(
            in_channels=in_channels,
            conv_channels=self.config.conv_channels,
            feature_dim=self.config.feature_dim,
            dropout_rate=0.30
        )

        app_logger.info(
            f"FrequencyStream initialized: Mode={self.config.channel_mode}, "
            f"Channels={in_channels}, OutputDim={self.config.feature_dim}"
        )

    @property
    def out_features(self) -> int:
        """
        Returns output feature dimension (512-d).
        """
        return self.config.feature_dim

    def compute_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the 2D log-magnitude spectrum directly without CNN extraction.
        Useful for visualization, forensic plotting, and Streamlit dashboard inspection.

        Parameters:
            x (torch.Tensor): Spatial RGB image tensor (B, 3, H, W) in [0.0, 1.0].

        Returns:
            torch.Tensor: 2D magnitude spectrum (B, C, H, W) in [0.0, 1.0].
        """
        return self.fft_module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass:
          Spatial Image (B, 3, H, W) -> 2D FFT/DCT -> Magnitude Spectrum (B, C, H, W) -> CNN -> (B, 512).

        Parameters:
            x (torch.Tensor): Spatial RGB image tensor of shape (B, 3, H, W) in [0.0, 1.0].

        Returns:
            torch.Tensor: Frequency feature embedding of shape (B, 512).
        """
        # Step 1: Compute 2D centralized log-magnitude Fourier and DCT spectrum
        spectrum = self.fft_module(x)

        # Step 2: Feed magnitude spectrum into Lightweight Frequency CNN
        freq_features = self.cnn(spectrum)

        return freq_features
