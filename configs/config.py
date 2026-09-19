# ==============================================================================
# File: configs/config.py
# Description: Central Configuration Module for GUARD (Generalizable Universal 
#              AI-generated & Real-image Detector).
# Architecture:
#   - Two-stream dual-branch architecture (Spatial: EfficientNet-B4 + Frequency: 2D FFT CNN)
#   - Cross-attention / Gated Fusion layer
#   - Binary Real/Fake classification head
# ==============================================================================

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Project Root Directory (resolved dynamically relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """
    Manages filesystem paths for datasets, model checkpoints, logs, and outputs.
    Centralizing paths avoids hardcoded strings throughout the codebase.
    """
    # Root workspace directory
    root_dir: Path = BASE_DIR
    
    # Directory where datasets (raw and preprocessed) are stored
    data_dir: Path = BASE_DIR / "data"
    
    # Directory to store trained model weights and checkpoints
    checkpoints_dir: Path = BASE_DIR / "checkpoints"
    
    # Directory to store runtime training logs, tensorboard runs, or evaluation reports
    logs_dir: Path = BASE_DIR / "logs"
    
    # Directory for generated visual artifacts (ROC curves, heatmaps, frequency plots)
    outputs_dir: Path = BASE_DIR / "outputs"

    def create_dirs(self) -> None:
        """Create directories on disk if they do not already exist."""
        for path in [self.data_dir, self.checkpoints_dir, self.logs_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FaceAlignmentConfig:
    """
    Configuration parameters for optional face detection, landmark extraction,
    and bounding-box alignment before feeding images to the spatial/frequency streams.
    """
    # Flag to enable/disable automated face cropping
    enable_face_crop: bool = True
    
    # Detector backend: 'mtcnn', 'retinaface', or 'opencv_haar' as fallback
    detector_backend: str = "retinaface"
    
    # Confidence threshold to filter out low-confidence face bounding boxes
    confidence_threshold: float = 0.90
    
    # Margin expansion ratio around the detected face bounding box (0.25 = +25% padding)
    # Adding margin ensures ear, forehead, and jaw boundary artifacts are preserved
    crop_margin: float = 0.25
    
    # Default alignment target size before input to the model (Height, Width)
    target_face_size: Tuple[int, int] = (380, 380)


@dataclass
class SpatialStreamConfig:
    """
    Configuration for the RGB Spatial Feature Extraction Stream.
    Uses an EfficientNet-B4 backbone from PyTorch Image Models (timm).
    """
    # timm model identifier for EfficientNet-B4
    backbone_name: str = "efficientnet_b4"
    
    # Whether to load pre-trained ImageNet-1k weights
    pretrained: bool = True
    
    # Spatial input resolution expected by EfficientNet-B4 (380x380 is standard)
    image_size: Tuple[int, int] = (380, 380)
    
    # Normalization constants (ImageNet defaults)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Dimensionality of the extracted feature vector from EfficientNet-B4 pooling stage
    feature_dim: int = 1792
    
    # Dropout rate applied to spatial representations
    dropout_rate: float = 0.30


@dataclass
class FrequencyStreamConfig:
    """
    Configuration for the Frequency Feature Extraction Stream.
    Transforms spatial RGB images into 2D Fast Fourier Transform (FFT) magnitude
    spectrum maps and extracts spectral artifact representations via a lightweight CNN.
    """
    # Input size for frequency spectrum generation (Height, Width)
    input_size: Tuple[int, int] = (380, 380)
    
    # Channel mode: 'grayscale' (1-channel log-magnitude) or 'rgb' (3-channel independent log-magnitude)
    channel_mode: str = "rgb"
    
    # Number of input channels to the frequency lightweight CNN (3 if RGB, 1 if Grayscale)
    in_channels: int = 3
    
    # Feature dimension output produced by the frequency lightweight CNN
    feature_dim: int = 512
    
    # CNN channel projection progression
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Epsilon constant added to prevent log(0) calculation during log-scale magnitude spectrum conversion
    eps: float = 1e-8


@dataclass
class FusionConfig:
    """
    Configuration for the Multi-Modal Spatial-Frequency Fusion Layer.
    Fuses spatial features (1792-d) and frequency features (512-d) using
    concatenation followed by cross-attention or gating mechanism.
    """
    # Type of fusion: 'cross_attention', 'gated', or 'concat_linear'
    fusion_type: str = "gated"
    
    # Hidden projection dimension for joint representation space
    hidden_dim: int = 1024
    
    # Number of attention heads (if using multi-head attention fusion)
    num_heads: int = 4
    
    # Dropout applied within the fusion block
    dropout_rate: float = 0.30


@dataclass
class HeadConfig:
    """
    Configuration for the final Binary Classification Head.
    Outputs the logit / probability of the input image being Real (0) or Fake (1).
    """
    # Number of output classes: 1 for binary classification with BCEWithLogitsLoss / Sigmoid
    num_classes: int = 1
    
    # Class labels mapping
    labels: Tuple[str, str] = ("Real", "AI-Generated / Manipulated")
    
    # Classification threshold (>= threshold is classified as Fake)
    decision_threshold: float = 0.50


@dataclass
class AugmentationConfig:
    """
    Configuration for the Albumentations Training-Time Augmentation Pipeline.
    Simulates real-world social media degradations, compression, and sensor noise
    to enforce generalizability and prevent overfitting to generator-specific artifacts.
    """
    # Probability of applying JPEG recompression artifact injection
    jpeg_compression_prob: float = 0.50
    jpeg_quality_range: Tuple[int, int] = (30, 95)
    
    # Probability of applying Gaussian / Motion Blur
    blur_prob: float = 0.30
    blur_limit: Tuple[int, int] = (3, 7)
    
    # Probability of applying Gaussian Noise / ISO sensor grain
    noise_prob: float = 0.30
    noise_var_limit: Tuple[float, float] = (10.0, 50.0)
    
    # Probability of applying Random Resizing / Downsampling-Upsampling
    downscale_prob: float = 0.30
    downscale_scale_range: Tuple[float, float] = (0.5, 0.95)
    
    # Geometric transformations
    horizontal_flip_prob: float = 0.50
    random_rotate_prob: float = 0.30
    rotate_limit: int = 15


@dataclass
class TrainingConfig:
    """
    Hyperparameters and optimization settings for training GUARD.
    """
    # Seed for full deterministic reproducibility
    seed: int = 42
    
    # Batch size per GPU / device
    batch_size: int = 16
    
    # Number of data loader worker threads
    num_workers: int = 4
    
    # Number of training epochs
    epochs: int = 50
    
    # Base learning rate for AdamW optimizer
    learning_rate: float = 1e-4
    
    # Weight decay for L2 regularization
    weight_decay: float = 1e-4
    
    # Learning rate scheduler type ('cosine', 'step', or 'plateau')
    scheduler_type: str = "cosine"
    
    # Minimum learning rate target for cosine annealing
    min_lr: float = 1e-6
    
    # Warmup epochs before standard schedule
    warmup_epochs: int = 3
    
    # Maximum gradient norm for gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision training enabled (FP16 / AMP)
    mixed_precision: bool = True
    
    # Checkpoint saving frequency (save every N epochs)
    save_frequency: int = 5


@dataclass
class EvaluationConfig:
    """
    Configuration for cross-dataset validation and degradation sweeps.
    """
    # Benchmark datasets targeted for validation
    benchmarks: List[str] = field(
        default_factory=lambda: ["FaceForensics++", "Celeb-DF-v2", "DFDC", "DiffusionForensics"]
    )
    
    # Quality degradation factors to sweep during robustness testing
    jpeg_sweep_qualities: List[int] = field(default_factory=lambda: [90, 70, 50, 30])
    blur_sweep_sigmas: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])


@dataclass
class UIConfig:
    """
    Settings for the Futuristic Tech Dashboard (Streamlit / Web App).
    """
    # Host binding address
    host: str = "127.0.0.1"
    
    # Port to run the dashboard
    port: int = 8501
    
    # Application Title
    app_title: str = "GUARD // AI-Generated & Real Image Detector"
    
    # Maximum uploaded image file size in Megabytes
    max_upload_size_mb: int = 25


@dataclass
class GuardConfig:
    """
    Master configuration class encapsulating all sub-configurations for GUARD.
    """
    paths: PathConfig = field(default_factory=PathConfig)
    face_alignment: FaceAlignmentConfig = field(default_factory=FaceAlignmentConfig)
    spatial: SpatialStreamConfig = field(default_factory=SpatialStreamConfig)
    frequency: FrequencyStreamConfig = field(default_factory=FrequencyStreamConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Default singleton instance of master configuration
default_config = GuardConfig()

# Expose key configurations when module is imported
__all__ = [
    "GuardConfig",
    "PathConfig",
    "FaceAlignmentConfig",
    "SpatialStreamConfig",
    "FrequencyStreamConfig",
    "FusionConfig",
    "HeadConfig",
    "AugmentationConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "UIConfig",
    "default_config",
]
