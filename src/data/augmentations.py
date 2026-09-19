# ==============================================================================
# File: src/data/augmentations.py
# Description: Training-Time Robustness Augmentation Pipeline for GUARD.
#
# Forensic Rationale:
#   Deepfake and AI-generation detectors frequently overfit to compression
#   artifacts or sensor quirks specific to the source training dataset.
#   This module implements Albumentations pipelines that simulate real-world
#   social media transmissions, re-compressions, optical blurs, and downsamplings
#   to enforce domain-invariant spatial and frequency feature learning.
# ==============================================================================

from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from configs.config import AugmentationConfig, default_config
from src.utils.logger import app_logger


def get_training_augmentations(
    config: Optional[AugmentationConfig] = None,
    target_size: Tuple[int, int] = (380, 380)
) -> A.Compose:
    """
    Constructs the stochastic training data augmentation pipeline using Albumentations.

    Includes:
      - JPEG re-compression simulation (quality 30-95).
      - Resizing / downscaling artifacts.
      - Optical Gaussian / Motion / Median blur.
      - Sensor ISO Gaussian noise.
      - Flips and slight rotational perturbations.
      - Standard resize to target resolution.

    Parameters:
        config (Optional[AugmentationConfig]): Augmentation hyperparameters dataclass.
        target_size (Tuple[int, int]): Final image output dimensions (H, W).

    Returns:
        A.Compose: Composed Albumentations pipeline.
    """
    cfg = config or default_config.augmentation
    target_h, target_w = target_size

    transforms_list = [
        # 1. Horizontal Flip (50% probability) - maintains facial symmetry invariants
        A.HorizontalFlip(p=cfg.horizontal_flip_prob),

        # 2. Slight rotation, scaling, and translation to prevent spatial bias
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.05, 0.05),
            rotate=(-cfg.rotate_limit, cfg.rotate_limit),
            border_mode=cv2.BORDER_REFLECT_101,
            p=cfg.random_rotate_prob
        ),

        # 3. JPEG recompression artifact injection (crucial for social media robustness)
        A.ImageCompression(
            quality_range=cfg.jpeg_quality_range,
            compression_type="jpeg",
            p=cfg.jpeg_compression_prob
        ),

        # 4. Blur perturbations (Gaussian, Motion, or Median)
        A.OneOf([
            A.GaussianBlur(blur_limit=cfg.blur_limit),
            A.MotionBlur(blur_limit=cfg.blur_limit),
            A.MedianBlur(blur_limit=5),
        ], p=cfg.blur_prob),

        # 5. Sensor noise injection (forces frequency stream to ignore high-frequency sensor grain)
        A.GaussNoise(
            std_range=(0.05, 0.20),
            p=cfg.noise_prob
        ),

        # 6. Downscale & upscale simulation (emulating low-resolution platforms like Twitter/X or TikTok)
        A.Downscale(
            scale_range=cfg.downscale_scale_range,
            p=cfg.downscale_prob
        ),

        # 7. Color jittering (slight brightness and contrast shifts)
        A.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.05,
            p=0.30
        ),

        # 8. Deterministic resize to final model dimensions
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_CUBIC),
    ]

    return A.Compose(transforms_list)


def get_validation_augmentations(
    target_size: Tuple[int, int] = (380, 380)
) -> A.Compose:
    """
    Constructs a deterministic validation / testing augmentation pipeline.
    Applies only aspect-preserving resizing with zero random degradations.

    Parameters:
        target_size (Tuple[int, int]): Final dimensions (H, W).

    Returns:
        A.Compose: Deterministic resizing pipeline.
    """
    target_h, target_w = target_size
    return A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_CUBIC)
    ])


def get_degradation_sweep_pipeline(
    jpeg_quality: Optional[int] = None,
    blur_limit: Optional[int] = None,
    downscale_scale: Optional[float] = None,
    target_size: Tuple[int, int] = (380, 380)
) -> A.Compose:
    """
    Builds controlled single-factor degradation pipelines for robustness stress testing
    during evaluation protocol sweeps.

    Parameters:
        jpeg_quality (Optional[int]): Fixed JPEG quality (e.g. 90, 70, 50, 30).
        blur_limit (Optional[int]): Fixed blur kernel size (e.g. 3, 5, 7).
        downscale_scale (Optional[float]): Fixed downsampling ratio (e.g. 0.5, 0.25).
        target_size (Tuple[int, int]): Output dimensions (H, W).

    Returns:
        A.Compose: Controlled degradation transform pipeline.
    """
    transforms = []

    # 1. Apply JPEG recompression if requested
    if jpeg_quality is not None:
        transforms.append(
            A.ImageCompression(
                quality_range=(jpeg_quality, jpeg_quality),
                compression_type="jpeg",
                p=1.0
            )
        )

    # 2. Apply Gaussian blur if requested
    if blur_limit is not None:
        transforms.append(
            A.GaussianBlur(blur_limit=(blur_limit, blur_limit), p=1.0)
        )

    # 3. Apply Downscale downsampling if requested
    if downscale_scale is not None:
        transforms.append(
            A.Downscale(
                scale_range=(downscale_scale, downscale_scale),
                p=1.0
            )
        )

    # 4. Final resize to ensure uniform model input
    target_h, target_w = target_size
    transforms.append(A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_CUBIC))

    return A.Compose(transforms)


class AugmentationPipeline:
    """
    High-level wrapper managing data transformations and tensor conversions.
    Converts RGB NumPy arrays (H, W, 3) in uint8 to PyTorch FloatTensors (3, H, W) in [0.0, 1.0].
    """

    def __init__(
        self,
        mode: str = "train",
        config: Optional[AugmentationConfig] = None,
        target_size: Tuple[int, int] = (380, 380)
    ):
        """
        Parameters:
            mode (str): 'train' for stochastic degradations, 'val' or 'test' for deterministic.
            config (Optional[AugmentationConfig]): Config dataclass.
            target_size (Tuple[int, int]): Output dimensions (H, W).
        """
        self.mode = mode.lower()
        self.target_size = target_size
        self.config = config or default_config.augmentation

        if self.mode == "train":
            self.transform = get_training_augmentations(self.config, target_size=self.target_size)
        else:
            self.transform = get_validation_augmentations(target_size=self.target_size)

        # PyTorch Tensor formatting: (H, W, C) -> (C, H, W) with float32 in [0.0, 1.0]
        self.to_tensor = ToTensorV2()

        app_logger.info(f"AugmentationPipeline initialized: Mode={self.mode}, TargetSize={self.target_size}")

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Executes the augmentation pipeline on an input RGB image array.

        Parameters:
            image (np.ndarray): Input RGB array of shape (H, W, 3) and dtype uint8.

        Returns:
            torch.Tensor: Augmented tensor of shape (3, H, W) and dtype float32 scaled to [0.0, 1.0].
        """
        # Apply Albumentations spatial and degradation transforms
        augmented = self.transform(image=image)["image"]

        # Normalize pixel values from [0, 255] uint8 to [0.0, 1.0] float32
        norm_img = augmented.astype(np.float32) / 255.0

        # Convert to PyTorch Tensor format: (H, W, 3) -> (3, H, W)
        tensor = self.to_tensor(image=norm_img)["image"]

        return tensor

    def visualize_augmentations(
        self,
        image: np.ndarray,
        num_samples: int = 4,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generates multiple stochastic augmentation samples of a single image
        and stacks them horizontally for visual validation and debugging.

        Parameters:
            image (np.ndarray): Original RGB image (H, W, 3).
            num_samples (int): Number of perturbed variants to create.
            output_path (Optional[str]): Optional filepath to save composite visualization.

        Returns:
            np.ndarray: Composite image showing original followed by perturbed variants.
        """
        target_h, target_w = self.target_size
        orig_resized = cv2.resize(image, (target_w, target_h))

        samples = [orig_resized]
        # Temporarily use training transform if in val mode
        active_transform = (
            self.transform
            if self.mode == "train"
            else get_training_augmentations(self.config, self.target_size)
        )

        for _ in range(num_samples):
            aug_sample = active_transform(image=image)["image"]
            samples.append(aug_sample)

        # Horizontally stack all samples
        composite = np.hstack(samples)

        if output_path is not None:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            app_logger.info(f"Saved augmentation visualization to: {output_path}")

        return composite
