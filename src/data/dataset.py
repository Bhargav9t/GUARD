# ==============================================================================
# File: src/data/dataset.py
# Description: PyTorch Dataset and DataLoader abstractions for GUARD.
# Seamlessly integrates FaceAligner, Albumentations, and multi-threaded loading.
# ==============================================================================

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.face_aligner import FaceAligner
from src.data.augmentations import AugmentationPipeline
from src.utils.logger import app_logger


class GUARDDataset(Dataset):
    """
    Standard PyTorch Dataset for Deepfake and AI-generated image forensics.
    Reads images from directory structures or explicit sample lists,
    executes face alignment/cropping, and applies training or validation augmentations.
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[Union[str, Path], int]]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        mode: str = "train",
        enable_face_align: bool = True,
        target_size: Tuple[int, int] = (380, 380),
    ):
        """
        Parameters:
            samples (Optional[List[Tuple[Path, int]]]): List of (image_path, label) pairs.
                Label 0 = Real, Label 1 = Fake.
            data_dir (Optional[Path]): Directory path containing 'real' and 'fake' subfolders.
                Used if `samples` is None.
            mode (str): 'train' for stochastic degradations, 'val' or 'test' for deterministic.
            enable_face_align (bool): Whether to pass inputs through FaceAligner.
            target_size (Tuple[int, int]): Dimensions (H, W) for network input.
        """
        super().__init__()
        self.mode = mode.lower()
        self.enable_face_align = enable_face_align
        self.target_size = target_size

        # 1. Initialize sample list
        if samples is not None:
            self.samples = [(str(p), int(l)) for p, l in samples]
        elif data_dir is not None:
            self.samples = self._load_samples_from_dir(Path(data_dir))
        else:
            self.samples = []

        # 2. Instantiate FaceAligner
        if self.enable_face_align:
            self.aligner = FaceAligner()
        else:
            self.aligner = None

        # 3. Instantiate Augmentation Pipeline
        self.augmentation = AugmentationPipeline(mode=self.mode, target_size=self.target_size)

        app_logger.info(
            f"GUARDDataset initialized: Mode={self.mode}, TotalSamples={len(self.samples)}, "
            f"FaceAlign={self.enable_face_align}"
        )

    @staticmethod
    def _load_samples_from_dir(data_dir: Path) -> List[Tuple[str, int]]:
        """
        Recursively discovers image files organized in 'real' (0) and 'fake' (1) subdirectories.

        Parameters:
            data_dir (Path): Root directory containing 'real' and 'fake' subfolders.

        Returns:
            List[Tuple[str, int]]: List of (filepath, binary_label) tuples.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        samples = []

        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"

        # Discover authentic images (Label 0)
        if real_dir.is_dir():
            for p in real_dir.rglob("*"):
                if p.suffix.lower() in valid_extensions:
                    samples.append((str(p), 0))

        # Discover manipulated / AI-generated images (Label 1)
        if fake_dir.is_dir():
            for p in fake_dir.rglob("*"):
                if p.suffix.lower() in valid_extensions:
                    samples.append((str(p), 1))

        return samples

    def __len__(self) -> int:
        """
        Returns total number of images in dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads, preprocesses, aligns, and augments a single image sample.

        Parameters:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]:
                - 'image': PyTorch Tensor of shape (3, H, W) scaled to [0.0, 1.0].
                - 'label': PyTorch FloatTensor of shape (1,) with value 0.0 (Real) or 1.0 (Fake).
                - 'path': Original file path string.
                - 'face_detected': Boolean flag indicating if a face was located.
        """
        img_path, label = self.samples[idx]

        # 1. Load image and perform face alignment if enabled
        if self.enable_face_align and self.aligner is not None:
            processed = self.aligner.process(img_path)
            img_rgb = processed["aligned_image"]
            face_detected = processed["face_detected"]
        else:
            # Fallback loading without face alignment
            img_rgb = FaceAligner().load_image(img_path)
            face_detected = False

        # 2. Apply Albumentations augmentation pipeline & convert to tensor (3, H, W)
        tensor_img = self.augmentation(img_rgb)

        # 3. Format label as (1,) FloatTensor for BCEWithLogitsLoss
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        return {
            "image": tensor_img,
            "label": label_tensor,
            "path": img_path,
            "face_detected": face_detected,
        }


def create_dataloader(
    dataset: GUARDDataset,
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Factory creating high-throughput PyTorch DataLoader.

    Parameters:
        dataset (GUARDDataset): Initialized dataset instance.
        batch_size (int): Number of images per batch.
        shuffle (Optional[bool]): Whether to randomize order. Defaults to True for train, False for val/test.
        num_workers (int): Background subprocess worker count for data fetching.
        pin_memory (bool): If True, pins host memory for accelerated CUDA transfers.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # Auto-determine shuffle behavior based on dataset mode
    if shuffle is None:
        shuffle = (dataset.mode == "train")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(dataset.mode == "train"),
    )
