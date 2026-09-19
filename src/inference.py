# ==============================================================================
# File: src/inference.py
# Description: Calibrated Inference Pipeline & Forensic Predictor for GUARD.
#
# Key Features:
#   - Safe model weight loading from 'checkpoints/guard_model.pth' with fallback
#   - Differentiable face alignment and canonical square cropping
#   - Calibrated sigmoid probability calculation with tunable decision threshold
#   - Dual-stream forensic breakdown: Spatial Stream (EfficientNet-B4) vs.
#     Frequency Stream (2D FFT/DCT) individual logit and confidence metrics
#   - Multi-modal explainability assets: 2D FFT spectrogram and Grad-CAM heatmap
# ==============================================================================

# Import operating system and typing utilities
import os
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

# Import scientific computing and image processing libraries
import numpy as np
import cv2
from PIL import Image

# Import PyTorch tensor operations
import torch

# Import GUARD configurations and core modules
from configs import default_config, GuardConfig
from src.model import GUARDModel, load_guard_checkpoint
from src.preprocessing import FaceAligner
from src.utils import get_device, app_logger


class ForensicInferenceEngine:
    """
    Production-grade inference engine for GUARD AI Image Forensics.
    Encapsulates image loading, face alignment, dual-stream inference,
    logit calibration, threshold-based classification, and explainability assets.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        config: Optional[GuardConfig] = None,
        enable_face_crop: bool = True
    ):
        """
        Initializes the inference engine and loads weights into memory.

        Parameters:
            checkpoint_path (Optional[Union[str, Path]]): Filepath to model weights.
            device (Optional[torch.device]): Hardware compute device (CUDA or CPU).
            config (Optional[GuardConfig]): Master configuration object.
            enable_face_crop (bool): Whether to perform automated face alignment.
        """
        # Assign configuration
        self.config = config or default_config
        # Resolve target device
        self.device = device or get_device()

        # Resolve checkpoint path dynamically
        if checkpoint_path is None:
            self.checkpoint_path = str(self.config.paths.checkpoints_dir / "guard_model.pth")
        else:
            self.checkpoint_path = str(checkpoint_path)

        # 1. Load GUARD neural network and calibrated weights
        self.model, self.is_trained, self.status_info = load_guard_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            config=self.config
        )

        # 2. Instantiate FaceAligner
        aligner_cfg = self.config.face_alignment
        aligner_cfg.enable_face_crop = enable_face_crop
        self.aligner = FaceAligner(config=aligner_cfg)

        app_logger.info(
            f"ForensicInferenceEngine initialized: Device={self.device}, "
            f"Calibrated={self.is_trained}, Checkpoint={self.checkpoint_path}"
        )

    def preprocess_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
        """
        Loads and prepares an input image into a normalized PyTorch tensor.

        Parameters:
            image_input: Filepath string/Path, NumPy array (H, W, 3) in RGB, or PIL Image.

        Returns:
            Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
                - tensor: PyTorch FloatTensor of shape (1, 3, 380, 380) in [0.0, 1.0].
                - aligned_rgb: Processed NumPy RGB array of shape (380, 380, 3) in uint8.
                - alignment_info: Dictionary detailing face detection status and bounding box.
        """
        # Convert PIL Image to NumPy array if needed
        if isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, (str, Path)):
            rgb_arr = self.aligner.load_image(str(image_input))
        elif isinstance(image_input, np.ndarray):
            rgb_arr = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Execute canonical face alignment and center-crop fallback
        alignment_result = self.aligner.process(rgb_arr)
        aligned_rgb = alignment_result["aligned_image"]

        # Convert uint8 (H, W, 3) to normalized float32 tensor (1, 3, 380, 380) in [0.0, 1.0]
        tensor = aligned_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # (C, H, W)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        return tensor, aligned_rgb, alignment_result

    @torch.no_grad()
    def inspect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes full dual-stream forensic inspection on an input image.

        Parameters:
            image_input: Target image to evaluate.
            threshold (float): Decision threshold in [0.0, 1.0] (default 0.50).
                Images with fake_probability >= threshold are classified as Fake.

        Returns:
            Dict[str, Any]: Comprehensive forensic verdict and telemetry dictionary.
        """
        # Step 1: Preprocess and format input tensor
        tensor, aligned_rgb, align_info = self.preprocess_image(image_input)

        # Step 2: Extract explainability assets and stream predictions
        explain_data = self.model.extract_explainability_data(tensor, threshold=threshold)
        pred_info = explain_data["prediction"]

        # Step 3: Package comprehensive results
        return {
            "is_fake": pred_info["is_fake"],
            "label": pred_info["label"],
            "fake_probability": pred_info["fake_probability"],
            "confidence": pred_info["confidence"],
            "threshold": threshold,
            "raw_logit": pred_info.get("logit", 0.0),
            "spatial_gating_ratio": pred_info.get("spatial_gating_ratio", 0.5),
            "frequency_gating_ratio": pred_info.get("frequency_gating_ratio", 0.5),
            "stream_breakdown": pred_info.get("stream_breakdown", {}),
            "aligned_image": aligned_rgb,
            "face_detected": align_info["face_detected"],
            "bbox": align_info.get("bbox"),
            "spectrum_tensor": explain_data["spectrum_tensor"],
            "spatial_heatmap_tensor": explain_data["spatial_heatmap_tensor"],
            "checkpoint_status": self.status_info,
        }


# Convenience singleton loader
_cached_engine: Optional[ForensicInferenceEngine] = None

def get_inference_engine(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> ForensicInferenceEngine:
    """
    Returns a cached instance of ForensicInferenceEngine to avoid redundant model reloading.
    """
    global _cached_engine
    if _cached_engine is None:
        _cached_engine = ForensicInferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device
        )
    return _cached_engine
