"""
Abstract BaseDetector Interface for GUARD.
Provides a unified contract for AI-generated media detection engines,
allowing off-the-shelf Hugging Face baselines and custom dual-stream
timm architectures to be swapped interchangeably without altering the UI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


@dataclass
class DetectionResult:
    """
    Standardized detection result dataclass.
    All detectors normalize their raw outputs into this contract.
    """
    is_ai: bool
    ai_probability: float               # Uncalibrated softmax probability [0.0, 1.0]
    confidence: float                   # Distance from uncertainty: max(p, 1-p)
    predicted_label: str                # Normalized: "AI-Generated" or "Real / Authentic"
    raw_scores: Dict[str, float]        # Original unnormalized dictionary from backend
    device: str                         # "cuda" or "cpu"
    dtype: str                          # "float16" or "float32"
    model_name: str                     # Identifier of active inference backbone
    inference_time_ms: float = 0.0      # Wall-clock latency for telemetry
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.predicted_label,
            "is_ai": self.is_ai,
            "ai_probability": round(self.ai_probability, 4),
            "confidence": round(self.confidence, 4),
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "latency_ms": round(self.inference_time_ms, 2),
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """
    Abstract Base Class for all GUARD detector backends.
    """

    @abstractmethod
    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.50,
    ) -> DetectionResult:
        """
        Runs inference on a single image and returns a normalized DetectionResult.

        Parameters:
            image: Input image as RGB numpy array (H, W, 3) or PIL Image.
            threshold: Decision boundary for AI classification (default: 0.50).
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        threshold: float = 0.50,
    ) -> List[DetectionResult]:
        """
        Runs inference on a batch of images for fast video frame processing.
        """
        pass

    @abstractmethod
    def get_spatial_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        target_class_idx: int = 0,
        method: str = "gradcam",
    ) -> np.ndarray:
        """
        Extracts a 2D spatial heatmap (H, W) in [0.0, 1.0] indicating
        which spatial regions drove the AI classification verdict.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name / identifier of the underlying model architecture."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Active computing hardware device (CUDA or CPU)."""
        pass

    @property
    @abstractmethod
    def is_fp16(self) -> bool:
        """True if running in half-precision (FP16)."""
        pass
