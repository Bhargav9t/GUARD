"""
Dual-Stream Timm Detector Adapter for GUARD.
Demonstrates the extensibility of the BaseDetector architecture:
allows the Phase 2 custom-trained dual-stream (Spatial EfficientNet + 2D FFT Frequency)
checkpoint to plug in seamlessly as a drop-in replacement for the Hugging Face baseline
without changing a single line of Streamlit UI code.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
import cv2

from src.detectors.base import BaseDetector, DetectionResult
from src.model import load_guard_checkpoint


class DualStreamDetector(BaseDetector):
    """
    Phase 2 Custom Dual-Stream Detector Adapter.
    Combines spatial deep features with 2D FFT spectral residual embeddings
    via adaptive gated attention fusion.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/guard_model.pth",
        force_cpu: bool = False,
    ):
        self._checkpoint_path = checkpoint_path
        if not force_cpu and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
            self._is_fp16 = False  # Keep FP32 for custom fusion stability
        else:
            self._device = torch.device("cpu")
            self._is_fp16 = False

        self.model, self.is_trained, self.status_info = load_guard_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self._device,
        )
        self.model.eval()

    def _prepare_tensor(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = image.copy()

        resized = cv2.resize(rgb, (380, 380), interpolation=cv2.INTER_AREA)
        t = resized.astype(np.float32) / 255.0
        t = np.transpose(t, (2, 0, 1))
        tensor = torch.from_numpy(t).unsqueeze(0).to(self._device)
        return tensor

    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.50,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        t_input = self._prepare_tensor(image)

        with torch.no_grad():
            pred = self.model.predict(t_input, threshold=threshold)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_ai = bool(pred["is_fake"])
        ai_prob = float(pred["fake_probability"])
        confidence = float(pred["confidence"])
        predicted_label = "AI-Generated" if is_ai else "Real / Human"

        return DetectionResult(
            is_ai=is_ai,
            ai_probability=ai_prob,
            confidence=confidence,
            predicted_label=predicted_label,
            raw_scores={
                "spatial_gating_ratio": float(pred["spatial_gating_ratio"]),
                "frequency_gating_ratio": float(pred["frequency_gating_ratio"]),
                "combined_dual_score": float(pred.get("combined_dual_score", ai_prob)),
            },
            device=str(self._device),
            dtype="float32",
            model_name="GUARD-DualStream-Phase2 (EfficientNet-B4 + 2D FFT)",
            inference_time_ms=latency_ms,
            metadata={
                "spatial_gating": pred["spatial_gating_ratio"],
                "frequency_gating": pred["frequency_gating_ratio"],
                "checkpoint": self._checkpoint_path,
            },
        )

    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        threshold: float = 0.50,
    ) -> List[DetectionResult]:
        return [self.predict_image(img, threshold=threshold) for img in images]

    def get_spatial_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        target_class_idx: int = 0,
        method: str = "gradcam",
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = image.copy()
        h_orig, w_orig = rgb.shape[:2]

        t_input = self._prepare_tensor(rgb)
        try:
            expl = self.model.extract_explainability_data(t_input, threshold=0.50)
            heatmap_tensor = expl["spatial_heatmap_tensor"]  # (1, 1, H, W)
            hm_np = heatmap_tensor.squeeze().cpu().numpy()
            resized = cv2.resize(hm_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            return np.clip(resized, 0.0, 1.0)
        except Exception:
            return np.ones((h_orig, w_orig), dtype=np.float32) * 0.5

    @property
    def model_name(self) -> str:
        return "GUARD-DualStream-Phase2 (Spatial + 2D FFT)"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_fp16(self) -> bool:
        return self._is_fp16
