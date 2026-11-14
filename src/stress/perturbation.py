"""
In-Memory Degradation Stress-Test Module for GUARD.
Assesses forensic perturbation stability under standard transmission degradation:
  1. JPEG Compression Q=85 (Mild messaging transmission)
  2. JPEG Compression Q=50 (Social media recompression)
  3. Resizing 0.75x (Screenshot / mobile downscaling & bicubic upsampling)
  4. Resizing 0.50x (Severe thumbnail downsampling & bicubic upsampling)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.detectors.base import BaseDetector, DetectionResult


@dataclass
class DegradationVariantResult:
    variant_name: str
    degradation_type: str
    parameters: str
    predicted_label: str
    is_ai: bool
    ai_probability: float
    confidence: float
    score_delta: float                  # p_degraded - p_clean
    verdict_matches_baseline: bool
    degraded_image_rgb: np.ndarray


@dataclass
class StressTestSummary:
    baseline_result: DetectionResult
    variant_results: List[DegradationVariantResult]
    total_perturbations: int
    matching_perturbations: int
    majority_verdict_flipped: bool      # True if majority flipped relative to clean
    is_stable: bool                     # True if at least 3/4 perturbations stay stable
    stability_score_pct: float          # Percentage of perturbations maintaining stability
    disclaimer: str = (
        "Perturbation Stability Notice: Assesses resilience against standard transmission "
        "compression (JPEG, Resizing) in-memory. This evaluates empirical heuristic stability, "
        "not formal mathematical adversarial robustness."
    )


def apply_jpeg_compression(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Simulates in-memory JPEG compression and re-decoding via OpenCV."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    success, encimg = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return image_rgb.copy()
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)


def apply_resize_degradation(image_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Simulates downscaling followed by bicubic upsampling back to original resolution."""
    h, w = image_rgb.shape[:2]
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    down = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def run_degradation_stress_test(
    image: Union[np.ndarray, Image.Image],
    detector: BaseDetector,
    threshold: float = 0.50,
) -> StressTestSummary:
    """
    Executes the 4-variant in-memory degradation stress test.
    """
    if isinstance(image, Image.Image):
        clean_rgb = np.array(image.convert("RGB"))
    else:
        clean_rgb = image.copy()

    # 1. Clean Baseline Inference
    baseline_result = detector.predict_image(clean_rgb, threshold=threshold)
    clean_prob = baseline_result.ai_probability
    clean_is_ai = baseline_result.is_ai

    # 2. Define the 4 specified perturbation tests
    perturbation_specs = [
        ("JPEG Compression (Q85)", "JPEG", "Quality=85 (Mild Recompression)", apply_jpeg_compression(clean_rgb, 85)),
        ("JPEG Compression (Q50)", "JPEG", "Quality=50 (Social Media Recompression)", apply_jpeg_compression(clean_rgb, 50)),
        ("Downscale-Upscale (0.75x)", "Resizing", "Scale=0.75x Area Down + Bicubic Up", apply_resize_degradation(clean_rgb, 0.75)),
        ("Downscale-Upscale (0.50x)", "Resizing", "Scale=0.50x Area Down + Bicubic Up", apply_resize_degradation(clean_rgb, 0.50)),
    ]

    variant_results = []
    matching_count = 0

    for name, deg_type, params, deg_img in perturbation_specs:
        res = detector.predict_image(deg_img, threshold=threshold)
        matches = (res.is_ai == clean_is_ai)
        if matches:
            matching_count += 1

        delta = res.ai_probability - clean_prob

        variant_results.append(
            DegradationVariantResult(
                variant_name=name,
                degradation_type=deg_type,
                parameters=params,
                predicted_label=res.predicted_label,
                is_ai=res.is_ai,
                ai_probability=res.ai_probability,
                confidence=res.confidence,
                score_delta=delta,
                verdict_matches_baseline=matches,
                degraded_image_rgb=deg_img,
            )
        )

    total_pert = len(perturbation_specs)  # 4
    stability_pct = (matching_count / total_pert) * 100.0

    # Majority verdict flipped if less than half match baseline (i.e. <= 1 out of 4 match)
    majority_flipped = (matching_count < 2)
    # Is stable if at least 3 out of 4 agree with baseline verdict
    is_stable = (matching_count >= 3)

    return StressTestSummary(
        baseline_result=baseline_result,
        variant_results=variant_results,
        total_perturbations=total_pert,
        matching_perturbations=matching_count,
        majority_verdict_flipped=majority_flipped,
        is_stable=is_stable,
        stability_score_pct=stability_pct,
    )
