"""
Detectors Package for GUARD.
Provides factory function get_detector() to instantiate the hybrid multi-evidence
engine, Hugging Face pretrained models, or the custom Phase 2 dual-stream engine.
"""

from typing import Optional
from src.detectors.base import BaseDetector, DetectionResult
from src.detectors.hf_detector import HuggingFaceDetector
from src.detectors.dual_stream_adapter import DualStreamDetector
from src.detectors.hybrid_engine import MultiEvidenceHybridDetector

SUPPORTED_ENGINES = {
    "GUARD Multi-Evidence Hybrid (Swin + SDXL + Midjourney + Physics) [RECOMMENDED]": ("hybrid", "umm-maybe/AI-image-detector"),
    "umm-maybe/AI-image-detector (General AI & Art Swin)": ("hf", "umm-maybe/AI-image-detector"),
    "Organika/sdxl-detector (SDXL & Photorealism Swin)": ("hf", "Organika/sdxl-detector"),
    "ideepankarsharma2003/Midjourney_v6_ImageClassifier (Midjourney v6)": ("hf", "ideepankarsharma2003/Midjourney_v6_ImageClassifier"),
    "GUARD Dual-Stream (Phase 2 Custom: Spatial + 2D FFT)": ("dual_stream", "checkpoints/guard_model.pth"),
}

def get_detector(
    engine_name: str = "GUARD Multi-Evidence Hybrid (Swin + SDXL + Midjourney + Physics) [RECOMMENDED]",
    force_cpu: bool = False,
    use_fp16: bool = True,
) -> BaseDetector:
    """
    Factory function to retrieve initialized detector backend.
    """
    engine_type, path_or_id = SUPPORTED_ENGINES.get(
        engine_name,
        ("hybrid", "umm-maybe/AI-image-detector")
    )

    if engine_type == "hybrid":
        return MultiEvidenceHybridDetector(
            primary_model_id=path_or_id,
            force_cpu=force_cpu,
            use_fp16_on_cuda=use_fp16,
        )
    elif engine_type == "hf":
        return HuggingFaceDetector(
            model_id=path_or_id,
            force_cpu=force_cpu,
            use_fp16_on_cuda=use_fp16,
        )
    elif engine_type == "dual_stream":
        return DualStreamDetector(
            checkpoint_path=path_or_id,
            force_cpu=force_cpu,
        )
    else:
        raise ValueError(f"Unknown detector engine type: {engine_type}")

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "HuggingFaceDetector",
    "DualStreamDetector",
    "MultiEvidenceHybridDetector",
    "get_detector",
    "SUPPORTED_ENGINES",
]
