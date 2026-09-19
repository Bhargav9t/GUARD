# ==============================================================================
# File: src/preprocessing/__init__.py
# Description: Preprocessing package initialization for GUARD.
# Exposes face alignment, extraction, and normalization tools.
# ==============================================================================

from .face_aligner import FaceAligner

__all__ = [
    "FaceAligner",
]
