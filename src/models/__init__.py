# ==============================================================================
# File: src/models/__init__.py
# Description: Model architecture package initialization for GUARD.
# Exposes neural network modules for spatial, frequency, fusion, head, and checkpoint loading.
# ==============================================================================

from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream, FourierTransform2D, FrequencyCNN
from .fusion import GatedFusion, CrossAttentionFusion, FusionLayer
from .classification_head import ClassificationHead
from .guard_model import GUARDModel, load_guard_checkpoint

__all__ = [
    "SpatialStream",
    "FrequencyStream",
    "FourierTransform2D",
    "FrequencyCNN",
    "GatedFusion",
    "CrossAttentionFusion",
    "FusionLayer",
    "ClassificationHead",
    "GUARDModel",
    "load_guard_checkpoint",
]
