# ==============================================================================
# File: src/model.py
# Description: Compatibility wrapper and public alias for GUARDModel.
# Exposes the master dual-stream spatial-frequency detector and checkpoint loaders.
# ==============================================================================

# Re-export core classes and factory helpers from the src.models package
from src.models.guard_model import (
    GUARDModel,
    load_guard_checkpoint,
)
from src.models.spatial_stream import SpatialStream
from src.models.frequency_stream import FrequencyStream, FourierTransform2D
from src.models.fusion import FusionLayer, GatedFusion, CrossAttentionFusion
from src.models.classification_head import ClassificationHead

__all__ = [
    "GUARDModel",
    "load_guard_checkpoint",
    "SpatialStream",
    "FrequencyStream",
    "FourierTransform2D",
    "FusionLayer",
    "GatedFusion",
    "CrossAttentionFusion",
    "ClassificationHead",
]
