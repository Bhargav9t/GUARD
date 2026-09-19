# ==============================================================================
# File: src/utils/__init__.py
# Description: Exposes utility functions for seed, device, logging, and metrics.
# ==============================================================================

from src.utils.seed import set_seed
from src.utils.device import get_device, get_device_info, print_device_summary
from src.utils.logger import setup_logger, app_logger
from src.utils.metrics import compute_eer, evaluate_predictions

__all__ = [
    "set_seed",
    "get_device",
    "get_device_info",
    "print_device_summary",
    "setup_logger",
    "app_logger",
    "compute_eer",
    "evaluate_predictions",
]
