# ==============================================================================
# File: src/engine/__init__.py
# Description: Training and Evaluation execution engine for GUARD.
# Exposes the Trainer and Evaluator classes.
# ==============================================================================

from src.engine.trainer import GUARDTrainer
from src.engine.evaluator import GUARDEvaluator

__all__ = [
    "GUARDTrainer",
    "GUARDEvaluator",
]
