"""
Stress-test package exports for GUARD.
"""

from src.stress.perturbation import (
    DegradationVariantResult,
    StressTestSummary,
    run_degradation_stress_test,
    apply_jpeg_compression,
    apply_resize_degradation,
)

__all__ = [
    "DegradationVariantResult",
    "StressTestSummary",
    "run_degradation_stress_test",
    "apply_jpeg_compression",
    "apply_resize_degradation",
]
