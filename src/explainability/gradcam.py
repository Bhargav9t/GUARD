"""
Spatial Heatmap Rendering & Visual Overlay Module for GUARD.
Converts 2D Grad-CAM or Activation Norm arrays into color-mapped overlays
blended onto the original image.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def create_heatmap_overlay(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.50,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blends a 2D float heatmap [0.0, 1.0] onto an image.

    Parameters:
        image: Original input image as RGB numpy array (H, W, 3) or PIL Image.
        heatmap: 2D float array (H, W) in range [0.0, 1.0].
        alpha: Blending weight for the heatmap overlay (0.0 to 1.0).
        colormap: OpenCV colormap (default: cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB image with dimensions of the original image.
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]

    # Ensure heatmap matches target image dimensions
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap strictly to [0, 255]
    norm_hm = np.clip(heatmap, 0.0, 1.0)
    uint8_hm = (norm_hm * 255.0).astype(np.uint8)

    # Apply colormap
    color_bgr = cv2.applyColorMap(uint8_hm, colormap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and colored heatmap
    blended = cv2.addWeighted(rgb, 1.0 - alpha, color_rgb, alpha, 0)
    return blended
