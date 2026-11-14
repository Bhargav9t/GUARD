"""
2D Fast Fourier Transform (FFT) Spectral Analysis Module for GUARD.
Computes log-magnitude spectra of uploaded images to expose high-frequency
artifacts, upsampling grids, checkerboard patterns, and synthetic spectral roll-off.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def compute_2d_fft_spectrum(
    image: Union[np.ndarray, Image.Image],
    colormap: int = cv2.COLORMAP_TURBO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D Fast Fourier Transform log-magnitude spectrum of an image.

    Parameters:
        image: Input RGB image (H, W, 3) or PIL Image.
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_TURBO / INFERNO).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - display_spectrum: RGB image (H, W, 3) colored with the colormap.
            - raw_log_magnitude: Grayscale 2D float array (H, W) in [0.0, 1.0].
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    # Convert to grayscale for standard 2D Fourier representation
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. 2D Fast Fourier Transform
    fft2d = np.fft.fft2(gray)

    # 2. Shift zero-frequency component to the center of the spectrum
    fft_shifted = np.fft.fftshift(fft2d)

    # 3. Compute magnitude spectrum with logarithmic dynamic range compression: S = log(1 + |F|)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # 4. Normalize to [0, 255] for visual presentation
    min_val = np.percentile(log_magnitude, 1)
    max_val = np.percentile(log_magnitude, 99.5)
    if max_val > min_val:
        norm_mag = np.clip((log_magnitude - min_val) / (max_val - min_val), 0.0, 1.0)
    else:
        norm_mag = np.zeros_like(log_magnitude)

    uint8_mag = (norm_mag * 255.0).astype(np.uint8)

    # 5. Apply colormap
    colored_bgr = cv2.applyColorMap(uint8_mag, colormap)
    display_spectrum = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    return display_spectrum, norm_mag
