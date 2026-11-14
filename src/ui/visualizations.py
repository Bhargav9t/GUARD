# ==============================================================================
# File: src/ui/visualizations.py
# Description: Multi-Modal Forensic Visualizations for the GUARD Dashboard.
# Generates false-color 2D FFT magnitude spectrograms, spatial Grad-CAM /
# activation heatmaps, and side-by-side diagnostic composites.
# ==============================================================================

from typing import Union, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def tensor_to_display_image(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor (3, H, W) or (1, 3, H, W) in [0.0, 1.0]
    to a standard RGB NumPy array of shape (H, W, 3) in uint8 [0, 255].

    Parameters:
        tensor (Union[torch.Tensor, np.ndarray]): Input tensor or array.

    Returns:
        np.ndarray: Display-ready RGB image array.
    """
    if isinstance(tensor, torch.Tensor):
        # Detach from computation graph and move to CPU
        arr = tensor.detach().cpu().numpy()
        # Remove batch dimension if shape is (1, C, H, W)
        if arr.ndim == 4:
            arr = arr[0]
        # Transpose from PyTorch channel-first (C, H, W) to NumPy channel-last (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(tensor)

    # If single channel (H, W, 1), squeeze or tile to 3 channels
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    # Scale from [0.0, 1.0] to [0, 255] uint8 if values are floating point
    if arr.dtype in [np.float32, np.float64]:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return arr


def create_spectrum_display(
    spectrum_tensor: Union[torch.Tensor, np.ndarray],
    colormap: int = cv2.COLORMAP_INFERNO
) -> np.ndarray:
    """
    Transforms a 2D Fourier log-magnitude spectrum map into a high-contrast
    false-color visualization for forensic frequency analysis.

    Parameters:
        spectrum_tensor (Union[torch.Tensor, np.ndarray]): 2D spectrum map (C, H, W) or (H, W).
        colormap (int): OpenCV false-color colormap constant (e.g. cv2.COLORMAP_INFERNO).

    Returns:
        np.ndarray: RGB false-color magnitude spectrogram ready for display (H, W, 3).
    """
    if isinstance(spectrum_tensor, torch.Tensor):
        arr = spectrum_tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        # Average across color channels if 3-channel spectrum
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)
    else:
        arr = np.asarray(spectrum_tensor)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=-1)

    # Normalize intensity to [0, 255] uint8
    norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0).astype(np.uint8)

    # Apply false-color thermal colormap (Inferno maps low frequency to black/purple and peaks to yellow/white)
    color_bgr = cv2.applyColorMap(norm, colormap)

    # Convert BGR output of applyColorMap to standard RGB for Streamlit
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    return color_rgb


def create_spatial_heatmap_display(
    image_rgb: np.ndarray,
    heatmap_tensor: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.50,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Generates a Grad-CAM / Spatial Feature activation overlay.
    Interpolates coarse feature maps (e.g. 12x12) up to source resolution (380x380)
    and alpha-blends it on top of the original RGB image to expose manipulation boundaries.

    Parameters:
        image_rgb (np.ndarray): Original or aligned RGB image (H, W, 3) in uint8.
        heatmap_tensor (Union[torch.Tensor, np.ndarray]): 2D spatial activations.
        alpha (float): Transparency blending factor (0.0 to 1.0).
        colormap (int): False-color colormap (default: JET).

    Returns:
        np.ndarray: Composite image with glowing activation overlay (H, W, 3).
    """
    if isinstance(heatmap_tensor, torch.Tensor):
        heat_arr = heatmap_tensor.detach().cpu().numpy()
        if heat_arr.ndim == 4:
            heat_arr = heat_arr[0, 0]
        elif heat_arr.ndim == 3:
            heat_arr = heat_arr[0]
    else:
        heat_arr = np.asarray(heatmap_tensor)
        if heat_arr.ndim == 3:
            heat_arr = heat_arr[..., 0]

    img_h, img_w = image_rgb.shape[:2]

    # Normalize activation values to [0.0, 1.0]
    heat_norm = (heat_arr - heat_arr.min()) / (heat_arr.max() - heat_arr.min() + 1e-8)

    # Resize coarse feature map to match exact input image dimensions using bicubic interpolation
    heat_resized = cv2.resize(heat_norm, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = (heat_resized * 255.0).astype(np.uint8)

    # Apply Jet color scheme (cold blue = low activation, red/yellow = strong manipulation cues)
    heat_bgr = cv2.applyColorMap(heat_uint8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # Compute alpha-blended composite: I_fused = (1 - alpha) * Image + alpha * Heatmap
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heat_rgb, alpha, 0)

    return blended
