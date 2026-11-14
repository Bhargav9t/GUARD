# ==============================================================================
# File: src/models/guard_model.py
# Description: Master End-to-End GUARD Detector Architecture.
# Unifies:
#   1. Spatial Stream (EfficientNet-B4)
#   2. Frequency Stream (2D Fast Fourier Transform + Residual CNN)
#   3. Gated Attention Fusion Layer
#   4. Binary Classification Head (Real vs. Fake)
#
# Key Capabilities:
#   - Full forward pass through both streams and fusion head
#   - Individual stream metric breakdown (Spatial Logits/Prob & Frequency Logits/Prob)
#   - Calibrated inference with tunable decision threshold
#   - Safe checkpoint loading with fallback warning state
# ==============================================================================

# Import typing primitives for static analysis and type annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

# Import PyTorch core tensor and neural network primitives
import torch
import torch.nn as nn

# Import project configurations and sub-modules
from configs.config import GuardConfig, default_config

try:
    # Use relative imports when loaded as part of the src.models package
    from .spatial_stream import SpatialStream
    from .frequency_stream import FrequencyStream
    from .fusion import FusionLayer, GatedFusion
    from .classification_head import ClassificationHead
    from ..utils.device import get_device
    from ..utils.logger import app_logger
except (ImportError, ValueError):
    # Fallback for direct standalone execution
    from src.models.spatial_stream import SpatialStream
    from src.models.frequency_stream import FrequencyStream
    from src.models.fusion import FusionLayer, GatedFusion
    from src.models.classification_head import ClassificationHead
    from src.utils.device import get_device
    from src.utils.logger import app_logger


class GUARDModel(nn.Module):
    """
    GUARD: Generalizable Universal AI-generated & Real-image Detector.
    Dual-branch spatial-frequency neural architecture with adaptive gated fusion.
    """

    def __init__(self, config: Optional[GuardConfig] = None):
        """
        Initializes all sub-modules of the GUARD architecture.

        Parameters:
            config (Optional[GuardConfig]): Master configuration object.
                If None, uses default_config.
        """
        # Call base nn.Module initialization
        super().__init__()

        # Assign configuration dataclass or default singleton
        self.config = config or default_config

        # 1. Instantiate RGB Spatial Stream (EfficientNet-B4 backbone)
        self.spatial_stream = SpatialStream(self.config.spatial)

        # 2. Instantiate Frequency Stream (2D FFT + Lightweight Residual CNN)
        self.frequency_stream = FrequencyStream(self.config.frequency)

        # 3. Instantiate Adaptive Fusion Layer (Gated Attention or Cross-Attention)
        self.fusion = FusionLayer(
            config=self.config.fusion,
            spatial_dim=self.spatial_stream.out_features,
            frequency_dim=self.frequency_stream.out_features,
        )

        # 4. Instantiate Binary Classification Head
        self.head = ClassificationHead(
            in_features=self.fusion.out_features,
            config=self.config.head,
        )

        # Log successful initialization
        app_logger.info("GUARDModel end-to-end architecture assembled successfully.")

    def forward(
        self,
        x: torch.Tensor,
        apply_norm: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through both streams, fusion, and classification head.

        Parameters:
            x (torch.Tensor): Input batch of RGB images of shape (B, 3, H, W) in [0.0, 1.0].
            apply_norm (bool): Whether to apply ImageNet normalization inside spatial stream.

        Returns:
            Dict[str, torch.Tensor]:
                - 'logits': Raw unbounded fused logit (B, 1).
                - 'probabilities': Continuous probability of being Fake in [0.0, 1.0] (B, 1).
                - 'spatial_features': 1D spatial feature embedding (B, 1792).
                - 'frequency_features': 1D frequency feature embedding (B, 512).
                - 'fused_features': Joint multi-modal feature vector (B, 1024).
                - 'gate_weights': Gating/attention coefficients (B, 1024).
        """
        # Step 1: Spatial Stream RGB feature extraction via EfficientNet-B4
        f_spatial = self.spatial_stream(x, apply_norm=apply_norm)

        # Step 2: Frequency Stream 2D FFT & spectral feature extraction
        f_frequency = self.frequency_stream(x)

        # Step 3: Adaptive Gated Multi-Modal Fusion
        f_fused, gate_weights = self.fusion(f_spatial, f_frequency)

        # Step 4: Binary Classification Head
        logits = self.head(f_fused)

        # Step 5: Sigmoid probability mapping
        probabilities = torch.sigmoid(logits)

        # Return full prediction dictionary
        return {
            "logits": logits,
            "probabilities": probabilities,
            "spatial_features": f_spatial,
            "frequency_features": f_frequency,
            "fused_features": f_fused,
            "gate_weights": gate_weights,
        }

    def compute_stream_breakdown(
        self,
        x: torch.Tensor,
        apply_norm: bool = True
    ) -> Dict[str, Any]:
        """
        Computes independent forensic metrics for the Spatial Stream and Frequency Stream.
        Projects each modality into the classification space to measure individual stream
        confidence, logits, and probabilities.

        Parameters:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W) or (3, H, W).
            apply_norm (bool): Whether to apply ImageNet normalization to spatial input.

        Returns:
            Dict[str, Any]:
                - 'spatial_logit': Raw logit output from spatial stream.
                - 'spatial_prob': Fake probability score from spatial stream [0.0, 1.0].
                - 'spatial_confidence': Certainty score for the spatial prediction [0.5, 1.0].
                - 'frequency_logit': Raw logit output from frequency stream.
                - 'frequency_prob': Fake probability score from frequency stream [0.0, 1.0].
                - 'frequency_confidence': Certainty score for the frequency prediction [0.5, 1.0].
        """
        # Ensure evaluation mode
        self.eval()

        # Add batch dimension if single image is passed: (3, H, W) -> (1, 3, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Extract features independently from spatial backbone
        f_spatial = self.spatial_stream(x, apply_norm=apply_norm)

        # Extract features independently from frequency backbone
        f_frequency = self.frequency_stream(x)

        # Evaluate spatial features through fusion projection into classification space
        out_spatial = self.fusion.forward_spatial_stream(f_spatial)
        spatial_logit_tensor = self.head(out_spatial)

        # Evaluate frequency features through fusion projection into classification space
        out_frequency = self.fusion.forward_frequency_stream(f_frequency)
        frequency_logit_tensor = self.head(out_frequency)

        # Compute continuous probabilities via sigmoid
        spatial_logit = float(spatial_logit_tensor[0, 0].item())
        spatial_prob = float(torch.sigmoid(spatial_logit_tensor)[0, 0].item())
        spatial_conf = spatial_prob if spatial_prob >= 0.5 else (1.0 - spatial_prob)

        frequency_logit = float(frequency_logit_tensor[0, 0].item())
        frequency_prob = float(torch.sigmoid(frequency_logit_tensor)[0, 0].item())
        frequency_conf = frequency_prob if frequency_prob >= 0.5 else (1.0 - frequency_prob)

        # Return structured breakdown dictionary
        return {
            "spatial_logit": spatial_logit,
            "spatial_prob": spatial_prob,
            "spatial_confidence": spatial_conf,
            "frequency_logit": frequency_logit,
            "frequency_prob": frequency_prob,
            "frequency_confidence": frequency_conf,
        }

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Inference routine returning predicted binary status, confidence,
        human-readable label, and individual stream telemetry.

        Parameters:
            x (torch.Tensor): Input image tensor (B, 3, H, W) or (3, H, W).
            threshold (Optional[float]): Classification decision boundary threshold.
                Scores >= threshold are classified as Fake.

        Returns:
            Dict[str, Any]:
                - 'is_fake' (bool): True if predicted Fake / AI-generated.
                - 'predicted_class' (int): 0 for Real, 1 for Fake.
                - 'label' (str): "Real" or "AI-Generated / Manipulated".
                - 'fake_probability' (float): Continuous probability of being Fake.
                - 'confidence' (float): Calibrated certainty score of the prediction.
                - 'threshold' (float): Active decision threshold used.
                - 'spatial_gating_ratio' (float): Gating weight allocated to spatial stream.
                - 'frequency_gating_ratio' (float): Gating weight allocated to frequency stream.
                - 'stream_breakdown' (Dict[str, Any]): Individual stream logits & confidences.
        """
        # Ensure model is in inference mode
        self.eval()

        # Add batch dimension if single image is passed: (3, H, W) -> (1, 3, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Execute full forward pass
        outputs = self.forward(x)
        prob = float(outputs["probabilities"][0, 0].item())
        logit = float(outputs["logits"][0, 0].item())

        # Determine active decision threshold (tunable)
        thresh = threshold if threshold is not None else self.config.head.decision_threshold

        # Binary decision: probability >= threshold indicates Fake
        is_fake = (prob >= thresh)
        pred_class = 1 if is_fake else 0
        label = self.config.head.labels[pred_class]

        # Calculate certainty confidence: distance from decision threshold
        confidence = prob if is_fake else (1.0 - prob)

        # Compute average gate weights
        gate_score = float(outputs["gate_weights"][0].mean().item())

        # Extract independent stream breakdown
        stream_breakdown = self.compute_stream_breakdown(x)

        # Dual-pathway parallel scorer combination (Score Fusion Mode)
        # Combines spatial stream score and frequency stream score using dynamic gating weights
        sp_prob = stream_breakdown.get("spatial_prob", prob)
        fq_prob = stream_breakdown.get("frequency_prob", prob)
        dual_score = (gate_score * sp_prob) + ((1.0 - gate_score) * fq_prob)
        stream_breakdown["combined_dual_score"] = dual_score
        stream_breakdown["dual_score_fake"] = bool(dual_score >= thresh)

        # Return comprehensive inference payload
        return {
            "is_fake": is_fake,
            "predicted_class": pred_class,
            "label": label,
            "logit": logit,
            "fake_probability": prob,
            "confidence": confidence,
            "threshold": thresh,
            "spatial_gating_ratio": gate_score,
            "frequency_gating_ratio": 1.0 - gate_score,
            "stream_breakdown": stream_breakdown,
            "combined_dual_score": dual_score,
        }

    def compute_gradcam(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes Grad-CAM (Gradient-Weighted Class Activation Mapping) on the spatial feature maps.
        Highlights the exact spatial regions (blending seams, warping, unnatural skin micro-textures)
        that drove the model's decision (artifact evidence map).

        Parameters:
            x (torch.Tensor): Input image tensor of shape (1, 3, H, W) or (3, H, W).
            target_class (Optional[int]): 1 for Fake class attribution, 0 for Real class attribution.
                If None, attributes with respect to the active predicted class.

        Returns:
            torch.Tensor: Grad-CAM heatmap tensor of shape (1, 1, H_cam, W_cam) in [0.0, 1.0].
        """
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x_in = x.clone().detach()

        # Temporarily enable gradients for Grad-CAM
        with torch.enable_grad():
            # 1. Forward through spatial stream up to feature maps
            spatial_fmaps = self.spatial_stream.forward_features(x_in)
            spatial_fmaps.requires_grad_(True)
            spatial_fmaps.retain_grad()

            # 2. Pool spatial features to 1D vector
            pooled_spatial = torch.mean(spatial_fmaps, dim=(-2, -1))
            proj_spatial = self.spatial_stream.proj(pooled_spatial)

            # 3. Frequency stream forward
            f_frequency = self.frequency_stream(x_in)

            # 4. Fusion and classification head
            f_fused, _ = self.fusion(proj_spatial, f_frequency)
            logit = self.head(f_fused)[0, 0]
            prob = torch.sigmoid(logit)

            # 5. Determine target score
            # If Fake (prob >= 0.5) or target_class == 1, target is +logit
            # If Real (prob < 0.5) or target_class == 0, target is -logit
            if target_class is None:
                target_score = logit if prob.item() >= 0.5 else -logit
            elif target_class == 1:
                target_score = logit
            else:
                target_score = -logit

            # 6. Backward pass to compute gradients with respect to spatial feature maps
            self.zero_grad()
            target_score.backward(retain_graph=False)

            grads = spatial_fmaps.grad
            if grads is not None:
                # Global average pooling of gradients: channel importance weights alpha_k
                alpha = torch.mean(grads, dim=(-2, -1), keepdim=True)
                # Weighted linear combination
                cam = torch.sum(alpha * spatial_fmaps, dim=1, keepdim=True)
                # Apply ReLU: only features that positively correlate with the target class
                cam = torch.relu(cam)
                # Normalize between 0 and 1
                c_min, c_max = cam.min(), cam.max()
                if c_max > c_min:
                    cam = (cam - c_min) / (c_max - c_min + 1e-8)
                else:
                    cam = torch.zeros_like(cam)
                return cam.detach()
            else:
                # Fallback to normalized channel mean
                mean_cam = torch.mean(spatial_fmaps, dim=1, keepdim=True)
                m_min, m_max = mean_cam.min(), mean_cam.max()
                return ((mean_cam - m_min) / (m_max - m_min + 1e-8)).detach()

    def extract_explainability_data(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extracts multi-modal forensic inspection assets for the UI Dashboard:
          - 2D Fourier Magnitude Spectrum map.
          - 2D Grad-CAM Spatial Artifact Evidence Heatmap.
          - Multi-modal Gating ratio breakdown and dual-stream metrics.

        Parameters:
            x (torch.Tensor): Input image tensor of shape (1, 3, H, W) or (3, H, W).
            threshold (Optional[float]): Decision boundary threshold.

        Returns:
            Dict[str, Any]: Dictionary containing visualization maps and diagnostic metrics.
        """
        # Set evaluation mode
        self.eval()

        # Add batch dimension if missing
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # 1. Standard prediction pass with active threshold
        with torch.no_grad():
            pred_info = self.predict(x, threshold=threshold)

            # 2. Extract 2D FFT Magnitude Spectrum (1, 3, H, W)
            spectrum = self.frequency_stream.compute_spectrum(x)

        # 3. Compute Grad-CAM Spatial Artifact Heatmap
        try:
            spatial_heatmap = self.compute_gradcam(x, target_class=pred_info["predicted_class"])
        except Exception as e:
            app_logger.warning(f"Grad-CAM computation fallback: {e}")
            with torch.no_grad():
                spatial_fmaps = self.spatial_stream.forward_features(x)
                spatial_heatmap = torch.mean(spatial_fmaps, dim=1, keepdim=True)

        # Return explainability assets
        return {
            "prediction": pred_info,
            "spectrum_tensor": spectrum,
            "spatial_heatmap_tensor": spatial_heatmap,
        }


def load_guard_checkpoint(
    checkpoint_path: Union[str, Path] = "checkpoints/guard_model.pth",
    device: Optional[torch.device] = None,
    config: Optional[GuardConfig] = None,
) -> Tuple[GUARDModel, bool, Dict[str, Any]]:
    """
    Safely loads trained GUARD model weights if available, or initializes an
    untrained model with a clear warning state when checkpoints are missing.

    Parameters:
        checkpoint_path (Union[str, Path]): Path to checkpoint weights file.
        device (Optional[torch.device]): Target compute device.
        config (Optional[GuardConfig]): Model configuration.

    Returns:
        Tuple[GUARDModel, bool, Dict[str, Any]]:
            - model: Instantiated GUARDModel moved to device and in eval mode.
            - is_trained: True if valid trained weights were loaded, False if untrained fallback.
            - status_info: Dictionary containing diagnostic loading metadata.
    """
    # Resolve hardware compute device
    target_device = device or get_device()

    # Assign config dataclass
    cfg = config or default_config

    # Instantiate model architecture
    model = GUARDModel(config=cfg)

    # Resolve file path dynamically: handle relative paths against configured checkpoints_dir
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        resolved_ckpt = cfg.paths.checkpoints_dir / ckpt_path.name
        if resolved_ckpt.is_file():
            ckpt_path = resolved_ckpt
        elif ckpt_path.is_file():
            ckpt_path = ckpt_path.resolve()

    fallback_best = cfg.paths.checkpoints_dir / "guard_best.pth"

    # Tracking metadata
    is_trained = False
    status_info: Dict[str, Any] = {
        "status": "untrained",
        "checkpoint_path": str(ckpt_path),
        "message": "Untrained base weights (ImageNet backbone only). Calibrate model using 'train.py' to eliminate false positives.",
        "epoch": None,
        "loss": None,
        "accuracy": None,
    }

    # 1. Check primary requested checkpoint path
    chosen_path = None
    if ckpt_path.is_file():
        chosen_path = ckpt_path
    # 2. Fall back to best checkpoint if available
    elif fallback_best.is_file():
        chosen_path = fallback_best

    # Load weights if a valid checkpoint was found
    if chosen_path is not None:
        try:
            checkpoint = torch.load(chosen_path, map_location=target_device)
            # Extract state dict (handling raw dict or nested dict)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                status_info["epoch"] = checkpoint.get("epoch")
                status_info["loss"] = checkpoint.get("loss")
                status_info["accuracy"] = checkpoint.get("accuracy")
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)

            is_trained = True
            status_info["status"] = "calibrated"
            status_info["checkpoint_path"] = str(chosen_path)
            status_info["message"] = f"Calibrated weights loaded successfully from: {chosen_path.name}"
            app_logger.info(f"Loaded trained GUARD weights from {chosen_path}")

        except Exception as err:
            is_trained = False
            status_info["status"] = "load_error"
            status_info["message"] = f"Failed to load weights from {chosen_path}: {str(err)}"
            app_logger.warning(f"Error loading checkpoint {chosen_path}: {err}")
    else:
        app_logger.warning(
            f"No checkpoint found at '{ckpt_path}'. Operating in UNTRAINED fallback state."
        )

    # Move model to target device and set to evaluation mode
    model.to(target_device)
    model.eval()

    return model, is_trained, status_info
