# ==============================================================================
# File: src/models/classification_head.py
# Description: Binary Classification Head for the GUARD framework.
# Maps fused multi-modal spatial-frequency representations to Real vs. Fake
# classification logits and calibrated probability confidence scores.
# ==============================================================================

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from configs.config import HeadConfig, default_config


class ClassificationHead(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classification Head.
    Takes fused representations (1024-d) and outputs binary Real/Fake predictions.
    """

    def __init__(
        self,
        in_features: int = 1024,
        config: Optional[HeadConfig] = None
    ):
        """
        Parameters:
            in_features (int): Dimensionality of the incoming fused feature vector.
            config (Optional[HeadConfig]): Classification head configuration dataclass.
        """
        super().__init__()
        self.config = config or default_config.head
        self.num_classes = self.config.num_classes
        self.labels = self.config.labels
        self.default_threshold = self.config.decision_threshold

        # Multi-stage projection with Layer Normalization and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw unbounded logits.
        Raw logits are numerically optimal for PyTorch BCEWithLogitsLoss.

        Parameters:
            x (torch.Tensor): Fused feature tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes continuous probability score in [0.0, 1.0] via Sigmoid activation.
        Score represents probability of image being Fake / AI-Generated.

        Parameters:
            x (torch.Tensor): Fused feature tensor (B, in_features).

        Returns:
            torch.Tensor: Confidence probabilities (B, 1) in [0.0, 1.0].
        """
        logits = self.forward(x)
        probabilities = torch.sigmoid(logits)
        return probabilities

    def predict(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates fused embeddings and returns binary class assignments and confidence.

        Parameters:
            x (torch.Tensor): Fused feature tensor (B, in_features).
            threshold (Optional[float]): Decision boundary threshold. Default is 0.50.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - predicted_classes: Tensor of shape (B, 1) with values 0 (Real) or 1 (Fake).
                - confidence: Tensor of shape (B, 1) representing class confidence.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        probs = self.predict_proba(x)

        # 0 = Real, 1 = Fake
        pred_labels = (probs >= thresh).long()

        # Class-specific confidence:
        # If predicted Fake (1): confidence is probs
        # If predicted Real (0): confidence is (1.0 - probs)
        confidence = torch.where(pred_labels == 1, probs, 1.0 - probs)

        return pred_labels, confidence
