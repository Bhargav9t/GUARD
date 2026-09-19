# ==============================================================================
# File: src/utils/metrics.py
# Description: Quantitative Evaluation Metrics for Deepfake and AI-Generated
#              Image Detection in GUARD.
# Computes Accuracy, Precision, Recall, F1, AUC-ROC, Equal Error Rate (EER),
# and Confusion Matrix values.
# ==============================================================================

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
)
import torch


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) and the corresponding decision threshold.
    EER is the operating point on the ROC curve where the False Acceptance Rate (FAR)
    equals the False Rejection Rate (FRR). A lower EER indicates superior detection capability.

    Parameters:
        y_true (np.ndarray): Binary ground truth labels (0 for Real, 1 for Fake).
        y_scores (np.ndarray): Predicted probabilities / confidence scores for Fake class.

    Returns:
        Tuple[float, float]: (eer_value, eer_threshold)
    """
    # Compute Receiver Operating Characteristic (ROC) curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    # False Rejection Rate (FRR) is equal to 1 - True Positive Rate (TPR)
    fnr = 1.0 - tpr

    # Find the index where FPR and FNR (FRR) are closest to each other
    eer_index = np.nanargmin(np.absolute(fnr - fpr))

    # Calculate average of FPR and FNR at this critical intersection point
    eer_val = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_thresh = float(thresholds[eer_index])

    return eer_val, eer_thresh


def evaluate_predictions(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Comprehensive evaluation routine calculating standard binary classification
    metrics and deepfake forensic metrics.

    Parameters:
        y_true (Union[np.ndarray, torch.Tensor]): Ground truth labels (0 = Real, 1 = Fake).
        y_pred_probs (Union[np.ndarray, torch.Tensor]): Predicted continuous probabilities [0.0, 1.0].
        threshold (float): Decision threshold to binarize predicted probabilities. Default is 0.50.

    Returns:
        Dict[str, float]: Dictionary mapping metric names to their scalar numerical values.
    """
    # 1. Convert PyTorch Tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    else:
        y_true = np.asarray(y_true).flatten()

    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy().flatten()
    else:
        y_pred_probs = np.asarray(y_pred_probs).flatten()

    # Ensure binary integer representation for ground truth labels
    y_true = y_true.astype(int)

    # 2. Binarize continuous probabilities using the selected decision threshold
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    # 3. Calculate core performance metrics
    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    # 4. Calculate threshold-independent metrics (AUC-ROC and Average Precision)
    # Check if both classes exist in y_true to prevent roc_auc_score exceptions
    if len(np.unique(y_true)) > 1:
        auc_roc = float(roc_auc_score(y_true, y_pred_probs))
        ap_score = float(average_precision_score(y_true, y_pred_probs))
        eer, eer_thresh = compute_eer(y_true, y_pred_probs)
    else:
        # Fallback if evaluation batch contains only single class samples
        auc_roc = 0.0
        ap_score = 0.0
        eer = 0.0
        eer_thresh = threshold

    # 5. Extract Confusion Matrix elements: TN, FP, FN, TP
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "average_precision": ap_score,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
