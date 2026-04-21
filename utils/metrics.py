"""
Metrics module for the secure deepfake detection system.

Provides watermark quality metrics (PSNR, NC) and detection metrics
(AUC, EER, ACC, IoU variants, token-level F1/P/R, mAP).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 16.3, 16.4
"""

import math
from typing import List

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


# ---------------------------------------------------------------------------
# Watermark quality metrics
# ---------------------------------------------------------------------------


def compute_psnr(original: Tensor, watermarked: Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio between two tensors.

    PSNR = 10 * log10(MAX² / MSE), where MAX = 1.0 for normalized images.
    Returns float('inf') when the two tensors are identical (MSE = 0).

    Args:
        original:    Reference tensor of any shape (e.g. (B, 3, H, W)).
        watermarked: Tensor of the same shape as *original*.

    Returns:
        PSNR value in decibels as a Python float, or float('inf') when
        MSE == 0.

    Raises:
        ValueError: If *original* and *watermarked* have different shapes.
    """
    if original.shape != watermarked.shape:
        raise ValueError(
            f"Shape mismatch in compute_psnr: "
            f"original {tuple(original.shape)} vs watermarked {tuple(watermarked.shape)}"
        )

    mse: float = torch.mean((original.float() - watermarked.float()) ** 2).item()

    if mse == 0.0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)


def compute_nc(m: Tensor, m_hat: Tensor) -> float:
    """Compute Normalized Correlation between two tensors.

    NC = dot(m, m_hat) / (||m||₂ * ||m_hat||₂)

    Returns a value in [-1.0, 1.0].  When both tensors are identical the
    result is exactly 1.0.

    Args:
        m:     Reference tensor of any shape.
        m_hat: Tensor of the same shape as *m*.

    Returns:
        NC value as a Python float in [-1.0, 1.0].

    Raises:
        ValueError: If *m* and *m_hat* have different shapes.
    """
    if m.shape != m_hat.shape:
        raise ValueError(
            f"Shape mismatch in compute_nc: "
            f"m {tuple(m.shape)} vs m_hat {tuple(m_hat.shape)}"
        )

    m_flat = m.float().flatten()
    m_hat_flat = m_hat.float().flatten()

    dot = torch.dot(m_flat, m_hat_flat).item()
    norm_m = torch.norm(m_flat).item()
    norm_m_hat = torch.norm(m_hat_flat).item()

    if norm_m == 0.0 or norm_m_hat == 0.0:
        return 0.0

    # Clamp to [-1, 1] to guard against floating-point rounding beyond the valid range
    nc = dot / (norm_m * norm_m_hat)
    return float(max(-1.0, min(1.0, nc)))


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC) for binary classification.

    Args:
        y_true:  1-D array of ground-truth binary labels (0 or 1).
        y_score: 1-D array of predicted probability scores for the positive class.

    Returns:
        AUC as a float in [0.0, 1.0].
    """
    return float(roc_auc_score(y_true, y_score))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) for binary classification.

    EER is the point on the ROC curve where the false positive rate equals
    the false negative rate (1 - true positive rate).

    Args:
        y_true:  1-D array of ground-truth binary labels (0 or 1).
        y_score: 1-D array of predicted probability scores for the positive class.

    Returns:
        EER as a float in [0.0, 1.0].
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    # Find the index where FPR and FNR are closest
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer


def compute_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: 1-D array of ground-truth binary labels (0 or 1).
        y_pred: 1-D array of predicted binary labels (0 or 1).

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# IoU / grounding metrics
# ---------------------------------------------------------------------------


def compute_iou_mean(iou_scores: List[float]) -> float:
    """Compute mean Intersection over Union across all samples.

    Args:
        iou_scores: List of per-sample IoU values in [0.0, 1.0].

    Returns:
        Mean IoU as a float.  Returns 0.0 for an empty list.
    """
    if not iou_scores:
        return 0.0
    return float(np.mean(iou_scores))


def compute_iou_at_50(iou_scores: List[float]) -> float:
    """Compute IoU@50: fraction of samples with IoU >= 0.50.

    Args:
        iou_scores: List of per-sample IoU values in [0.0, 1.0].

    Returns:
        IoU@50 as a float in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not iou_scores:
        return 0.0
    return float(np.mean([s >= 0.50 for s in iou_scores]))


def compute_iou_at_75(iou_scores: List[float]) -> float:
    """Compute IoU@75: fraction of samples with IoU >= 0.75.

    Args:
        iou_scores: List of per-sample IoU values in [0.0, 1.0].

    Returns:
        IoU@75 as a float in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not iou_scores:
        return 0.0
    return float(np.mean([s >= 0.75 for s in iou_scores]))


def compute_iou_at_95(iou_scores: List[float]) -> float:
    """Compute IoU@95: fraction of samples with IoU >= 0.95.

    Args:
        iou_scores: List of per-sample IoU values in [0.0, 1.0].

    Returns:
        IoU@95 as a float in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not iou_scores:
        return 0.0
    return float(np.mean([s >= 0.95 for s in iou_scores]))


# ---------------------------------------------------------------------------
# Token-level text grounding metrics
# ---------------------------------------------------------------------------


def compute_token_precision(tp: int, fp: int) -> float:
    """Compute token-level precision.

    Precision = TP / (TP + FP).  Returns 0.0 when TP + FP == 0.

    Args:
        tp: Number of true positive token predictions.
        fp: Number of false positive token predictions.

    Returns:
        Precision as a float in [0.0, 1.0].
    """
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)


def compute_token_recall(tp: int, fn: int) -> float:
    """Compute token-level recall.

    Recall = TP / (TP + FN).  Returns 0.0 when TP + FN == 0.

    Args:
        tp: Number of true positive token predictions.
        fn: Number of false negative token predictions.

    Returns:
        Recall as a float in [0.0, 1.0].
    """
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


def compute_token_f1(tp: int, fp: int, fn: int) -> float:
    """Compute token-level F1 score.

    F1 = 2 * precision * recall / (precision + recall).
    Returns 0.0 when both precision and recall are 0.

    Args:
        tp: Number of true positive token predictions.
        fp: Number of false positive token predictions.
        fn: Number of false negative token predictions.

    Returns:
        F1 score as a float in [0.0, 1.0].
    """
    precision = compute_token_precision(tp, fp)
    recall = compute_token_recall(tp, fn)
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


# ---------------------------------------------------------------------------
# Mean Average Precision
# ---------------------------------------------------------------------------


def compute_map(ap_meter) -> float:
    """Compute mean Average Precision (mAP) from an AveragePrecisionMeter.

    Delegates to ``ap_meter.value().mean()`` as defined in
    ``tools/multilabel_metrics.py``.

    Args:
        ap_meter: An instance of ``AveragePrecisionMeter`` that has already
                  accumulated scores and targets via ``ap_meter.add(...)``.

    Returns:
        mAP as a Python float.
    """
    ap_values = ap_meter.value()
    if isinstance(ap_values, (int, float)):
        return float(ap_values)
    return float(ap_values.mean().item())
