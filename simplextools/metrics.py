"""
Metrics for evaluating model performance with different thresholds on the Simplex.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


def get_simplex_predictions(probs_np_func, tau_np_func):
    """
    Computes predictions using the multidimensional threshold rule.
    Handles both single sample (1D) and batch (2D) inputs.

    Args:
        probs_np_func (np.ndarray): Array of probability outputs (n_samples, n_classes) or (n_classes,).
        tau_np_func (np.ndarray): Threshold vector for each class (n_classes,).

    Returns:
        np.ndarray: Array of predicted class indices for each sample.
    """
    probs_np_func = np.asarray(probs_np_func)
    tau_np_func = np.asarray(tau_np_func)

    # If single sample, reshape to (1, n_classes)
    if probs_np_func.ndim == 1:
        probs_np_func = probs_np_func.reshape(1, -1)
    if tau_np_func.ndim == 1:
        # tau is fine as (n_classes,)
        pass
    elif tau_np_func.ndim == 2 and tau_np_func.shape[0] == 1:
        tau_np_func = tau_np_func.flatten()

    return np.argmax(probs_np_func - tau_np_func, axis=1)


def argmax_accuracy(probs, labels):
    """Calculate standard classification metrics using argmax prediction.

    Args:
        probs (torch.Tensor or np.ndarray): Model probability outputs
        labels (torch.Tensor or np.ndarray): True labels

    Returns:
        tuple: (predictions, accuracy) where predictions are the argmax class indices
    """
    # Convert to numpy if tensors
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get standard predictions (argmax)
    preds = np.argmax(probs, axis=1)

    # Calculate accuracy
    accuracy = np.mean(preds == labels)

    return accuracy


def calculate_simplex_metrics(probs, labels, tau, metric="accuracy"):
    """
    Calculate accuracy or macro F1 using custom thresholds on the probability simplex.
    """
    if metric not in ["accuracy", "f1"]:
        raise ValueError("Metric must be either 'accuracy' or 'f1'")

    # Convert to numpy if needed
    for arr in (probs, labels, tau):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()

    predictions = get_simplex_predictions(probs, tau)
    if metric == "accuracy":
        return np.mean(predictions == labels)
    return f1_score(labels, predictions, average="macro")
