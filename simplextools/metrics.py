"""
Metrics for evaluating model performance with different thresholds on the Simplex.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


def get_simplex_predictions(probs_np_func, tau_np_func):
    """
    Computes predictions using the multidimensional threshold rule from the paper.

    A sample is classified to the class 'j' that maximizes the margin, 
    defined as (probability[j] - threshold[j]). This is mathematically
    equivalent to the rule: z^j - z^k > τ^j - τ^k for all k != j.

    Args:
        probs_np_func (np.ndarray): Array of probability outputs (n_samples, n_classes).
        tau_np_func (np.ndarray): Threshold vector for each class (n_classes,).

    Returns:
        np.ndarray: Array of predicted class indices for each sample.
    """
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
    """Calculate metrics by using custom thresholds on the probability simplex.

    This function implements decision regions in the probability simplex based on
    the threshold vector tau.
    A prediction is made for class i if:
        p_i > tau_i and no other class has a higher margin over its threshold.

    Args:
        probs  (torch.Tensor or np.ndarray): Probability outputs from model (N_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True class labels (N_samples,)
        tau (np.ndarray): Threshold values for each class (n_classes,)
        metric (str): Metric to return, either "accuracy" or "f1" (default: "accuracy")

    Returns:
        float: The calculated metric (accuracy or macro F1 score)
    """

    # Validate metric type
    if metric not in ["accuracy", "f1"]:
        raise ValueError("Metric must be either 'accuracy' or 'f1'")

    # Convert inputs to numpy if they're torch tensors
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(tau, torch.Tensor):
        tau = tau.cpu().numpy()

    n_samples = probs.shape[0]
    n_classes = probs.shape[1]

    # Initialize predictions array
    predictions = np.zeros(n_samples, dtype=int)

    # For each sample, find which region it belongs to
    for i in range(n_samples):
        # Check if any probability exceeds its threshold
        exceeds_threshold = probs[i] > tau

        if np.any(exceeds_threshold):
            # If multiple probabilities exceed thresholds, take argmax
            candidates = np.where(exceeds_threshold)[0]
            predictions[i] = candidates[np.argmax(probs[i][candidates])]
        else:
            # If no probability exceeds threshold, take overall argmax
            predictions[i] = np.argmax(probs[i])

    # Calculate the requested metric
    if metric == "accuracy":
        return np.mean(predictions == labels)
    else:  # metric == "f1"
        return f1_score(labels, predictions, average="macro")
