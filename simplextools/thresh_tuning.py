"""
Tools for threshold tuning and optimization in the probability simplex.
"""

import numpy as np
import torch
from p_tqdm import p_map
from simplextools.metrics import calculate_simplex_metrics
import os
import itertools

def generate_simplex_grid(n_classes=3, resolution=20):
    """
    Generate a grid of points in the probability simplex.
    
    Each point (τ₁, τ₂, τ₃, ...) satisfies:
    - Sum(τᵢ) = 1 
    - τᵢ ≥ 0 for all i
    
    Args:
        n_classes (int):  Number of classes (dimensionality of simplex).
        resolution (int): Number of steps along each dimension.
                          The number of points generated is C(resolution + n_classes - 1, n_classes - 1).
        
    Returns:
        list: List of tau points in the simplex as numpy arrays.
    """
    if n_classes <= 0:
        return []
    if resolution < 0:
        raise ValueError("Resolution must be non-negative.")
    if resolution == 0:
        point = np.zeros(n_classes)
        if n_classes > 0:
            point[-1] = 1.0
        return [point]

    # This generates all combinations of integers that sum to the resolution.
    # np.bincount is used to count the occurrences of each class index.
    combs = itertools.combinations_with_replacement(range(n_classes), resolution)
    
    tau_points = [
        np.bincount(c, minlength=n_classes) / resolution
        for c in combs
    ]
    
    return tau_points
    
def find_optimal_threshold(probs, labels, resolution=10, n_jobs=None, metric='accuracy'):
    """Find the optimal threshold (tau) on the simplex maximizing the given metric(s)."""
    if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    n_classes = probs.shape[1]
    tau_points = generate_simplex_grid(n_classes, resolution)
    if n_jobs is None: n_jobs = max(1, os.cpu_count())
    metrics = [metric] if isinstance(metric, str) else metric

    def _worker(tau, probs, labels):
        return {m: calculate_simplex_metrics(probs, labels, tau, metric=m) for m in metrics}

    all_results = p_map(
        _worker, tau_points, [probs]*len(tau_points), [labels]*len(tau_points),
        num_cpus=n_jobs, desc=f"Evaluating thresholds for {metrics}"
    )

    results = {}
    for m in metrics:
        scores = [res[m] for res in all_results]
        best_idx = np.argmax(scores)
        results[m] = (tau_points[best_idx], scores[best_idx], scores, tau_points)
        print(f"Best score for {m}: {scores[best_idx]:.4f} with tau={tau_points[best_idx]}")

    return results[metric] if isinstance(metric, str) else results
