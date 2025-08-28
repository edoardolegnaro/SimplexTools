"""
Tools for threshold tuning and optimization in the probability simplex.
"""

import numpy as np
import torch
from p_tqdm import p_map
from simplextools.metrics import calculate_simplex_metrics
import os

def generate_simplex_grid(n_classes=3, resolution=20):
    """
    Generate a grid of points in the probability simplex.
    
    Each point (τ₁, τ₂, τ₃, ...) satisfies:
    - Sum(τᵢ) = 1 
    - τᵢ ≥ 0 for all i
    
    Args:
        n_classes (int):  Number of classes (dimensionality of simplex)
        resolution (int): Number of steps along each dimension.
                          The number of points generated is C(resolution + n_classes - 1, n_classes - 1).
        
    Returns:
        list: List of tau points in the simplex as numpy arrays
    """
    if n_classes <= 0:
        return []
    if resolution < 0:
        raise ValueError("Resolution must be non-negative.")

    if resolution == 0:
        # For resolution 0, return a single point, typically a vertex or barycenter.
        point = np.zeros(n_classes)
        if n_classes > 0:
            point[-1] = 1.0
        return [point]

    if n_classes == 2:
        # For binary case, simplex is just a line
        return [np.array([i/resolution, 1-i/resolution]) 
                for i in range(resolution+1)]
    
    elif n_classes == 3:
        # For 3 classes, generate points in a triangle
        tau_points = []
        step_val = 1.0 / resolution
        
        # Iterate through possible values for t1 and t2
        for i in range(resolution + 1):
            t1 = i * step_val
            # Ensure t2 doesn't make the sum exceed 1
            for j in range(resolution + 1 - i):
                t2 = j * step_val
                t3 = 1.0 - t1 - t2
                
                # Due to floating point precision issues, ensure t3 is not effectively negative
                if t3 < -1e-9:
                    continue
                t3 = max(0.0, t3)
                
                # Re-normalize to ensure sum is exactly 1, handling potential precision errors
                current_point = np.array([t1, t2, t3])
                current_sum = np.sum(current_point)
                if current_sum > 1e-9:
                    current_point /= current_sum
                else:
                    current_point = np.zeros(n_classes)
                    if n_classes > 0:
                        current_point[0] = 1.0

                tau_points.append(current_point)
                
        return tau_points
    
    else:  # Recursive approach for n_classes > 3
        tau_points = []
        step_val = 1.0 / resolution

        # Recursive helper function to generate integer partitions
        def _generate_partitions_recursive(components_left, resolution_sum_remaining, current_path_steps):
            if components_left == 1:
                # Last component takes all remaining resolution sum
                if resolution_sum_remaining >= 0:
                    final_path_steps = current_path_steps + [resolution_sum_remaining]
                    # Convert steps to float values
                    point = np.array([s * step_val for s in final_path_steps])
                    # Normalize to ensure sum is exactly 1 due to potential float precision
                    s = np.sum(point)
                    if s > 1e-9:
                        point /= s
                    else:
                        point = np.zeros(n_classes)
                        if n_classes > 0:
                            point[0] = 1.0
                    tau_points.append(point)
                return

            # For the current component, iterate through possible step values
            for i in range(resolution_sum_remaining + 1):
                _generate_partitions_recursive(
                    components_left - 1,
                    resolution_sum_remaining - i,
                    current_path_steps + [i]
                )
        
        _generate_partitions_recursive(n_classes, resolution, [])
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
