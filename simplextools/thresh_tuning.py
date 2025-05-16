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
    """Find the optimal threshold (tau) value on the simplex
       that maximizes the specified metric(s).
    
    Args:
        probs (np.ndarray): Probability outputs from model (n_samples, n_classes)
        labels (np.ndarray): True class labels (n_samples,)
        resolution (int): Grid resolution for searching tau values
        n_jobs (int, optional): Number of parallel jobs for computation
                               (defaults to number of CPU cores)
        metric (str or list): Metric or list of metrics to optimize.
                              Supported metrics are 'accuracy', 'f1', etc.
                               (as defined in calculate_simplex_metrics)
        
    Returns:
        tuple or dict: If a single metric is provided, returns a tuple:
                       (best_tau, best_score, all_scores, all_tau_points).
                       If a list of metrics is provided, returns a dictionary
                       where keys are metric names and values are tuples:
                       (best_tau, best_score, all_scores, all_tau_points).
    """
    # Convert to numpy if tensors
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    n_classes = probs.shape[1]
    
    # Generate grid points in the simplex
    tau_points = generate_simplex_grid(n_classes, resolution)
    
    # Set up parallel processing
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count())

    metrics_to_compute = [metric] if isinstance(metric, str) else metric

    # Define the worker function for p_map internally to ensure correct version is used by workers
    def _worker_function_for_pmap(tau_param, probs_param, labels_param):
        results = {}
        for m in metrics_to_compute:
            results[m] = calculate_simplex_metrics(probs_param, labels_param, tau_param, metric=m)
        return results
    
    # Compute scores in parallel
    print(f"Computing scores for {len(tau_points)} points and metrics: {metrics_to_compute}...")
    all_results_per_tau = p_map(
        _worker_function_for_pmap,
        tau_points,
        [probs] * len(tau_points),
        [labels] * len(tau_points),
        num_cpus=n_jobs,
        desc=f"Evaluating thresholds for {metrics_to_compute}"
    )
    
    # Process results
    output_results = {}
    for m_idx, m_name in enumerate(metrics_to_compute):
        current_metric_scores = [res[m_name] for res in all_results_per_tau]
        
        best_idx = np.argmax(current_metric_scores)
        best_tau_for_metric = tau_points[best_idx]
        best_score_for_metric = current_metric_scores[best_idx]
        
        print(f"Best score for {m_name}: {best_score_for_metric:.4f} with tau={best_tau_for_metric}")
        output_results[m_name] = (best_tau_for_metric, best_score_for_metric, current_metric_scores, tau_points)

    if isinstance(metric, str): # Single metric was passed
        return output_results[metric]
    else: # List of metrics was passed
        return output_results
