"""
Visualization utilities for plotting points on the probability simplex.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize

def get_barycentric_coordinates():
    """Get the 2D Cartesian coordinates for the vertices of an equilateral triangle
    representing the probability simplex.
    
    Returns:
        tuple: (v0, v1, v2) Cartesian coordinates of vertices.
               v0 corresponds to class 0, v1 to class 1, v2 to class 2.
    """
    v0 = np.array([0, 0])               # Typically class 0 (e.g., bottom-left)
    v1 = np.array([1, 0])               # Typically class 1 (e.g., bottom-right)
    v2 = np.array([0.5, np.sqrt(3) / 2])  # Typically class 2 (e.g., top)
    return v0, v1, v2

def barycentric_to_cartesian(points_bary):
    """Convert barycentric coordinates to 2D Cartesian coordinates."""
    v0, v1, v2 = get_barycentric_coordinates()
    verts = np.stack([v0, v1, v2], axis=0)

    points_bary = np.asarray(points_bary)
    if points_bary.ndim == 1:
        points_bary = points_bary.reshape(1, -1)
    if points_bary.shape[1] != 3:
        raise ValueError("Barycentric points must have 3 coordinates.")
    
    return points_bary @ verts

def pca_project_to_2d(points):
    """Project high-dimensional points to 2D using PCA-like approach for simplex visualization"""
    # Center the data
    n_points = points.shape[0]
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute covariance matrix and eigenvectors
    cov = np.dot(centered.T, centered) / n_points
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to 2D using the top 2 principal components
    return np.dot(centered, eigenvectors[:, :2])

def plot_barycentric_point(
    tau: np.ndarray,
    label_map: dict[int, str] = None,
    point_kwargs: dict = None,
    vertex_kwargs: dict = None,
    ax: plt.Axes = None,
    show_regions: bool = False,
    region_n: int = 200,
    region_colors: list[str] = None,
    region_alpha: float = 0.3,
    show_outline: bool = True
) -> plt.Axes:
    """
    Draw the 2D simplex (equilateral triangle) and scatter one point τ.
    Optional: fill the three regions where each barycentric component
    dominates relative to τ.

    Args:
      tau           (3,) barycentric point summing to 1
      label_map     maps vertex indices to labels
      show_regions  if True, color Voronoi‐like regions
      region_n      grid resolution for coloring
      region_colors list of 3 colors
      region_alpha  transparency of region fill
      show_outline  if True, draw the simplex outline and vertex labels
    """
    # --- Convert input to NumPy array ---
    tau = np.asarray(tau)
    # --- End conversion ---

    # --- Add validation check ---
    if not np.isclose(np.sum(tau), 1.0):
        raise ValueError(f"tau must sum to 1, got {tau} (sum={np.sum(tau)})")
    if not np.all(tau >= 0):
        raise ValueError(f"tau components must be non-negative, got {tau}")
    # --- End validation check ---

    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if show_regions:
        region_colors = region_colors or ["lightcoral", "lightgreen", "lightskyblue"]
        b1g, b2g = np.meshgrid(np.linspace(0, 1, region_n), np.linspace(0, 1, region_n))
        b3g = 1 - b1g - b2g
        mask = (b1g >= 0) & (b2g >= 0) & (b3g >= 0)
        b1, b2, b3 = b1g[mask], b2g[mask], b3g[mask]
        pts = np.vstack([b1, b2, b3]).T @ verts
        x1, x2, x3 = tau
        r1 = (b1 - b2 > x1 - x2) & (b1 - b3 > x1 - x3)
        r2 = (b2 - b1 > x2 - x1) & (b2 - b3 > x2 - x3)
        r3 = (b3 - b1 > x3 - x1) & (b3 - b2 > x3 - x2)
        ax.scatter(pts[r1, 0], pts[r1, 1], s=6, color=region_colors[0], alpha=region_alpha)
        ax.scatter(pts[r2, 0], pts[r2, 1], s=6, color=region_colors[1], alpha=region_alpha)
        ax.scatter(pts[r3, 0], pts[r3, 1], s=6, color=region_colors[2], alpha=region_alpha)

    if show_outline:
        ax.add_patch(plt.Polygon(verts, fill=False, edgecolor='k', lw=1.5))
        if label_map:
            center = verts.mean(axis=0)
            shift = 0.05
            vk = vertex_kwargs or {}
            for i, v in enumerate(verts):
                pos = v + (v - center) / np.linalg.norm(v - center) * shift
                ax.text(pos[0], pos[1], label_map.get(i, str(i)), ha='center', va='center', **vk)

    pk = point_kwargs or {"c": "C0", "s": 80}
    p = tau @ verts
    ax.scatter(p[0], p[1], **pk)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')
    return ax

def plot_points_on_simplex(probs_bary, labels=None, ax=None, class_colors=None, 
                           label_map=None, tau=None, marker_size=5, data_alpha=0.7,
                           region_alpha=0.2, tau_marker_kwargs=None):
    """Plot multiple data points on the probability simplex, optionally showing
    decision regions defined by a 'tau' threshold.

    Args:
        probs_bary (np.ndarray): Probability outputs for data points (n_samples, 3).
        labels (np.ndarray, optional): True class labels for data points.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        class_colors (dict, optional): Mapping from class indices to colors for data points
                                     and for deriving region colors.
        label_map (dict, optional): Mapping from class indices to labels for simplex vertices.
        tau (np.ndarray, optional): Barycentric threshold (3,) to display and define regions.
                                  Defaults to (1/3, 1/3, 1/3) if None.
        marker_size (float): Size of markers for data points.
        data_alpha (float): Alpha for data points.
        region_alpha (float): Alpha for decision regions.
        tau_marker_kwargs (dict, optional): Matplotlib kwargs for the tau marker.
        
    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    if probs_bary.shape[1] != 3:
        raise ValueError("Input probabilities must be for a 3-class problem.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8.6)) # Adjusted for aspect ratio

    # Default class colors for data points if not provided
    if class_colors is None:
        class_colors = {0: 'red', 1: 'green', 2: 'blue'} # Bright defaults for points
    
    if label_map is None: # For vertex labels
        label_map = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

    # Point that defines the regions: tau or default barycenter
    region_defining_point_bary = tau if tau is not None else np.array([1/3, 1/3, 1/3])
    
    # Base colors for regions, derived from class_colors for consistency
    # Fallback to light versions if class_colors are very dark or not fully specified
    _default_region_bases = ['lightcoral', 'lightgreen', 'lightblue'] 
    prepared_region_base_colors = [
        class_colors.get(0, _default_region_bases[0]),
        class_colors.get(1, _default_region_bases[1]),
        class_colors.get(2, _default_region_bases[2])
    ]
    
    # Marker style for tau (if provided)
    if tau_marker_kwargs is None:
        tau_marker_kwargs = {'c': 'black', 's': 130, 'marker': '*', 'edgecolor': 'white', 'linewidth': 1.5, 'zorder': 10}
    
    # Marker style for default barycenter (if tau is None)
    default_center_marker_kwargs = {'c': 'dimgray', 's': 100, 'marker': 'o', 'edgecolor': 'white', 'linewidth': 1, 'zorder': 10}

    # Draw simplex outline, labels, regions, and the tau/center marker
    ax = plot_barycentric_point(
        point_bary=region_defining_point_bary,
        ax=ax,
        label_map=label_map,
        show_regions=True,
        region_alpha=region_alpha,
        region_colors_input=prepared_region_base_colors,
        point_kwargs=tau_marker_kwargs if tau is not None else default_center_marker_kwargs,
        show_outline=True
    )

    # Plot all data points (probs_bary)
    points_cartesian = barycentric_to_cartesian(probs_bary)

    if labels is not None:
        unique_labels = sorted(np.unique(labels)) # Sort for consistent legend order
        plotted_labels = []
        for label_val in unique_labels:
            if label_val not in class_colors: # Skip if no color defined for this label
                print(f"Warning: No color defined in class_colors for label {label_val}. Skipping these points.")
                continue
            mask = (labels == label_val)
            ax.scatter(points_cartesian[mask, 0], points_cartesian[mask, 1],
                       color=class_colors[label_val], 
                       s=marker_size, alpha=data_alpha, marker='.', zorder=3,
                       label=label_map.get(label_val, f"Class {label_val}"))
            plotted_labels.append(label_val)
        if plotted_labels: # Add legend if any points were plotted
             ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    else: # Plot all points in a single default color if no labels
        ax.scatter(points_cartesian[:, 0], points_cartesian[:, 1],
                   color='gray', s=marker_size, alpha=data_alpha, marker='.', zorder=3)
    
    plt.tight_layout() # Adjust layout to prevent cutting off legend/labels
    return ax


def plot_score_heatmap(scores, tau_points, metric="accuracy", best_tau=None, figsize=(10, 8), vmin=None, vmax=None, title=None):
    """Create a heatmap visualization of performance metrics across the simplex.
    
    Args:
        scores (np.ndarray): Array of metric values (accuracy or F1 scores)
        tau_points (list): List of tau points in the simplex
        metric (str): Metric type, either "accuracy" or "f1" (default: "accuracy")
        best_tau (np.ndarray, optional): The best tau value
        figsize (tuple): Figure size
        vmin (float, optional): Minimum value for colorbar
        vmax (float, optional): Maximum value for colorbar
        title (str, optional): Plot title
        
    Returns:
        tuple: (fig, best_tau, best_score)
    """
    # Validate metric type
    if metric not in ["accuracy", "f1"]:
        raise ValueError("Metric must be either 'accuracy' or 'f1'")
    
    # Find the best tau if not provided
    if best_tau is None:
        best_idx = np.argmax(scores)
        best_tau = tau_points[best_idx]
        best_score = scores[best_idx]
    else:
        best_score = scores[np.argmax(scores)]
    
    # Define simplex vertices in 2D
    v1 = np.array([0, 0])               # Bottom-left corner
    v2 = np.array([1, 0])               # Bottom-right corner
    v3 = np.array([0.5, np.sqrt(3)/2])  # Top corner
    
    # Convert tau points to cartesian coordinates
    points_2d = barycentric_to_cartesian(np.array(tau_points))
    
    # Create the triangulation for irregular grid
    tri = Triangulation(points_2d[:, 0], points_2d[:, 1])
    
    # Define color normalization and colormap
    # Adjust vmin to your specific needs based on the data
    if vmin is None:
        vmin = max(0.5, np.min(scores) - 0.05)  # More flexible lower bound for F1
    if vmax is None:
        vmax = min(1.0, np.max(scores) + 0.01)  # Cap at 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.cm.plasma
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap using triangulation
    contour = ax.tripcolor(tri, scores, cmap=cmap, norm=norm, shading='gouraud')
    
    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax)
    metric_label = "Accuracy" if metric == "accuracy" else "Macro F1 Score"
    cbar.set_label(metric_label, rotation=270, labelpad=20)
    
    # Mark the simplex vertices
    ax.scatter([v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], c='black', s=50)
    
    # Mark the standard point (1/3, 1/3, 1/3)
    std_tau = np.array([1/3, 1/3, 1/3])
    std_2d = std_tau[0] * v1 + std_tau[1] * v2 + std_tau[2] * v3
    ax.scatter(std_2d[0], std_2d[1], c='red', s=100, marker='o', 
              edgecolor='white', linewidth=1, label='Standard τ = [1/3, 1/3, 1/3]')
    
    # Mark the best tau
    best_2d = best_tau[0] * v1 + best_tau[1] * v2 + best_tau[2] * v3
    ax.scatter(best_2d[0], best_2d[1], c='lime', s=150, marker='*', 
              edgecolor='white', linewidth=1.5, 
              label=f'Best τ = [{best_tau[0]:.2f}, {best_tau[1]:.2f}, {best_tau[2]:.2f}]')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set title and axis properties
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f"{metric_label} Across Simplex", fontsize=16)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig, best_tau, best_score