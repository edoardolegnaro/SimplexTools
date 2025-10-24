import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from simplextools.metrics import get_simplex_predictions
from simplextools.thresh_tuning import generate_simplex_grid


def calculate_ovr_rates_for_tau(probs: np.ndarray, labels: np.ndarray, tau: np.ndarray):
    """
    Calculates the TPR and FPR for each class for a single tau vector.

    Args:
        probs (np.ndarray): Model probability outputs (n_samples, n_classes).
        labels (np.ndarray): True labels (n_samples,).
        tau (np.ndarray): A single threshold vector on the simplex (n_classes,).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing (tpr_rates, fpr_rates)
                                       where each is a numpy array of shape (n_classes,).
    """
    n_classes = probs.shape[1]

    # Get predictions for the given tau
    predictions = get_simplex_predictions(probs, tau)

    tpr_rates = np.zeros(n_classes)
    fpr_rates = np.zeros(n_classes)

    for i in range(n_classes):
        # TP: Correctly predicted as class i
        tp = np.sum((predictions == i) & (labels == i))

        # FP: Incorrectly predicted as class i
        fp = np.sum((predictions == i) & (labels != i))

        # FN: Incorrectly predicted as not class i when it was
        fn = np.sum((predictions != i) & (labels == i))

        # TN: Correctly predicted as not class i
        tn = np.sum((predictions != i) & (labels != i))

        # TPR
        tpr_rates[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FPR
        fpr_rates[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr_rates, fpr_rates


def generate_multiclass_roc_points(
    probs: np.ndarray, labels: np.ndarray, resolution: int = 20
):
    """
    Generates ROC points for each class by iterating over thresholds on the simplex.

    Args:
        probs (np.ndarray): Model probability outputs (n_samples, n_classes).
        labels (np.ndarray): True labels (n_samples,).
        resolution (int): The resolution of the grid for tau vectors.

    Returns:
        dict: A dictionary where keys are class indices and values are lists of (FPR, TPR) tuples.
    """
    n_samples, n_classes = probs.shape

    tau_points = generate_simplex_grid(n_classes, resolution)

    roc_points = {i: [] for i in range(n_classes)}

    print(f"Generating ROC points for {len(tau_points)} thresholds...")
    for tau in tqdm(tau_points):
        tpr_rates, fpr_rates = calculate_ovr_rates_for_tau(probs, labels, tau)
        for i in range(n_classes):
            roc_points[i].append((fpr_rates[i], tpr_rates[i]))

    return roc_points


def plot_multiclass_roc(roc_points: dict, label_map: dict = None):
    """
    Plots the generated multiclass ROC points.
    """
    n_classes = len(roc_points)
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        points = np.array(roc_points[i])
        label = label_map.get(i, f"Class {i}") if label_map else f"Class {i}"
        plt.scatter(
            points[:, 0], points[:, 1], alpha=0.5, s=15, label=f"{label} points"
        )

    # Plot the baseline random guess line
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Multiclass ROC Points on the Simplex")
    plt.legend()
    plt.grid(True)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.show()


if __name__ == "__main__":
    # --- Example Usage ---
    N_SAMPLES = 1000
    N_CLASSES = 3
    RESOLUTION = 25

    # Dummy probability data and labels
    # Probabilities skewed towards the true label
    np.random.seed(42)
    dummy_probs = np.random.dirichlet(alpha=np.ones(N_CLASSES) * 0.8, size=N_SAMPLES)
    dummy_labels = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

    for i in range(N_SAMPLES):
        dummy_probs[i, dummy_labels[i]] += 0.5
    dummy_probs /= dummy_probs.sum(axis=1, keepdims=True)

    # Generate the ROC points
    roc_data = generate_multiclass_roc_points(
        dummy_probs, dummy_labels, resolution=RESOLUTION
    )

    # Plot the results
    class_labels = {0: "Red", 1: "Green", 2: "Blue"}
    plot_multiclass_roc(roc_data, label_map=class_labels)
