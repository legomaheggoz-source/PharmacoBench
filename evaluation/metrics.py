"""
Evaluation Metrics

Standard metrics for drug sensitivity prediction:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Pearson correlation
- Spearman correlation
"""

import logging
from typing import Dict, Union

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def compute_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute R² (Coefficient of Determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R² value (can be negative for poor fits)
    """
    return r2_score(y_true, y_pred)


def compute_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Pearson correlation coefficient.

    Measures linear correlation between predictions and ground truth.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Pearson correlation coefficient
    """
    if len(y_true) < 2:
        return 0.0

    correlation, _ = stats.pearsonr(y_true, y_pred)
    return correlation if not np.isnan(correlation) else 0.0


def compute_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Measures monotonic relationship between predictions and ground truth.
    More robust to outliers than Pearson.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2:
        return 0.0

    correlation, _ = stats.spearmanr(y_true, y_pred)
    return correlation if not np.isnan(correlation) else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict with all metric values
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    metrics = {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "pearson": compute_pearson(y_true, y_pred),
        "spearman": compute_spearman(y_true, y_pred),
        "n_samples": len(y_true),
    }

    return metrics


def format_metrics(
    metrics: Dict[str, float],
    precision: int = 4,
) -> str:
    """
    Format metrics for display.

    Args:
        metrics: Dict of metric values
        precision: Decimal precision

    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if name == "n_samples":
            lines.append(f"  {name}: {int(value)}")
        else:
            lines.append(f"  {name}: {value:.{precision}f}")
    return "\n".join(lines)


class MetricsTracker:
    """
    Track metrics across multiple evaluations.

    Useful for computing mean and std across cross-validation folds.
    """

    def __init__(self):
        self.history: list = []

    def add(self, metrics: Dict[str, float]) -> None:
        """Add metrics from one evaluation."""
        self.history.append(metrics)

    def get_mean(self) -> Dict[str, float]:
        """Compute mean across all evaluations."""
        if not self.history:
            return {}

        mean_metrics = {}
        for key in self.history[0]:
            if key != "n_samples":
                values = [m[key] for m in self.history]
                mean_metrics[key] = np.mean(values)
            else:
                mean_metrics[key] = sum(m[key] for m in self.history)

        return mean_metrics

    def get_std(self) -> Dict[str, float]:
        """Compute std across all evaluations."""
        if not self.history:
            return {}

        std_metrics = {}
        for key in self.history[0]:
            if key != "n_samples":
                values = [m[key] for m in self.history]
                std_metrics[key] = np.std(values)

        return std_metrics

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get mean ± std summary."""
        return {
            "mean": self.get_mean(),
            "std": self.get_std(),
            "n_folds": len(self.history),
        }

    def reset(self) -> None:
        """Clear history."""
        self.history = []


def main():
    """Test metrics."""
    np.random.seed(42)

    # Generate sample predictions
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.5  # Add noise

    print("Sample metrics:")
    metrics = compute_all_metrics(y_true, y_pred)
    print(format_metrics(metrics))

    # Test tracker
    tracker = MetricsTracker()
    for _ in range(5):
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.5
        tracker.add(compute_all_metrics(y_true, y_pred))

    print("\nCross-validation summary:")
    summary = tracker.get_summary()
    print(f"  RMSE: {summary['mean']['rmse']:.4f} ± {summary['std']['rmse']:.4f}")
    print(f"  Pearson: {summary['mean']['pearson']:.4f} ± {summary['std']['pearson']:.4f}")


if __name__ == "__main__":
    main()
