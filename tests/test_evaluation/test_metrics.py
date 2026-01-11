"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from evaluation.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    calculate_pearson,
    calculate_spearman,
    calculate_all_metrics,
)


class TestRMSE:
    """Test suite for RMSE calculation."""

    def test_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rmse = calculate_rmse(y_true, y_pred)

        assert rmse == 0.0

    def test_known_value(self):
        """Test RMSE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1

        rmse = calculate_rmse(y_true, y_pred)

        assert rmse == 1.0

    def test_positive_value(self):
        """Test that RMSE is always positive."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        rmse = calculate_rmse(y_true, y_pred)

        assert rmse >= 0

    def test_symmetry(self):
        """Test that RMSE is symmetric."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        rmse1 = calculate_rmse(y_true, y_pred)
        rmse2 = calculate_rmse(y_pred, y_true)

        assert rmse1 == rmse2


class TestMAE:
    """Test suite for MAE calculation."""

    def test_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mae = calculate_mae(y_true, y_pred)

        assert mae == 0.0

    def test_known_value(self):
        """Test MAE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1

        mae = calculate_mae(y_true, y_pred)

        assert mae == 1.0

    def test_less_than_rmse(self):
        """Test that MAE <= RMSE."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)

        assert mae <= rmse


class TestR2:
    """Test suite for R² calculation."""

    def test_perfect_prediction(self):
        """Test R² with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = calculate_r2(y_true, y_pred)

        assert r2 == 1.0

    def test_mean_prediction(self):
        """Test R² when predicting the mean (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))

        r2 = calculate_r2(y_true, y_pred)

        assert abs(r2) < 1e-10

    def test_range(self):
        """Test that R² is typically in [-inf, 1]."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        r2 = calculate_r2(y_true, y_pred)

        assert r2 <= 1.0


class TestPearson:
    """Test suite for Pearson correlation."""

    def test_perfect_correlation(self):
        """Test Pearson with perfect positive correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x

        pearson = calculate_pearson(y_true, y_pred)

        assert abs(pearson - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        """Test Pearson with perfect negative correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        pearson = calculate_pearson(y_true, y_pred)

        assert abs(pearson - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Test Pearson with no correlation."""
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = np.random.randn(1000)

        pearson = calculate_pearson(y_true, y_pred)

        # Should be close to 0 for independent random variables
        assert abs(pearson) < 0.1

    def test_range(self):
        """Test that Pearson is in [-1, 1]."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        pearson = calculate_pearson(y_true, y_pred)

        assert -1.0 <= pearson <= 1.0


class TestSpearman:
    """Test suite for Spearman correlation."""

    def test_perfect_monotonic(self):
        """Test Spearman with perfect monotonic relationship."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # y = x^2, monotonic

        spearman = calculate_spearman(y_true, y_pred)

        assert abs(spearman - 1.0) < 1e-10

    def test_range(self):
        """Test that Spearman is in [-1, 1]."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        spearman = calculate_spearman(y_true, y_pred)

        assert -1.0 <= spearman <= 1.0

    def test_robust_to_outliers(self):
        """Test that Spearman is more robust to outliers than Pearson."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        pearson = calculate_pearson(y_true, y_pred)
        spearman = calculate_spearman(y_true, y_pred)

        # Spearman should be higher because it ignores magnitude
        assert spearman > pearson


class TestAllMetrics:
    """Test suite for combined metrics calculation."""

    def test_returns_all_metrics(self):
        """Test that all metrics are returned."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metrics = calculate_all_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "pearson" in metrics
        assert "spearman" in metrics

    def test_consistent_values(self):
        """Test that combined function returns consistent values."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        metrics = calculate_all_metrics(y_true, y_pred)

        assert metrics["rmse"] == calculate_rmse(y_true, y_pred)
        assert metrics["mae"] == calculate_mae(y_true, y_pred)
        assert metrics["r2"] == calculate_r2(y_true, y_pred)
        assert metrics["pearson"] == calculate_pearson(y_true, y_pred)
        assert metrics["spearman"] == calculate_spearman(y_true, y_pred)

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
            calculate_all_metrics(y_true, y_pred)

    def test_single_element(self):
        """Test handling of single element arrays."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])

        # Should handle gracefully (may produce NaN for correlation)
        metrics = calculate_all_metrics(y_true, y_pred)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
