"""
Tests for data preprocessor.
"""

import pytest
import pandas as pd
import numpy as np
from data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def raw_ic50_data(self):
        """Generate raw IC50 data (not log-transformed)."""
        np.random.seed(42)
        return pd.DataFrame({
            "DRUG_NAME": ["Drug_A"] * 50 + ["Drug_B"] * 50,
            "CELL_LINE_NAME": [f"Cell_{i % 10}" for i in range(100)],
            "IC50": np.random.exponential(scale=10, size=100),
            "AUC": np.random.uniform(0, 1, 100),
        })

    def test_log_transform_ic50(self, preprocessor, raw_ic50_data):
        """Test log transformation of IC50 values."""
        processed = preprocessor.transform_ic50(raw_ic50_data, target_col="IC50", log_transform=True)

        assert "LN_IC50" in processed.columns
        # Log values should be smaller than original exponential values
        assert processed["LN_IC50"].mean() < raw_ic50_data["IC50"].mean()

    def test_no_log_transform(self, preprocessor, raw_ic50_data):
        """Test without log transformation."""
        processed = preprocessor.transform_ic50(raw_ic50_data, target_col="IC50", log_transform=False)

        # Should just copy the values
        assert "LN_IC50" in processed.columns
        np.testing.assert_array_almost_equal(processed["LN_IC50"].values, raw_ic50_data["IC50"].values)

    def test_normalize_features_zscore(self, preprocessor):
        """Test z-score normalization."""
        features = pd.DataFrame({
            "gene_1": [1, 2, 3, 4, 5],
            "gene_2": [10, 20, 30, 40, 50],
        })

        normalized = preprocessor.normalize_features(features, method="zscore")

        # Z-score normalized should have mean ~0 and std ~1
        for col in normalized.columns:
            assert abs(normalized[col].mean()) < 1e-10
            assert abs(normalized[col].std() - 1) < 0.1

    def test_normalize_features_minmax(self, preprocessor):
        """Test min-max normalization."""
        features = pd.DataFrame({
            "gene_1": [1, 2, 3, 4, 5],
            "gene_2": [10, 20, 30, 40, 50],
        })

        normalized = preprocessor.normalize_features(features, method="minmax")

        # Min-max normalized should be in [0, 1]
        for col in normalized.columns:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1

    def test_handle_missing_values_drop(self, preprocessor):
        """Test dropping rows with missing values."""
        data = pd.DataFrame({
            "A": [1, 2, np.nan, 4],
            "B": [np.nan, 2, 3, 4],
            "C": [1, 2, 3, 4],
        })

        cleaned = preprocessor.handle_missing_values(data, strategy="drop")

        assert len(cleaned) == 2  # Only rows without NaN
        assert not cleaned.isnull().any().any()

    def test_handle_missing_values_mean(self, preprocessor):
        """Test imputing missing values with mean."""
        data = pd.DataFrame({
            "A": [1.0, 2.0, np.nan, 4.0],
            "B": [1.0, np.nan, 3.0, 4.0],
        })

        cleaned = preprocessor.handle_missing_values(data, strategy="mean")

        assert len(cleaned) == 4
        assert not cleaned.isnull().any().any()
        # NaN in A should be replaced with mean of [1, 2, 4] = 2.33...
        assert abs(cleaned.loc[2, "A"] - 7/3) < 0.01

    def test_filter_by_count(self, preprocessor, sample_gdsc_data):
        """Test filtering by minimum count."""
        # Add some rare drugs
        rare_drug_data = pd.DataFrame({
            "DRUG_NAME": ["RareDrug"] * 2,
            "CELL_LINE_NAME": ["Cell_1", "Cell_2"],
            "LN_IC50": [-1.0, -2.0],
            "DRUG_ID": [999, 999],
            "SANGER_MODEL_ID": [1, 2],
        })
        data = pd.concat([sample_gdsc_data, rare_drug_data], ignore_index=True)

        filtered = preprocessor.filter_by_count(data, column="DRUG_NAME", min_count=3)

        assert "RareDrug" not in filtered["DRUG_NAME"].values

    def test_remove_outliers_zscore(self, preprocessor):
        """Test outlier removal using z-score."""
        data = pd.DataFrame({
            "LN_IC50": [0, 1, 2, 1, 0, 100],  # 100 is an outlier
        })

        cleaned = preprocessor.remove_outliers(data, column="LN_IC50", method="zscore", threshold=3)

        assert len(cleaned) < len(data)
        assert 100 not in cleaned["LN_IC50"].values

    def test_remove_outliers_iqr(self, preprocessor):
        """Test outlier removal using IQR."""
        data = pd.DataFrame({
            "LN_IC50": [0, 1, 2, 1, 0, 1, 2, 100],  # 100 is an outlier
        })

        cleaned = preprocessor.remove_outliers(data, column="LN_IC50", method="iqr", threshold=1.5)

        assert 100 not in cleaned["LN_IC50"].values

    def test_full_pipeline(self, preprocessor, raw_ic50_data):
        """Test running full preprocessing pipeline."""
        processed = preprocessor.preprocess(
            raw_ic50_data,
            ic50_col="IC50",
            log_transform=True,
            remove_outliers=True,
        )

        assert "LN_IC50" in processed.columns
        assert len(processed) <= len(raw_ic50_data)
