"""
Tests for data splitting strategies.
"""

import pytest
import pandas as pd
import numpy as np
from data.splitters import DataSplitter


class TestDataSplitter:
    """Test suite for DataSplitter class."""

    def test_init_valid_strategy(self):
        """Test initialization with valid strategies."""
        for strategy in ["random", "drug_blind", "cell_blind", "disjoint"]:
            splitter = DataSplitter(strategy=strategy)
            assert splitter.strategy == strategy

    def test_init_invalid_strategy(self):
        """Test initialization with invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            DataSplitter(strategy="invalid_strategy")

    def test_random_split_proportions(self, sample_gdsc_data):
        """Test that random split maintains correct proportions."""
        splitter = DataSplitter(strategy="random", train_ratio=0.8, val_ratio=0.1)
        train, val, test = splitter.split(sample_gdsc_data)

        total = len(sample_gdsc_data)
        # Allow for some variance due to rounding
        assert abs(len(train) / total - 0.8) < 0.1
        assert abs(len(val) / total - 0.1) < 0.1
        assert abs(len(test) / total - 0.1) < 0.1

    def test_random_split_no_overlap(self, sample_gdsc_data):
        """Test that random split has no overlapping indices."""
        splitter = DataSplitter(strategy="random")
        train, val, test = splitter.split(sample_gdsc_data)

        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_drug_blind_split(self, sample_gdsc_data):
        """Test that drug-blind split has no overlapping drugs."""
        splitter = DataSplitter(strategy="drug_blind")
        train, val, test = splitter.split(sample_gdsc_data)

        train_drugs = set(train["DRUG_NAME"].unique())
        val_drugs = set(val["DRUG_NAME"].unique())
        test_drugs = set(test["DRUG_NAME"].unique())

        # No drug overlap between train and test
        assert len(train_drugs & test_drugs) == 0
        # No drug overlap between val and test
        assert len(val_drugs & test_drugs) == 0

    def test_cell_blind_split(self, sample_gdsc_data):
        """Test that cell-blind split has no overlapping cell lines."""
        splitter = DataSplitter(strategy="cell_blind")
        train, val, test = splitter.split(sample_gdsc_data)

        train_cells = set(train["CELL_LINE_NAME"].unique())
        val_cells = set(val["CELL_LINE_NAME"].unique())
        test_cells = set(test["CELL_LINE_NAME"].unique())

        # No cell line overlap between train and test
        assert len(train_cells & test_cells) == 0
        # No cell line overlap between val and test
        assert len(val_cells & test_cells) == 0

    def test_disjoint_split(self, sample_gdsc_data):
        """Test that disjoint split has no overlapping drugs or cell lines."""
        splitter = DataSplitter(strategy="disjoint")
        train, val, test = splitter.split(sample_gdsc_data)

        train_drugs = set(train["DRUG_NAME"].unique())
        test_drugs = set(test["DRUG_NAME"].unique())
        train_cells = set(train["CELL_LINE_NAME"].unique())
        test_cells = set(test["CELL_LINE_NAME"].unique())

        # Neither drugs nor cells overlap
        assert len(train_drugs & test_drugs) == 0
        assert len(train_cells & test_cells) == 0

    def test_split_reproducibility(self, sample_gdsc_data):
        """Test that splits are reproducible with same random state."""
        splitter1 = DataSplitter(strategy="random", random_state=42)
        splitter2 = DataSplitter(strategy="random", random_state=42)

        train1, val1, test1 = splitter1.split(sample_gdsc_data)
        train2, val2, test2 = splitter2.split(sample_gdsc_data)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.reset_index(drop=True), val2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

    def test_split_different_random_states(self, sample_gdsc_data):
        """Test that different random states produce different splits."""
        splitter1 = DataSplitter(strategy="random", random_state=42)
        splitter2 = DataSplitter(strategy="random", random_state=123)

        train1, _, _ = splitter1.split(sample_gdsc_data)
        train2, _, _ = splitter2.split(sample_gdsc_data)

        # Should have different samples (with high probability)
        assert not train1.index.equals(train2.index)

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        splitter = DataSplitter(strategy="random")
        empty_df = pd.DataFrame(columns=["DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"])

        train, val, test = splitter.split(empty_df)

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_small_dataframe(self):
        """Test handling of very small dataframe."""
        splitter = DataSplitter(strategy="random")
        small_df = pd.DataFrame({
            "DRUG_NAME": ["Drug_1", "Drug_2", "Drug_3"],
            "CELL_LINE_NAME": ["Cell_1", "Cell_2", "Cell_3"],
            "LN_IC50": [-1.0, -2.0, -3.0],
        })

        train, val, test = splitter.split(small_df)

        # Should still produce some split
        assert len(train) + len(val) + len(test) == len(small_df)
