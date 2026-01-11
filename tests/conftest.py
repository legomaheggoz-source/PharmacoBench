"""
Pytest configuration and fixtures for PharmacoBench tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_gdsc_data():
    """Generate sample GDSC-like data for testing."""
    np.random.seed(42)
    n_samples = 100

    drugs = [f"Drug_{i}" for i in range(10)]
    cell_lines = [f"CellLine_{i}" for i in range(20)]

    data = {
        "DRUG_NAME": np.random.choice(drugs, n_samples),
        "CELL_LINE_NAME": np.random.choice(cell_lines, n_samples),
        "LN_IC50": np.random.normal(-2, 1.5, n_samples),
        "DRUG_ID": np.random.randint(1, 100, n_samples),
        "SANGER_MODEL_ID": np.random.randint(1, 100, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature matrix for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    return np.random.randn(n_samples, n_features)


@pytest.fixture
def sample_targets():
    """Generate sample target values for testing."""
    np.random.seed(42)
    n_samples = 100

    return np.random.normal(-2, 1.5, n_samples)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_train_test_data(sample_features, sample_targets):
    """Generate train/test split data."""
    n_train = 80
    n_test = 20

    X_train = sample_features[:n_train]
    X_test = sample_features[n_train:]
    y_train = sample_targets[:n_train]
    y_test = sample_targets[n_train:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_smiles_list():
    """Sample SMILES strings for molecular testing."""
    return [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
    ]
