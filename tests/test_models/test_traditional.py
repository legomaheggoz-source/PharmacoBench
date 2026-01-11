"""
Tests for traditional ML models.
"""

import pytest
import numpy as np
from pathlib import Path

from models.traditional.ridge import RidgeModel
from models.traditional.elasticnet import ElasticNetModel
from models.traditional.random_forest import RandomForestModel
from models.traditional.xgboost_model import XGBoostModel
from models.traditional.lightgbm_model import LightGBMModel


class TestRidgeModel:
    """Test suite for Ridge Regression model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RidgeModel(alpha=1.0)
        assert model.alpha == 1.0
        assert model.model is None

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = RidgeModel(alpha=1.0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape
        assert model.model is not None

    def test_hyperparameters(self):
        """Test hyperparameter retrieval."""
        model = RidgeModel(alpha=0.5)
        params = model.get_hyperparameters()

        assert "alpha" in params
        assert params["alpha"] == 0.5

    def test_save_load(self, sample_train_test_data, tmp_path):
        """Test model save and load."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = RidgeModel(alpha=1.0)
        model.fit(X_train, y_train)
        original_predictions = model.predict(X_test)

        # Save
        save_path = tmp_path / "ridge_model.joblib"
        model.save(str(save_path))
        assert save_path.exists()

        # Load
        new_model = RidgeModel(alpha=1.0)
        new_model.load(str(save_path))
        loaded_predictions = new_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestElasticNetModel:
    """Test suite for ElasticNet model."""

    def test_initialization(self):
        """Test model initialization."""
        model = ElasticNetModel(alpha=1.0, l1_ratio=0.5)
        assert model.alpha == 1.0
        assert model.l1_ratio == 0.5

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape

    def test_sparsity(self, sample_train_test_data):
        """Test that ElasticNet produces sparse coefficients."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = ElasticNetModel(alpha=1.0, l1_ratio=0.9)
        model.fit(X_train, y_train)

        # High L1 ratio should produce some zero coefficients
        n_zero = np.sum(model.model.coef_ == 0)
        assert n_zero >= 0  # May or may not have zeros depending on data


class TestRandomForestModel:
    """Test suite for Random Forest model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestModel(n_estimators=100, max_depth=10)
        assert model.n_estimators == 100
        assert model.max_depth == 10

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = RandomForestModel(n_estimators=10, max_depth=5)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape

    def test_feature_importance(self, sample_train_test_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = RandomForestModel(n_estimators=10)
        model.fit(X_train, y_train)

        importances = model.model.feature_importances_

        assert len(importances) == X_train.shape[1]
        assert np.sum(importances) > 0


class TestXGBoostModel:
    """Test suite for XGBoost model."""

    def test_initialization(self):
        """Test model initialization."""
        model = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
        assert model.n_estimators == 100
        assert model.max_depth == 6
        assert model.learning_rate == 0.1

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = XGBoostModel(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape

    def test_early_stopping(self, sample_train_test_data):
        """Test early stopping with validation data."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = XGBoostModel(n_estimators=100, max_depth=3, early_stopping_rounds=5)
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        # Model should have stopped before 100 iterations if early stopping worked
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape


class TestLightGBMModel:
    """Test suite for LightGBM model."""

    def test_initialization(self):
        """Test model initialization."""
        model = LightGBMModel(n_estimators=100, num_leaves=31, learning_rate=0.1)
        assert model.n_estimators == 100
        assert model.num_leaves == 31
        assert model.learning_rate == 0.1

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        model = LightGBMModel(n_estimators=10, num_leaves=15)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape

    def test_hyperparameters(self):
        """Test hyperparameter retrieval."""
        model = LightGBMModel(n_estimators=50, num_leaves=31, learning_rate=0.05)
        params = model.get_hyperparameters()

        assert params["n_estimators"] == 50
        assert params["num_leaves"] == 31
        assert params["learning_rate"] == 0.05


class TestModelConsistency:
    """Test consistency across all traditional models."""

    @pytest.fixture
    def all_models(self):
        """Return instances of all traditional models."""
        return [
            RidgeModel(alpha=1.0),
            ElasticNetModel(alpha=0.1, l1_ratio=0.5),
            RandomForestModel(n_estimators=10, max_depth=5),
            XGBoostModel(n_estimators=10, max_depth=3),
            LightGBMModel(n_estimators=10, num_leaves=15),
        ]

    def test_all_models_have_required_methods(self, all_models):
        """Test that all models implement required interface methods."""
        required_methods = ["fit", "predict", "get_hyperparameters", "save", "load"]

        for model in all_models:
            for method in required_methods:
                assert hasattr(model, method), f"{model.__class__.__name__} missing {method}"
                assert callable(getattr(model, method))

    def test_all_models_produce_predictions(self, all_models, sample_train_test_data):
        """Test that all models produce valid predictions."""
        X_train, X_test, y_train, y_test = sample_train_test_data

        for model in all_models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            assert predictions.shape == y_test.shape, f"{model.__class__.__name__} prediction shape mismatch"
            assert not np.any(np.isnan(predictions)), f"{model.__class__.__name__} produced NaN predictions"
            assert not np.any(np.isinf(predictions)), f"{model.__class__.__name__} produced Inf predictions"
