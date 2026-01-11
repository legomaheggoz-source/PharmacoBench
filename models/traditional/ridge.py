"""
Ridge Regression Model

Linear regression with L2 regularization.
Serves as the interpretable baseline for benchmarking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RidgeModel(BaseModel):
    """
    Ridge Regression model for drug sensitivity prediction.

    Ridge regression adds L2 penalty to prevent overfitting
    and handles multicollinearity in high-dimensional gene expression data.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        **kwargs,
    ):
        """
        Initialize Ridge model.

        Args:
            alpha: Regularization strength (larger = more regularization)
            fit_intercept: Whether to fit intercept
            max_iter: Maximum iterations for solver
            tol: Tolerance for stopping criterion
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )

        self.model_ = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RidgeModel":
        """
        Train Ridge regression model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Ignored (Ridge doesn't use validation)
            y_val: Ignored

        Returns:
            self
        """
        logger.info(f"Training Ridge model on {X_train.shape[0]} samples, {X_train.shape[1]} features")

        self.model_.fit(X_train, y_train)
        self.is_fitted_ = True

        logger.info("Ridge model training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_.predict(X)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "alpha": self.model_.alpha,
            "fit_intercept": self.model_.fit_intercept,
            "max_iter": self.model_.max_iter,
            "tol": self.model_.tol,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid for tuning."""
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature coefficients as importance.

        Returns:
            Absolute coefficient values
        """
        if not self.is_fitted_:
            return None
        return np.abs(self.model_.coef_)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "RidgeModel":
        """Load model from disk."""
        path = Path(path)
        self.model_ = joblib.load(path)
        self.is_fitted_ = True
        logger.info(f"Model loaded from {path}")
        return self
