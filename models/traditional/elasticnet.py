"""
ElasticNet Model

Linear regression with combined L1 and L2 regularization.
Provides sparsity (L1) while handling multicollinearity (L2).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ElasticNetModel(BaseModel):
    """
    ElasticNet model for drug sensitivity prediction.

    Combines L1 (Lasso) and L2 (Ridge) penalties for feature selection
    and regularization. Good for high-dimensional gene expression data.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        selection: str = "cyclic",
        **kwargs,
    ):
        """
        Initialize ElasticNet model.

        Args:
            alpha: Regularization strength
            l1_ratio: Mix of L1/L2 (0=Ridge, 1=Lasso, 0.5=balanced)
            fit_intercept: Whether to fit intercept
            max_iter: Maximum iterations
            tol: Tolerance for stopping
            selection: Coefficient update order ('cyclic' or 'random')
        """
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            **kwargs,
        )

        self.model_ = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ElasticNetModel":
        """
        Train ElasticNet model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Ignored
            y_val: Ignored

        Returns:
            self
        """
        logger.info(f"Training ElasticNet on {X_train.shape[0]} samples")

        self.model_.fit(X_train, y_train)
        self.is_fitted_ = True

        # Log sparsity
        n_nonzero = np.sum(self.model_.coef_ != 0)
        logger.info(f"ElasticNet: {n_nonzero}/{len(self.model_.coef_)} non-zero coefficients")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "alpha": self.model_.alpha,
            "l1_ratio": self.model_.l1_ratio,
            "fit_intercept": self.model_.fit_intercept,
            "max_iter": self.model_.max_iter,
            "tol": self.model_.tol,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9],
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get absolute coefficient values."""
        if not self.is_fitted_:
            return None
        return np.abs(self.model_.coef_)

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)

    def load(self, path: Union[str, Path]) -> "ElasticNetModel":
        """Load model."""
        self.model_ = joblib.load(path)
        self.is_fitted_ = True
        return self
