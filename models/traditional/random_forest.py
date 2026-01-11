"""
Random Forest Model

Ensemble of decision trees for non-linear regression.
Handles feature interactions well without requiring feature scaling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for drug sensitivity prediction.

    Ensemble of decision trees with bagging and feature randomization.
    Provides non-linear modeling and built-in feature importance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float] = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None=unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider per split
            n_jobs: Parallel jobs (-1=all cores)
            random_state: Random seed
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

        self.model_ = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestModel":
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Ignored (RF doesn't use validation during training)
            y_val: Ignored

        Returns:
            self
        """
        logger.info(
            f"Training Random Forest ({self.hyperparameters['n_estimators']} trees) "
            f"on {X_train.shape[0]} samples"
        )

        self.model_.fit(X_train, y_train)
        self.is_fitted_ = True

        logger.info("Random Forest training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "n_estimators": self.model_.n_estimators,
            "max_depth": self.model_.max_depth,
            "min_samples_split": self.model_.min_samples_split,
            "min_samples_leaf": self.model_.min_samples_leaf,
            "max_features": self.model_.max_features,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances from tree ensemble."""
        if not self.is_fitted_:
            return None
        return self.model_.feature_importances_

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)

    def load(self, path: Union[str, Path]) -> "RandomForestModel":
        """Load model."""
        self.model_ = joblib.load(path)
        self.is_fitted_ = True
        return self
