"""
XGBoost Model

Gradient boosting framework optimized for speed and performance.
Handles missing values and provides strong tabular performance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for drug sensitivity prediction.

    Gradient boosting with regularization for preventing overfitting.
    Excellent performance on tabular data.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            min_child_weight: Minimum sum of instance weight in child
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            n_jobs: Parallel jobs
            random_state: Random seed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

        self.model_ = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:squarederror",
            tree_method="hist",  # Fast histogram-based method
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets

        Returns:
            self
        """
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted_ = True

        logger.info("XGBoost training complete")
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
            "learning_rate": self.model_.learning_rate,
            "min_child_weight": self.model_.min_child_weight,
            "subsample": self.model_.subsample,
            "colsample_bytree": self.model_.colsample_bytree,
            "reg_alpha": self.model_.reg_alpha,
            "reg_lambda": self.model_.reg_lambda,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if not self.is_fitted_:
            return None
        return self.model_.feature_importances_

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path))

    def load(self, path: Union[str, Path]) -> "XGBoostModel":
        """Load model."""
        self.model_.load_model(str(path))
        self.is_fitted_ = True
        return self
