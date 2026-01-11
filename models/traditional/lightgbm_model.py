"""
LightGBM Model

Fast gradient boosting framework with leaf-wise tree growth.
Memory efficient and excellent for large datasets.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM model for drug sensitivity prediction.

    Leaf-wise tree growth for faster training.
    Excellent memory efficiency and speed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting iterations
            num_leaves: Maximum leaves per tree
            max_depth: Maximum tree depth (-1=no limit)
            learning_rate: Step size shrinkage
            min_child_samples: Minimum samples in leaf
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            n_jobs: Parallel jobs
            random_state: Random seed
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        super().__init__(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

        self.model_ = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMModel":
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            self
        """
        logger.info(f"Training LightGBM on {X_train.shape[0]} samples")

        callbacks = [lgb.log_evaluation(period=0)]  # Suppress output

        if X_val is not None and y_val is not None:
            self.model_.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model_.fit(X_train, y_train, callbacks=callbacks)

        self.is_fitted_ = True
        logger.info("LightGBM training complete")
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
            "num_leaves": self.model_.num_leaves,
            "max_depth": self.model_.max_depth,
            "learning_rate": self.model_.learning_rate,
            "min_child_samples": self.model_.min_child_samples,
            "subsample": self.model_.subsample,
            "colsample_bytree": self.model_.colsample_bytree,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "n_estimators": [100, 200, 500],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.1],
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
        self.model_.booster_.save_model(str(path))

    def load(self, path: Union[str, Path]) -> "LightGBMModel":
        """Load model."""
        self.model_ = lgb.Booster(model_file=str(path))
        self.is_fitted_ = True
        return self
