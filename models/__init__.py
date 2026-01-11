"""
PharmacoBench Models

Machine learning models for drug sensitivity prediction.
All models implement the BaseModel interface.

Note: Heavy ML libraries (xgboost, lightgbm, torch) are optional.
The Streamlit demo works without them using simulated predictions.
"""

from models.base_model import BaseModel

# Registry of available models (populated based on installed packages)
MODEL_REGISTRY = {}

# Core sklearn models (always available)
from models.traditional.ridge import RidgeModel
from models.traditional.elasticnet import ElasticNetModel
from models.traditional.random_forest import RandomForestModel

MODEL_REGISTRY["ridge"] = RidgeModel
MODEL_REGISTRY["elasticnet"] = ElasticNetModel
MODEL_REGISTRY["random_forest"] = RandomForestModel

# XGBoost (optional - pulls CUDA dependencies)
try:
    from models.traditional.xgboost_model import XGBoostModel
    MODEL_REGISTRY["xgboost"] = XGBoostModel
except ImportError:
    XGBoostModel = None  # Not installed

# LightGBM (optional)
try:
    from models.traditional.lightgbm_model import LightGBMModel
    MODEL_REGISTRY["lightgbm"] = LightGBMModel
except ImportError:
    LightGBMModel = None  # Not installed

# MLP (requires PyTorch)
try:
    from models.deep_learning.mlp import MLPModel
    MODEL_REGISTRY["mlp"] = MLPModel
except ImportError:
    MLPModel = None  # PyTorch not installed

# Graph models (requires PyTorch Geometric)
try:
    from models.deep_learning.graphdrp import GraphDRPModel
    from models.deep_learning.deepcdr import DeepCDRModel
    MODEL_REGISTRY["graphdrp"] = GraphDRPModel
    MODEL_REGISTRY["deepcdr"] = DeepCDRModel
except ImportError:
    GraphDRPModel = None
    DeepCDRModel = None

__all__ = [
    "BaseModel",
    "RidgeModel",
    "ElasticNetModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "MLPModel",
    "MODEL_REGISTRY",
]
