# PharmacoBench API Reference

## Data Module

### GDSCDownloader

Downloads GDSC datasets from cancerrxgene.org.

```python
from data.downloader import GDSCDownloader

downloader = GDSCDownloader(cache_dir="data/cache")
```

#### Methods

**download_all()**
```python
def download_all(self, force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download all GDSC datasets.

    Args:
        force: If True, redownload even if cached

    Returns:
        Dictionary with keys: 'gdsc1_ic50', 'gdsc2_ic50', 'cell_lines', 'drug_info'
    """
```

**download_ic50(version)**
```python
def download_ic50(self, version: str = "gdsc1") -> pd.DataFrame:
    """
    Download IC50 data for specified GDSC version.

    Args:
        version: 'gdsc1' or 'gdsc2'

    Returns:
        DataFrame with columns: DRUG_NAME, CELL_LINE_NAME, IC50, LN_IC50, ...
    """
```

### DataPreprocessor

Preprocesses raw GDSC data.

```python
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
```

#### Methods

**preprocess()**
```python
def preprocess(
    self,
    df: pd.DataFrame,
    ic50_col: str = "IC50",
    log_transform: bool = True,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        df: Raw IC50 data
        ic50_col: Column name for IC50 values
        log_transform: Apply natural log transformation
        remove_outliers: Remove Z-score outliers
        outlier_threshold: Z-score cutoff

    Returns:
        Preprocessed DataFrame with LN_IC50 column
    """
```

**normalize_features()**
```python
def normalize_features(
    self,
    features: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Normalize feature matrix.

    Args:
        features: Feature DataFrame
        method: 'zscore' or 'minmax'

    Returns:
        Normalized features
    """
```

### FeatureEngineer

Extracts features from raw data.

```python
from data.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
```

#### Methods

**smiles_to_fingerprint()**
```python
def smiles_to_fingerprint(
    self,
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Convert SMILES to Morgan fingerprint.

    Args:
        smiles: SMILES string
        radius: Morgan radius
        n_bits: Fingerprint length

    Returns:
        Binary fingerprint array
    """
```

**smiles_to_graph()**
```python
def smiles_to_graph(self, smiles: str) -> Data:
    """
    Convert SMILES to PyTorch Geometric graph.

    Args:
        smiles: SMILES string

    Returns:
        PyG Data object with:
        - x: Node features (atom properties)
        - edge_index: Bond connections
        - edge_attr: Bond properties
    """
```

### DataSplitter

Creates train/val/test splits.

```python
from data.splitters import DataSplitter

splitter = DataSplitter(
    strategy="random",
    train_ratio=0.8,
    val_ratio=0.1,
    random_state=42,
)
```

#### Strategies

| Strategy | Description |
|----------|-------------|
| `"random"` | Standard random split |
| `"drug_blind"` | Test drugs unseen in training |
| `"cell_blind"` | Test cell lines unseen in training |
| `"disjoint"` | Neither drugs nor cells overlap |

#### Methods

**split()**
```python
def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data according to strategy.

    Args:
        df: DataFrame with DRUG_NAME and CELL_LINE_NAME columns

    Returns:
        (train_df, val_df, test_df)
    """
```

---

## Models Module

### BaseModel

Abstract base class for all models.

```python
from models.base_model import BaseModel
```

#### Abstract Methods

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameter dictionary."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
```

### Traditional Models

#### RidgeModel

```python
from models.traditional.ridge import RidgeModel

model = RidgeModel(alpha=1.0)
```

**Parameters:**
- `alpha` (float): Regularization strength. Default: 1.0

#### ElasticNetModel

```python
from models.traditional.elasticnet import ElasticNetModel

model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
```

**Parameters:**
- `alpha` (float): Regularization strength. Default: 0.1
- `l1_ratio` (float): L1/L2 mixing (0=L2, 1=L1). Default: 0.5

#### RandomForestModel

```python
from models.traditional.random_forest import RandomForestModel

model = RandomForestModel(n_estimators=100, max_depth=None)
```

**Parameters:**
- `n_estimators` (int): Number of trees. Default: 100
- `max_depth` (int or None): Maximum tree depth. Default: None

#### XGBoostModel

```python
from models.traditional.xgboost_model import XGBoostModel

model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=10,
)
```

**Parameters:**
- `n_estimators` (int): Boosting rounds. Default: 100
- `max_depth` (int): Tree depth. Default: 6
- `learning_rate` (float): Step size. Default: 0.1
- `early_stopping_rounds` (int or None): Early stopping. Default: None

#### LightGBMModel

```python
from models.traditional.lightgbm_model import LightGBMModel

model = LightGBMModel(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
)
```

**Parameters:**
- `n_estimators` (int): Boosting rounds. Default: 100
- `num_leaves` (int): Maximum leaves. Default: 31
- `learning_rate` (float): Step size. Default: 0.1

### Deep Learning Models

#### MLPModel

```python
from models.deep_learning.mlp import MLPModel

model = MLPModel(
    input_dim=1000,
    hidden_dims=[256, 128],
    dropout=0.2,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
)
```

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dims` (List[int]): Hidden layer sizes
- `dropout` (float): Dropout rate. Default: 0.2
- `learning_rate` (float): Optimizer LR. Default: 0.001
- `batch_size` (int): Training batch size. Default: 32
- `epochs` (int): Training epochs. Default: 100

#### GraphDRPModel

```python
from models.deep_learning.graphdrp import GraphDRPModel

model = GraphDRPModel(
    drug_features=9,
    cell_features=735,
    hidden_dim=128,
    n_layers=3,
    dropout=0.2,
)
```

**Parameters:**
- `drug_features` (int): Drug node feature dim. Default: 9
- `cell_features` (int): Cell line feature dim
- `hidden_dim` (int): Hidden layer size. Default: 128
- `n_layers` (int): GCN layers. Default: 3
- `dropout` (float): Dropout rate. Default: 0.2

#### DeepCDRModel

```python
from models.deep_learning.deepcdr import DeepCDRModel

model = DeepCDRModel(
    drug_features=9,
    mutation_features=735,
    expression_features=697,
    methylation_features=808,
    hidden_dim=128,
)
```

**Parameters:**
- `drug_features` (int): Drug node feature dim
- `mutation_features` (int): Mutation vector size
- `expression_features` (int): Expression vector size
- `methylation_features` (int): Methylation vector size
- `hidden_dim` (int): Hidden layer size. Default: 128

---

## Evaluation Module

### Metrics

```python
from evaluation.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    calculate_pearson,
    calculate_spearman,
    calculate_all_metrics,
)
```

#### Functions

**calculate_rmse()**
```python
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
```

**calculate_mae()**
```python
def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
```

**calculate_r2()**
```python
def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
```

**calculate_pearson()**
```python
def calculate_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
```

**calculate_spearman()**
```python
def calculate_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
```

**calculate_all_metrics()**
```python
def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all metrics.

    Returns:
        {'rmse': float, 'mae': float, 'r2': float, 'pearson': float, 'spearman': float}
    """
```

### BenchmarkRunner

Orchestrates model training and evaluation.

```python
from evaluation.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner()
```

#### Methods

**add_model()**
```python
def add_model(self, name: str, model: BaseModel) -> None:
    """Add a model to the benchmark."""
```

**remove_model()**
```python
def remove_model(self, name: str) -> None:
    """Remove a model from the benchmark."""
```

**run()**
```python
def run(
    self,
    df: pd.DataFrame,
    X_features: np.ndarray,
    y_target: np.ndarray,
    strategies: List[str] = ["random"],
    n_cv_folds: int = 0,
) -> BenchmarkSummary:
    """
    Run benchmarks.

    Args:
        df: DataFrame with DRUG_NAME, CELL_LINE_NAME columns
        X_features: Feature matrix (n_samples, n_features)
        y_target: Target values (n_samples,)
        strategies: List of split strategies
        n_cv_folds: Cross-validation folds (0 = no CV)

    Returns:
        BenchmarkSummary with results
    """
```

### BenchmarkSummary

Results container.

```python
summary.results  # List[ModelResult]
summary.to_dataframe()  # pd.DataFrame
summary.get_best_model(metric="rmse")  # ModelResult
```

### ModelResult

Individual result.

```python
@dataclass
class ModelResult:
    model_name: str
    split_strategy: str
    rmse: float
    mae: float
    r2: float
    pearson: float
    spearman: float
    train_time: float
```

---

## Streamlit Components

### Main Entry Point

```python
# app/main.py
streamlit run app/main.py
```

### Pages

| Page | File | Description |
|------|------|-------------|
| Home | `main.py` | Welcome and navigation |
| Dashboard | `pages/1_Dashboard.py` | KPIs and overview |
| Data Explorer | `pages/2_Data_Explorer.py` | Dataset statistics |
| Model Training | `pages/3_Model_Training.py` | Training interface |
| Benchmark Results | `pages/4_Benchmark_Results.py` | Performance comparison |
| Drug Analysis | `pages/5_Drug_Analysis.py` | Drug profiles |
| Cell Line Analysis | `pages/6_Cell_Line_Analysis.py` | Cell line profiles |

### Caching

Use Streamlit caching for expensive operations:

```python
@st.cache_data
def load_data():
    """Cached data loading."""
    return pd.read_csv("data.csv")

@st.cache_resource
def load_model():
    """Cached model loading."""
    return joblib.load("model.joblib")
```
