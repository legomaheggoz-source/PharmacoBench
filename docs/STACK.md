# PharmacoBench Technology Stack

## Overview

This document describes the technology choices for PharmacoBench and the rationale behind each decision.

## Core Framework

### Streamlit (v1.28+)
**Purpose**: Web application framework

**Why Streamlit?**
- Rapid prototyping for data-heavy applications
- Native Python - no JavaScript required
- Built-in caching with `@st.cache_data`
- Multi-page app support
- Free hosting on HuggingFace Spaces

**Alternatives Considered:**
- Dash (Plotly): More complex, steeper learning curve
- Flask: Would require separate frontend
- Gradio: Less flexible for complex dashboards

## Data Processing

### Pandas (v2.0+)
**Purpose**: Tabular data manipulation

**Key Usage:**
- Loading GDSC Excel files
- Data cleaning and transformation
- Aggregations and groupby operations

### NumPy (v1.24+)
**Purpose**: Numerical operations

**Key Usage:**
- Feature matrix operations
- Vectorized computations
- Random number generation for splits

### SciPy (v1.10+)
**Purpose**: Scientific computing

**Key Usage:**
- Statistical tests
- Correlation calculations
- Sparse matrix operations

## Machine Learning

### Scikit-learn (v1.3+)
**Purpose**: Traditional ML models and utilities

**Models:**
- Ridge Regression
- ElasticNet
- Random Forest

**Utilities:**
- Train/test splitting
- StandardScaler, MinMaxScaler
- Metrics (RMSE, MAE, R²)

### XGBoost (v2.0+)
**Purpose**: Gradient boosting

**Why XGBoost?**
- State-of-the-art tabular performance
- Built-in early stopping
- GPU acceleration support
- Feature importance

### LightGBM (v4.0+)
**Purpose**: Fast gradient boosting

**Why LightGBM?**
- Faster training than XGBoost
- Lower memory usage
- Excellent for large datasets
- Histogram-based splitting

## Deep Learning

### PyTorch (v2.0+)
**Purpose**: Neural network framework

**Why PyTorch?**
- Dynamic computation graphs
- Pythonic API
- Strong research community
- Good debugging experience

**Models:**
- MLP (Multi-Layer Perceptron)

### PyTorch Geometric (v2.4+)
**Purpose**: Graph neural networks

**Why PyG?**
- Native PyTorch integration
- Rich library of GNN layers
- Molecular graph support
- Efficient batching

**Models:**
- GraphDRP (GCN-based drug encoder)
- DeepCDR (Multi-omics GNN hybrid)

## Chemistry

### RDKit (v2023.03+)
**Purpose**: Molecular processing

**Key Usage:**
- SMILES parsing
- Molecular fingerprints
- 2D structure visualization
- Graph feature extraction

**Why RDKit?**
- Industry standard for cheminformatics
- Open source
- Comprehensive functionality
- Python bindings

## Visualization

### Plotly (v5.18+)
**Purpose**: Interactive charts

**Why Plotly?**
- Interactive by default
- Native Streamlit integration
- Beautiful out-of-the-box styling
- Export capabilities

**Chart Types Used:**
- Heatmaps (model × split performance)
- Scatter plots (actual vs predicted)
- Box/violin plots (distributions)
- Bar charts (rankings)
- Radar charts (multi-metric comparison)

## Hyperparameter Optimization

### Optuna (v3.4+)
**Purpose**: Automated hyperparameter tuning

**Why Optuna?**
- Efficient Bayesian optimization
- Pruning for early stopping
- Native sklearn/PyTorch integration
- Visualization tools

## Testing

### Pytest (v7.4+)
**Purpose**: Test framework

**Why Pytest?**
- Fixture system for test setup
- Parametrized tests
- Rich plugin ecosystem
- Clear failure messages

### Pytest-cov (v4.1+)
**Purpose**: Code coverage

## Code Quality

### Ruff (v0.1+)
**Purpose**: Linting and formatting

**Why Ruff?**
- Extremely fast (Rust-based)
- Replaces flake8, isort, pyupgrade
- Easy configuration
- Auto-fix support

## Deployment

### Docker
**Purpose**: Container deployment

**Base Image**: `python:3.10-slim`

**Why Docker?**
- Reproducible environments
- Required for HuggingFace Spaces
- Easy local testing

### HuggingFace Spaces
**Purpose**: Free cloud hosting

**Why HuggingFace?**
- Free tier available
- Streamlit support
- Secrets management
- GitHub integration

## Data Sources

### GDSC (cancerrxgene.org)
**Datasets:**
- GDSC1: 320 drugs, 987 cell lines
- GDSC2: 175 drugs, 809 cell lines

**Data Types:**
- Drug response (IC50)
- Gene expression (RNA-seq)
- Mutation data
- Drug SMILES structures

## File Formats

| Format | Usage |
|--------|-------|
| `.xlsx` | GDSC raw data |
| `.csv` | Processed data, results |
| `.parquet` | Cached features (optional) |
| `.pt` | PyTorch model checkpoints |
| `.joblib` | Sklearn model saves |
| `.json` | Configuration files |

## Architecture Decisions

### 1. Modular Model Interface
All models implement `BaseModel` abstract class:
- Consistent API across models
- Easy to add new models
- Swappable for benchmarking

### 2. Caching Strategy
- Download cache: Avoid re-downloading GDSC data
- Preprocessing cache: Skip repeated transformations
- Streamlit cache: Fast dashboard reloads

### 3. Demo Data Fallback
Each Streamlit page can run with generated demo data:
- Works without GDSC download
- Fast initial load times
- Consistent demo experience

### 4. Optional Dependencies
GNN features gracefully degrade:
```python
try:
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
```

## Version Compatibility

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 |
| Streamlit | 1.28 | 1.30+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.8 | 12.0+ (optional) |

## Security Considerations

1. **No hardcoded tokens**: HuggingFace token in Spaces Secrets only
2. **Input validation**: Sanitize user inputs in Streamlit
3. **Dependency pinning**: Lock versions in requirements.txt
4. **No sensitive data**: GDSC is public data
