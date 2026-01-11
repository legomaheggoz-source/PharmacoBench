# PharmacoBench Product Guide

## Overview

PharmacoBench is a comprehensive benchmarking platform for drug sensitivity prediction, designed to evaluate and compare machine learning models on the GDSC (Genomics of Drug Sensitivity in Cancer) dataset.

## Target Users

- **Computational Biologists**: Researchers developing new drug response prediction methods
- **Pharmaceutical Scientists**: Teams evaluating ML approaches for drug discovery
- **ML Engineers**: Practitioners benchmarking models on biomedical data
- **Students/Educators**: Learning about drug sensitivity prediction

## Key Features

### 1. Data Management
- Automatic download of GDSC1 and GDSC2 datasets
- Preprocessing pipeline with log transformation and normalization
- Caching for faster subsequent runs

### 2. Model Library
Eight pre-implemented models covering traditional ML and deep learning:

| Model | Type | Best For |
|-------|------|----------|
| Ridge Regression | Linear | Interpretable baseline |
| ElasticNet | Linear | Feature selection |
| Random Forest | Ensemble | Non-linear patterns |
| XGBoost | Boosting | Tabular data |
| LightGBM | Boosting | Large datasets |
| MLP | Neural Network | Deep learning baseline |
| GraphDRP | GNN | Drug structure encoding |
| DeepCDR | Hybrid GNN | Multi-omics integration |

### 3. Evaluation Strategies
Four split strategies for rigorous model assessment:

- **Random Split**: Standard 80/10/10 baseline
- **Drug-Blind**: Test on unseen drugs (new drug development)
- **Cell-Blind**: Test on unseen cell lines (personalized medicine)
- **Disjoint**: Neither drugs nor cells overlap (strictest test)

### 4. Interactive Dashboard
Six pages for comprehensive analysis:

1. **Dashboard**: Overview of benchmark results with KPIs
2. **Data Explorer**: Dataset statistics and distributions
3. **Model Training**: Configure and run training
4. **Benchmark Results**: Compare model performance
5. **Drug Analysis**: Individual drug profiles
6. **Cell Line Analysis**: Cell line characteristics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/legomaheggoz-source/PharmacoBench.git
cd PharmacoBench

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/main.py
```

### Running Benchmarks

```python
from data.downloader import GDSCDownloader
from data.preprocessor import DataPreprocessor
from data.splitters import DataSplitter
from models.traditional.ridge import RidgeModel
from evaluation.benchmark_runner import BenchmarkRunner

# Download and preprocess data
downloader = GDSCDownloader()
data = downloader.download_all()

preprocessor = DataPreprocessor()
processed = preprocessor.preprocess(data)

# Create benchmark runner
runner = BenchmarkRunner()
runner.add_model("ridge", RidgeModel(alpha=1.0))

# Run benchmarks
results = runner.run(
    df=processed,
    X_features=features,
    y_target=processed["LN_IC50"].values,
    strategies=["random", "drug_blind"],
)

print(results.to_dataframe())
```

## Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| RMSE | Root Mean Squared Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| R² | Coefficient of Determination | Higher is better (max 1) |
| Pearson | Linear correlation | Higher is better |
| Spearman | Rank correlation | Higher is better |

## Expected Performance

Based on published benchmarks:

| Model | Random RMSE | Drug-Blind RMSE |
|-------|-------------|-----------------|
| Ridge | ~1.2 | ~1.35 |
| XGBoost | ~0.98 | ~1.13 |
| DeepCDR | ~0.88 | ~1.10 |

## Troubleshooting

### Common Issues

**Data download fails:**
- Check internet connection
- Verify GDSC URLs are accessible
- Use cached data if available

**Out of memory:**
- Reduce batch size for deep learning models
- Use fewer cross-validation folds
- Subset the data for initial testing

**PyTorch Geometric not found:**
- GNN models (GraphDRP, DeepCDR) require PyTorch Geometric
- Install with: `pip install torch-geometric`

## Contributing

We welcome contributions! Please see our GitHub repository for guidelines.

## License

MIT License - see LICENSE file for details.
