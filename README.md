# PharmacoBench

**Comparative Auditor for In-Silico Drug Sensitivity Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

## Overview

PharmacoBench is a comprehensive benchmarking platform for evaluating machine learning architectures on drug sensitivity prediction. It addresses a critical challenge in drug discovery: **90% of drugs fail in clinical trials**, costing ~$2.6B per successful drug.

### Key Features

- **8 ML Models**: From Ridge Regression baseline to state-of-the-art Graph Neural Networks
- **4 Evaluation Strategies**: Random, drug-blind, cell-blind, and disjoint splits
- **GDSC Dataset**: 495 drugs tested across 1,000+ cancer cell lines
- **Interactive Dashboard**: Aurora Solar-inspired design with Plotly visualizations
- **Reproducible**: Unified interfaces, documented preprocessing, cached results

## Models Benchmarked

| Model | Type | Purpose |
|-------|------|---------|
| Ridge Regression | Linear | Interpretability baseline |
| ElasticNet | Linear | Sparsity + regularization |
| Random Forest | Ensemble | Non-linear baseline |
| XGBoost | Gradient Boosting | Strong tabular performance |
| LightGBM | Gradient Boosting | Fast training |
| MLP | Deep Learning | Neural network baseline |
| GraphDRP | Graph Neural Network | Drug structure encoding |
| DeepCDR | Hybrid GNN | State-of-the-art multi-omics |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/legomaheggoz-source/PharmacoBench.git
cd PharmacoBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app/main.py
```

### Docker

```bash
docker build -t pharmacobench .
docker run -p 7860:7860 pharmacobench
```

## Live Demo

Visit the live demo on HuggingFace Spaces:
**[https://huggingface.co/spaces/legomaheggo/PharmacoBench](https://huggingface.co/spaces/legomaheggo/PharmacoBench)**

## Data

PharmacoBench uses the **Genomics of Drug Sensitivity in Cancer (GDSC)** dataset:

- **GDSC1**: 320 drugs, 987 cell lines
- **GDSC2**: 175 drugs, 809 cell lines
- **Features**: Gene expression profiles, mutation data, SMILES molecular structures
- **Target**: IC50 (Half-Maximal Inhibitory Concentration)

Data is automatically downloaded from [cancerrxgene.org](https://www.cancerrxgene.org/) on first run.

## Evaluation Strategies

| Strategy | Description | Clinical Relevance |
|----------|-------------|-------------------|
| Random | Standard 80/10/10 split | Baseline (optimistic) |
| Drug-Blind | Test drugs never seen in training | New drug development |
| Cell-Blind | Test cell lines never seen | Personalized medicine |
| Disjoint | Neither drugs nor cells overlap | Strictest generalization |

## Project Structure

```
PharmacoBench/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── pages/             # Dashboard pages
│   └── components/        # UI components
├── data/                   # Data pipeline
│   ├── downloader.py      # GDSC data fetching
│   ├── preprocessor.py    # Data cleaning
│   └── splitters.py       # Split strategies
├── models/                 # ML models
│   ├── traditional/       # sklearn-based
│   └── deep_learning/     # PyTorch-based
├── evaluation/             # Benchmarking
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Documentation

- [Product Guide](docs/PRODUCT.md) - User guide and features
- [Technical Stack](docs/STACK.md) - Architecture decisions
- [Problem Statement](docs/PROBLEM.md) - Drug sensitivity prediction
- [Solution Design](docs/SOLUTION.md) - Model selection rationale
- [API Reference](docs/API.md) - Module interfaces

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use PharmacoBench in your research, please cite:

```bibtex
@software{pharmacobench2026,
  title = {PharmacoBench: Comparative Auditor for In-Silico Drug Sensitivity Prediction},
  year = {2026},
  url = {https://github.com/legomaheggoz-source/PharmacoBench}
}
```

## Acknowledgments

- GDSC Dataset: [Genomics of Drug Sensitivity in Cancer](https://www.cancerrxgene.org/)
- GraphDRP: [Graph Convolutional Networks for Drug Response Prediction](https://github.com/hauldhut/GraphDRP)
- DeepCDR: [Deep Learning for Cancer Drug Response](https://github.com/kimmo1019/DeepCDR)
