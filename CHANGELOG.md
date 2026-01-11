# Changelog

All notable changes to PharmacoBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

*No unreleased changes*

## [1.0.0] - 2026-01-11

### Added
- REST API with FastAPI
  - Prediction endpoints (single and batch)
  - Benchmark results endpoints with filtering
  - Models listing endpoint
  - OpenAPI/Swagger documentation
- Comprehensive User Guide (`docs/USER_GUIDE.md`)
- Project Summary with motivation and portfolio blurb (`docs/PROJECT_SUMMARY.md`)

### Fixed
- HuggingFace Spaces deployment timeout (30 min)
  - Root cause: xgboost pulls nvidia-nccl-cu12 (290MB CUDA dependency)
  - Made xgboost and lightgbm optional in requirements.txt
  - Made model imports lazy with try/except in models/__init__.py
- CORS and XSRF settings for HuggingFace compatibility
- Type hint errors in deep learning models (future annotations)

### Changed
- Streamlit config optimized for HuggingFace Spaces deployment
- Debug logging added to main.py for deployment diagnostics

## [0.2.1] - 2026-01-11

### Added
- Plan 2 features: Custom data upload, prediction interface
- CONTRIBUTING.md with development guidelines
- CHANGELOG.md to track changes

## [0.2.0] - 2026-01-11

### Added
- Reusable UI components (MetricCard, FilterPanel, ErrorBoundary, LoadingSkeleton)
- External Aurora CSS theme file (`app/styles/aurora.css`)
- Page utilities module (`app/utils/page_utils.py`)
- Loading states and error handling across pages
- Data refresh buttons with timestamps
- Accessibility features (focus states, high contrast, reduced motion)
- Custom Data page for uploading user data
- Interactive Prediction page for drug sensitivity predictions

### Changed
- Data Explorer sidebar filters now actually filter the data
- Sidebar stats show dynamic model/split counts
- Improved dashboard with refresh capability

### Fixed
- HuggingFace deployment timeout (removed heavy imports from startup)
- Data Explorer filters were capturing values but not applying them

## [0.1.0] - 2026-01-10

### Added
- Initial release of PharmacoBench
- 8 ML models: Ridge, ElasticNet, Random Forest, XGBoost, LightGBM, MLP, GraphDRP, DeepCDR
- 4 split strategies: Random, Drug-Blind, Cell-Blind, Disjoint
- 6 Streamlit dashboard pages:
  - Dashboard (overview metrics)
  - Data Explorer (GDSC dataset exploration)
  - Model Training (configure and train models)
  - Benchmark Results (compare model performance)
  - Drug Analysis (individual drug profiles)
  - Cell Line Analysis (cell line characteristics)
- GDSC data downloader connecting to cancerrxgene.org
- Data preprocessing pipeline
- Aurora Solar-inspired light theme
- Comprehensive documentation (6 docs in docs/)
- GitHub Actions CI/CD pipelines
- HuggingFace Spaces deployment

### Technical Stack
- Streamlit 1.28+
- scikit-learn, XGBoost, LightGBM
- Plotly for visualizations
- PyTorch + PyTorch Geometric (optional for deep learning models)
- RDKit (optional for molecular visualization)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.0 | 2026-01-11 | UX polish, components, custom data |
| 0.1.0 | 2026-01-10 | Initial release |

---

## Upgrade Guide

### 0.1.0 to 0.2.0

No breaking changes. Simply pull the latest code:

```bash
git pull origin main
pip install -r requirements.txt
```

New features:
- Use the new Custom Data page to upload your own datasets
- Use the Prediction page for making predictions
- Components in `app/components/` can be imported for custom pages

---

## Contributors

- Initial development by the PharmacoBench team
- Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)
