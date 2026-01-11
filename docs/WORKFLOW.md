# PharmacoBench Development Workflow

## Project Structure

```
PharmacoBench/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── pages/             # Multi-page app
│   ├── components/        # Reusable UI components
│   └── styles/            # CSS styling
├── data/                  # Data pipeline
│   ├── downloader.py      # GDSC data fetching
│   ├── preprocessor.py    # Data cleaning
│   ├── feature_engineer.py # Feature extraction
│   ├── splitters.py       # Split strategies
│   └── cache/             # Local cache (gitignored)
├── models/                # ML models
│   ├── base_model.py      # Abstract interface
│   ├── traditional/       # sklearn-based
│   └── deep_learning/     # PyTorch-based
├── evaluation/            # Benchmarking
│   ├── metrics.py         # RMSE, MAE, etc.
│   └── benchmark_runner.py # Orchestration
├── tests/                 # Test suite
├── docs/                  # Documentation
└── .github/workflows/     # CI/CD
```

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA for GPU training

### Local Setup

```bash
# Clone repository
git clone https://github.com/legomaheggoz-source/PharmacoBench.git
cd PharmacoBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov ruff

# Run tests
pytest tests/ -v

# Run linting
ruff check .

# Start development server
streamlit run app/main.py
```

## Git Workflow

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructure
- `test`: Test additions
- `chore`: Maintenance

Examples:
```
feat(models): add LightGBM model implementation
fix(splitters): correct disjoint split edge case
docs(readme): update installation instructions
```

### Pull Request Process

1. Create feature branch from `develop`
2. Implement changes with tests
3. Run linting and tests locally
4. Push and create PR
5. Wait for CI checks
6. Request review
7. Merge after approval

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_models/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Structure

```python
# tests/test_models/test_ridge.py

import pytest
from models.traditional.ridge import RidgeModel

class TestRidgeModel:
    def test_initialization(self):
        model = RidgeModel(alpha=1.0)
        assert model.alpha == 1.0

    def test_fit_predict(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = RidgeModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
```

### Fixtures

Common fixtures in `conftest.py`:
- `sample_gdsc_data`: Mock GDSC dataframe
- `sample_features`: Feature matrix
- `sample_targets`: Target values
- `temp_cache_dir`: Temporary directory

## Code Quality

### Linting with Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

### Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]
```

## CI/CD Pipeline

### On Pull Request

1. **Lint**: Run ruff check
2. **Test**: Run pytest
3. **Build**: Verify Docker build

### On Merge to Main

1. All PR checks
2. Deploy to HuggingFace Spaces

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install ruff
      - run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v --cov=.
```

## Adding New Models

### 1. Create Model File

```python
# models/traditional/new_model.py

from models.base_model import BaseModel
import numpy as np

class NewModel(BaseModel):
    def __init__(self, param1=1.0):
        self.param1 = param1
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        pass

    def predict(self, X):
        return self.model.predict(X)

    def get_hyperparameters(self):
        return {"param1": self.param1}

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
```

### 2. Add Tests

```python
# tests/test_models/test_new_model.py

class TestNewModel:
    def test_fit_predict(self, sample_train_test_data):
        model = NewModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
```

### 3. Register in Benchmark Runner

```python
# evaluation/benchmark_runner.py

from models.traditional.new_model import NewModel

# Add to default models
DEFAULT_MODELS = {
    ...,
    "new_model": NewModel(),
}
```

### 4. Update Documentation

- Add to SOLUTION.md model list
- Update README model table

## Deployment

### Local Docker

```bash
# Build image
docker build -t pharmacobench .

# Run container
docker run -p 7860:7860 pharmacobench
```

### HuggingFace Spaces

1. Push to GitHub main branch
2. GitHub Action triggers deploy
3. Spaces pulls and rebuilds

### Secrets Management

Never commit secrets! Use:
- HuggingFace Spaces: Settings → Secrets
- GitHub: Settings → Secrets → Actions

Required secrets:
- `HF_TOKEN`: HuggingFace write token

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the right directory
cd PharmacoBench
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**CUDA issues:**
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""
```

**Memory issues:**
```python
# Reduce batch size
model = MLPModel(batch_size=16)
```

### Getting Help

1. Check existing GitHub issues
2. Search documentation
3. Open new issue with:
   - Python version
   - OS
   - Error message
   - Minimal reproduction

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create PR to main
4. After merge, create GitHub Release
5. Tag triggers deploy to HuggingFace
