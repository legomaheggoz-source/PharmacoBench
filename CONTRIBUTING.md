# Contributing to PharmacoBench

Thank you for your interest in contributing to PharmacoBench! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- pip or conda

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PharmacoBench.git
   cd PharmacoBench
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/legomaheggoz-source/PharmacoBench.git
   ```

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov ruff

# For full model training (optional)
pip install torch torch-geometric rdkit optuna
```

### Verify Installation

```bash
# Run the app locally
streamlit run app/main.py

# Run tests
pytest tests/
```

## Making Changes

### Branch Naming

Create a descriptive branch name:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

```bash
git checkout -b feature/add-new-model
```

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Co-Authored-By: Your Name <email@example.com>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(models): add support for transformer-based drug encoding
fix(data): handle missing values in IC50 preprocessing
docs(api): update model interface documentation
```

## Pull Request Process

1. Update your fork with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your changes:
   ```bash
   git push origin feature/your-feature
   ```

3. Create a Pull Request on GitHub with:
   - Clear title describing the change
   - Description of what and why
   - Link to any related issues
   - Screenshots for UI changes

4. Address reviewer feedback

5. Once approved, maintainers will merge

## Code Style

### Python

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

Key style guidelines:
- Use type hints for function signatures
- Write docstrings for all public functions
- Keep lines under 100 characters
- Use descriptive variable names

### Example

```python
def predict_sensitivity(
    drug_features: np.ndarray,
    cell_features: np.ndarray,
    model: BaseModel,
) -> np.ndarray:
    """
    Predict drug sensitivity for drug-cell pairs.

    Args:
        drug_features: Drug feature matrix (n_drugs, n_features)
        cell_features: Cell line feature matrix (n_cells, n_features)
        model: Trained prediction model

    Returns:
        Predicted IC50 values (n_pairs,)
    """
    combined = np.concatenate([drug_features, cell_features], axis=1)
    return model.predict(combined)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.

# Run specific test file
pytest tests/test_models/test_ridge.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Place tests in `tests/` directory mirroring source structure
- Use descriptive test names
- Include both positive and edge case tests

```python
def test_ridge_model_fit_and_predict():
    """Test Ridge model training and prediction."""
    model = RidgeModel(alpha=1.0)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)

    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    assert predictions.shape == (100,)
    assert np.isfinite(predictions).all()
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """
    Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: If arg1 is empty
    """
```

### Updating Documentation

- Update `docs/` files for any API changes
- Add docstrings to all new public functions
- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` on GitHub.

### Priority Areas

1. **Models**: Implement new ML models following `BaseModel` interface
2. **Visualizations**: Add new Plotly visualizations
3. **Data Loaders**: Support additional data sources
4. **Tests**: Increase test coverage
5. **Documentation**: Improve guides and examples

### Adding a New Model

1. Create file in `models/traditional/` or `models/deep_learning/`
2. Implement `BaseModel` interface
3. Register in `models/__init__.py`
4. Add tests in `tests/test_models/`
5. Update documentation

```python
from models.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        pass

    def predict(self, X_test) -> np.ndarray:
        # Prediction logic
        pass

    def get_hyperparameters(self) -> Dict:
        return {"param1": self.param1}

    def save(self, path: str):
        # Save model
        pass

    def load(self, path: str):
        # Load model
        pass
```

## Questions?

- Open an issue on GitHub
- Check existing documentation in `docs/`

Thank you for contributing to PharmacoBench!
