# PharmacoBench User Guide

## Welcome to PharmacoBench

PharmacoBench is an interactive web application for benchmarking machine learning models on drug sensitivity prediction. This guide will walk you through all features of the live dashboard.

**Live Demo:** [https://huggingface.co/spaces/legomaheggo/PharmacoBench](https://huggingface.co/spaces/legomaheggo/PharmacoBench)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Page-by-Page Guide](#page-by-page-guide)
4. [Understanding the Data](#understanding-the-data)
5. [Understanding the Models](#understanding-the-models)
6. [Understanding Evaluation Strategies](#understanding-evaluation-strategies)
7. [Interpreting Results](#interpreting-results)
8. [Using Custom Data](#using-custom-data)
9. [Making Predictions](#making-predictions)
10. [FAQ](#faq)

---

## Getting Started

### Accessing the Dashboard

**Option 1: Live Demo (Recommended)**
Visit the HuggingFace Spaces deployment:
- [https://huggingface.co/spaces/legomaheggo/PharmacoBench](https://huggingface.co/spaces/legomaheggo/PharmacoBench)

**Option 2: Run Locally**
```bash
git clone https://github.com/legomaheggoz-source/PharmacoBench.git
cd PharmacoBench
pip install -r requirements.txt
streamlit run app/main.py
```

### Navigation

The sidebar on the left contains:
- **Quick Stats**: Number of models and evaluation strategies available
- **Resources**: Links to documentation and data sources
- **Page Navigation**: Click any page name to navigate

---

## Dashboard Overview

The main dashboard (Home page) provides:

- **Hero Section**: Project introduction and purpose
- **Feature Cards**: Key statistics (8 models, 4 splits, 1000+ cell lines)
- **Models Table**: Complete list of available ML models with their types and purposes
- **Getting Started Guide**: Quick links to main features

---

## Page-by-Page Guide

### 1. Dashboard (Home)

**Purpose:** Overview of the platform and its capabilities.

**Features:**
- Project introduction
- Summary statistics
- Model listing with descriptions
- Navigation hints to other pages

---

### 2. Data Explorer

**Purpose:** Explore the GDSC (Genomics of Drug Sensitivity in Cancer) dataset.

**Features:**

**Overview Tab:**
- Total records, unique drugs, cell lines, and tissue types
- Data source information
- Quick statistics

**Distribution Tab:**
- IC50 distribution histogram (log-transformed)
- IC50 by tissue type (violin plot)
- Helps understand drug sensitivity ranges

**Filtering:**
- Filter by data source (GDSC1, GDSC2)
- Filter by tissue type
- Filter by IC50 range
- All visualizations update dynamically

**How to Use:**
1. Use sidebar filters to focus on specific subsets
2. Examine distributions to understand data characteristics
3. Compare IC50 values across tissue types

---

### 3. Model Training

**Purpose:** Configure and simulate model training runs.

**Features:**

**Model Selection:**
- Choose from 8 available models
- View model descriptions and typical use cases

**Hyperparameter Configuration:**
- Model-specific hyperparameters
- Sensible defaults provided
- Tooltips explain each parameter

**Training Simulation:**
- Progress visualization
- Simulated training metrics
- Results display

**How to Use:**
1. Select a model from the dropdown
2. Adjust hyperparameters if desired
3. Click "Train Model" to see simulated training

---

### 4. Benchmark Results

**Purpose:** Compare model performance across different evaluation strategies.

**Features:**

**Performance Heatmap:**
- Models vs. split strategies
- Color-coded by RMSE (lower = better = greener)
- Quick visual comparison

**Detailed Metrics Table:**
- All 5 metrics (RMSE, MAE, R², Pearson, Spearman)
- Sortable columns
- Filter by model or strategy

**Scatter Plot:**
- Actual vs. Predicted IC50 values
- Perfect prediction reference line
- Shows model accuracy visually

**How to Use:**
1. Start with the heatmap for quick comparison
2. Select specific models to compare in detail
3. Use the scatter plot to understand prediction quality
4. Lower RMSE and higher R² indicate better models

---

### 5. Drug Analysis

**Purpose:** Explore individual drugs and their sensitivity profiles.

**Features:**

**Drug Selector:**
- Search or browse available drugs
- View drug metadata

**Sensitivity Profile:**
- IC50 distribution across cell lines
- Most/least sensitive cell lines
- Tissue-specific response patterns

**Drug Statistics:**
- Number of cell lines tested
- Mean, median, std of IC50 values
- Response range

**How to Use:**
1. Select a drug of interest
2. Examine its sensitivity distribution
3. Identify which cell lines/tissues respond best
4. Compare with other drugs

---

### 6. Cell Line Analysis

**Purpose:** Explore cell line characteristics and drug response patterns.

**Features:**

**Cell Line Selector:**
- Search or browse cell lines
- Filter by tissue type

**Response Profile:**
- Which drugs the cell line responds to
- Resistance patterns
- IC50 distribution across tested drugs

**Genomic Context:**
- Tissue of origin
- Number of drugs tested

**How to Use:**
1. Select a cell line
2. View its drug response profile
3. Identify drugs it's sensitive/resistant to
4. Compare across tissue types

---

### 7. Custom Data

**Purpose:** Upload and analyze your own drug sensitivity data.

**Features:**

**File Upload:**
- Supports CSV and Excel formats
- Automatic column detection
- Validation feedback

**Required Columns:**
- `DRUG_NAME`: Drug identifier
- `CELL_LINE_NAME`: Cell line identifier
- `IC50` or `LN_IC50`: Sensitivity value

**Optional Columns:**
- `TISSUE`: Tissue type for stratified analysis

**Analysis Tools:**
- Same visualizations as Data Explorer
- Custom statistics
- Export processed data

**How to Use:**
1. Prepare your data with required columns
2. Upload CSV or Excel file
3. Review validation results
4. Click "Load Data for Analysis"
5. Explore your data with built-in tools

---

### 8. Predict

**Purpose:** Make drug sensitivity predictions using trained models.

**Features:**

**Single Prediction:**
- Select a drug and cell line
- Choose prediction model
- Get IC50 prediction with confidence

**Batch Prediction:**
- Upload file with drug-cell line pairs
- Get predictions for all pairs
- Download results as CSV

**Prediction Output:**
- Predicted IC50 value
- Confidence interval
- Sensitivity classification (Sensitive/Intermediate/Resistant)

**How to Use:**
1. Choose single or batch mode
2. Select drug(s) and cell line(s)
3. Choose a model
4. View predictions and confidence levels

---

## Understanding the Data

### What is IC50?

**IC50** (Half-maximal Inhibitory Concentration) measures drug potency:
- The drug concentration needed to inhibit 50% of cell growth
- **Lower IC50 = More potent drug** (less drug needed for effect)
- Typically measured in micromolar (μM)

### Why Log Transform?

IC50 values span several orders of magnitude. Log transformation:
- Makes distributions more normal
- Enables better statistical analysis
- Standard practice in pharmacology

### GDSC Dataset

The Genomics of Drug Sensitivity in Cancer (GDSC) dataset:
- **GDSC1**: ~320 drugs, ~987 cell lines
- **GDSC2**: ~175 drugs, ~809 cell lines
- Combined: ~495 unique drugs, ~1000 cell lines
- Source: [cancerrxgene.org](https://www.cancerrxgene.org/)

---

## Understanding the Models

### Traditional Machine Learning

| Model | Strengths | When to Use |
|-------|-----------|-------------|
| **Ridge Regression** | Interpretable, fast, stable | Baseline, feature importance |
| **ElasticNet** | Feature selection, handles multicollinearity | High-dimensional data |
| **Random Forest** | Non-linear, robust to outliers | General purpose |
| **XGBoost** | State-of-the-art for tabular | Production models |
| **LightGBM** | Fast training, memory efficient | Large datasets |

### Deep Learning

| Model | Strengths | When to Use |
|-------|-----------|-------------|
| **MLP** | Learns complex patterns | Deep learning baseline |
| **GraphDRP** | Encodes molecular structure | Drug structure matters |
| **DeepCDR** | Multi-omics integration | Maximum performance |

---

## Understanding Evaluation Strategies

### Why Different Splits Matter

| Strategy | What It Tests | Clinical Relevance |
|----------|---------------|-------------------|
| **Random** | General performance | Optimistic baseline |
| **Drug-Blind** | New drug generalization | New drug development |
| **Cell-Blind** | New patient generalization | Personalized medicine |
| **Disjoint** | Full generalization | Strictest real-world test |

### Interpreting Performance Gaps

- **Random vs Drug-Blind gap**: How well model generalizes to new drugs
- **Random vs Cell-Blind gap**: How well model generalizes to new patients
- **Small gap** = Good generalization
- **Large gap** = Model may be overfitting

---

## Interpreting Results

### Metrics Guide

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **RMSE** | 0 to ∞ | Lower = better; typical: 0.8-1.5 |
| **MAE** | 0 to ∞ | Lower = better; less sensitive to outliers |
| **R²** | -∞ to 1 | Higher = better; 1 = perfect |
| **Pearson** | -1 to 1 | Higher = better linear correlation |
| **Spearman** | -1 to 1 | Higher = better rank correlation |

### What's "Good" Performance?

Based on published literature:
- **RMSE < 1.0**: Excellent
- **RMSE 1.0-1.2**: Good
- **RMSE 1.2-1.5**: Acceptable
- **RMSE > 1.5**: Needs improvement

- **R² > 0.85**: Excellent
- **R² 0.75-0.85**: Good
- **R² 0.60-0.75**: Moderate
- **R² < 0.60**: Limited predictive power

---

## Using Custom Data

### Data Format Requirements

Your CSV/Excel file must have:

```csv
DRUG_NAME,CELL_LINE_NAME,IC50
DrugA,CellLine1,0.5
DrugA,CellLine2,1.2
DrugB,CellLine1,0.8
```

### Optional Enhancements

Add a `TISSUE` column for tissue-specific analysis:

```csv
DRUG_NAME,CELL_LINE_NAME,IC50,TISSUE
DrugA,CellLine1,0.5,Lung
DrugA,CellLine2,1.2,Breast
```

### Tips for Success

1. Ensure no missing values in required columns
2. Use consistent naming (case-sensitive)
3. IC50 can be raw or log-transformed (auto-detected)
4. Large files (>100MB) may take time to process

---

## Making Predictions

### Single Prediction Workflow

1. Navigate to **Predict** page
2. Select **Single Prediction** tab
3. Choose a drug from the dropdown
4. Choose a cell line from the dropdown
5. Select a model (XGBoost recommended for accuracy)
6. Click **Predict**
7. View results with confidence interval

### Batch Prediction Workflow

1. Prepare a CSV with `DRUG_NAME` and `CELL_LINE_NAME` columns
2. Navigate to **Predict** page
3. Select **Batch Prediction** tab
4. Upload your file
5. Select a model
6. Click **Predict All**
7. Download results CSV

### Understanding Confidence

- **High confidence**: Narrow interval, reliable prediction
- **Low confidence**: Wide interval, treat with caution
- Confidence based on model uncertainty and data similarity

---

## FAQ

### General Questions

**Q: What is PharmacoBench for?**
A: Benchmarking and comparing machine learning models for predicting how cancer cells respond to drugs.

**Q: Who should use this?**
A: Researchers in computational biology, pharmaceutical scientists, ML engineers, and students learning about drug sensitivity prediction.

**Q: Is this for production drug discovery?**
A: PharmacoBench is a benchmarking and educational tool. Production systems require additional validation, regulatory compliance, and domain expertise.

### Technical Questions

**Q: Why do different splits give different results?**
A: Different splits test different types of generalization. Drug-blind tests new drug prediction, cell-blind tests new patient prediction.

**Q: Which model should I use?**
A: Start with XGBoost or LightGBM for tabular data. Use GraphDRP/DeepCDR if molecular structure is important and you have PyTorch Geometric installed.

**Q: Can I add my own models?**
A: Yes! Implement the `BaseModel` interface and add to the model registry. See CONTRIBUTING.md for details.

### Data Questions

**Q: Where does the data come from?**
A: The GDSC (Genomics of Drug Sensitivity in Cancer) project at the Wellcome Sanger Institute.

**Q: Can I use my own data?**
A: Yes! Use the Custom Data page to upload CSV/Excel files with drug sensitivity measurements.

**Q: How often is GDSC updated?**
A: GDSC releases periodic updates. PharmacoBench caches downloaded data locally.

---

## Getting Help

- **Documentation**: See `docs/` folder for technical documentation
- **Issues**: Report bugs at [GitHub Issues](https://github.com/legomaheggoz-source/PharmacoBench/issues)
- **Contributing**: See CONTRIBUTING.md for development guidelines

---

## Acknowledgments

- GDSC data from the Wellcome Sanger Institute
- Built with Streamlit, scikit-learn, PyTorch, and Plotly
- Inspired by the need for reproducible ML benchmarking in drug discovery

---

*PharmacoBench - Making drug sensitivity prediction transparent and reproducible.*
