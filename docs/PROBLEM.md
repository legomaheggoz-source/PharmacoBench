# The Drug Sensitivity Prediction Problem

## The Challenge

### The $2.6 Billion Question

Developing a new drug costs approximately **$2.6 billion** and takes **10-15 years**. Yet, roughly **90% of drugs fail** in clinical trials, often due to lack of efficacy or unexpected toxicity.

A major cause of these failures: **cell line models used in early-stage research don't translate to human patients**.

### What is Drug Sensitivity?

Drug sensitivity measures how effectively a drug kills cancer cells. It's typically quantified as:

**IC50 (Half-Maximal Inhibitory Concentration)**: The drug concentration needed to reduce cell viability by 50%.

- **Lower IC50** = More sensitive = Drug works better
- **Higher IC50** = More resistant = Drug works poorly

### The Prediction Task

Given:
- A drug (molecular structure, targets)
- A cancer cell line (genomic features, tissue type)

Predict:
- The IC50 value (or log-transformed ln(IC50))

### Why is This Hard?

1. **High Dimensionality**: Genomic features include ~35,000+ genes
2. **Complex Interactions**: Drug-gene interactions are non-linear
3. **Heterogeneity**: Cancer cells vary widely within and across tumors
4. **Data Sparsity**: Not all drug-cell line combinations are tested
5. **Generalization**: Models must predict for new drugs or cell lines

## The GDSC Dataset

The **Genomics of Drug Sensitivity in Cancer (GDSC)** project provides one of the largest public datasets for this problem.

### Dataset Statistics

| Metric | GDSC1 | GDSC2 | Combined |
|--------|-------|-------|----------|
| Drugs | 320 | 175 | ~495 |
| Cell Lines | 987 | 809 | ~1,000 |
| Total Tests | 226K | 135K | ~360K |

### Available Features

**Cell Line Features:**
- Gene expression (RNA-seq): ~17,000 genes
- Mutations: Binary (mutated/wild-type)
- Copy number variations
- DNA methylation

**Drug Features:**
- SMILES strings (molecular structure)
- Drug targets
- Mechanism of action
- Molecular descriptors

### Data Challenges

1. **Missing Values**: Not all combinations tested
2. **Batch Effects**: GDSC1 and GDSC2 have different protocols
3. **IC50 Distribution**: Highly skewed, requires log transformation
4. **Outliers**: Some extreme values from experimental error

## Evaluation Challenges

### The Generalization Problem

Simple random splits produce **overly optimistic** results because:
- Training and test sets share drugs and cell lines
- Models memorize specific drug-cell relationships
- Real-world use involves NEW drugs or NEW patients

### Split Strategies

| Strategy | Description | Realistic For |
|----------|-------------|---------------|
| Random | Standard 80/10/10 | Baseline (optimistic) |
| Drug-Blind | Test drugs unseen | New drug development |
| Cell-Blind | Test cells unseen | Personalized medicine |
| Disjoint | Neither overlaps | Strictest generalization |

### Performance Gap

Published benchmarks show significant performance drops:

| Model | Random RMSE | Drug-Blind RMSE | Gap |
|-------|-------------|-----------------|-----|
| Ridge | 1.20 | 1.35 | +12% |
| XGBoost | 0.98 | 1.13 | +15% |
| DeepCDR | 0.88 | 1.08 | +23% |

Deep learning models often show **larger gaps**, suggesting overfitting.

## Clinical Relevance

### Drug Development Pipeline

1. **Target Discovery**: Identify cancer vulnerabilities
2. **Lead Optimization**: Design drug candidates
3. **Preclinical Testing**: Cell line and animal studies
4. **Clinical Trials**: Human testing (Phase I-III)

**Where ML Fits**: Stages 2-3, predicting which drugs will work before expensive trials.

### Personalized Medicine

For existing drugs, predict:
- Which patients will respond to which drugs
- Optimal drug combinations
- Resistance mechanisms

## State of the Art

### Current Best Models

| Model | Key Innovation | RMSE (Random) |
|-------|----------------|---------------|
| DeepCDR | Multi-omics + GNN | ~0.88 |
| GraphDRP | Drug graph encoding | ~0.92 |
| tCNNS | Drug + Cell CNN | ~0.95 |

### Open Challenges

1. **Interpretability**: Why does a model predict sensitivity?
2. **Uncertainty**: How confident is the prediction?
3. **Generalization**: Performance on truly novel drugs
4. **Multi-task**: Predicting multiple endpoints
5. **Real-world Translation**: From cell lines to patients

## Why PharmacoBench?

### The Gap

- Many papers report only random-split results
- Comparison across papers is difficult
- Code is often not reproducible
- No standardized evaluation framework

### Our Solution

PharmacoBench provides:
1. **Unified Interface**: Same API for all models
2. **Rigorous Evaluation**: All 4 split strategies
3. **Reproducibility**: Documented preprocessing
4. **Accessibility**: Free, web-deployed dashboard

## Further Reading

1. Yang et al., "Genomics of Drug Sensitivity in Cancer (GDSC)" (2013)
2. Liu et al., "DeepCDR: Drug-response prediction" (2020)
3. Nguyen et al., "GraphDRP: Graph drug-response prediction" (2021)
4. Sharifi-Noghabi et al., "Drug sensitivity prediction with MOLI" (2019)

## References

- GDSC Database: https://www.cancerrxgene.org/
- DepMap Portal: https://depmap.org/portal/
- DrugBank: https://www.drugbank.com/
