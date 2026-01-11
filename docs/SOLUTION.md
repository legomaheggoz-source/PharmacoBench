# PharmacoBench Solution Architecture

## Model Selection Rationale

### Overview

PharmacoBench includes 8 models spanning three categories:
1. **Linear Models**: Interpretable baselines
2. **Ensemble Methods**: Strong tabular performers
3. **Deep Learning**: State-of-the-art architectures

### Selection Criteria

1. **Performance**: Published benchmarks on GDSC
2. **Interpretability**: Ability to explain predictions
3. **Scalability**: Handles 300K+ samples
4. **Reproducibility**: Clear algorithm, available code
5. **Free/Open Source**: No licensing costs

## Model Details

### 1. Ridge Regression

**Type**: Linear, L2 regularization

**Purpose**: Interpretable baseline

**Why Include?**
- Fastest to train
- Highly interpretable (linear coefficients)
- Provides "floor" performance
- Surprisingly competitive on some tasks

**Hyperparameters:**
- `alpha`: Regularization strength (0.001 - 100)

**Expected RMSE**: ~1.2 (random split)

### 2. ElasticNet

**Type**: Linear, L1+L2 regularization

**Purpose**: Feature selection baseline

**Why Include?**
- Sparse solutions (L1 component)
- Identifies important genes
- Better than Ridge when many features are irrelevant

**Hyperparameters:**
- `alpha`: Overall regularization (0.001 - 1)
- `l1_ratio`: Balance L1/L2 (0.1 - 0.9)

**Expected RMSE**: ~1.18 (random split)

### 3. Random Forest

**Type**: Ensemble of decision trees

**Purpose**: Non-linear baseline

**Why Include?**
- Handles non-linear relationships
- Built-in feature importance
- Robust to hyperparameters
- No feature scaling needed

**Hyperparameters:**
- `n_estimators`: Number of trees (100 - 500)
- `max_depth`: Tree depth (10, 20, None)

**Expected RMSE**: ~1.05 (random split)

### 4. XGBoost

**Type**: Gradient boosted trees

**Purpose**: Strong tabular performance

**Why Include?**
- Often wins tabular benchmarks
- Efficient implementation
- Built-in regularization
- Handles missing values

**Hyperparameters:**
- `n_estimators`: Boosting rounds (100 - 300)
- `max_depth`: Tree depth (3 - 9)
- `learning_rate`: Step size (0.01 - 0.3)
- `early_stopping_rounds`: Prevent overfitting

**Expected RMSE**: ~0.98 (random split)

### 5. LightGBM

**Type**: Histogram-based gradient boosting

**Purpose**: Scalable boosting

**Why Include?**
- Faster than XGBoost
- Lower memory usage
- Handles categorical features
- Competitive performance

**Hyperparameters:**
- `n_estimators`: Boosting rounds (100 - 300)
- `num_leaves`: Leaf nodes (31 - 127)
- `learning_rate`: Step size (0.01 - 0.3)

**Expected RMSE**: ~0.99 (random split)

### 6. MLP (Multi-Layer Perceptron)

**Type**: Fully-connected neural network

**Purpose**: Deep learning baseline

**Why Include?**
- Simple neural network baseline
- Can learn complex patterns
- Foundation for deeper architectures
- GPU acceleration

**Architecture:**
```
Input → Dense(256) → ReLU → Dropout
      → Dense(128) → ReLU → Dropout
      → Dense(1)
```

**Hyperparameters:**
- `hidden_dims`: Layer sizes ([256, 128])
- `dropout`: Regularization (0.2 - 0.3)
- `learning_rate`: Optimizer step (0.001)
- `batch_size`: Training batch (32 - 128)
- `epochs`: Training iterations (50 - 200)

**Expected RMSE**: ~1.0 (random split)

### 7. GraphDRP

**Type**: Graph Neural Network

**Purpose**: Drug structure encoding

**Why Include?**
- Encodes molecular topology
- Learns from SMILES structure
- Better drug generalization
- Published on GDSC

**Architecture:**
```
Drug (SMILES) → Molecular Graph → GCN Layers → Drug Embedding
Cell Line     → Gene Expression  → CNN Layers  → Cell Embedding
                                                      ↓
                             Concatenate → Dense → IC50
```

**Key Innovation:**
- GCN (Graph Convolutional Network) for drugs
- Captures atom connectivity and bond types

**Hyperparameters:**
- `hidden_dim`: Embedding size (64 - 128)
- `n_layers`: GCN depth (2 - 4)
- `dropout`: Regularization (0.2 - 0.3)

**Expected RMSE**: ~0.92 (random split)

### 8. DeepCDR

**Type**: Hybrid GNN + Multi-omics

**Purpose**: State-of-the-art performance

**Why Include?**
- Top performer on GDSC
- Integrates multiple data types
- Comprehensive feature usage
- Published benchmark

**Architecture:**
```
Drug → Graph Attention Network → Drug Embedding
Mutations → Dense Network → Mutation Embedding
Expression → 1D CNN → Expression Embedding
Methylation → Dense → Methylation Embedding
                              ↓
                  Concatenate All → Dense Layers → IC50
```

**Key Innovation:**
- GAT (Graph Attention) for drugs
- Multi-omics integration (mutations + expression + methylation)

**Hyperparameters:**
- `hidden_dim`: Embedding size (64 - 128)
- `n_heads`: Attention heads (4 - 8)
- `dropout`: Regularization (0.2 - 0.3)

**Expected RMSE**: ~0.88 (random split)

## Evaluation Framework

### Split Strategies

| Strategy | Implementation |
|----------|----------------|
| Random | sklearn `train_test_split` |
| Drug-Blind | Group by `DRUG_NAME`, split groups |
| Cell-Blind | Group by `CELL_LINE_NAME`, split groups |
| Disjoint | Intersect drug-blind and cell-blind |

### Metrics

All models evaluated on:
1. **RMSE**: Primary metric (lower is better)
2. **MAE**: Alternative error metric
3. **R²**: Variance explained
4. **Pearson**: Linear correlation
5. **Spearman**: Rank correlation

### Cross-Validation

Optional 5-fold CV for robust estimates:
- Each fold maintains split strategy constraints
- Reports mean ± std across folds

## Preprocessing Pipeline

### IC50 Transformation
```python
# Log transform (required for GDSC data)
df["LN_IC50"] = np.log(df["IC50"])
```

### Cell Line Features
```python
# Z-score normalization for expression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
expression_normalized = scaler.fit_transform(expression)
```

### Drug Features
```python
# SMILES to molecular fingerprint
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

### Drug Graphs (for GNN models)
```python
# SMILES to PyTorch Geometric graph
from data.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
graph = engineer.smiles_to_graph(smiles)
# Returns: Data(x=node_features, edge_index=edges, edge_attr=bond_types)
```

## Training Recommendations

### For Quick Experimentation
1. Start with Ridge (fastest, good baseline)
2. Use random split only
3. Subset data to 10K samples

### For Rigorous Benchmarking
1. Run all 8 models
2. Use all 4 split strategies
3. Enable 5-fold CV
4. Use full dataset

### For Production
1. Tune hyperparameters with Optuna
2. Use disjoint split for final evaluation
3. Consider ensemble of top models
4. Quantify uncertainty (dropout at test time)

## Known Limitations

1. **Data Leakage Risk**: Some drugs appear in both GDSC1 and GDSC2
2. **Cell Line Bias**: More blood cancer lines than others
3. **Drug Target Overlap**: Many drugs target same pathways
4. **Feature Completeness**: Not all cell lines have all omics data

## Future Extensions

1. **Additional Models**: Transformer-based, attention mechanisms
2. **Transfer Learning**: Pre-train on larger datasets
3. **Uncertainty Quantification**: Bayesian neural networks
4. **Multi-task Learning**: Predict multiple endpoints
5. **Explainability**: SHAP values, attention visualization
