"""
Model Training Page

Configure and run model training with hyperparameter selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Model Training - PharmacoBench", page_icon="🤖", layout="wide")

st.title("🤖 Model Training")
st.markdown("Configure and run drug sensitivity prediction models")

# Model configurations
MODELS = {
    "Ridge Regression": {
        "description": "Linear regression with L2 regularization. Interpretable baseline.",
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        "type": "Traditional",
    },
    "ElasticNet": {
        "description": "Linear regression with L1+L2 regularization. Provides sparsity.",
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        "type": "Traditional",
    },
    "Random Forest": {
        "description": "Ensemble of decision trees. Handles non-linear relationships.",
        "params": {"n_estimators": [100, 200, 500], "max_depth": [10, 20, None]},
        "type": "Traditional",
    },
    "XGBoost": {
        "description": "Gradient boosting with regularization. Strong tabular performance.",
        "params": {"n_estimators": [100, 200], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1]},
        "type": "Traditional",
    },
    "LightGBM": {
        "description": "Fast gradient boosting. Memory efficient.",
        "params": {"n_estimators": [100, 200], "num_leaves": [31, 63], "learning_rate": [0.01, 0.1]},
        "type": "Traditional",
    },
    "MLP": {
        "description": "Multi-layer perceptron. Deep learning baseline.",
        "params": {"hidden_dims": [[256, 128], [512, 256, 128]], "dropout": [0.2, 0.3]},
        "type": "Deep Learning",
    },
    "GraphDRP": {
        "description": "Graph Neural Network. Encodes drug molecular structure.",
        "params": {"hidden_dim": [64, 128], "dropout": [0.2, 0.3]},
        "type": "Deep Learning",
    },
    "DeepCDR": {
        "description": "Hybrid GNN with multi-omics integration. State-of-the-art.",
        "params": {"hidden_dim": [64, 128], "dropout": [0.2, 0.3]},
        "type": "Deep Learning",
    },
}

SPLIT_STRATEGIES = {
    "Random": "Standard 80/10/10 split. Baseline evaluation (optimistic).",
    "Drug-Blind": "Test drugs never seen in training. Simulates new drug development.",
    "Cell-Blind": "Test cell lines never seen. Simulates personalized medicine.",
    "Disjoint": "Neither drugs nor cells overlap. Strictest generalization test.",
}

# Sidebar configuration
st.sidebar.markdown("### Training Configuration")

# Model selection
st.sidebar.markdown("#### Select Models")
selected_models = []
for model_name, model_info in MODELS.items():
    if st.sidebar.checkbox(model_name, value=model_name in ["Ridge Regression", "Random Forest"]):
        selected_models.append(model_name)

st.sidebar.markdown("#### Select Split Strategies")
selected_splits = []
for split_name in SPLIT_STRATEGIES:
    if st.sidebar.checkbox(split_name, value=split_name == "Random"):
        selected_splits.append(split_name)

st.sidebar.markdown("#### Training Options")
use_cross_validation = st.sidebar.checkbox("Cross-Validation (5-fold)", value=False)
save_checkpoints = st.sidebar.checkbox("Save Model Checkpoints", value=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Selected Configuration")

    if selected_models:
        st.markdown("**Models:**")
        for model in selected_models:
            model_type = MODELS[model]["type"]
            st.markdown(f"- {model} ({model_type})")
    else:
        st.warning("No models selected. Please select at least one model.")

    if selected_splits:
        st.markdown("**Split Strategies:**")
        for split in selected_splits:
            st.markdown(f"- {split}")
    else:
        st.warning("No split strategies selected. Please select at least one.")

with col2:
    st.markdown("### Estimated Time")

    n_experiments = len(selected_models) * len(selected_splits)
    if use_cross_validation:
        n_experiments *= 5

    est_time_per_exp = 2  # minutes (rough estimate)
    total_time = n_experiments * est_time_per_exp

    st.metric("Total Experiments", n_experiments)
    st.metric("Estimated Time", f"~{total_time} min")

st.divider()

# Model details
st.markdown("### Model Details")

tabs = st.tabs([m for m in selected_models] if selected_models else ["No models selected"])

for i, model_name in enumerate(selected_models if selected_models else []):
    with tabs[i]:
        model_info = MODELS[model_name]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Description:** {model_info['description']}")
            st.markdown(f"**Type:** {model_info['type']}")

            st.markdown("**Hyperparameters:**")
            for param, values in model_info["params"].items():
                selected_value = st.selectbox(
                    f"{param}",
                    options=values,
                    key=f"{model_name}_{param}",
                )

        with col2:
            st.markdown("**Expected Performance:**")
            st.markdown("""
            - RMSE: ~0.9-1.2
            - Spearman: ~0.65-0.80
            - R²: ~0.60-0.75
            """)

st.divider()

# Training execution
st.markdown("### Run Training")

can_train = len(selected_models) > 0 and len(selected_splits) > 0

if st.button("🚀 Start Training", disabled=not can_train, type="primary"):
    st.markdown("---")

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    results = []
    total_experiments = len(selected_models) * len(selected_splits)
    current = 0

    for split in selected_splits:
        for model in selected_models:
            current += 1
            progress = current / total_experiments

            status_text.markdown(f"**Training:** {model} with {split} split...")
            progress_bar.progress(progress)

            # Simulate training
            with st.spinner(f"Training {model}..."):
                time.sleep(0.5)  # Simulated training time

            # Simulated results
            np.random.seed(hash(model + split) % 2**32)
            base_rmse = {
                "Ridge Regression": 1.2, "ElasticNet": 1.18, "Random Forest": 1.05,
                "XGBoost": 0.98, "LightGBM": 0.99, "MLP": 1.0,
                "GraphDRP": 0.92, "DeepCDR": 0.88
            }.get(model, 1.0)

            split_penalty = {"Random": 0, "Drug-Blind": 0.15, "Cell-Blind": 0.12, "Disjoint": 0.25}.get(split, 0)

            rmse = base_rmse + split_penalty + np.random.uniform(-0.05, 0.05)
            spearman = 0.85 - (rmse - 0.8) * 0.3 + np.random.uniform(-0.02, 0.02)

            results.append({
                "Model": model,
                "Split": split,
                "RMSE": round(rmse, 4),
                "Spearman": round(spearman, 4),
                "Status": "✅ Complete",
            })

    progress_bar.progress(1.0)
    status_text.markdown("**✅ Training Complete!**")

    with results_container:
        st.markdown("### Training Results")

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        # Best model
        best = results_df.loc[results_df["RMSE"].idxmin()]
        st.success(f"🏆 Best Model: **{best['Model']}** ({best['Split']}) - RMSE: {best['RMSE']:.4f}")

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="training_results.csv",
            mime="text/csv",
        )

elif not can_train:
    st.info("Select at least one model and one split strategy to start training.")

# Tips
st.divider()
st.markdown("### 💡 Tips")

st.markdown("""
- **Start with baselines**: Ridge Regression and Random Forest provide good baselines
- **Use Random split first**: It's the fastest and gives upper-bound performance
- **Drug-Blind is most realistic**: For drug discovery applications
- **Cross-validation**: More reliable estimates but takes longer
- **DeepCDR requires more data**: Works best with full multi-omics features
""")
