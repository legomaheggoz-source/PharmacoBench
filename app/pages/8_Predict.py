"""
Prediction Interface Page

Make drug sensitivity predictions using trained models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Predict - PharmacoBench", page_icon="🔮", layout="wide")

st.title("🔮 Drug Sensitivity Prediction")
st.markdown("Predict drug sensitivity for cell line-drug pairs")

# Model options (simulated - in production these would be loaded models)
AVAILABLE_MODELS = {
    "Ridge Regression": {"type": "Linear", "rmse": 1.20, "description": "Fast, interpretable baseline"},
    "Random Forest": {"type": "Ensemble", "rmse": 1.05, "description": "Robust non-linear model"},
    "XGBoost": {"type": "Boosting", "rmse": 0.98, "description": "Strong tabular performance"},
    "LightGBM": {"type": "Boosting", "rmse": 0.99, "description": "Fast training, efficient"},
    "DeepCDR": {"type": "Deep Learning", "rmse": 0.88, "description": "State-of-the-art GNN"},
}

# Sample drugs and cell lines for demo
SAMPLE_DRUGS = [
    "Gefitinib", "Erlotinib", "Lapatinib", "Imatinib", "Sorafenib",
    "Vemurafenib", "Crizotinib", "Dasatinib", "Nilotinib", "Bosutinib"
]

SAMPLE_CELL_LINES = [
    "A549", "MCF7", "HeLa", "PC3", "HCT116",
    "SK-MEL-28", "K562", "U2OS", "HepG2", "MDA-MB-231"
]


def simulate_prediction(drug: str, cell_line: str, model: str) -> dict:
    """Simulate a prediction (in production, this would call actual model)."""
    np.random.seed(hash(drug + cell_line + model) % 2**32)

    # Base prediction varies by model quality
    model_rmse = AVAILABLE_MODELS[model]["rmse"]
    base_pred = -2.0 + np.random.normal(0, model_rmse * 0.5)

    # Add some drug-specific variation
    drug_effect = hash(drug) % 10 / 10 - 0.5
    cell_effect = hash(cell_line) % 10 / 10 - 0.5

    prediction = base_pred + drug_effect + cell_effect

    # Simulate confidence interval
    ci_width = model_rmse * 0.3

    return {
        "prediction": round(prediction, 3),
        "ci_lower": round(prediction - ci_width, 3),
        "ci_upper": round(prediction + ci_width, 3),
        "sensitivity": "Sensitive" if prediction < -2 else ("Intermediate" if prediction < -1 else "Resistant")
    }


# Sidebar - Model selection
st.sidebar.markdown("### Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    options=list(AVAILABLE_MODELS.keys()),
    index=2,  # Default to XGBoost
    help="Select the model to use for predictions"
)

model_info = AVAILABLE_MODELS[selected_model]
st.sidebar.markdown(f"**Type:** {model_info['type']}")
st.sidebar.markdown(f"**Benchmark RMSE:** {model_info['rmse']}")
st.sidebar.caption(model_info['description'])

st.sidebar.divider()
st.sidebar.markdown("### Prediction Mode")
prediction_mode = st.sidebar.radio(
    "Mode",
    ["Single Prediction", "Batch Prediction"],
    help="Single: one drug-cell pair. Batch: multiple predictions at once."
)

# Main content
if prediction_mode == "Single Prediction":
    st.markdown("### Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Select Drug")

        drug_input_method = st.radio(
            "Input method",
            ["Select from list", "Enter SMILES"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if drug_input_method == "Select from list":
            selected_drug = st.selectbox(
                "Drug",
                options=SAMPLE_DRUGS,
                help="Select a drug from the database"
            )
        else:
            smiles_input = st.text_input(
                "SMILES String",
                placeholder="Enter SMILES (e.g., CC(=O)NC1=CC=C(O)C=C1)",
                help="Enter the molecular SMILES string"
            )
            selected_drug = smiles_input if smiles_input else "Custom Drug"

    with col2:
        st.markdown("#### Select Cell Line")

        cell_input_method = st.radio(
            "Input method",
            ["Select from list", "Enter name"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if cell_input_method == "Select from list":
            selected_cell = st.selectbox(
                "Cell Line",
                options=SAMPLE_CELL_LINES,
                help="Select a cell line from the database"
            )
        else:
            cell_input = st.text_input(
                "Cell Line Name",
                placeholder="Enter cell line name",
                help="Enter a cell line identifier"
            )
            selected_cell = cell_input if cell_input else "Custom Cell Line"

    st.divider()

    # Predict button
    if st.button("🔮 Predict Sensitivity", type="primary", use_container_width=True):
        if selected_drug and selected_cell:
            with st.spinner(f"Running {selected_model} prediction..."):
                result = simulate_prediction(selected_drug, selected_cell, selected_model)

            st.markdown("### Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Color based on sensitivity
                color = "#22c55e" if result["sensitivity"] == "Sensitive" else (
                    "#eab308" if result["sensitivity"] == "Intermediate" else "#ef4444"
                )
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}20, {color}10);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                ">
                    <div style="color: #64748b; font-size: 0.875rem; text-transform: uppercase;">
                        Predicted ln(IC50)
                    </div>
                    <div style="color: {color}; font-size: 2.5rem; font-weight: 700;">
                        {result['prediction']}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.875rem;">
                        95% CI: [{result['ci_lower']}, {result['ci_upper']}]
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Classification", result["sensitivity"])
                st.caption("Based on standard IC50 thresholds")

            with col3:
                st.metric("Model Used", selected_model)
                st.caption(f"Expected RMSE: {model_info['rmse']}")

            # Interpretation
            st.markdown("---")
            st.markdown("### Interpretation")

            if result["sensitivity"] == "Sensitive":
                st.success(f"""
                **{selected_drug}** is predicted to be **highly effective** against **{selected_cell}**.

                The predicted IC50 ({result['prediction']}) indicates the cell line is sensitive to this drug at relatively low concentrations.
                """)
            elif result["sensitivity"] == "Intermediate":
                st.warning(f"""
                **{selected_drug}** shows **moderate activity** against **{selected_cell}**.

                The predicted IC50 ({result['prediction']}) suggests partial sensitivity. Higher doses may be required for therapeutic effect.
                """)
            else:
                st.error(f"""
                **{selected_drug}** is predicted to be **ineffective** against **{selected_cell}**.

                The predicted IC50 ({result['prediction']}) indicates resistance. Alternative drugs should be considered.
                """)
        else:
            st.warning("Please select both a drug and a cell line.")

else:  # Batch Prediction
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with drug-cell line pairs or select multiple from lists.")

    batch_method = st.radio(
        "Input method",
        ["Upload CSV", "Select Multiple"],
        horizontal=True
    )

    if batch_method == "Upload CSV":
        st.markdown("""
        **CSV Format:**
        ```
        DRUG_NAME,CELL_LINE_NAME
        Gefitinib,A549
        Erlotinib,MCF7
        ...
        ```
        """)

        uploaded_file = st.file_uploader(
            "Upload pairs CSV",
            type=["csv"],
            help="CSV with DRUG_NAME and CELL_LINE_NAME columns"
        )

        if uploaded_file:
            pairs_df = pd.read_csv(uploaded_file)
            st.dataframe(pairs_df.head(), use_container_width=True)

            if st.button("🔮 Run Batch Predictions", type="primary"):
                results = []
                progress = st.progress(0)

                for i, row in pairs_df.iterrows():
                    result = simulate_prediction(row["DRUG_NAME"], row["CELL_LINE_NAME"], selected_model)
                    results.append({
                        "DRUG_NAME": row["DRUG_NAME"],
                        "CELL_LINE_NAME": row["CELL_LINE_NAME"],
                        "Predicted_LN_IC50": result["prediction"],
                        "CI_Lower": result["ci_lower"],
                        "CI_Upper": result["ci_upper"],
                        "Sensitivity": result["sensitivity"]
                    })
                    progress.progress((i + 1) / len(pairs_df))

                results_df = pd.DataFrame(results)

                st.markdown("### Batch Results")
                st.dataframe(results_df, use_container_width=True)

                # Summary
                sens_counts = results_df["Sensitivity"].value_counts()
                fig = px.pie(
                    values=sens_counts.values,
                    names=sens_counts.index,
                    title="Sensitivity Distribution",
                    color=sens_counts.index,
                    color_discrete_map={
                        "Sensitive": "#22c55e",
                        "Intermediate": "#eab308",
                        "Resistant": "#ef4444"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

    else:  # Select Multiple
        col1, col2 = st.columns(2)

        with col1:
            selected_drugs = st.multiselect(
                "Select Drugs",
                options=SAMPLE_DRUGS,
                default=SAMPLE_DRUGS[:3]
            )

        with col2:
            selected_cells = st.multiselect(
                "Select Cell Lines",
                options=SAMPLE_CELL_LINES,
                default=SAMPLE_CELL_LINES[:3]
            )

        if selected_drugs and selected_cells:
            n_pairs = len(selected_drugs) * len(selected_cells)
            st.info(f"This will generate {n_pairs} predictions ({len(selected_drugs)} drugs × {len(selected_cells)} cell lines)")

            if st.button("🔮 Run Predictions", type="primary"):
                results = []
                progress = st.progress(0)
                total = len(selected_drugs) * len(selected_cells)

                for i, drug in enumerate(selected_drugs):
                    for j, cell in enumerate(selected_cells):
                        result = simulate_prediction(drug, cell, selected_model)
                        results.append({
                            "DRUG_NAME": drug,
                            "CELL_LINE_NAME": cell,
                            "Predicted_LN_IC50": result["prediction"],
                            "Sensitivity": result["sensitivity"]
                        })
                        progress.progress((i * len(selected_cells) + j + 1) / total)

                results_df = pd.DataFrame(results)

                # Heatmap
                pivot = results_df.pivot(index="DRUG_NAME", columns="CELL_LINE_NAME", values="Predicted_LN_IC50")

                fig = px.imshow(
                    pivot,
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto",
                    title="Drug Sensitivity Heatmap",
                    labels={"color": "ln(IC50)"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.dataframe(results_df, use_container_width=True)

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

# Footer
st.divider()
st.caption("""
**Note:** Predictions are simulated for demonstration. In production, actual trained models would be loaded and used.
Sensitivity thresholds: Sensitive (ln(IC50) < -2), Intermediate (-2 to -1), Resistant (> -1)
""")
