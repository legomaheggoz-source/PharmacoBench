"""
Dashboard Page

Overview of benchmark results and key metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Dashboard - PharmacoBench", page_icon="📊", layout="wide")

# Page header with refresh capability
col1, col2 = st.columns([5, 1])
with col1:
    st.title("📊 Dashboard")
    st.markdown("Overview of drug sensitivity prediction benchmarks")
with col2:
    if st.button("Refresh", help="Refresh data from cache"):
        st.cache_data.clear()
        st.rerun()

# Show last updated timestamp
if "dashboard_last_updated" not in st.session_state:
    st.session_state["dashboard_last_updated"] = datetime.now()
st.caption(f"Last updated: {st.session_state['dashboard_last_updated'].strftime('%Y-%m-%d %H:%M')}")

# Demo data (will be replaced with actual results)
@st.cache_data
def get_demo_results():
    """Generate demo benchmark results."""
    models = ["Ridge", "ElasticNet", "Random Forest", "XGBoost", "LightGBM", "MLP", "GraphDRP", "DeepCDR"]
    splits = ["Random", "Drug-Blind", "Cell-Blind", "Disjoint"]

    data = []
    np.random.seed(42)

    for model in models:
        for split in splits:
            # Simulate realistic metrics
            base_rmse = {
                "Ridge": 1.2, "ElasticNet": 1.18, "Random Forest": 1.05,
                "XGBoost": 0.98, "LightGBM": 0.99, "MLP": 1.0,
                "GraphDRP": 0.92, "DeepCDR": 0.88
            }[model]

            split_penalty = {"Random": 0, "Drug-Blind": 0.15, "Cell-Blind": 0.12, "Disjoint": 0.25}[split]

            rmse = base_rmse + split_penalty + np.random.uniform(-0.05, 0.05)
            spearman = 0.85 - (rmse - 0.8) * 0.3 + np.random.uniform(-0.02, 0.02)

            data.append({
                "Model": model,
                "Split": split,
                "RMSE": round(rmse, 3),
                "Spearman": round(min(0.95, max(0.5, spearman)), 3),
                "R²": round(1 - rmse**2 / 2, 3),
            })

    return pd.DataFrame(data)


# Load data with error handling
try:
    with st.spinner("Loading benchmark results..."):
        results_df = get_demo_results()
        st.session_state["dashboard_last_updated"] = datetime.now()
except Exception as e:
    st.error(f"Failed to load benchmark results: {str(e)}")
    st.info("Please try refreshing the page or contact support if the issue persists.")
    st.stop()

if results_df is None or len(results_df) == 0:
    st.warning("No benchmark results available.")
    st.stop()

# KPI Cards
st.markdown("### Key Metrics")

col1, col2, col3, col4 = st.columns(4)

best_model = results_df.loc[results_df["RMSE"].idxmin()]

with col1:
    st.metric(
        label="Best Model",
        value=best_model["Model"],
        delta=f"RMSE: {best_model['RMSE']:.3f}",
    )

with col2:
    avg_rmse = results_df["RMSE"].mean()
    st.metric(
        label="Avg RMSE",
        value=f"{avg_rmse:.3f}",
        delta=f"{(avg_rmse - 1.0):.3f}",
        delta_color="inverse",
    )

with col3:
    avg_spearman = results_df["Spearman"].mean()
    st.metric(
        label="Avg Spearman",
        value=f"{avg_spearman:.3f}",
        delta=f"{(avg_spearman - 0.7):.3f}",
    )

with col4:
    n_models = results_df["Model"].nunique()
    st.metric(
        label="Models Tested",
        value=n_models,
    )

st.divider()

# Performance Heatmap
st.markdown("### Performance Heatmap (RMSE)")

pivot_rmse = results_df.pivot(index="Model", columns="Split", values="RMSE")

fig_heatmap = px.imshow(
    pivot_rmse,
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    labels={"color": "RMSE"},
)
fig_heatmap.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# Model Comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("### RMSE by Model")

    model_avg = results_df.groupby("Model")["RMSE"].mean().sort_values()

    fig_bar = px.bar(
        x=model_avg.index,
        y=model_avg.values,
        labels={"x": "Model", "y": "Average RMSE"},
        color=model_avg.values,
        color_continuous_scale="RdYlGn_r",
    )
    fig_bar.update_layout(
        showlegend=False,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Spearman by Split Strategy")

    split_avg = results_df.groupby("Split")["Spearman"].mean().sort_values(ascending=False)

    fig_bar2 = px.bar(
        x=split_avg.index,
        y=split_avg.values,
        labels={"x": "Split Strategy", "y": "Average Spearman"},
        color=split_avg.values,
        color_continuous_scale="RdYlGn",
    )
    fig_bar2.update_layout(
        showlegend=False,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_bar2, use_container_width=True)

st.divider()

# Radar Chart
st.markdown("### Model Performance Comparison")

# Prepare data for radar chart
metrics = ["RMSE", "Spearman", "R²"]
models_to_compare = ["Ridge", "Random Forest", "XGBoost", "DeepCDR"]

random_split = results_df[results_df["Split"] == "Random"]

fig_radar = go.Figure()

for model in models_to_compare:
    model_data = random_split[random_split["Model"] == model].iloc[0]

    # Normalize metrics (invert RMSE so higher is better)
    values = [
        1 - (model_data["RMSE"] - 0.8) / 0.5,  # Normalize RMSE
        model_data["Spearman"],
        max(0, model_data["R²"]),
    ]

    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=metrics + [metrics[0]],
        fill="toself",
        name=model,
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    height=400,
    margin=dict(l=50, r=50, t=30, b=30),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# Results Table
st.markdown("### Detailed Results")

st.dataframe(
    results_df.sort_values("RMSE"),
    use_container_width=True,
    hide_index=True,
)

# Export button
csv = results_df.to_csv(index=False)
st.download_button(
    label="📥 Download Results (CSV)",
    data=csv,
    file_name="pharmacobench_results.csv",
    mime="text/csv",
)
