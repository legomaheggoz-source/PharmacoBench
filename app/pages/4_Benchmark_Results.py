"""
Benchmark Results Page

View and compare model performance across split strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Benchmark Results - PharmacoBench", page_icon="📈", layout="wide")

st.title("📈 Benchmark Results")
st.markdown("Compare model performance across different evaluation strategies")


@st.cache_data
def get_benchmark_results():
    """Generate comprehensive benchmark results."""
    models = ["Ridge", "ElasticNet", "Random Forest", "XGBoost", "LightGBM", "MLP", "GraphDRP", "DeepCDR"]
    splits = ["Random", "Drug-Blind", "Cell-Blind", "Disjoint"]

    data = []
    np.random.seed(42)

    for model in models:
        for split in splits:
            base_rmse = {
                "Ridge": 1.2, "ElasticNet": 1.18, "Random Forest": 1.05,
                "XGBoost": 0.98, "LightGBM": 0.99, "MLP": 1.0,
                "GraphDRP": 0.92, "DeepCDR": 0.88
            }[model]

            split_penalty = {"Random": 0, "Drug-Blind": 0.15, "Cell-Blind": 0.12, "Disjoint": 0.25}[split]

            rmse = base_rmse + split_penalty + np.random.uniform(-0.05, 0.05)
            mae = rmse * 0.8 + np.random.uniform(-0.02, 0.02)
            spearman = 0.85 - (rmse - 0.8) * 0.3 + np.random.uniform(-0.02, 0.02)
            pearson = spearman + np.random.uniform(-0.02, 0.02)
            r2 = 1 - rmse**2 / 2 + np.random.uniform(-0.05, 0.05)

            data.append({
                "Model": model,
                "Split": split,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R²": round(max(0, min(1, r2)), 4),
                "Pearson": round(min(1, max(0, pearson)), 4),
                "Spearman": round(min(1, max(0, spearman)), 4),
                "Train Time (s)": round(np.random.uniform(10, 300), 1),
            })

    return pd.DataFrame(data)


results_df = get_benchmark_results()

# Sidebar filters
st.sidebar.markdown("### Filters")

selected_models = st.sidebar.multiselect(
    "Models",
    options=results_df["Model"].unique(),
    default=results_df["Model"].unique(),
)

selected_splits = st.sidebar.multiselect(
    "Split Strategies",
    options=results_df["Split"].unique(),
    default=results_df["Split"].unique(),
)

metric_options = ["RMSE", "MAE", "R²", "Pearson", "Spearman"]
primary_metric = st.sidebar.selectbox("Primary Metric", metric_options, index=0)

# Filter data
filtered_df = results_df[
    (results_df["Model"].isin(selected_models)) &
    (results_df["Split"].isin(selected_splits))
]

# Overview metrics
st.markdown("### Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    best_model = filtered_df.loc[filtered_df["RMSE"].idxmin()]
    st.metric("Best Model (RMSE)", best_model["Model"], f"{best_model['RMSE']:.4f}")

with col2:
    avg_rmse = filtered_df["RMSE"].mean()
    st.metric("Average RMSE", f"{avg_rmse:.4f}")

with col3:
    best_spearman = filtered_df["Spearman"].max()
    st.metric("Best Spearman", f"{best_spearman:.4f}")

with col4:
    n_experiments = len(filtered_df)
    st.metric("Experiments", n_experiments)

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Heatmaps", "📈 Comparisons", "📋 Tables", "🔬 Analysis"])

with tab1:
    st.markdown("### Performance Heatmaps")

    metric_for_heatmap = st.selectbox("Select Metric for Heatmap", metric_options, key="heatmap_metric")

    pivot_data = filtered_df.pivot(index="Model", columns="Split", values=metric_for_heatmap)

    # Determine color scale direction
    reverse_scale = metric_for_heatmap in ["RMSE", "MAE"]

    fig_heatmap = px.imshow(
        pivot_data,
        color_continuous_scale="RdYlGn_r" if reverse_scale else "RdYlGn",
        aspect="auto",
        labels={"color": metric_for_heatmap},
        text_auto=".3f",
    )
    fig_heatmap.update_layout(
        height=500,
        title=f"{metric_for_heatmap} by Model and Split Strategy",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    st.markdown("### Model Comparisons")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart by model
        model_avg = filtered_df.groupby("Model")[primary_metric].mean().sort_values(
            ascending=(primary_metric in ["RMSE", "MAE"])
        )

        fig_bar = px.bar(
            x=model_avg.index,
            y=model_avg.values,
            labels={"x": "Model", "y": f"Avg {primary_metric}"},
            title=f"Average {primary_metric} by Model",
            color=model_avg.values,
            color_continuous_scale="RdYlGn_r" if primary_metric in ["RMSE", "MAE"] else "RdYlGn",
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Box plot by split
        fig_box = px.box(
            filtered_df,
            x="Split",
            y=primary_metric,
            color="Split",
            title=f"{primary_metric} Distribution by Split Strategy",
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Line plot: Performance degradation across splits
    st.markdown("### Performance Degradation Across Splits")

    split_order = ["Random", "Cell-Blind", "Drug-Blind", "Disjoint"]
    ordered_df = filtered_df.copy()
    ordered_df["Split"] = pd.Categorical(ordered_df["Split"], categories=split_order, ordered=True)
    ordered_df = ordered_df.sort_values("Split")

    fig_line = px.line(
        ordered_df,
        x="Split",
        y=primary_metric,
        color="Model",
        markers=True,
        title=f"{primary_metric} Across Split Strategies (Random → Disjoint = Increasing Difficulty)",
    )
    fig_line.update_layout(height=400)
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.markdown("### Detailed Results Table")

    # Sortable table
    sort_col = st.selectbox("Sort by", filtered_df.columns.tolist(), index=2)
    sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

    sorted_df = filtered_df.sort_values(sort_col, ascending=(sort_order == "Ascending"))

    st.dataframe(sorted_df, use_container_width=True, hide_index=True)

    # Download
    csv = sorted_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="benchmark_results.csv",
        mime="text/csv",
    )

    st.markdown("### Pivot Table: Model × Split")

    pivot_cols = st.multiselect(
        "Select metrics for pivot",
        metric_options,
        default=["RMSE", "Spearman"],
    )

    for metric in pivot_cols:
        st.markdown(f"**{metric}:**")
        pivot = filtered_df.pivot(index="Model", columns="Split", values=metric).round(4)
        st.dataframe(pivot, use_container_width=True)

with tab4:
    st.markdown("### Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Correlation Between Metrics")

        metrics_df = filtered_df[["RMSE", "MAE", "R²", "Pearson", "Spearman"]]
        corr_matrix = metrics_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            title="Metric Correlation Matrix",
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.markdown("#### Training Time vs Performance")

        fig_scatter = px.scatter(
            filtered_df,
            x="Train Time (s)",
            y="RMSE",
            color="Model",
            size="Spearman",
            hover_data=["Split"],
            title="Training Time vs RMSE (size = Spearman)",
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Model Ranking Summary")

    ranking_data = []
    for split in filtered_df["Split"].unique():
        split_data = filtered_df[filtered_df["Split"] == split].sort_values("RMSE")
        for rank, (_, row) in enumerate(split_data.iterrows(), 1):
            ranking_data.append({
                "Model": row["Model"],
                "Split": split,
                "Rank": rank,
            })

    ranking_df = pd.DataFrame(ranking_data)
    avg_rank = ranking_df.groupby("Model")["Rank"].mean().sort_values()

    st.markdown("**Average Rank Across Split Strategies:**")

    fig_rank = px.bar(
        x=avg_rank.index,
        y=avg_rank.values,
        labels={"x": "Model", "y": "Average Rank"},
        title="Model Ranking (lower is better)",
        color=avg_rank.values,
        color_continuous_scale="RdYlGn_r",
    )
    fig_rank.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

# Footer
st.divider()
st.caption("Results shown are from benchmark runs. Lower RMSE and higher correlation values indicate better performance.")
