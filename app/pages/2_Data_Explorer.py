"""
Data Explorer Page

Explore GDSC dataset statistics, distributions, and coverage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer - PharmacoBench", page_icon="🔍", layout="wide")

st.title("🔍 Data Explorer")
st.markdown("Explore the GDSC drug sensitivity dataset")


@st.cache_data
def get_demo_data():
    """Generate demo GDSC-like data."""
    np.random.seed(42)

    n_samples = 200000
    n_drugs = 300
    n_cell_lines = 900

    drugs = [f"Drug_{i}" for i in range(n_drugs)]
    cell_lines = [f"CellLine_{i}" for i in range(n_cell_lines)]
    tissues = ["Lung", "Breast", "Colon", "Blood", "Brain", "Skin", "Liver", "Kidney", "Ovary", "Pancreas"]

    data = {
        "DRUG_NAME": np.random.choice(drugs, n_samples),
        "CELL_LINE_NAME": np.random.choice(cell_lines, n_samples),
        "LN_IC50": np.random.normal(-2, 1.5, n_samples),
        "TISSUE": np.random.choice(tissues, n_samples),
        "SOURCE": np.random.choice(["GDSC1", "GDSC2"], n_samples, p=[0.6, 0.4]),
    }

    return pd.DataFrame(data)


# Load data
with st.spinner("Loading data..."):
    df = get_demo_data()

# Overview metrics
st.markdown("### Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Unique Drugs", f"{df['DRUG_NAME'].nunique():,}")

with col3:
    st.metric("Unique Cell Lines", f"{df['CELL_LINE_NAME'].nunique():,}")

with col4:
    st.metric("Tissue Types", f"{df['TISSUE'].nunique():,}")

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "💊 Drug Coverage", "🧫 Cell Line Coverage", "🔬 Data Quality"])

with tab1:
    st.markdown("### IC50 Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            df,
            x="LN_IC50",
            nbins=50,
            color="SOURCE",
            marginal="box",
            labels={"LN_IC50": "ln(IC50)"},
            title="IC50 Distribution by Data Source",
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_violin = px.violin(
            df,
            x="TISSUE",
            y="LN_IC50",
            color="TISSUE",
            box=True,
            title="IC50 Distribution by Tissue Type",
        )
        fig_violin.update_layout(height=400, showlegend=False)
        fig_violin.update_xaxes(tickangle=45)
        st.plotly_chart(fig_violin, use_container_width=True)

    # Statistics table
    st.markdown("### IC50 Statistics by Source")

    stats = df.groupby("SOURCE")["LN_IC50"].agg(["count", "mean", "std", "min", "max"]).round(3)
    stats.columns = ["Count", "Mean", "Std", "Min", "Max"]
    st.dataframe(stats, use_container_width=True)

with tab2:
    st.markdown("### Drug Coverage Analysis")

    drug_counts = df.groupby("DRUG_NAME").agg({
        "CELL_LINE_NAME": "nunique",
        "LN_IC50": ["count", "mean", "std"],
    }).round(3)
    drug_counts.columns = ["Cell Lines Tested", "Total Tests", "Mean IC50", "Std IC50"]
    drug_counts = drug_counts.sort_values("Total Tests", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_drug = px.bar(
            drug_counts.head(30).reset_index(),
            x="DRUG_NAME",
            y="Total Tests",
            color="Mean IC50",
            color_continuous_scale="RdYlGn_r",
            title="Top 30 Most Tested Drugs",
        )
        fig_drug.update_layout(height=400)
        fig_drug.update_xaxes(tickangle=45)
        st.plotly_chart(fig_drug, use_container_width=True)

    with col2:
        st.markdown("### Coverage Summary")
        st.metric("Avg Cell Lines/Drug", f"{drug_counts['Cell Lines Tested'].mean():.0f}")
        st.metric("Max Cell Lines/Drug", f"{drug_counts['Cell Lines Tested'].max():.0f}")
        st.metric("Min Cell Lines/Drug", f"{drug_counts['Cell Lines Tested'].min():.0f}")

    st.markdown("### Drug Coverage Table")
    st.dataframe(drug_counts.head(50), use_container_width=True)

with tab3:
    st.markdown("### Cell Line Coverage Analysis")

    cell_counts = df.groupby("CELL_LINE_NAME").agg({
        "DRUG_NAME": "nunique",
        "LN_IC50": ["count", "mean", "std"],
        "TISSUE": "first",
    }).round(3)
    cell_counts.columns = ["Drugs Tested", "Total Tests", "Mean IC50", "Std IC50", "Tissue"]
    cell_counts = cell_counts.sort_values("Total Tests", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Cell lines by tissue
        tissue_counts = df.groupby("TISSUE")["CELL_LINE_NAME"].nunique().sort_values(ascending=False)

        fig_tissue = px.pie(
            values=tissue_counts.values,
            names=tissue_counts.index,
            title="Cell Lines by Tissue Type",
            hole=0.4,
        )
        fig_tissue.update_layout(height=400)
        st.plotly_chart(fig_tissue, use_container_width=True)

    with col2:
        st.markdown("### Coverage Summary")
        st.metric("Avg Drugs/Cell Line", f"{cell_counts['Drugs Tested'].mean():.0f}")
        st.metric("Max Drugs/Cell Line", f"{cell_counts['Drugs Tested'].max():.0f}")
        st.metric("Min Drugs/Cell Line", f"{cell_counts['Drugs Tested'].min():.0f}")

    st.markdown("### Cell Line Coverage Table")
    st.dataframe(cell_counts.head(50), use_container_width=True)

with tab4:
    st.markdown("### Data Quality Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Missing Values")

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        quality_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
        })
        st.dataframe(quality_df, use_container_width=True)

    with col2:
        st.markdown("#### Data Source Distribution")

        source_counts = df["SOURCE"].value_counts()

        fig_source = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Records by Data Source",
        )
        fig_source.update_layout(height=300)
        st.plotly_chart(fig_source, use_container_width=True)

    st.markdown("#### IC50 Outlier Analysis")

    # Calculate Z-scores
    z_scores = np.abs((df["LN_IC50"] - df["LN_IC50"].mean()) / df["LN_IC50"].std())

    outlier_counts = pd.DataFrame({
        "Threshold": ["Z > 2", "Z > 3", "Z > 4"],
        "Count": [
            (z_scores > 2).sum(),
            (z_scores > 3).sum(),
            (z_scores > 4).sum(),
        ],
        "Percentage": [
            f"{(z_scores > 2).sum() / len(df) * 100:.2f}%",
            f"{(z_scores > 3).sum() / len(df) * 100:.2f}%",
            f"{(z_scores > 4).sum() / len(df) * 100:.2f}%",
        ],
    })
    st.dataframe(outlier_counts, use_container_width=True)

# Sidebar filters
st.sidebar.markdown("### Filters")

selected_source = st.sidebar.multiselect(
    "Data Source",
    options=df["SOURCE"].unique(),
    default=df["SOURCE"].unique(),
)

selected_tissues = st.sidebar.multiselect(
    "Tissue Types",
    options=df["TISSUE"].unique(),
    default=df["TISSUE"].unique()[:5],
)

ic50_range = st.sidebar.slider(
    "IC50 Range",
    float(df["LN_IC50"].min()),
    float(df["LN_IC50"].max()),
    (float(df["LN_IC50"].quantile(0.05)), float(df["LN_IC50"].quantile(0.95))),
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered:** {len(df):,} records")
