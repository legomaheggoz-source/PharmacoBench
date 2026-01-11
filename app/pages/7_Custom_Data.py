"""
Custom Data Page

Upload and analyze your own drug sensitivity data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import StringIO
from typing import Tuple

st.set_page_config(page_title="Custom Data - PharmacoBench", page_icon="📁", layout="wide")

st.title("📁 Custom Data")
st.markdown("Upload your own drug sensitivity data for analysis")

# Initialize session state for custom data
if "custom_data" not in st.session_state:
    st.session_state["custom_data"] = None
if "custom_data_name" not in st.session_state:
    st.session_state["custom_data_name"] = None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate uploaded dataframe has required columns."""
    required_cols = {"DRUG_NAME", "CELL_LINE_NAME"}
    ic50_cols = {"IC50", "LN_IC50", "LOG_IC50", "ic50", "ln_ic50"}

    # Check for required columns
    df_cols_upper = {col.upper() for col in df.columns}

    missing = required_cols - df_cols_upper
    if missing:
        return False, f"Missing required columns: {missing}. Your data needs DRUG_NAME and CELL_LINE_NAME columns."

    # Check for IC50 column
    has_ic50 = bool(ic50_cols & df_cols_upper)
    if not has_ic50:
        return False, "Missing IC50 column. Please include IC50, LN_IC50, or LOG_IC50 column."

    return True, "Data validated successfully!"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to expected format."""
    # Create mapping for case-insensitive matching
    col_mapping = {}
    for col in df.columns:
        upper_col = col.upper()
        if upper_col == "DRUG_NAME" or upper_col == "DRUGNAME" or upper_col == "DRUG":
            col_mapping[col] = "DRUG_NAME"
        elif upper_col == "CELL_LINE_NAME" or upper_col == "CELLLINENAME" or upper_col == "CELL_LINE":
            col_mapping[col] = "CELL_LINE_NAME"
        elif upper_col in ("IC50", "LN_IC50", "LOG_IC50"):
            col_mapping[col] = "LN_IC50"
        elif upper_col == "TISSUE" or upper_col == "TISSUE_TYPE":
            col_mapping[col] = "TISSUE"

    df = df.rename(columns=col_mapping)

    # Convert IC50 to LN_IC50 if needed
    if "IC50" in df.columns and "LN_IC50" not in df.columns:
        df["LN_IC50"] = np.log(df["IC50"])

    return df


# File upload section
st.markdown("### Upload Data")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload drug sensitivity data with DRUG_NAME, CELL_LINE_NAME, and IC50/LN_IC50 columns"
    )

with col2:
    st.markdown("#### Required Columns")
    st.markdown("""
    - `DRUG_NAME` - Drug identifier
    - `CELL_LINE_NAME` - Cell line identifier
    - `IC50` or `LN_IC50` - Sensitivity value

    **Optional:**
    - `TISSUE` - Tissue type
    """)

if uploaded_file is not None:
    try:
        # Load file based on type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.markdown("---")
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Validate
        is_valid, message = validate_dataframe(df)

        if is_valid:
            st.success(message)

            # Standardize columns
            df = standardize_columns(df)

            # Store in session state
            if st.button("Load Data for Analysis", type="primary"):
                st.session_state["custom_data"] = df
                st.session_state["custom_data_name"] = uploaded_file.name
                st.success(f"Data loaded! {len(df):,} records from {uploaded_file.name}")
                st.rerun()
        else:
            st.error(message)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

st.divider()

# Show loaded data analysis
if st.session_state["custom_data"] is not None:
    df = st.session_state["custom_data"]

    st.markdown("### Loaded Data Analysis")
    st.caption(f"Source: {st.session_state['custom_data_name']}")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Drugs", f"{df['DRUG_NAME'].nunique():,}")
    with col3:
        st.metric("Unique Cell Lines", f"{df['CELL_LINE_NAME'].nunique():,}")
    with col4:
        if "TISSUE" in df.columns:
            st.metric("Tissue Types", f"{df['TISSUE'].nunique():,}")
        else:
            st.metric("Tissue Types", "N/A")

    st.divider()

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "💊 Drug Summary", "🧫 Cell Line Summary"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df,
                x="LN_IC50",
                nbins=50,
                title="IC50 Distribution",
                labels={"LN_IC50": "ln(IC50)"}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            if "TISSUE" in df.columns:
                fig_violin = px.violin(
                    df,
                    x="TISSUE",
                    y="LN_IC50",
                    color="TISSUE",
                    title="IC50 by Tissue Type",
                    box=True
                )
                fig_violin.update_layout(height=400, showlegend=False)
                fig_violin.update_xaxes(tickangle=45)
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                st.info("Add TISSUE column to see tissue-specific analysis")

    with tab2:
        drug_stats = df.groupby("DRUG_NAME").agg({
            "LN_IC50": ["count", "mean", "std"],
            "CELL_LINE_NAME": "nunique"
        }).round(3)
        drug_stats.columns = ["Tests", "Mean IC50", "Std IC50", "Cell Lines"]
        drug_stats = drug_stats.sort_values("Tests", ascending=False)

        st.dataframe(drug_stats, use_container_width=True)

        # Top drugs bar chart
        fig_drugs = px.bar(
            drug_stats.head(20).reset_index(),
            x="DRUG_NAME",
            y="Tests",
            color="Mean IC50",
            color_continuous_scale="RdYlGn_r",
            title="Top 20 Most Tested Drugs"
        )
        fig_drugs.update_layout(height=400)
        fig_drugs.update_xaxes(tickangle=45)
        st.plotly_chart(fig_drugs, use_container_width=True)

    with tab3:
        cell_stats = df.groupby("CELL_LINE_NAME").agg({
            "LN_IC50": ["count", "mean", "std"],
            "DRUG_NAME": "nunique"
        }).round(3)
        cell_stats.columns = ["Tests", "Mean IC50", "Std IC50", "Drugs Tested"]
        cell_stats = cell_stats.sort_values("Tests", ascending=False)

        st.dataframe(cell_stats.head(50), use_container_width=True)

    st.divider()

    # Export processed data
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data (CSV)",
            data=csv,
            file_name="processed_sensitivity_data.csv",
            mime="text/csv"
        )

    with col2:
        if st.button("Clear Loaded Data"):
            st.session_state["custom_data"] = None
            st.session_state["custom_data_name"] = None
            st.rerun()

else:
    st.info("Upload a file above to begin analysis, or use the sample data below.")

    # Provide sample data option
    st.markdown("### Try with Sample Data")

    if st.button("Load Sample Data"):
        np.random.seed(42)
        n_samples = 1000

        sample_data = {
            "DRUG_NAME": np.random.choice([f"Drug_{i}" for i in range(20)], n_samples),
            "CELL_LINE_NAME": np.random.choice([f"Cell_{i}" for i in range(50)], n_samples),
            "LN_IC50": np.random.normal(-2, 1.5, n_samples),
            "TISSUE": np.random.choice(["Lung", "Breast", "Colon", "Blood", "Skin"], n_samples)
        }

        st.session_state["custom_data"] = pd.DataFrame(sample_data)
        st.session_state["custom_data_name"] = "sample_data.csv"
        st.success("Sample data loaded!")
        st.rerun()

# Footer
st.divider()
st.caption("Supported formats: CSV, Excel (.xlsx, .xls)")
