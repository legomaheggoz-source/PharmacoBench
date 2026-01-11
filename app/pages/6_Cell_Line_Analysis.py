"""
Cell Line Analysis Page

Analyze individual cell lines, view genomic features, and drug sensitivity profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cell Line Analysis - PharmacoBench", page_icon="🧫", layout="wide")

st.title("🧫 Cell Line Analysis")
st.markdown("Explore cell line characteristics and drug sensitivity profiles")


@st.cache_data
def get_cell_line_data():
    """Generate demo cell line data."""
    np.random.seed(42)

    tissues = ["Lung", "Breast", "Blood", "Colon", "Skin", "Brain", "Liver", "Kidney", "Ovary", "Pancreas"]
    subtypes = {
        "Lung": ["NSCLC", "SCLC", "Adenocarcinoma"],
        "Breast": ["Triple Negative", "HER2+", "ER+"],
        "Blood": ["AML", "CML", "ALL", "CLL"],
        "Colon": ["Adenocarcinoma", "Carcinoma"],
        "Skin": ["Melanoma", "SCC", "BCC"],
        "Brain": ["Glioblastoma", "Astrocytoma", "Medulloblastoma"],
        "Liver": ["HCC", "Cholangiocarcinoma"],
        "Kidney": ["RCC", "Wilms"],
        "Ovary": ["Serous", "Mucinous", "Clear Cell"],
        "Pancreas": ["PDAC", "Neuroendocrine"],
    }

    mutations = ["KRAS", "TP53", "EGFR", "BRAF", "PIK3CA", "PTEN", "RB1", "MYC", "BRCA1", "BRCA2"]

    cell_lines = []
    n_cell_lines = 100

    for i in range(n_cell_lines):
        tissue = np.random.choice(tissues)
        subtype = np.random.choice(subtypes[tissue])

        # Generate mutation profile
        mutation_profile = {mut: np.random.choice([0, 1], p=[0.8, 0.2]) for mut in mutations}

        # Generate expression features (simplified)
        expression_mean = np.random.uniform(-1, 1)

        cell_lines.append({
            "CELL_LINE_NAME": f"CellLine_{i}",
            "TISSUE": tissue,
            "SUBTYPE": subtype,
            "EXPRESSION_MEAN": round(expression_mean, 3),
            "MUTATION_COUNT": sum(mutation_profile.values()),
            **mutation_profile,
        })

    return pd.DataFrame(cell_lines), mutations


@st.cache_data
def get_sensitivity_data(cell_line_name):
    """Generate drug sensitivity data for a specific cell line."""
    np.random.seed(hash(cell_line_name) % 2**32)

    drugs = [
        "Gefitinib", "Erlotinib", "Lapatinib", "Imatinib", "Sorafenib",
        "Vemurafenib", "Crizotinib", "Dasatinib", "Nilotinib", "Bosutinib",
        "Sunitinib", "Pazopanib", "Axitinib", "Cabozantinib", "Lenvatinib",
    ]

    targets = {
        "Gefitinib": "EGFR", "Erlotinib": "EGFR", "Lapatinib": "EGFR/HER2",
        "Imatinib": "BCR-ABL", "Sorafenib": "Multi-kinase", "Vemurafenib": "BRAF",
        "Crizotinib": "ALK/MET", "Dasatinib": "BCR-ABL/SRC", "Nilotinib": "BCR-ABL",
        "Bosutinib": "BCR-ABL/SRC", "Sunitinib": "Multi-kinase", "Pazopanib": "VEGFR",
        "Axitinib": "VEGFR", "Cabozantinib": "MET/VEGFR", "Lenvatinib": "Multi-kinase",
    }

    data = []
    for drug in drugs:
        ic50 = -2 + np.random.normal(0, 1)
        data.append({
            "DRUG_NAME": drug,
            "TARGET": targets[drug],
            "LN_IC50": round(ic50, 3),
            "SENSITIVITY": "Sensitive" if ic50 < -2 else ("Intermediate" if ic50 < -1 else "Resistant"),
        })

    return pd.DataFrame(data)


cell_df, mutation_list = get_cell_line_data()

# Sidebar - Cell line selection
st.sidebar.markdown("### Select Cell Line")

# Filter by tissue first
tissues = ["All"] + cell_df["TISSUE"].unique().tolist()
selected_tissue_filter = st.sidebar.selectbox("Filter by Tissue", tissues)

if selected_tissue_filter != "All":
    filtered_cells = cell_df[cell_df["TISSUE"] == selected_tissue_filter]["CELL_LINE_NAME"].tolist()
else:
    filtered_cells = cell_df["CELL_LINE_NAME"].tolist()

selected_cell = st.sidebar.selectbox("Cell Line", filtered_cells)

st.sidebar.markdown("---")

# Cell line info
cell_info = cell_df[cell_df["CELL_LINE_NAME"] == selected_cell].iloc[0]

st.sidebar.markdown(f"**Tissue:** {cell_info['TISSUE']}")
st.sidebar.markdown(f"**Subtype:** {cell_info['SUBTYPE']}")
st.sidebar.markdown(f"**Mutations:** {cell_info['MUTATION_COUNT']}")

# Main content
sensitivity_data = get_sensitivity_data(selected_cell)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Cell Line Information")

    st.markdown(f"**Name:** {selected_cell}")
    st.markdown(f"**Tissue:** {cell_info['TISSUE']}")
    st.markdown(f"**Subtype:** {cell_info['SUBTYPE']}")

    st.markdown("---")
    st.markdown("### Mutation Profile")

    # Display mutations as chips
    mutated = [mut for mut in mutation_list if cell_info[mut] == 1]
    wild_type = [mut for mut in mutation_list if cell_info[mut] == 0]

    if mutated:
        st.markdown("**Mutated:**")
        st.markdown(" ".join([f"`{m}`" for m in mutated]))
    else:
        st.info("No mutations detected in screened genes")

    st.markdown("---")
    st.markdown("### Expression Profile")
    st.metric("Mean Expression Z-score", f"{cell_info['EXPRESSION_MEAN']:.3f}")

with col2:
    st.markdown("### Drug Sensitivity Profile")

    # Waterfall plot of drug sensitivities
    sorted_data = sensitivity_data.sort_values("LN_IC50")

    colors = sorted_data["LN_IC50"].apply(
        lambda x: "#22c55e" if x < -2 else ("#eab308" if x < -1 else "#ef4444")
    )

    fig_waterfall = go.Figure(data=[
        go.Bar(
            x=sorted_data["DRUG_NAME"],
            y=sorted_data["LN_IC50"],
            marker_color=colors,
            text=sorted_data["LN_IC50"].round(2),
            textposition="outside",
        )
    ])

    fig_waterfall.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Sensitive threshold")
    fig_waterfall.add_hline(y=-1, line_dash="dash", line_color="orange", annotation_text="Intermediate threshold")

    fig_waterfall.update_layout(
        height=400,
        title=f"{selected_cell} Drug Sensitivity (Waterfall Plot)",
        xaxis_title="Drug",
        yaxis_title="ln(IC50)",
        xaxis_tickangle=45,
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

st.divider()

# Sensitivity breakdown
st.markdown("### Sensitivity Summary")

col1, col2, col3 = st.columns(3)

sensitive_count = (sensitivity_data["SENSITIVITY"] == "Sensitive").sum()
intermediate_count = (sensitivity_data["SENSITIVITY"] == "Intermediate").sum()
resistant_count = (sensitivity_data["SENSITIVITY"] == "Resistant").sum()

with col1:
    st.metric("Sensitive", sensitive_count, delta=f"{sensitive_count/len(sensitivity_data)*100:.0f}%")

with col2:
    st.metric("Intermediate", intermediate_count, delta=f"{intermediate_count/len(sensitivity_data)*100:.0f}%")

with col3:
    st.metric("Resistant", resistant_count, delta=f"{resistant_count/len(sensitivity_data)*100:.0f}%")

col1, col2 = st.columns(2)

with col1:
    # Pie chart of sensitivity distribution
    sens_counts = sensitivity_data["SENSITIVITY"].value_counts()

    fig_pie = px.pie(
        values=sens_counts.values,
        names=sens_counts.index,
        title="Sensitivity Distribution",
        color=sens_counts.index,
        color_discrete_map={
            "Sensitive": "#22c55e",
            "Intermediate": "#eab308",
            "Resistant": "#ef4444",
        },
        hole=0.4,
    )
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Target breakdown
    target_sensitivity = sensitivity_data.groupby("TARGET")["LN_IC50"].mean().sort_values()

    fig_target = px.bar(
        x=target_sensitivity.index,
        y=target_sensitivity.values,
        labels={"x": "Drug Target", "y": "Mean ln(IC50)"},
        title="Sensitivity by Drug Target",
        color=target_sensitivity.values,
        color_continuous_scale="RdYlGn_r",
    )
    fig_target.update_layout(height=350, showlegend=False)
    fig_target.update_xaxes(tickangle=45)
    st.plotly_chart(fig_target, use_container_width=True)

st.divider()

# Detailed sensitivity table
st.markdown("### Drug Sensitivity Details")

# Add styling to the dataframe
styled_df = sensitivity_data[["DRUG_NAME", "TARGET", "LN_IC50", "SENSITIVITY"]].copy()
styled_df = styled_df.sort_values("LN_IC50")

st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.divider()

# Compare with other cell lines
st.markdown("### Compare with Other Cell Lines")

comparison_cells = st.multiselect(
    "Select cell lines to compare",
    [c for c in filtered_cells if c != selected_cell],
    default=filtered_cells[1:3] if len(filtered_cells) > 2 else [],
)

if comparison_cells:
    # Get sensitivity for comparison cells
    all_cells = [selected_cell] + comparison_cells
    comparison_data = []

    for cell in all_cells:
        cell_sens = get_sensitivity_data(cell)
        cell_sens["CELL_LINE"] = cell
        comparison_data.append(cell_sens)

    comparison_df = pd.concat(comparison_data, ignore_index=True)

    fig_comparison = px.box(
        comparison_df,
        x="CELL_LINE",
        y="LN_IC50",
        color="CELL_LINE",
        title="IC50 Distribution Comparison",
        points="all",
    )
    fig_comparison.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Heatmap of drug sensitivities across cell lines
    st.markdown("### Drug Sensitivity Heatmap")

    pivot_data = comparison_df.pivot(index="DRUG_NAME", columns="CELL_LINE", values="LN_IC50")

    fig_heatmap = px.imshow(
        pivot_data,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"color": "ln(IC50)"},
        text_auto=".2f",
    )
    fig_heatmap.update_layout(height=500, title="Drug Sensitivity Across Cell Lines")
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# Tissue-level analysis
st.markdown("### Tissue-Level Statistics")

tissue_stats = cell_df.groupby("TISSUE").agg({
    "CELL_LINE_NAME": "count",
    "MUTATION_COUNT": "mean",
    "EXPRESSION_MEAN": "mean",
}).round(3)
tissue_stats.columns = ["Cell Line Count", "Avg Mutations", "Avg Expression"]
tissue_stats = tissue_stats.sort_values("Cell Line Count", ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(tissue_stats, use_container_width=True)

with col2:
    fig_tissue = px.bar(
        x=tissue_stats.index,
        y=tissue_stats["Cell Line Count"],
        labels={"x": "Tissue", "y": "Cell Line Count"},
        title="Cell Lines by Tissue Type",
        color=tissue_stats["Cell Line Count"],
        color_continuous_scale="Blues",
    )
    fig_tissue.update_layout(height=300, showlegend=False)
    fig_tissue.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tissue, use_container_width=True)

# Export
st.divider()
csv = sensitivity_data.to_csv(index=False)
st.download_button(
    label=f"📥 Download {selected_cell} Sensitivity Data",
    data=csv,
    file_name=f"{selected_cell}_drug_sensitivity.csv",
    mime="text/csv",
)
