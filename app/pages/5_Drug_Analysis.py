"""
Drug Analysis Page

Analyze individual drugs, view structures, and sensitivity profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Drug Analysis - PharmacoBench", page_icon="💊", layout="wide")

st.title("💊 Drug Analysis")
st.markdown("Explore drug structures and sensitivity profiles")


@st.cache_data
def get_drug_data():
    """Generate demo drug data."""
    np.random.seed(42)

    drugs = [
        {"name": "Gefitinib", "smiles": "COC1=C(OC2CCCC2)C=C2C(NC3=CC(Cl)=C(F)C=C3)=NC=NC2=C1", "target": "EGFR", "phase": "Approved"},
        {"name": "Erlotinib", "smiles": "COC1=C(OCCOC)C=C2C(NC3=CC=CC(C#C)=C3)=NC=NC2=C1", "target": "EGFR", "phase": "Approved"},
        {"name": "Lapatinib", "smiles": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(Cl)=C(OCC5=CC(F)=CC=C5)C=C4", "target": "EGFR/HER2", "phase": "Approved"},
        {"name": "Imatinib", "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "target": "BCR-ABL", "phase": "Approved"},
        {"name": "Sorafenib", "smiles": "CNC(=O)C1=CC(OC2=CC=C(NC(=O)NC3=CC(Cl)=C(C(F)(F)F)C=C3)C=C2)=CC=N1", "target": "Multi-kinase", "phase": "Approved"},
        {"name": "Vemurafenib", "smiles": "CCCS(=O)(=O)NC1=CC=C(F)C(C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(Cl)C=C4)=C1", "target": "BRAF", "phase": "Approved"},
        {"name": "Crizotinib", "smiles": "CC(OC1=C(N)N=CC(=C1)C2=CN(N=C2)C3CCNCC3)C4=C(Cl)C=CC(F)=C4Cl", "target": "ALK/MET", "phase": "Approved"},
        {"name": "Dasatinib", "smiles": "CC1=NC(NC2=NC=C(S2)C(=O)NC3=C(C)C=CC=C3Cl)=CC(=N1)N4CCN(CCO)CC4", "target": "BCR-ABL/SRC", "phase": "Approved"},
        {"name": "Nilotinib", "smiles": "CC1=CN=C(NC2=CC(NC(=O)C3=CC(C(F)(F)F)=CC=N3)=CC(=C2)C(F)(F)F)N=C1C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5", "target": "BCR-ABL", "phase": "Approved"},
        {"name": "Bosutinib", "smiles": "COC1=CC2=C(C=C1OC)C(=NC=N2)NC3=CC(OC)=C(Cl)C=C3Cl", "target": "BCR-ABL/SRC", "phase": "Approved"},
    ]

    n_cell_lines = 100
    tissues = ["Lung", "Breast", "Blood", "Colon", "Skin"]

    data = []
    for drug in drugs:
        for i in range(n_cell_lines):
            # Different sensitivity patterns per drug
            base_ic50 = -2 + np.random.uniform(-0.5, 0.5)
            tissue = np.random.choice(tissues)

            # Some drugs are more tissue-specific
            if drug["name"] == "Gefitinib" and tissue == "Lung":
                base_ic50 -= 1  # More effective in lung
            elif drug["name"] == "Imatinib" and tissue == "Blood":
                base_ic50 -= 1.5  # More effective in blood cancers

            data.append({
                "DRUG_NAME": drug["name"],
                "SMILES": drug["smiles"],
                "TARGET": drug["target"],
                "PHASE": drug["phase"],
                "CELL_LINE": f"Cell_{i}",
                "TISSUE": tissue,
                "LN_IC50": base_ic50 + np.random.normal(0, 0.5),
            })

    return pd.DataFrame(data)


df = get_drug_data()

# Sidebar - Drug selection
st.sidebar.markdown("### Select Drug")

drugs_list = df["DRUG_NAME"].unique().tolist()
selected_drug = st.sidebar.selectbox("Drug", drugs_list)

st.sidebar.markdown("---")

# Drug info
drug_info = df[df["DRUG_NAME"] == selected_drug].iloc[0]

st.sidebar.markdown(f"**Target:** {drug_info['TARGET']}")
st.sidebar.markdown(f"**Phase:** {drug_info['PHASE']}")

# Main content
drug_data = df[df["DRUG_NAME"] == selected_drug]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Drug Information")

    st.markdown(f"**Name:** {selected_drug}")
    st.markdown(f"**Target:** {drug_info['TARGET']}")
    st.markdown(f"**Phase:** {drug_info['PHASE']}")

    st.markdown("---")
    st.markdown("### SMILES Structure")
    st.code(drug_info['SMILES'], language=None)

    # Molecular structure placeholder
    st.markdown("### 2D Structure")
    st.info("📷 Molecular structure visualization requires RDKit. Install with: `pip install rdkit`")

    # Try to render structure
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        import base64

        mol = Chem.MolFromSmiles(drug_info['SMILES'])
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            st.image(buf.getvalue(), caption=selected_drug)
    except ImportError:
        pass

with col2:
    st.markdown("### Sensitivity Profile")

    # IC50 distribution
    fig_hist = px.histogram(
        drug_data,
        x="LN_IC50",
        color="TISSUE",
        nbins=30,
        marginal="box",
        title=f"{selected_drug} IC50 Distribution by Tissue",
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# Tissue-specific analysis
st.markdown("### Tissue-Specific Sensitivity")

col1, col2 = st.columns(2)

with col1:
    # Violin plot
    fig_violin = px.violin(
        drug_data,
        x="TISSUE",
        y="LN_IC50",
        color="TISSUE",
        box=True,
        title="IC50 by Tissue Type",
    )
    fig_violin.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

with col2:
    # Summary statistics
    tissue_stats = drug_data.groupby("TISSUE")["LN_IC50"].agg([
        "count", "mean", "std", "min", "max"
    ]).round(3)
    tissue_stats.columns = ["Count", "Mean IC50", "Std", "Min", "Max"]
    tissue_stats = tissue_stats.sort_values("Mean IC50")

    st.markdown("### Statistics by Tissue")
    st.dataframe(tissue_stats, use_container_width=True)

    # Most sensitive tissue
    most_sensitive = tissue_stats["Mean IC50"].idxmin()
    st.success(f"🎯 Most sensitive tissue: **{most_sensitive}** (Mean IC50: {tissue_stats.loc[most_sensitive, 'Mean IC50']:.3f})")

st.divider()

# Cell line ranking
st.markdown("### Cell Line Sensitivity Ranking")

cell_ranking = drug_data.sort_values("LN_IC50")[["CELL_LINE", "TISSUE", "LN_IC50"]].head(20)
cell_ranking["Rank"] = range(1, len(cell_ranking) + 1)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Most Sensitive (Top 20)")
    st.dataframe(cell_ranking, use_container_width=True, hide_index=True)

with col2:
    # Bar chart of top sensitive
    fig_bar = px.bar(
        cell_ranking,
        x="CELL_LINE",
        y="LN_IC50",
        color="TISSUE",
        title="Top 20 Most Sensitive Cell Lines",
    )
    fig_bar.update_layout(height=400)
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Drug comparison
st.markdown("### Compare with Other Drugs")

comparison_drugs = st.multiselect(
    "Select drugs to compare",
    [d for d in drugs_list if d != selected_drug],
    default=[drugs_list[1]] if len(drugs_list) > 1 else [],
)

if comparison_drugs:
    comparison_data = df[df["DRUG_NAME"].isin([selected_drug] + comparison_drugs)]

    fig_comparison = px.box(
        comparison_data,
        x="DRUG_NAME",
        y="LN_IC50",
        color="DRUG_NAME",
        title="IC50 Comparison Across Drugs",
    )
    fig_comparison.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_comparison, use_container_width=True)

# Export
st.divider()
csv = drug_data.to_csv(index=False)
st.download_button(
    label=f"📥 Download {selected_drug} Data",
    data=csv,
    file_name=f"{selected_drug}_sensitivity_data.csv",
    mime="text/csv",
)
