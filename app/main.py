"""
PharmacoBench - Main Streamlit Application

Comparative Auditor for In-Silico Drug Sensitivity Prediction
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PharmacoBench",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/legomaheggoz-source/PharmacoBench",
        "Report a bug": "https://github.com/legomaheggoz-source/PharmacoBench/issues",
        "About": "PharmacoBench - Benchmarking ML models for drug sensitivity prediction"
    }
)

# Load custom CSS
def load_css():
    """Load Aurora-inspired custom CSS."""
    css = """
    <style>
    /* Aurora Solar-inspired theme */
    :root {
        --primary-bg: #f8fafc;
        --card-bg: #ffffff;
        --primary-accent: #3b82f6;
        --secondary-accent: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }

    /* Main container */
    .main {
        padding: 1rem 2rem;
    }

    /* Card styling */
    .metric-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .app-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }

    .app-header p {
        color: rgba(255,255,255,0.9);
        margin: 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-bg);
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-accent);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #2563eb;
    }

    /* Table styling */
    .dataframe {
        border: none !important;
    }

    .dataframe th {
        background-color: var(--primary-bg) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-success {
        background-color: #dcfce7;
        color: #166534;
    }

    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }

    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Render application header."""
    st.markdown("""
    <div class="app-header">
        <h1>🧬 PharmacoBench</h1>
        <p>Comparative Auditor for In-Silico Drug Sensitivity Prediction</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/dna-helix.png", width=80)
        st.title("PharmacoBench")
        st.caption("v1.0.0")

        st.divider()

        # Navigation links (handled by Streamlit's built-in page navigation)
        st.markdown("### Navigation")
        st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", icon="📊")
        st.page_link("pages/2_Data_Explorer.py", label="🔍 Data Explorer", icon="🔍")
        st.page_link("pages/3_Model_Training.py", label="🤖 Model Training", icon="🤖")
        st.page_link("pages/4_Benchmark_Results.py", label="📈 Benchmark Results", icon="📈")
        st.page_link("pages/5_Drug_Analysis.py", label="💊 Drug Analysis", icon="💊")
        st.page_link("pages/6_Cell_Line_Analysis.py", label="🧫 Cell Line Analysis", icon="🧫")

        st.divider()

        # Quick stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "8")
        with col2:
            st.metric("Splits", "4")

        st.divider()

        # Links
        st.markdown("### Resources")
        st.markdown("[📚 Documentation](https://github.com/legomaheggoz-source/PharmacoBench)")
        st.markdown("[🐛 Report Issue](https://github.com/legomaheggoz-source/PharmacoBench/issues)")
        st.markdown("[📊 GDSC Data](https://www.cancerrxgene.org/)")


def main():
    """Main application entry point."""
    # Load custom CSS
    load_css()

    # Render header
    render_header()

    # Main content
    st.markdown("## Welcome to PharmacoBench")

    st.markdown("""
    PharmacoBench is a comprehensive benchmarking platform for evaluating machine learning
    architectures on drug sensitivity prediction. It addresses a critical challenge in drug
    discovery: **90% of drugs fail in clinical trials**, costing ~$2.6B per successful drug.
    """)

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">ML Models</div>
            <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #64748b;">
                From Ridge Regression to Graph Neural Networks
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Split Strategies</div>
            <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #64748b;">
                Random, Drug-Blind, Cell-Blind, Disjoint
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1000+</div>
            <div class="metric-label">Cell Lines</div>
            <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #64748b;">
                GDSC1 + GDSC2 combined datasets
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Models overview
    st.markdown("### Models Benchmarked")

    models_data = {
        "Model": ["Ridge Regression", "ElasticNet", "Random Forest", "XGBoost",
                  "LightGBM", "MLP", "GraphDRP", "DeepCDR"],
        "Type": ["Linear", "Linear", "Ensemble", "Gradient Boosting",
                 "Gradient Boosting", "Deep Learning", "GNN", "Hybrid GNN"],
        "Library": ["sklearn", "sklearn", "sklearn", "xgboost",
                    "lightgbm", "PyTorch", "PyTorch Geometric", "PyTorch Geometric"],
        "Purpose": ["Interpretability baseline", "Sparsity + regularization",
                    "Non-linear baseline", "Strong tabular performance",
                    "Fast training", "Neural network baseline",
                    "Drug structure encoding", "State-of-the-art multi-omics"],
    }

    st.dataframe(
        models_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Getting started
    st.markdown("### Getting Started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 1. Explore the Data
        Navigate to **Data Explorer** to:
        - View dataset statistics
        - Explore IC50 distributions
        - Analyze drug and cell line coverage
        """)

        st.markdown("""
        #### 2. Train Models
        Use **Model Training** to:
        - Select models to benchmark
        - Configure hyperparameters
        - Run training with progress tracking
        """)

    with col2:
        st.markdown("""
        #### 3. View Results
        Check **Benchmark Results** to:
        - Compare model performance
        - View metric heatmaps
        - Analyze across split strategies
        """)

        st.markdown("""
        #### 4. Deep Dive
        Use **Drug/Cell Line Analysis** to:
        - View 2D molecular structures
        - Explore sensitivity profiles
        - Identify patterns
        """)

    # Footer
    st.divider()
    st.caption("""
    PharmacoBench | MIT License | Built with Streamlit |
    [GitHub](https://github.com/legomaheggoz-source/PharmacoBench) |
    Data: [GDSC](https://www.cancerrxgene.org/)
    """)


if __name__ == "__main__":
    main()
