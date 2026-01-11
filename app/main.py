"""
PharmacoBench - Main Streamlit Application

Comparative Auditor for In-Silico Drug Sensitivity Prediction
"""

import sys
import traceback
from pathlib import Path

# Print startup info for debugging HuggingFace issues
print("=" * 50, flush=True)
print("PharmacoBench Starting...", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working directory: {Path.cwd()}", flush=True)
print("=" * 50, flush=True)

try:
    import streamlit as st
    print("Streamlit imported successfully", flush=True)
except Exception as e:
    print(f"FATAL: Failed to import streamlit: {e}", flush=True)
    traceback.print_exc()
    raise

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root added to path: {project_root}", flush=True)

# Dynamic counts - use lightweight approach to avoid heavy imports at startup
# This prevents loading ML libraries (sklearn, xgboost, etc.) just for counts
NUM_MODELS = 8  # Ridge, ElasticNet, RF, XGBoost, LightGBM, MLP, GraphDRP, DeepCDR
NUM_SPLITS = 4  # Random, Drug-Blind, Cell-Blind, Disjoint

print("Setting page config...", flush=True)

# Page configuration
st.set_page_config(
    page_title="PharmacoBench",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/legomaheggoz-source/PharmacoBench",
        "Report a bug": "https://github.com/legomaheggoz-source/PharmacoBench/issues",
        "About": "PharmacoBench - Benchmarking ML models for drug sensitivity prediction"
    }
)


def load_css():
    """Load Aurora Solar-inspired light theme CSS from external file."""
    from pathlib import Path

    css_path = Path(__file__).parent / "styles" / "aurora.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback to inline CSS if file not found
        _load_fallback_css()


def _load_fallback_css():
    """Fallback inline CSS if external file not found."""
    css = """
    <style>
    /* Aurora Solar-inspired light theme (fallback) */

    /* Root variables */
    :root {
        --aurora-white: #ffffff;
        --aurora-light: #f9fafb;
        --aurora-gray-50: #f3f4f6;
        --aurora-gray-100: #e5e7eb;
        --aurora-gray-200: #d1d5db;
        --aurora-gray-500: #6b7280;
        --aurora-gray-700: #374151;
        --aurora-gray-900: #111827;
        --aurora-blue: #0ea5e9;
        --aurora-blue-light: #e0f2fe;
        --aurora-green: #10b981;
        --aurora-green-light: #d1fae5;
    }

    /* Global styles */
    .stApp {
        background-color: var(--aurora-white);
    }

    /* Remove default padding for cleaner look */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--aurora-gray-900);
        font-weight: 600;
    }

    /* Clean metric cards */
    [data-testid="stMetric"] {
        background-color: var(--aurora-light);
        border: 1px solid var(--aurora-gray-100);
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--aurora-gray-500);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricValue"] {
        color: var(--aurora-gray-900);
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--aurora-light);
        border-right: 1px solid var(--aurora-gray-100);
    }

    [data-testid="stSidebar"] h1 {
        color: var(--aurora-gray-900);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--aurora-blue);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background-color: #0284c7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: var(--aurora-blue);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--aurora-light);
        border-radius: 8px;
        color: var(--aurora-gray-700);
        border: 1px solid var(--aurora-gray-100);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--aurora-blue-light);
        color: var(--aurora-blue);
        border-color: var(--aurora-blue);
    }

    /* Selectbox and inputs */
    [data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--aurora-gray-100);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--aurora-light);
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: var(--aurora-gray-100);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }

    /* Success message */
    .element-container .stSuccess {
        background-color: var(--aurora-green-light);
        color: #065f46;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom hero section */
    .hero-section {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }

    .hero-section h1 {
        color: white;
        margin-bottom: 0.75rem;
        font-size: 2rem;
    }

    .hero-section p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0;
    }

    /* Feature cards */
    .feature-card {
        background: var(--aurora-light);
        border: 1px solid var(--aurora-gray-100);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }

    .feature-card h3 {
        color: var(--aurora-blue);
        font-size: 2.5rem;
        margin-bottom: 0.25rem;
    }

    .feature-card .label {
        color: var(--aurora-gray-500);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    .feature-card p {
        color: var(--aurora-gray-700);
        font-size: 0.875rem;
        margin-top: 0.75rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    print("main() started, loading CSS...", flush=True)
    try:
        load_css()
        print("CSS loaded successfully", flush=True)
    except Exception as e:
        print(f"WARNING: CSS loading failed: {e}", flush=True)
        # Continue without custom CSS

    print("Rendering sidebar...", flush=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 💊 PharmacoBench")
        st.caption("Drug Sensitivity Benchmarking")

        st.divider()

        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", str(NUM_MODELS), help="ML models available for benchmarking")
        with col2:
            st.metric("Splits", str(NUM_SPLITS), help="Evaluation split strategies")

        st.divider()

        st.markdown("### Resources")
        st.markdown("📚 [Documentation](https://github.com/legomaheggoz-source/PharmacoBench)")
        st.markdown("📊 [GDSC Data](https://www.cancerrxgene.org/)")
        st.markdown("🐛 [Report Issue](https://github.com/legomaheggoz-source/PharmacoBench/issues)")

    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1>💊 PharmacoBench</h1>
        <p>Comparative Auditor for In-Silico Drug Sensitivity Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    PharmacoBench is a comprehensive benchmarking platform for evaluating machine learning
    architectures on drug sensitivity prediction. It addresses a critical challenge in drug
    discovery: **90% of drugs fail in clinical trials**, costing ~$2.6B per successful drug.
    """)

    st.divider()

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{NUM_MODELS}</h3>
            <div class="label">ML Models</div>
            <p>From Ridge Regression baseline to state-of-the-art Graph Neural Networks</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{NUM_SPLITS}</h3>
            <div class="label">Split Strategies</div>
            <p>Random, Drug-Blind, Cell-Blind, and Disjoint evaluation methods</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>1000+</h3>
            <div class="label">Cell Lines</div>
            <p>GDSC1 + GDSC2 combined with ~495 drugs tested</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Models table
    st.markdown("### Models Benchmarked")

    import pandas as pd
    models_df = pd.DataFrame({
        "Model": ["Ridge Regression", "ElasticNet", "Random Forest", "XGBoost",
                  "LightGBM", "MLP", "GraphDRP", "DeepCDR"],
        "Type": ["Linear", "Linear", "Ensemble", "Boosting",
                 "Boosting", "Neural Net", "GNN", "Hybrid GNN"],
        "Purpose": ["Interpretability baseline", "Feature selection",
                    "Non-linear baseline", "Strong tabular performance",
                    "Fast training", "Deep learning baseline",
                    "Drug structure encoding", "State-of-the-art"]
    })

    st.dataframe(models_df, use_container_width=True, hide_index=True)

    st.divider()

    # Getting started
    st.markdown("### Getting Started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📊 Dashboard**
        View benchmark results overview and key performance metrics

        **🔍 Data Explorer**
        Explore GDSC dataset statistics, distributions, and coverage

        **🤖 Model Training**
        Configure and run model training with custom hyperparameters
        """)

    with col2:
        st.markdown("""
        **📈 Benchmark Results**
        Compare model performance across different split strategies

        **💊 Drug Analysis**
        Analyze individual drugs, view structures and sensitivity profiles

        **🧫 Cell Line Analysis**
        Explore cell line characteristics and drug response patterns
        """)

    # Footer
    st.divider()
    st.caption("""
    PharmacoBench | MIT License | Built with Streamlit
    [GitHub](https://github.com/legomaheggoz-source/PharmacoBench) •
    [GDSC Data](https://www.cancerrxgene.org/)
    """)


if __name__ == "__main__":
    try:
        print("Calling main()...", flush=True)
        main()
        print("main() completed successfully", flush=True)
    except Exception as e:
        print(f"FATAL ERROR in main(): {e}", flush=True)
        traceback.print_exc()
        st.error(f"Application error: {e}")
        raise
