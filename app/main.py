"""
PharmacoBench - Main Streamlit Application

Comparative Auditor for In-Silico Drug Sensitivity Prediction
"""

import streamlit as st

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
    """Load Aurora Solar-inspired light theme CSS."""
    css = """
    <style>
    /* Aurora Solar-inspired light theme */

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
    load_css()

    # Sidebar
    with st.sidebar:
        st.markdown("## 💊 PharmacoBench")
        st.caption("Drug Sensitivity Benchmarking")

        st.divider()

        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "8")
        with col2:
            st.metric("Splits", "4")

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
        st.markdown("""
        <div class="feature-card">
            <h3>8</h3>
            <div class="label">ML Models</div>
            <p>From Ridge Regression baseline to state-of-the-art Graph Neural Networks</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>4</h3>
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
    main()
