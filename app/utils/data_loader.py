"""
Data Loader Utility

Provides unified data loading for Streamlit pages.
Supports both demo data and real GDSC data with caching.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for PharmacoBench.

    Supports:
    - Demo data generation for quick testing
    - Real GDSC data loading (when available)
    - Caching for performance
    - Fallback to demo if real data unavailable
    """

    def __init__(self, use_real_data: bool = False, cache_dir: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            use_real_data: Whether to try loading real GDSC data
            cache_dir: Directory for cached data files
        """
        self.use_real_data = use_real_data
        self.cache_dir = cache_dir or Path("data/cache")

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_demo_ic50_data(n_samples: int = 200000) -> pd.DataFrame:
        """
        Generate demo IC50 data similar to GDSC format.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with drug sensitivity data
        """
        np.random.seed(42)

        n_drugs = 300
        n_cell_lines = 900

        drugs = [f"Drug_{i}" for i in range(n_drugs)]
        cell_lines = [f"CellLine_{i}" for i in range(n_cell_lines)]
        tissues = [
            "Lung", "Breast", "Colon", "Blood", "Brain",
            "Skin", "Liver", "Kidney", "Ovary", "Pancreas"
        ]

        data = {
            "DRUG_NAME": np.random.choice(drugs, n_samples),
            "CELL_LINE_NAME": np.random.choice(cell_lines, n_samples),
            "LN_IC50": np.random.normal(-2, 1.5, n_samples),
            "TISSUE": np.random.choice(tissues, n_samples),
            "SOURCE": np.random.choice(["GDSC1", "GDSC2"], n_samples, p=[0.6, 0.4]),
        }

        return pd.DataFrame(data)

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_demo_benchmark_results() -> pd.DataFrame:
        """
        Generate demo benchmark results for all models and splits.

        Returns:
            DataFrame with benchmark metrics
        """
        models = [
            "Ridge", "ElasticNet", "Random Forest", "XGBoost",
            "LightGBM", "MLP", "GraphDRP", "DeepCDR"
        ]
        splits = ["Random", "Drug-Blind", "Cell-Blind", "Disjoint"]

        data = []
        np.random.seed(42)

        base_rmse = {
            "Ridge": 1.2, "ElasticNet": 1.18, "Random Forest": 1.05,
            "XGBoost": 0.98, "LightGBM": 0.99, "MLP": 1.0,
            "GraphDRP": 0.92, "DeepCDR": 0.88
        }

        split_penalty = {
            "Random": 0, "Drug-Blind": 0.15, "Cell-Blind": 0.12, "Disjoint": 0.25
        }

        for model in models:
            for split in splits:
                rmse = base_rmse[model] + split_penalty[split] + np.random.uniform(-0.05, 0.05)
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

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_demo_drug_data() -> pd.DataFrame:
        """
        Generate demo drug data with SMILES and targets.

        Returns:
            DataFrame with drug information and sensitivity
        """
        np.random.seed(42)

        drugs = [
            {"name": "Gefitinib", "target": "EGFR", "phase": "Approved"},
            {"name": "Erlotinib", "target": "EGFR", "phase": "Approved"},
            {"name": "Lapatinib", "target": "EGFR/HER2", "phase": "Approved"},
            {"name": "Imatinib", "target": "BCR-ABL", "phase": "Approved"},
            {"name": "Sorafenib", "target": "Multi-kinase", "phase": "Approved"},
            {"name": "Vemurafenib", "target": "BRAF", "phase": "Approved"},
            {"name": "Crizotinib", "target": "ALK/MET", "phase": "Approved"},
            {"name": "Dasatinib", "target": "BCR-ABL/SRC", "phase": "Approved"},
            {"name": "Nilotinib", "target": "BCR-ABL", "phase": "Approved"},
            {"name": "Bosutinib", "target": "BCR-ABL/SRC", "phase": "Approved"},
        ]

        n_cell_lines = 100
        tissues = ["Lung", "Breast", "Blood", "Colon", "Skin"]

        data = []
        for drug in drugs:
            for i in range(n_cell_lines):
                base_ic50 = -2 + np.random.uniform(-0.5, 0.5)
                tissue = np.random.choice(tissues)

                # Drug-specific effects
                if drug["name"] == "Gefitinib" and tissue == "Lung":
                    base_ic50 -= 1
                elif drug["name"] == "Imatinib" and tissue == "Blood":
                    base_ic50 -= 1.5

                data.append({
                    "DRUG_NAME": drug["name"],
                    "TARGET": drug["target"],
                    "PHASE": drug["phase"],
                    "CELL_LINE": f"Cell_{i}",
                    "TISSUE": tissue,
                    "LN_IC50": base_ic50 + np.random.normal(0, 0.5),
                })

        return pd.DataFrame(data)

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_demo_cell_line_data() -> Tuple[pd.DataFrame, list]:
        """
        Generate demo cell line data with mutations and expression.

        Returns:
            Tuple of (cell_line_df, mutation_list)
        """
        np.random.seed(42)

        tissues = [
            "Lung", "Breast", "Blood", "Colon", "Skin",
            "Brain", "Liver", "Kidney", "Ovary", "Pancreas"
        ]

        subtypes = {
            "Lung": ["NSCLC", "SCLC", "Adenocarcinoma"],
            "Breast": ["Triple Negative", "HER2+", "ER+"],
            "Blood": ["AML", "CML", "ALL", "CLL"],
            "Colon": ["Adenocarcinoma", "Carcinoma"],
            "Skin": ["Melanoma", "SCC", "BCC"],
            "Brain": ["Glioblastoma", "Astrocytoma"],
            "Liver": ["HCC", "Cholangiocarcinoma"],
            "Kidney": ["RCC", "Wilms"],
            "Ovary": ["Serous", "Mucinous"],
            "Pancreas": ["PDAC", "Neuroendocrine"],
        }

        mutations = ["KRAS", "TP53", "EGFR", "BRAF", "PIK3CA", "PTEN", "RB1", "MYC", "BRCA1", "BRCA2"]

        cell_lines = []
        n_cell_lines = 100

        for i in range(n_cell_lines):
            tissue = np.random.choice(tissues)
            subtype = np.random.choice(subtypes[tissue])

            mutation_profile = {mut: np.random.choice([0, 1], p=[0.8, 0.2]) for mut in mutations}
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

    def load_gdsc_data(self) -> Optional[pd.DataFrame]:
        """
        Try to load real GDSC data using the downloader.

        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Lazy import to avoid loading heavy dependencies
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from data.downloader import GDSCDownloader

            downloader = GDSCDownloader()
            df = downloader.load_combined_ic50()

            if df is not None and len(df) > 0:
                return df

        except ImportError as e:
            logger.warning(f"Could not import GDSC downloader: {e}")
        except Exception as e:
            logger.warning(f"Could not load GDSC data: {e}")

        return None


# Convenience functions for direct use in pages
@st.cache_data(ttl=3600)
def get_ic50_data(use_real: bool = False) -> pd.DataFrame:
    """
    Get IC50 data (demo or real).

    Args:
        use_real: Whether to try loading real GDSC data

    Returns:
        DataFrame with IC50 data
    """
    loader = DataLoader(use_real_data=use_real)

    if use_real:
        real_data = loader.load_gdsc_data()
        if real_data is not None:
            return real_data

    return DataLoader.load_demo_ic50_data()


@st.cache_data(ttl=3600)
def get_benchmark_results() -> pd.DataFrame:
    """Get benchmark results (currently demo data)."""
    return DataLoader.load_demo_benchmark_results()


@st.cache_data(ttl=3600)
def get_drug_data() -> pd.DataFrame:
    """Get drug data with sensitivity profiles."""
    return DataLoader.load_demo_drug_data()


@st.cache_data(ttl=3600)
def get_cell_line_data() -> Tuple[pd.DataFrame, list]:
    """Get cell line data with mutations."""
    return DataLoader.load_demo_cell_line_data()


def search_dataframe(
    df: pd.DataFrame,
    query: str,
    columns: list,
    max_results: int = 50,
) -> pd.DataFrame:
    """
    Search dataframe using fuzzy matching.

    Args:
        df: DataFrame to search
        query: Search query
        columns: Columns to search in
        max_results: Maximum results to return

    Returns:
        Filtered DataFrame matching query
    """
    if not query:
        return df.head(max_results)

    query_lower = query.lower()
    mask = pd.Series([False] * len(df))

    for col in columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False)

    return df[mask].head(max_results)
