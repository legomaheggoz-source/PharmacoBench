"""
Feature Engineer

Extracts and transforms features for drug sensitivity prediction:
- Gene expression features (dimensionality reduction)
- Molecular graph features from SMILES
- Multi-omics feature integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Optional imports for molecular features
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Molecular features will be limited.")

# Optional imports for PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import from_smiles
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not available. Graph features disabled.")


class FeatureEngineer:
    """
    Feature engineering for drug sensitivity prediction.

    Handles:
    - Gene expression feature extraction (PCA, variance filtering)
    - SMILES to molecular graph conversion
    - Morgan fingerprints and molecular descriptors
    - Multi-omics feature concatenation
    """

    def __init__(
        self,
        n_gene_features: int = 256,
        morgan_radius: int = 2,
        morgan_bits: int = 1024,
        use_pca: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            n_gene_features: Number of gene expression features after reduction
            morgan_radius: Radius for Morgan fingerprints
            morgan_bits: Number of bits for Morgan fingerprints
            use_pca: Whether to use PCA for gene expression reduction
        """
        self.n_gene_features = n_gene_features
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.use_pca = use_pca

        # Fitted transformers
        self.gene_scaler_: Optional[StandardScaler] = None
        self.gene_pca_: Optional[PCA] = None
        self.selected_genes_: Optional[List[str]] = None

    def fit_gene_features(
        self,
        expression_df: pd.DataFrame,
        variance_threshold: float = 0.1,
    ) -> "FeatureEngineer":
        """
        Fit gene expression feature transformer.

        Args:
            expression_df: DataFrame with genes as columns, samples as rows
            variance_threshold: Minimum variance to keep a gene

        Returns:
            self
        """
        # Filter low-variance genes
        variances = expression_df.var()
        high_var_genes = variances[variances > variance_threshold].index.tolist()
        expression_df = expression_df[high_var_genes]

        logger.info(f"Kept {len(high_var_genes)} genes after variance filtering")

        # Standardize
        self.gene_scaler_ = StandardScaler()
        scaled_data = self.gene_scaler_.fit_transform(expression_df)

        # PCA if configured
        if self.use_pca and scaled_data.shape[1] > self.n_gene_features:
            self.gene_pca_ = PCA(n_components=self.n_gene_features)
            self.gene_pca_.fit(scaled_data)
            explained = self.gene_pca_.explained_variance_ratio_.sum()
            logger.info(f"PCA: {self.n_gene_features} components explain {explained:.2%} variance")

        self.selected_genes_ = high_var_genes
        return self

    def transform_gene_features(
        self,
        expression_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform gene expression data to features.

        Args:
            expression_df: DataFrame with genes as columns

        Returns:
            Feature array (n_samples, n_features)
        """
        if self.gene_scaler_ is None:
            raise ValueError("Not fitted. Call fit_gene_features first.")

        # Select same genes
        if self.selected_genes_:
            # Handle missing genes
            available_genes = [g for g in self.selected_genes_ if g in expression_df.columns]
            if len(available_genes) < len(self.selected_genes_):
                logger.warning(
                    f"Missing {len(self.selected_genes_) - len(available_genes)} genes"
                )
            expression_df = expression_df[available_genes]

        # Standardize
        scaled_data = self.gene_scaler_.transform(expression_df)

        # PCA if fitted
        if self.gene_pca_ is not None:
            scaled_data = self.gene_pca_.transform(scaled_data)

        return scaled_data

    @staticmethod
    def smiles_to_morgan_fingerprint(
        smiles: str,
        radius: int = 2,
        n_bits: int = 1024,
    ) -> Optional[np.ndarray]:
        """
        Convert SMILES string to Morgan fingerprint.

        Args:
            smiles: SMILES string
            radius: Fingerprint radius
            n_bits: Number of bits

        Returns:
            Fingerprint array or None if invalid SMILES
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except Exception as e:
            logger.debug(f"Failed to compute fingerprint for {smiles}: {e}")
            return None

    @staticmethod
    def smiles_to_descriptors(smiles: str) -> Optional[Dict[str, float]]:
        """
        Compute molecular descriptors from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Dict of descriptor names to values, or None if invalid
        """
        if not RDKIT_AVAILABLE:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
            }
        except Exception as e:
            logger.debug(f"Failed to compute descriptors for {smiles}: {e}")
            return None

    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional["Data"]:
        """
        Convert SMILES to PyTorch Geometric graph.

        Args:
            smiles: SMILES string

        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available")
            return None

        try:
            data = from_smiles(smiles, with_hydrogen=False, kekulize=True)
            return data
        except Exception as e:
            logger.debug(f"Failed to create graph for {smiles}: {e}")
            return None

    def compute_drug_features(
        self,
        compounds_df: pd.DataFrame,
        smiles_column: str = "SMILES",
        name_column: str = "DRUG_NAME",
        feature_type: str = "fingerprint",
    ) -> Dict[str, np.ndarray]:
        """
        Compute features for all drugs.

        Args:
            compounds_df: DataFrame with drug information
            smiles_column: Column with SMILES strings
            name_column: Column with drug names
            feature_type: 'fingerprint', 'descriptors', or 'graph'

        Returns:
            Dict mapping drug names to feature arrays
        """
        features = {}

        for _, row in compounds_df.iterrows():
            drug_name = row.get(name_column)
            smiles = row.get(smiles_column)

            if pd.isna(smiles) or not smiles:
                continue

            if feature_type == "fingerprint":
                feat = self.smiles_to_morgan_fingerprint(
                    smiles, self.morgan_radius, self.morgan_bits
                )
            elif feature_type == "descriptors":
                desc = self.smiles_to_descriptors(smiles)
                if desc:
                    feat = np.array(list(desc.values()))
                else:
                    feat = None
            elif feature_type == "graph":
                feat = self.smiles_to_graph(smiles)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if feat is not None:
                features[drug_name] = feat

        logger.info(f"Computed {feature_type} features for {len(features)} drugs")
        return features

    def create_feature_matrix(
        self,
        ic50_df: pd.DataFrame,
        cell_features: Dict[str, np.ndarray],
        drug_features: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix by concatenating cell and drug features.

        Args:
            ic50_df: DataFrame with drug-cell line pairs and IC50
            cell_features: Dict mapping cell line names to feature arrays
            drug_features: Dict mapping drug names to feature arrays

        Returns:
            Tuple of (X features, y targets)
        """
        X_list = []
        y_list = []

        for _, row in ic50_df.iterrows():
            drug_name = row["DRUG_NAME"]
            cell_name = row["CELL_LINE_NAME"]
            ic50 = row["LN_IC50"]

            if drug_name not in drug_features or cell_name not in cell_features:
                continue

            # Concatenate drug and cell features
            drug_feat = drug_features[drug_name]
            cell_feat = cell_features[cell_name]

            if isinstance(drug_feat, np.ndarray) and isinstance(cell_feat, np.ndarray):
                combined = np.concatenate([drug_feat, cell_feat])
                X_list.append(combined)
                y_list.append(ic50)

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Created feature matrix: {X.shape}")
        return X, y

    @staticmethod
    def create_binary_mutation_features(
        mutation_df: pd.DataFrame,
        cell_line_column: str = "CELL_LINE_NAME",
    ) -> Dict[str, np.ndarray]:
        """
        Create binary mutation feature vectors.

        Args:
            mutation_df: DataFrame with mutation data
            cell_line_column: Column with cell line names

        Returns:
            Dict mapping cell line names to binary mutation vectors
        """
        # Pivot to get cell line × gene mutation matrix
        mutation_matrix = mutation_df.pivot_table(
            index=cell_line_column,
            columns="GENE_NAME" if "GENE_NAME" in mutation_df.columns else mutation_df.columns[1],
            values="MUTATION" if "MUTATION" in mutation_df.columns else 1,
            aggfunc="max",
            fill_value=0,
        )

        features = {}
        for cell_line in mutation_matrix.index:
            features[cell_line] = mutation_matrix.loc[cell_line].values.astype(np.float32)

        logger.info(
            f"Created mutation features: {len(features)} cell lines, "
            f"{mutation_matrix.shape[1]} genes"
        )
        return features


def main():
    """Test the feature engineer."""
    logging.basicConfig(level=logging.INFO)

    # Test Morgan fingerprints
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    if RDKIT_AVAILABLE:
        fp = FeatureEngineer.smiles_to_morgan_fingerprint(test_smiles)
        print(f"Morgan fingerprint shape: {fp.shape if fp is not None else 'N/A'}")

        desc = FeatureEngineer.smiles_to_descriptors(test_smiles)
        print(f"Descriptors: {desc}")
    else:
        print("RDKit not available")

    if TORCH_GEOMETRIC_AVAILABLE:
        graph = FeatureEngineer.smiles_to_graph(test_smiles)
        print(f"Graph: {graph}")
    else:
        print("PyTorch Geometric not available")

    # Test gene feature extraction
    np.random.seed(42)
    expression_df = pd.DataFrame(
        np.random.randn(100, 500),
        columns=[f"Gene_{i}" for i in range(500)],
    )

    fe = FeatureEngineer(n_gene_features=50)
    fe.fit_gene_features(expression_df)
    features = fe.transform_gene_features(expression_df)
    print(f"Gene features shape: {features.shape}")


if __name__ == "__main__":
    main()
