"""
Data Preprocessor

Preprocesses GDSC drug sensitivity data including IC50 transformation,
missing value handling, and normalization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses GDSC data for machine learning.

    Transformations include:
    - Log IC50 transformation (natural log)
    - Missing value handling
    - Cell line filtering (minimum drug coverage)
    - Gene expression normalization (Z-score)
    - Outlier removal
    """

    def __init__(
        self,
        min_drugs_per_cell_line: int = 50,
        min_cell_lines_per_drug: int = 50,
        ic50_column: str = "LN_IC50",
        remove_outliers: bool = True,
        outlier_zscore_threshold: float = 4.0,
    ):
        """
        Initialize preprocessor.

        Args:
            min_drugs_per_cell_line: Minimum drugs tested per cell line to include
            min_cell_lines_per_drug: Minimum cell lines tested per drug to include
            ic50_column: Column name for IC50 values
            remove_outliers: Whether to remove outliers based on Z-score
            outlier_zscore_threshold: Z-score threshold for outlier removal
        """
        self.min_drugs_per_cell_line = min_drugs_per_cell_line
        self.min_cell_lines_per_drug = min_cell_lines_per_drug
        self.ic50_column = ic50_column
        self.remove_outliers = remove_outliers
        self.outlier_zscore_threshold = outlier_zscore_threshold

        # Statistics computed during fit
        self.ic50_mean_: Optional[float] = None
        self.ic50_std_: Optional[float] = None
        self.valid_cell_lines_: Optional[List[str]] = None
        self.valid_drugs_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor on training data.

        Computes statistics needed for transformation.

        Args:
            df: DataFrame with IC50 data

        Returns:
            self
        """
        df = self._validate_dataframe(df)

        # Compute IC50 statistics (excluding NaN)
        ic50_values = df[self.ic50_column].dropna()
        self.ic50_mean_ = ic50_values.mean()
        self.ic50_std_ = ic50_values.std()

        # Identify valid cell lines (sufficient drug coverage)
        cell_line_counts = df.groupby("CELL_LINE_NAME")["DRUG_NAME"].nunique()
        self.valid_cell_lines_ = cell_line_counts[
            cell_line_counts >= self.min_drugs_per_cell_line
        ].index.tolist()

        # Identify valid drugs (sufficient cell line coverage)
        drug_counts = df.groupby("DRUG_NAME")["CELL_LINE_NAME"].nunique()
        self.valid_drugs_ = drug_counts[
            drug_counts >= self.min_cell_lines_per_drug
        ].index.tolist()

        logger.info(
            f"Fit complete. Valid cell lines: {len(self.valid_cell_lines_)}, "
            f"Valid drugs: {len(self.valid_drugs_)}"
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted statistics.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if self.ic50_mean_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        df = self._validate_dataframe(df)
        df = df.copy()

        # Filter to valid cell lines and drugs
        df = df[df["CELL_LINE_NAME"].isin(self.valid_cell_lines_)]
        df = df[df["DRUG_NAME"].isin(self.valid_drugs_)]

        # Remove missing IC50 values
        initial_count = len(df)
        df = df.dropna(subset=[self.ic50_column])
        logger.info(f"Removed {initial_count - len(df)} rows with missing IC50")

        # Remove outliers if configured
        if self.remove_outliers:
            df = self._remove_outliers(df)

        logger.info(f"Transform complete. {len(df)} records remaining.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns exist."""
        required_cols = ["DRUG_NAME", "CELL_LINE_NAME", self.ic50_column]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on Z-score threshold."""
        ic50_zscore = np.abs(stats.zscore(df[self.ic50_column]))
        mask = ic50_zscore < self.outlier_zscore_threshold
        removed = (~mask).sum()
        if removed > 0:
            logger.info(f"Removed {removed} outliers (Z > {self.outlier_zscore_threshold})")
        return df[mask]

    @staticmethod
    def log_transform_ic50(
        df: pd.DataFrame,
        ic50_column: str = "IC50",
        output_column: str = "LN_IC50",
    ) -> pd.DataFrame:
        """
        Apply natural log transformation to IC50 values.

        Args:
            df: DataFrame with IC50 column
            ic50_column: Name of IC50 column
            output_column: Name for output column

        Returns:
            DataFrame with log-transformed IC50
        """
        df = df.copy()
        # Handle zeros/negatives by clipping to small positive value
        ic50_values = df[ic50_column].clip(lower=1e-10)
        df[output_column] = np.log(ic50_values)
        return df

    @staticmethod
    def normalize_gene_expression(
        expression_df: pd.DataFrame,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Normalize gene expression data.

        Args:
            expression_df: DataFrame with genes as columns, samples as rows
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized DataFrame
        """
        expression_df = expression_df.copy()

        if method == "zscore":
            # Z-score normalization per gene
            means = expression_df.mean()
            stds = expression_df.std()
            stds = stds.replace(0, 1)  # Avoid division by zero
            expression_df = (expression_df - means) / stds

        elif method == "minmax":
            # Min-max scaling per gene
            mins = expression_df.min()
            maxs = expression_df.max()
            ranges = maxs - mins
            ranges = ranges.replace(0, 1)
            expression_df = (expression_df - mins) / ranges

        elif method == "robust":
            # Robust scaling using median and IQR
            medians = expression_df.median()
            q75 = expression_df.quantile(0.75)
            q25 = expression_df.quantile(0.25)
            iqr = q75 - q25
            iqr = iqr.replace(0, 1)
            expression_df = (expression_df - medians) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return expression_df

    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for the dataset.

        Args:
            df: DataFrame with IC50 data

        Returns:
            Dict with statistics
        """
        df = self._validate_dataframe(df)

        stats_dict = {
            "total_records": len(df),
            "unique_drugs": df["DRUG_NAME"].nunique(),
            "unique_cell_lines": df["CELL_LINE_NAME"].nunique(),
            "ic50_stats": {
                "mean": df[self.ic50_column].mean(),
                "std": df[self.ic50_column].std(),
                "min": df[self.ic50_column].min(),
                "max": df[self.ic50_column].max(),
                "median": df[self.ic50_column].median(),
            },
            "missing_values": df[self.ic50_column].isna().sum(),
            "coverage": {
                "drugs_per_cell_line": df.groupby("CELL_LINE_NAME")["DRUG_NAME"].nunique().describe().to_dict(),
                "cell_lines_per_drug": df.groupby("DRUG_NAME")["CELL_LINE_NAME"].nunique().describe().to_dict(),
            },
        }

        return stats_dict

    def create_drug_cell_matrix(
        self,
        df: pd.DataFrame,
        value_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create drug × cell line matrix.

        Args:
            df: DataFrame with IC50 data
            value_column: Column to use as values (default: ic50_column)

        Returns:
            DataFrame with drugs as rows, cell lines as columns
        """
        if value_column is None:
            value_column = self.ic50_column

        df = self._validate_dataframe(df)

        matrix = df.pivot_table(
            index="DRUG_NAME",
            columns="CELL_LINE_NAME",
            values=value_column,
            aggfunc="mean",
        )

        logger.info(f"Created matrix: {matrix.shape[0]} drugs × {matrix.shape[1]} cell lines")
        return matrix


def main():
    """Test the preprocessor."""
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    data = {
        "DRUG_NAME": np.random.choice([f"Drug_{i}" for i in range(10)], n_samples),
        "CELL_LINE_NAME": np.random.choice([f"Cell_{i}" for i in range(20)], n_samples),
        "LN_IC50": np.random.normal(-2, 1, n_samples),
    }
    df = pd.DataFrame(data)

    # Add some missing values
    df.loc[np.random.choice(n_samples, 50), "LN_IC50"] = np.nan

    print("Sample data:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    # Test preprocessor
    preprocessor = DataPreprocessor(
        min_drugs_per_cell_line=3,
        min_cell_lines_per_drug=3,
    )

    processed_df = preprocessor.fit_transform(df)
    print(f"\nProcessed shape: {processed_df.shape}")

    stats = preprocessor.get_data_statistics(processed_df)
    print(f"\nStatistics:")
    print(f"  Unique drugs: {stats['unique_drugs']}")
    print(f"  Unique cell lines: {stats['unique_cell_lines']}")
    print(f"  IC50 mean: {stats['ic50_stats']['mean']:.3f}")


if __name__ == "__main__":
    main()
