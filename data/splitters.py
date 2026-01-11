"""
Data Splitters

Implements 4 train/test splitting strategies for drug sensitivity prediction:
1. Random Split - Standard random split
2. Drug-Blind Split - Test drugs never seen in training
3. Cell-Blind Split - Test cell lines never seen in training
4. Disjoint Split - Neither drugs nor cell lines overlap
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for split results."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_type: str
    metadata: Dict

    @property
    def train_drugs(self) -> List[str]:
        return self.train_df["DRUG_NAME"].unique().tolist()

    @property
    def test_drugs(self) -> List[str]:
        return self.test_df["DRUG_NAME"].unique().tolist()

    @property
    def train_cell_lines(self) -> List[str]:
        return self.train_df["CELL_LINE_NAME"].unique().tolist()

    @property
    def test_cell_lines(self) -> List[str]:
        return self.test_df["CELL_LINE_NAME"].unique().tolist()

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Split: {self.split_type}\n"
            f"  Train: {len(self.train_df)} samples, "
            f"{len(self.train_drugs)} drugs, {len(self.train_cell_lines)} cell lines\n"
            f"  Val: {len(self.val_df)} samples\n"
            f"  Test: {len(self.test_df)} samples, "
            f"{len(self.test_drugs)} drugs, {len(self.test_cell_lines)} cell lines"
        )


class DataSplitter:
    """
    Implements multiple train/test splitting strategies.

    Splitting Strategies:
    - random: Standard random split (baseline, overly optimistic)
    - drug_blind: Test drugs never appear in training (new drug development)
    - cell_blind: Test cell lines never appear in training (personalized medicine)
    - disjoint: Neither drugs nor cell lines overlap (strictest generalization)
    """

    VALID_STRATEGIES = ["random", "drug_blind", "cell_blind", "disjoint"]

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize splitter.

        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set (from training)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        strategy: str = "random",
    ) -> SplitResult:
        """
        Split data using specified strategy.

        Args:
            df: DataFrame with columns DRUG_NAME, CELL_LINE_NAME, LN_IC50
            strategy: One of 'random', 'drug_blind', 'cell_blind', 'disjoint'

        Returns:
            SplitResult with train, val, test DataFrames
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Valid options: {self.VALID_STRATEGIES}"
            )

        self._validate_dataframe(df)

        if strategy == "random":
            return self._random_split(df)
        elif strategy == "drug_blind":
            return self._drug_blind_split(df)
        elif strategy == "cell_blind":
            return self._cell_blind_split(df)
        elif strategy == "disjoint":
            return self._disjoint_split(df)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required = ["DRUG_NAME", "CELL_LINE_NAME"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _random_split(self, df: pd.DataFrame) -> SplitResult:
        """
        Standard random split.

        Drug-cell pairs are randomly assigned to train/val/test.
        """
        np.random.seed(self.random_state)

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Second split: train vs val
        effective_val_size = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=effective_val_size,
            random_state=self.random_state,
        )

        metadata = {
            "overlap": {
                "drugs": bool(set(train_df["DRUG_NAME"]) & set(test_df["DRUG_NAME"])),
                "cell_lines": bool(set(train_df["CELL_LINE_NAME"]) & set(test_df["CELL_LINE_NAME"])),
            }
        }

        logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return SplitResult(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            split_type="random",
            metadata=metadata,
        )

    def _drug_blind_split(self, df: pd.DataFrame) -> SplitResult:
        """
        Drug-blind split.

        Test set contains drugs that never appear in training.
        Simulates predicting response for novel drugs.
        """
        np.random.seed(self.random_state)

        # Get unique drugs and shuffle
        drugs = df["DRUG_NAME"].unique()
        np.random.shuffle(drugs)

        # Split drugs
        n_test = int(len(drugs) * self.test_size)
        n_val = int(len(drugs) * self.val_size)

        test_drugs = set(drugs[:n_test])
        val_drugs = set(drugs[n_test:n_test + n_val])
        train_drugs = set(drugs[n_test + n_val:])

        # Split data based on drug assignment
        test_df = df[df["DRUG_NAME"].isin(test_drugs)]
        val_df = df[df["DRUG_NAME"].isin(val_drugs)]
        train_df = df[df["DRUG_NAME"].isin(train_drugs)]

        metadata = {
            "test_drugs": list(test_drugs),
            "n_test_drugs": len(test_drugs),
            "n_train_drugs": len(train_drugs),
            "overlap": {
                "drugs": False,  # By construction
                "cell_lines": bool(set(train_df["CELL_LINE_NAME"]) & set(test_df["CELL_LINE_NAME"])),
            }
        }

        logger.info(
            f"Drug-blind split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
            f"test_drugs={len(test_drugs)}"
        )

        return SplitResult(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            split_type="drug_blind",
            metadata=metadata,
        )

    def _cell_blind_split(self, df: pd.DataFrame) -> SplitResult:
        """
        Cell-blind split.

        Test set contains cell lines that never appear in training.
        Simulates predicting response for novel cell lines/patients.
        """
        np.random.seed(self.random_state)

        # Get unique cell lines and shuffle
        cell_lines = df["CELL_LINE_NAME"].unique()
        np.random.shuffle(cell_lines)

        # Split cell lines
        n_test = int(len(cell_lines) * self.test_size)
        n_val = int(len(cell_lines) * self.val_size)

        test_cell_lines = set(cell_lines[:n_test])
        val_cell_lines = set(cell_lines[n_test:n_test + n_val])
        train_cell_lines = set(cell_lines[n_test + n_val:])

        # Split data based on cell line assignment
        test_df = df[df["CELL_LINE_NAME"].isin(test_cell_lines)]
        val_df = df[df["CELL_LINE_NAME"].isin(val_cell_lines)]
        train_df = df[df["CELL_LINE_NAME"].isin(train_cell_lines)]

        metadata = {
            "test_cell_lines": list(test_cell_lines),
            "n_test_cell_lines": len(test_cell_lines),
            "n_train_cell_lines": len(train_cell_lines),
            "overlap": {
                "drugs": bool(set(train_df["DRUG_NAME"]) & set(test_df["DRUG_NAME"])),
                "cell_lines": False,  # By construction
            }
        }

        logger.info(
            f"Cell-blind split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
            f"test_cell_lines={len(test_cell_lines)}"
        )

        return SplitResult(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            split_type="cell_blind",
            metadata=metadata,
        )

    def _disjoint_split(self, df: pd.DataFrame) -> SplitResult:
        """
        Disjoint split.

        Neither drugs nor cell lines in test set appear in training.
        Strictest evaluation of generalization.
        """
        np.random.seed(self.random_state)

        # Get unique drugs and cell lines
        drugs = df["DRUG_NAME"].unique()
        cell_lines = df["CELL_LINE_NAME"].unique()

        np.random.shuffle(drugs)
        np.random.shuffle(cell_lines)

        # Use smaller fractions for disjoint (since we're removing both)
        disjoint_frac = np.sqrt(self.test_size)  # e.g., 0.2 -> 0.45

        n_test_drugs = int(len(drugs) * disjoint_frac)
        n_test_cells = int(len(cell_lines) * disjoint_frac)

        n_val_drugs = int(len(drugs) * np.sqrt(self.val_size))
        n_val_cells = int(len(cell_lines) * np.sqrt(self.val_size))

        test_drugs = set(drugs[:n_test_drugs])
        test_cell_lines = set(cell_lines[:n_test_cells])

        val_drugs = set(drugs[n_test_drugs:n_test_drugs + n_val_drugs])
        val_cell_lines = set(cell_lines[n_test_cells:n_test_cells + n_val_cells])

        train_drugs = set(drugs[n_test_drugs + n_val_drugs:])
        train_cell_lines = set(cell_lines[n_test_cells + n_val_cells:])

        # Split data
        test_df = df[
            df["DRUG_NAME"].isin(test_drugs) &
            df["CELL_LINE_NAME"].isin(test_cell_lines)
        ]
        val_df = df[
            df["DRUG_NAME"].isin(val_drugs) &
            df["CELL_LINE_NAME"].isin(val_cell_lines)
        ]
        train_df = df[
            df["DRUG_NAME"].isin(train_drugs) &
            df["CELL_LINE_NAME"].isin(train_cell_lines)
        ]

        metadata = {
            "test_drugs": list(test_drugs),
            "test_cell_lines": list(test_cell_lines),
            "n_test_drugs": len(test_drugs),
            "n_test_cell_lines": len(test_cell_lines),
            "overlap": {
                "drugs": False,  # By construction
                "cell_lines": False,  # By construction
            }
        }

        logger.info(
            f"Disjoint split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
            f"test_drugs={len(test_drugs)}, test_cell_lines={len(test_cell_lines)}"
        )

        return SplitResult(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            split_type="disjoint",
            metadata=metadata,
        )

    def cross_validation_split(
        self,
        df: pd.DataFrame,
        strategy: str = "random",
        n_folds: int = 5,
    ) -> List[SplitResult]:
        """
        Generate cross-validation splits.

        Args:
            df: DataFrame with drug sensitivity data
            strategy: Split strategy to use
            n_folds: Number of CV folds

        Returns:
            List of SplitResults, one per fold
        """
        if strategy == "random":
            return self._random_cv(df, n_folds)
        elif strategy == "drug_blind":
            return self._drug_blind_cv(df, n_folds)
        elif strategy == "cell_blind":
            return self._cell_blind_cv(df, n_folds)
        else:
            raise ValueError(f"CV not implemented for strategy: {strategy}")

    def _random_cv(self, df: pd.DataFrame, n_folds: int) -> List[SplitResult]:
        """Random k-fold cross-validation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            train_val_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Split train into train/val
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - 1/n_folds),
                random_state=self.random_state,
            )

            results.append(SplitResult(
                train_df=train_df.reset_index(drop=True),
                val_df=val_df.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
                split_type=f"random_cv_fold_{fold}",
                metadata={"fold": fold, "n_folds": n_folds},
            ))

        return results

    def _drug_blind_cv(self, df: pd.DataFrame, n_folds: int) -> List[SplitResult]:
        """Drug-blind k-fold cross-validation."""
        drugs = df["DRUG_NAME"].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(drugs)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(drugs)):
            test_drugs = set(drugs[test_idx])
            train_val_drugs = set(drugs[train_idx])

            # Further split train drugs for validation
            train_val_list = list(train_val_drugs)
            n_val = int(len(train_val_list) * self.val_size / (1 - 1/n_folds))
            val_drugs = set(train_val_list[:n_val])
            train_drugs = set(train_val_list[n_val:])

            test_df = df[df["DRUG_NAME"].isin(test_drugs)]
            val_df = df[df["DRUG_NAME"].isin(val_drugs)]
            train_df = df[df["DRUG_NAME"].isin(train_drugs)]

            results.append(SplitResult(
                train_df=train_df.reset_index(drop=True),
                val_df=val_df.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
                split_type=f"drug_blind_cv_fold_{fold}",
                metadata={"fold": fold, "n_folds": n_folds, "test_drugs": list(test_drugs)},
            ))

        return results

    def _cell_blind_cv(self, df: pd.DataFrame, n_folds: int) -> List[SplitResult]:
        """Cell-blind k-fold cross-validation."""
        cell_lines = df["CELL_LINE_NAME"].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(cell_lines)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(cell_lines)):
            test_cells = set(cell_lines[test_idx])
            train_val_cells = set(cell_lines[train_idx])

            # Further split for validation
            train_val_list = list(train_val_cells)
            n_val = int(len(train_val_list) * self.val_size / (1 - 1/n_folds))
            val_cells = set(train_val_list[:n_val])
            train_cells = set(train_val_list[n_val:])

            test_df = df[df["CELL_LINE_NAME"].isin(test_cells)]
            val_df = df[df["CELL_LINE_NAME"].isin(val_cells)]
            train_df = df[df["CELL_LINE_NAME"].isin(train_cells)]

            results.append(SplitResult(
                train_df=train_df.reset_index(drop=True),
                val_df=val_df.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
                split_type=f"cell_blind_cv_fold_{fold}",
                metadata={"fold": fold, "n_folds": n_folds, "test_cell_lines": list(test_cells)},
            ))

        return results


def main():
    """Test the splitter."""
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    n_samples = 10000

    data = {
        "DRUG_NAME": np.random.choice([f"Drug_{i}" for i in range(50)], n_samples),
        "CELL_LINE_NAME": np.random.choice([f"Cell_{i}" for i in range(100)], n_samples),
        "LN_IC50": np.random.normal(-2, 1, n_samples),
    }
    df = pd.DataFrame(data)

    print(f"Total samples: {len(df)}")
    print(f"Unique drugs: {df['DRUG_NAME'].nunique()}")
    print(f"Unique cell lines: {df['CELL_LINE_NAME'].nunique()}")

    splitter = DataSplitter()

    for strategy in DataSplitter.VALID_STRATEGIES:
        print(f"\n{'='*50}")
        result = splitter.split(df, strategy=strategy)
        print(result.summary())


if __name__ == "__main__":
    main()
