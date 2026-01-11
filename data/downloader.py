"""
GDSC Data Downloader

Downloads and caches GDSC1 and GDSC2 drug sensitivity data from cancerrxgene.org.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GDSCDownloader:
    """
    Downloads and manages GDSC (Genomics of Drug Sensitivity in Cancer) data.

    Data is cached locally to avoid repeated downloads. Cache is validated
    by file existence and optional TTL (time-to-live).
    """

    # GDSC Data URLs (Release 8.5)
    GDSC_URLS = {
        "gdsc1_ic50": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "gdsc2_ic50": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
        "cell_lines": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
        "compounds": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv",
    }

    # Alternative URLs (backup)
    GDSC_URLS_BACKUP = {
        "gdsc1_ic50": "https://www.cancerrxgene.org/gdsc1/GDSC1_fitted_dose_response.xlsx",
        "gdsc2_ic50": "https://www.cancerrxgene.org/gdsc2/GDSC2_fitted_dose_response.xlsx",
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl_days: int = 30,
    ):
        """
        Initialize GDSC downloader.

        Args:
            cache_dir: Directory to cache downloaded files. Defaults to data/cache/
            cache_ttl_days: Cache time-to-live in days. Set to -1 to disable expiry.
        """
        if cache_dir is None:
            cache_dir = os.environ.get("DATA_CACHE_DIR", "data/cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=cache_ttl_days) if cache_ttl_days > 0 else None

        logger.info(f"GDSC Downloader initialized. Cache: {self.cache_dir}")

    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get the cache file path for a dataset."""
        extension = ".csv" if dataset_name == "compounds" else ".xlsx"
        return self.cache_dir / f"{dataset_name}{extension}"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file exists and is not expired."""
        if not cache_path.exists():
            return False

        if self.cache_ttl is None:
            return True

        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - file_mtime < self.cache_ttl

    def _download_file(
        self,
        url: str,
        dest_path: Path,
        chunk_size: int = 8192,
    ) -> bool:
        """
        Download a file from URL with progress bar.

        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Download chunk size in bytes

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            logger.info(f"Downloading from {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded: {dest_path}")
            return True

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def download(self, dataset_name: str, force: bool = False) -> Optional[Path]:
        """
        Download a specific GDSC dataset.

        Args:
            dataset_name: One of 'gdsc1_ic50', 'gdsc2_ic50', 'cell_lines', 'compounds'
            force: Force re-download even if cached

        Returns:
            Path to downloaded file, or None if failed
        """
        if dataset_name not in self.GDSC_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(self.GDSC_URLS.keys())}")

        cache_path = self._get_cache_path(dataset_name)

        if not force and self._is_cache_valid(cache_path):
            logger.info(f"Using cached: {cache_path}")
            return cache_path

        # Try primary URL
        url = self.GDSC_URLS[dataset_name]
        if self._download_file(url, cache_path):
            return cache_path

        # Try backup URL if available
        if dataset_name in self.GDSC_URLS_BACKUP:
            logger.info("Trying backup URL...")
            url = self.GDSC_URLS_BACKUP[dataset_name]
            if self._download_file(url, cache_path):
                return cache_path

        logger.error(f"Failed to download {dataset_name}")
        return None

    def download_all(self, force: bool = False) -> Dict[str, Optional[Path]]:
        """
        Download all GDSC datasets.

        Args:
            force: Force re-download even if cached

        Returns:
            Dict mapping dataset names to file paths (or None if failed)
        """
        results = {}
        for name in self.GDSC_URLS:
            results[name] = self.download(name, force=force)
        return results

    def load_gdsc1_ic50(self, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load GDSC1 IC50 data as DataFrame.

        Returns:
            DataFrame with columns: DRUG_NAME, CELL_LINE_NAME, LN_IC50, etc.
        """
        path = self.download("gdsc1_ic50", force=force_download)
        if path is None:
            return None

        try:
            df = pd.read_excel(path)
            logger.info(f"Loaded GDSC1: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load GDSC1: {e}")
            return None

    def load_gdsc2_ic50(self, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load GDSC2 IC50 data as DataFrame.

        Returns:
            DataFrame with columns: DRUG_NAME, CELL_LINE_NAME, LN_IC50, etc.
        """
        path = self.download("gdsc2_ic50", force=force_download)
        if path is None:
            return None

        try:
            df = pd.read_excel(path)
            logger.info(f"Loaded GDSC2: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load GDSC2: {e}")
            return None

    def load_cell_lines(self, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load cell line details.

        Returns:
            DataFrame with cell line metadata (tissue type, cancer type, etc.)
        """
        path = self.download("cell_lines", force=force_download)
        if path is None:
            return None

        try:
            df = pd.read_excel(path)
            logger.info(f"Loaded cell lines: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load cell lines: {e}")
            return None

    def load_compounds(self, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load compound/drug information including SMILES.

        Returns:
            DataFrame with drug metadata (name, SMILES, targets, etc.)
        """
        path = self.download("compounds", force=force_download)
        if path is None:
            return None

        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded compounds: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load compounds: {e}")
            return None

    def load_combined_ic50(
        self,
        prefer_gdsc2: bool = True,
        force_download: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Load and combine GDSC1 and GDSC2 IC50 data.

        For duplicate drug-cell line pairs, prefers GDSC2 data by default
        (as recommended by GDSC documentation for improved experimental design).

        Args:
            prefer_gdsc2: If True, prefer GDSC2 for duplicates. Otherwise prefer GDSC1.
            force_download: Force re-download of data

        Returns:
            Combined DataFrame with IC50 data from both datasets
        """
        gdsc1 = self.load_gdsc1_ic50(force_download)
        gdsc2 = self.load_gdsc2_ic50(force_download)

        if gdsc1 is None and gdsc2 is None:
            logger.error("Failed to load any GDSC data")
            return None

        if gdsc1 is None:
            return gdsc2
        if gdsc2 is None:
            return gdsc1

        # Add source column
        gdsc1 = gdsc1.copy()
        gdsc2 = gdsc2.copy()
        gdsc1["SOURCE"] = "GDSC1"
        gdsc2["SOURCE"] = "GDSC2"

        # Combine datasets
        combined = pd.concat([gdsc1, gdsc2], ignore_index=True)

        # Handle duplicates (same drug-cell line pair)
        # Identify key columns for deduplication
        key_cols = ["DRUG_NAME", "CELL_LINE_NAME"]
        if all(col in combined.columns for col in key_cols):
            if prefer_gdsc2:
                # Sort so GDSC2 comes last, then drop duplicates keeping last
                combined = combined.sort_values("SOURCE")
                combined = combined.drop_duplicates(subset=key_cols, keep="last")
            else:
                combined = combined.sort_values("SOURCE", ascending=False)
                combined = combined.drop_duplicates(subset=key_cols, keep="last")

        logger.info(f"Combined GDSC data: {len(combined)} records")
        return combined

    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of available data.

        Returns:
            Dict with counts of drugs, cell lines, and data points per dataset
        """
        summary = {
            "cached_files": [],
            "datasets": {},
        }

        for name in self.GDSC_URLS:
            cache_path = self._get_cache_path(name)
            if cache_path.exists():
                summary["cached_files"].append(name)

        # Load and summarize each dataset
        gdsc1 = self.load_gdsc1_ic50()
        if gdsc1 is not None:
            summary["datasets"]["gdsc1"] = {
                "records": len(gdsc1),
                "drugs": gdsc1["DRUG_NAME"].nunique() if "DRUG_NAME" in gdsc1.columns else 0,
                "cell_lines": gdsc1["CELL_LINE_NAME"].nunique() if "CELL_LINE_NAME" in gdsc1.columns else 0,
            }

        gdsc2 = self.load_gdsc2_ic50()
        if gdsc2 is not None:
            summary["datasets"]["gdsc2"] = {
                "records": len(gdsc2),
                "drugs": gdsc2["DRUG_NAME"].nunique() if "DRUG_NAME" in gdsc2.columns else 0,
                "cell_lines": gdsc2["CELL_LINE_NAME"].nunique() if "CELL_LINE_NAME" in gdsc2.columns else 0,
            }

        compounds = self.load_compounds()
        if compounds is not None:
            summary["datasets"]["compounds"] = {
                "records": len(compounds),
            }

        cell_lines = self.load_cell_lines()
        if cell_lines is not None:
            summary["datasets"]["cell_lines"] = {
                "records": len(cell_lines),
            }

        return summary


def main():
    """Test the downloader."""
    logging.basicConfig(level=logging.INFO)

    downloader = GDSCDownloader()
    summary = downloader.get_data_summary()

    print("\nGDSC Data Summary:")
    print(f"Cached files: {summary['cached_files']}")
    for name, stats in summary["datasets"].items():
        print(f"\n{name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
