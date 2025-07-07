"""
Data loading utilities for AIS data
"""
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from ..config import config


class AISDataLoader:
    """Handles loading and basic validation of AIS data"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.required_columns = [
            "MMSI",
            "LAT",
            "LON",
            "SOG",
            "COG",
            "VesselType",
            "BaseDateTime",
            "Length",
            "Width",
        ]

    def extract_zip_file(self, zip_path: Union[str, Path], extract_dir: Union[str, Path]) -> Path:
        """
        Extract ZIP file containing AIS data

        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract to

        Returns:
            Path to extracted CSV file
        """
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        logger.info(f"Extracting {zip_path} to {extract_dir}")

        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the CSV file
        csv_files = list(extract_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in extracted directory: {extract_dir}")

        csv_path = csv_files[0]  # Take the first CSV file
        logger.info(f"Found CSV file: {csv_path}")

        return csv_path

    def create_sample_data(
        self, input_path: Union[str, Path], output_path: Union[str, Path], sample_size: float = None
    ) -> Path:
        """
        Create a sample dataset for testing

        Args:
            input_path: Path to full dataset
            output_path: Path to save sample
            sample_size: Fraction of data to sample

        Returns:
            Path to sample file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        sample_size = sample_size or self.config.SAMPLE_SIZE

        if not input_path.exists():
            raise FileNotFoundError(f"Source CSV file not found: {input_path}")

        logger.info(f"Creating sample dataset with {sample_size*100:.1f}% of data")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = pd.read_csv(input_path, chunksize=100000)
        sample_chunks = []

        for chunk in chunks:
            if "VesselType" in chunk.columns:
                # Preserve vessel type distribution
                sample = chunk.groupby("VesselType", group_keys=False).apply(
                    lambda x: x.sample(frac=sample_size, random_state=self.config.RANDOM_STATE)
                    if len(x) > 0
                    else x
                )
                sample_chunks.append(sample)
            else:
                # Simple random sampling if no VesselType column
                sample = chunk.sample(frac=sample_size, random_state=self.config.RANDOM_STATE)
                sample_chunks.append(sample)

        if not sample_chunks:
            logger.warning("No data chunks processed for sampling")
            sample_df = pd.DataFrame()
        else:
            sample_df = pd.concat(sample_chunks, ignore_index=True)

        sample_df.to_csv(output_path, index=False)
        logger.info(f"Created sample dataset with {len(sample_df)} records at {output_path}")

        return output_path

    def load_csv_data(
        self, filepath: Union[str, Path], test_mode: bool = None, chunksize: int = 500000
    ) -> pd.DataFrame:
        """
        Load CSV data with chunking for large files

        Args:
            filepath: Path to CSV file
            test_mode: Whether to load in test mode
            chunksize: Size of chunks for large files

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        test_mode = test_mode if test_mode is not None else self.config.TEST_MODE

        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        logger.info(f"Loading data from {filepath}")

        if test_mode:
            logger.info("Loading in test mode - using sample data")
            df = pd.read_csv(filepath)
        else:
            logger.info("Loading full dataset in chunks")
            chunks = pd.read_csv(filepath, chunksize=chunksize, low_memory=False)
            df = pd.concat(chunks, ignore_index=True)

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def validate_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that required columns are present and drop rows with missing values

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with validated columns
        """
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info("All required columns present")

        # Drop rows with missing required values
        initial_count = len(df)
        df = df.dropna(subset=self.required_columns)

        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with missing required values")

        return df

    def basic_data_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data validation and cleaning

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        logger.info(f"Starting basic validation with {initial_count} records")

        # MMSI should be 9 digits
        df = df[df["MMSI"].astype(str).str.len() == self.config.MIN_MMSI_LENGTH]
        logger.info(f"After MMSI validation: {len(df)} records")

        # Convert numeric columns
        numeric_cols = ["LAT", "LON", "SOG", "COG", "Heading", "Length", "Width", "Draft"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with invalid critical values
        df = df.dropna(subset=["LAT", "LON", "SOG", "COG", "Length", "Width"])
        logger.info(f"After numeric conversion: {len(df)} records")

        # Remove zero coordinates (common AIS noise)
        df = df[(df["LAT"] != 0) & (df["LON"] != 0)]
        logger.info(f"After removing zero coordinates: {len(df)} records")

        # Speed validation
        df = df[(df["SOG"] >= self.config.MIN_SOG) & (df["SOG"] <= self.config.MAX_SOG)]
        logger.info(f"After speed validation: {len(df)} records")

        # Remove vessels with zero dimensions
        df = df[(df["Length"] > 0) & (df["Width"] > 0)]
        logger.info(f"After dimension validation: {len(df)} records")

        # Geographic filtering to US waters
        df = df[
            (df["LAT"] >= self.config.MIN_LAT)
            & (df["LAT"] <= self.config.MAX_LAT)
            & (df["LON"] >= self.config.MIN_LON)
            & (df["LON"] <= self.config.MAX_LON)
        ]
        logger.info(f"After geographic filtering: {len(df)} records")

        # Convert timestamp
        if "BaseDateTime" in df.columns:
            df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
            df = df.dropna(subset=["BaseDateTime"])
            logger.info(f"After timestamp conversion: {len(df)} records")

        final_count = len(df)
        logger.info(
            f"Validation complete: {final_count} records "
            f"({final_count/initial_count*100:.1f}% retained)"
        )

        return df

    def load_and_validate(self, filepath: Union[str, Path], test_mode: bool = None) -> pd.DataFrame:
        """
        Complete data loading and validation pipeline

        Args:
            filepath: Path to data file
            test_mode: Whether to run in test mode

        Returns:
            Loaded and validated DataFrame
        """
        # Load data
        df = self.load_csv_data(filepath, test_mode)

        # Validate columns
        df = self.validate_required_columns(df)

        # Basic validation
        df = self.basic_data_validation(df)

        # Log final statistics
        logger.info(f"Final dataset: {len(df)} records, {df['MMSI'].nunique()} unique vessels")
        logger.info(f"Vessel type distribution:\n{df['VesselType'].value_counts()}")

        return df


def setup_data_paths(zip_file_path: str = None, csv_file_name: str = None) -> tuple:
    """
    Setup data paths for loading

    Args:
        zip_file_path: Path to ZIP file
        csv_file_name: Name of CSV file in ZIP

    Returns:
        Tuple of (csv_path, sample_path)
    """
    zip_file_path = zip_file_path or config.ZIP_FILE_PATH
    csv_file_name = csv_file_name or config.CSV_FILE_NAME

    if not zip_file_path:
        raise ValueError("ZIP file path must be specified")
    if not csv_file_name:
        raise ValueError("CSV file name must be specified")

    extract_dir = config.RAW_DATA_DIR / "extracted"
    csv_path = extract_dir / csv_file_name
    sample_path = config.PROCESSED_DATA_DIR / "sample_ais_data.csv"

    return csv_path, sample_path
