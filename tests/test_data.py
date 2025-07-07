"""
Tests for data loading and preprocessing modules
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path before importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config  # noqa: E402
from src.data import AISDataLoader, AISPreprocessor  # noqa: E402


class TestAISDataLoader:
    """Test AIS data loading functionality"""

    @pytest.fixture
    def sample_ais_data(self):
        """Create sample AIS data for testing"""
        np.random.seed(42)

        data = {
            "MMSI": [123456789, 123456789, 987654321, 987654321, 111222333],
            "LAT": [34.05, 34.06, 40.71, 40.72, 25.77],
            "LON": [-118.25, -118.24, -74.00, -73.99, -80.19],
            "SOG": [10.5, 11.2, 8.3, 9.1, 15.6],
            "COG": [45.0, 47.0, 180.0, 182.0, 90.0],
            "VesselType": [30, 30, 37, 37, 70],
            "BaseDateTime": pd.date_range("2024-01-01", periods=5, freq="H"),
            "Length": [200, 200, 50, 50, 300],
            "Width": [30, 30, 10, 10, 40],
            "Draft": [10, 10, 3, 3, 15],
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def config_for_testing(self):
        """Create test configuration"""
        return Config(TEST_MODE=True, SAMPLE_SIZE=0.5)

    @pytest.fixture
    def temp_csv_file(self, sample_ais_data):
        """Create temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_ais_data.to_csv(f.name, index=False)
            return f.name

    @pytest.fixture
    def temp_zip_file(self, temp_csv_file):
        """Create temporary ZIP file containing CSV"""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            with zipfile.ZipFile(f.name, "w") as zf:
                zf.write(temp_csv_file, Path(temp_csv_file).name)
            return f.name

    def test_data_loader_initialization(self, config_for_testing):
        """Test data loader initialization"""
        loader = AISDataLoader(config_for_testing)
        assert loader.config == config_for_testing
        assert len(loader.required_columns) == 9

    def test_extract_zip_file(self, temp_zip_file, temp_csv_file):
        """Test ZIP file extraction"""
        loader = AISDataLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_path = loader.extract_zip_file(temp_zip_file, tmpdir)
            assert extracted_path.exists()
            assert extracted_path.suffix == ".csv"

        # Clean up
        Path(temp_csv_file).unlink()
        Path(temp_zip_file).unlink()

    def test_create_sample_data(self, temp_csv_file, config_for_testing):
        """Test sample data creation"""
        loader = AISDataLoader(config_for_testing)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_path = loader.create_sample_data(temp_csv_file, f.name)

            # Check that sample file was created
            assert Path(sample_path).exists()

            # Check that sample is smaller than original
            original_df = pd.read_csv(temp_csv_file)
            sample_df = pd.read_csv(sample_path)
            assert len(sample_df) <= len(original_df)

            # Clean up
            Path(f.name).unlink()

        Path(temp_csv_file).unlink()

    def test_load_csv_data(self, temp_csv_file, sample_ais_data):
        """Test CSV data loading"""
        loader = AISDataLoader()

        # Test normal loading
        df = loader.load_csv_data(temp_csv_file, test_mode=False)
        assert len(df) == len(sample_ais_data)
        assert list(df.columns) == list(sample_ais_data.columns)

        # Clean up
        Path(temp_csv_file).unlink()

    def test_validate_required_columns(self, sample_ais_data):
        """Test required columns validation"""
        loader = AISDataLoader()

        # Test with all required columns
        df_valid = loader.validate_required_columns(sample_ais_data)
        assert len(df_valid) == len(sample_ais_data)

        # Test with missing required column
        df_missing = sample_ais_data.drop("MMSI", axis=1)
        with pytest.raises(ValueError):
            loader.validate_required_columns(df_missing)

    def test_basic_data_validation(self, sample_ais_data):
        """Test basic data validation"""
        loader = AISDataLoader()

        # Test with valid data
        df_valid = loader.basic_data_validation(sample_ais_data.copy())
        assert len(df_valid) > 0

        # Test with invalid MMSI (wrong length)
        df_invalid = sample_ais_data.copy()
        df_invalid.loc[0, "MMSI"] = 12345  # Only 5 digits
        df_validated = loader.basic_data_validation(df_invalid)
        assert len(df_validated) == len(sample_ais_data) - 1

        # Test with zero coordinates
        df_zero_coords = sample_ais_data.copy()
        df_zero_coords.loc[0, "LAT"] = 0
        df_zero_coords.loc[0, "LON"] = 0
        df_validated = loader.basic_data_validation(df_zero_coords)
        assert len(df_validated) == len(sample_ais_data) - 1


class TestAISPreprocessor:
    """Test AIS data preprocessing functionality"""

    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed AIS data"""
        np.random.seed(42)

        data = {
            "MMSI": [123456789] * 10 + [987654321] * 8,
            "LAT": np.random.uniform(34, 35, 18),
            "LON": np.random.uniform(-119, -118, 18),
            "SOG": np.random.uniform(5, 25, 18),
            "COG": np.random.uniform(0, 360, 18),
            "VesselType": [30.0] * 10 + [37.0] * 8,
            "BaseDateTime": (
                list(pd.date_range("2024-01-01", periods=10, freq="30min"))
                + list(pd.date_range("2024-01-01 06:00:00", periods=8, freq="45min"))
            ),
            "Length": [200] * 10 + [50] * 8,
            "Width": [30] * 10 + [10] * 8,
            "Draft": [10] * 10 + [3] * 8,
        }

        return pd.DataFrame(data)

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = AISPreprocessor()
        assert preprocessor.config is not None

    def test_reduce_vessel_types(self, sample_processed_data):
        """Test vessel type reduction"""
        preprocessor = AISPreprocessor()

        # Add some vessel types that should be grouped as 'OTHER'
        df_test = sample_processed_data.copy()
        df_test.loc[0, "VesselType"] = 99.0  # Should become 'OTHER'

        df_reduced = preprocessor.reduce_vessel_types(df_test)

        # Check that vessel types are reduced
        vessel_types = df_reduced["VesselType"].unique()
        assert "OTHER" in vessel_types
        assert "30" in vessel_types
        assert "37" in vessel_types
        assert 99.0 not in vessel_types

    def test_process_trajectories(self, sample_processed_data):
        """Test trajectory processing"""
        preprocessor = AISPreprocessor()

        df_with_trajectories = preprocessor.process_trajectories(sample_processed_data.copy())

        # Check that trajectory IDs were added
        assert "traj_id" in df_with_trajectories.columns
        assert df_with_trajectories["traj_id"].nunique() >= 1

        # Check that short trajectories are filtered out
        traj_lengths = df_with_trajectories.groupby(["MMSI", "traj_id"]).size()
        assert all(traj_lengths >= preprocessor.config.MIN_TRAJECTORY_LENGTH)

    def test_remove_outliers(self, sample_processed_data):
        """Test outlier removal"""
        preprocessor = AISPreprocessor()

        # Add outliers
        df_with_outliers = sample_processed_data.copy()
        df_with_outliers.loc[0, "SOG"] = 100.0  # Extreme speed
        df_with_outliers.loc[1, "Length"] = 10000.0  # Extreme length

        df_no_outliers = preprocessor.remove_outliers(df_with_outliers)

        # Check that outliers were removed
        assert len(df_no_outliers) < len(df_with_outliers)
        assert df_no_outliers["SOG"].max() < 100.0

    def test_handle_missing_values(self, sample_processed_data):
        """Test missing value handling"""
        preprocessor = AISPreprocessor()

        # Add missing values
        df_with_missing = sample_processed_data.copy()
        df_with_missing.loc[0, "Draft"] = np.nan
        df_with_missing.loc[1, "Heading"] = np.nan

        df_handled = preprocessor.handle_missing_values(df_with_missing)

        # Check that missing values were handled
        # Draft should be filled with median or similar
        if "Draft" in df_handled.columns:
            assert df_handled["Draft"].isnull().sum() == 0

    def test_preprocess_pipeline(self, sample_processed_data):
        """Test complete preprocessing pipeline"""
        preprocessor = AISPreprocessor()

        df_processed = preprocessor.preprocess_pipeline(sample_processed_data.copy())

        # Check that pipeline completed successfully
        assert len(df_processed) > 0
        assert "traj_id" in df_processed.columns
        assert "VesselType" in df_processed.columns

        # Check that vessel types are reduced
        vessel_types = df_processed["VesselType"].unique()
        assert all(isinstance(vt, str) for vt in vessel_types)

    def test_get_preprocessing_summary(self, sample_processed_data):
        """Test preprocessing summary generation"""
        preprocessor = AISPreprocessor()

        original_df = sample_processed_data.copy()
        processed_df = preprocessor.preprocess_pipeline(sample_processed_data.copy())

        summary = preprocessor.get_preprocessing_summary(original_df, processed_df)

        # Check that summary contains expected keys
        expected_keys = [
            "original_records",
            "processed_records",
            "records_retained",
            "original_vessels",
            "processed_vessels",
            "original_vessel_types",
            "processed_vessel_types",
        ]

        for key in expected_keys:
            assert key in summary

        # Check that values make sense
        assert summary["records_retained"] <= 1.0
        assert summary["processed_records"] <= summary["original_records"]


if __name__ == "__main__":
    pytest.main([__file__])
