"""
Tests for configuration module
"""
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add src to path before importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, config, load_config_from_yaml  # noqa: E402


class TestConfig:
    """Test configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        cfg = Config()

        assert cfg.DEBUG is False
        assert cfg.TEST_MODE is False
        assert cfg.RANDOM_STATE == 42
        assert cfg.MIN_MMSI_LENGTH == 9
        assert cfg.MAX_SOG == 50.0
        assert len(cfg.TARGET_VESSEL_TYPES) == 5

    def test_test_config(self):
        """Test test mode configuration"""
        cfg = Config()
        test_cfg = cfg.get_test_config()

        assert test_cfg.TEST_MODE is True
        assert test_cfg.CV_FOLDS == 3
        assert test_cfg.EPOCHS == test_cfg.TEST_EPOCHS
        assert test_cfg.N_ESTIMATORS == test_cfg.TEST_TREES

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid values
        with pytest.raises(Exception):
            Config(MIN_SOG=-10)  # Should be >= 0

        with pytest.raises(Exception):
            Config(MAX_SOG=0)  # Should be > MIN_SOG

    def test_directory_creation(self):
        """Test that directories are created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg = Config(
                DATA_DIR=tmpdir_path / "data",
                MODELS_DIR=tmpdir_path / "models",
                PLOT_DIR=tmpdir_path / "plots",
            )

            assert cfg.DATA_DIR.exists()
            assert cfg.MODELS_DIR.exists()
            assert cfg.PLOT_DIR.exists()


class TestConfigLoading:
    """Test configuration loading from files"""

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file"""
        config_data = {
            "DEBUG": True,
            "TEST_MODE": True,
            "EPOCHS": 20,
            "TARGET_VESSEL_TYPES": [30.0, 31.0, 37.0],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            cfg = load_config_from_yaml(config_path)

            assert cfg.DEBUG is True
            assert cfg.TEST_MODE is True
            assert cfg.EPOCHS == 20
            assert cfg.TARGET_VESSEL_TYPES == [30.0, 31.0, 37.0]

        finally:
            Path(config_path).unlink()  # Clean up

    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file"""
        with pytest.raises(FileNotFoundError):
            load_config_from_yaml("nonexistent_file.yaml")


class TestDefaultConfigInstance:
    """Test the default config instance"""

    def test_default_instance(self):
        """Test that default config instance is accessible"""
        assert config is not None
        assert isinstance(config, Config)
        assert config.RANDOM_STATE == 42

    def test_config_immutability(self):
        """Test that config can be modified for testing"""
        original_debug = config.DEBUG

        # This should work (config is mutable)
        config.DEBUG = not original_debug
        assert config.DEBUG != original_debug

        # Reset to original value
        config.DEBUG = original_debug


if __name__ == "__main__":
    pytest.main([__file__])
