"""
Configuration settings for MVESLINT
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Main configuration class using Pydantic for validation"""

    # Environment
    DEBUG: bool = False
    TEST_MODE: bool = False

    # Data paths
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"

    # Input data configuration
    ZIP_FILE_PATH: Optional[str] = None
    CSV_FILE_NAME: Optional[str] = None
    SAMPLE_SIZE: float = 0.01  # For test mode

    # Data filtering parameters
    MIN_MMSI_LENGTH: int = 9
    MAX_SOG: float = 50.0  # knots
    MIN_SOG: float = 0.0
    MAX_COG: float = 360.0
    MIN_COG: float = 0.0

    # Geographic bounds (US waters)
    MIN_LAT: float = 24.0
    MAX_LAT: float = 49.0
    MIN_LON: float = -130.0
    MAX_LON: float = -65.0

    # Trajectory processing
    TIME_GAP_THRESHOLD: int = 3600  # seconds
    MIN_TRAJECTORY_LENGTH: int = 5

    # Target vessel types to keep
    TARGET_VESSEL_TYPES: List[float] = [37.0, 31.0, 52.0, 30.0, 70.0]

    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5

    # Deep learning parameters
    EPOCHS: int = 15
    TEST_EPOCHS: int = 2
    BATCH_SIZE: int = 128
    LSTM_BATCH_SIZE: int = 64
    EARLY_STOPPING_PATIENCE: int = 3
    LEARNING_RATE: float = 0.001
    LSTM_LEARNING_RATE: float = 0.0003

    # Traditional ML parameters
    N_ESTIMATORS: int = 150
    TEST_TREES: int = 10
    RF_MAX_DEPTH: int = 12
    RF_TEST_MAX_DEPTH: int = 5
    LGB_MAX_DEPTH: int = 8
    LGB_TEST_MAX_DEPTH: int = 3
    LGB_LEARNING_RATE: float = 0.05
    LGB_TEST_LEARNING_RATE: float = 0.1

    # SMOTE parameters
    SMOTE_K_NEIGHBORS: int = 5

    # Feature extraction
    STOP_SPEED_THRESHOLD: float = 0.5  # knots
    HIGH_SPEED_THRESHOLD: float = 25.0  # knots

    # Kalman filter parameters
    OBSERVATION_COVARIANCE: float = 0.5
    TRANSITION_COVARIANCE: float = 0.2

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    # Output
    SAVE_PLOTS: bool = True
    PLOT_DIR: Path = Path("plots")

    model_config = {"env_file": ".env", "case_sensitive": True}

    @field_validator("DATA_DIR", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "MODELS_DIR", "PLOT_DIR")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    def get_test_config(self) -> "Config":
        """Get configuration optimized for testing with small datasets"""
        test_config = self.model_copy()
        test_config.TEST_MODE = True
        test_config.CV_FOLDS = max(2, min(3, 5))  # Adaptive fold count
        test_config.EPOCHS = self.TEST_EPOCHS
        test_config.N_ESTIMATORS = self.TEST_TREES
        test_config.RF_MAX_DEPTH = self.RF_TEST_MAX_DEPTH
        test_config.LGB_MAX_DEPTH = self.LGB_TEST_MAX_DEPTH
        test_config.LGB_LEARNING_RATE = self.LGB_TEST_LEARNING_RATE
        test_config.MIN_TRAJECTORY_LENGTH = 3  # Reduce for small datasets
        test_config.SAMPLE_SIZE = 0.02  # Slightly larger sample
        test_config.SMOTE_K_NEIGHBORS = 3  # Reduce for small datasets
        test_config.BATCH_SIZE = 32  # Smaller batch size
        test_config.LSTM_BATCH_SIZE = 16  # Even smaller for LSTM
        return test_config


@dataclass
class ModelConfig:
    """Configuration for individual models"""

    name: str
    params: Dict[str, Any]
    enabled: bool = True


def load_config_from_yaml(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    return Config(**yaml_config)


# Default configuration instance
config = Config()


# Model configurations
CNN_CONFIG = ModelConfig(
    name="cnn",
    params={
        "filters": [128, 64],
        "kernel_sizes": [5, 3],
        "pool_size": 2,
        "dense_units": 128,
        "dropout_rate": 0.5,
        "activation": "relu",
    },
)

LSTM_CONFIG = ModelConfig(
    name="bi_lstm_attention",
    params={
        "lstm_units": 64,
        "dense_units": 64,
        "dropout_rate": 0.3,
        "attention": True,
        "bidirectional": True,
    },
)

RF_CONFIG = ModelConfig(
    name="random_forest", params={"class_weight": "balanced_subsample", "n_jobs": -1}
)

LGB_CONFIG = ModelConfig(
    name="lightgbm", params={"class_weight": "balanced", "n_jobs": -1, "verbosity": -1}
)

META_LEARNER_CONFIG = ModelConfig(
    name="meta_learner",
    params={"max_iter": 1000, "class_weight": "balanced", "n_jobs": -1, "solver": "lbfgs"},
)
