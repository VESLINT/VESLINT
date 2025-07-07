"""
Feature extraction utilities for vessel trajectory analysis
"""
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from loguru import logger
from scipy.stats import circmean, circstd
from sklearn.impute import SimpleImputer

from ..config import config


class VesselFeatureExtractor:
    """Extract features from vessel trajectories for classification"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    def calculate_initial_compass_bearing(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate initial compass bearing between two points

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Bearing in degrees (0-360)
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def extract_static_features(self, group: pd.DataFrame) -> dict:
        """
        Extract static vessel features (vessel characteristics)

        Args:
            group: DataFrame with vessel data

        Returns:
            Dictionary with static features
        """
        features = {}

        # Basic vessel dimensions
        length = group["Length"].iloc[0]
        width = group["Width"].iloc[0]

        features.update(
            {
                "Length": length,
                "Width": width,
                "L_W_ratio": length / max(1, width),
                "Size_category": length * width,
            }
        )

        # Draft if available
        if "Draft" in group.columns:
            draft = group["Draft"].iloc[0]
            features["Draft"] = draft
            features["Beam_to_draft"] = width / max(1, draft)
        else:
            features["Draft"] = 0
            features["Beam_to_draft"] = 0

        return features

    def extract_movement_features(self, group: pd.DataFrame) -> dict:
        """
        Extract movement-based features from trajectory

        Args:
            group: DataFrame with trajectory data

        Returns:
            Dictionary with movement features
        """
        features = {}

        # Sort by time
        group = group.sort_values("BaseDateTime")

        # Extract coordinates and movement data
        lat = group["LAT_smoothed"].values
        lon = group["LON_smoothed"].values
        sog = group["SOG"].values
        times = group["BaseDateTime"]
        n_points = len(group)

        # Time-based features
        duration = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600  # hours

        # Movement metrics
        total_distance = 0
        bearings = []
        speed_changes = []
        accelerations = []

        # Calculate movement statistics
        for i in range(1, n_points):
            # Distance
            dist = self.haversine_distance(lat[i - 1], lon[i - 1], lat[i], lon[i])
            total_distance += dist

            # Bearing
            bearing = self.calculate_initial_compass_bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])
            bearings.append(bearing)

            # Speed and acceleration
            time_diff = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600
            if time_diff > 0:
                speed_change = sog[i] - sog[i - 1]
                acceleration = speed_change / time_diff
                speed_changes.append(speed_change)
                accelerations.append(acceleration)

        # Navigation metrics
        if len(bearings) > 1:
            bearing_changes = np.abs(np.diff(bearings))
            bearing_changes = np.minimum(bearing_changes, 360 - bearing_changes)
        else:
            bearing_changes = [0]

        # Basic trajectory metrics
        features.update(
            {
                "Duration_hr": duration,
                "Total_distance_km": total_distance,
                "Avg_speed": total_distance / max(1, duration),
                "Point_density": n_points / max(1, duration),
            }
        )

        # Speed features
        features.update(
            {
                "Speed_mean": np.mean(sog),
                "Speed_std": np.std(sog),
                "Speed_max": np.max(sog),
                "Speed_min": np.min(sog),
                "Speed_range": np.max(sog) - np.min(sog),
            }
        )

        # Acceleration features
        features.update(
            {
                "Accel_mean": np.mean(accelerations) if accelerations else 0,
                "Accel_max": np.max(accelerations) if accelerations else 0,
                "Accel_std": np.std(accelerations) if accelerations else 0,
            }
        )

        # Direction features
        features.update(
            {
                "Bearing_mean": circmean(bearings, high=360) if bearings else 0,
                "Bearing_std": circstd(bearings, high=360) if bearings else 0,
                "Turn_rate_mean": np.mean(bearing_changes),
                "Turn_rate_max": np.max(bearing_changes),
            }
        )

        return features

    def extract_operational_features(self, group: pd.DataFrame) -> dict:
        """
        Extract operational pattern features

        Args:
            group: DataFrame with vessel data

        Returns:
            Dictionary with operational features
        """
        features = {}

        sog = group["SOG"].values
        cog = group["COG"].values
        n_points = len(group)

        # Stop detection
        stop_count = np.sum(sog < self.config.STOP_SPEED_THRESHOLD)
        stop_ratio = stop_count / n_points

        # High speed operation
        high_speed_count = np.sum(sog > self.config.HIGH_SPEED_THRESHOLD)
        high_speed_ratio = high_speed_count / n_points

        # Calculate accelerations for maneuvering intensity
        accelerations = []
        times = group["BaseDateTime"]
        for i in range(1, len(sog)):
            time_diff = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600
            if time_diff > 0:
                acceleration = (sog[i] - sog[i - 1]) / time_diff
                accelerations.append(acceleration)

        # Calculate bearing changes for turn analysis
        bearing_changes = []
        for i in range(1, len(cog)):
            change = abs(cog[i] - cog[i - 1])
            change = min(change, 360 - change)  # Handle circularity
            bearing_changes.append(change)

        # Operational patterns
        features.update(
            {
                "Stop_ratio": stop_ratio,
                "High_speed_ratio": high_speed_ratio,
                "Maneuvering_intensity": (
                    np.mean(np.abs(accelerations)) * np.mean(bearing_changes)
                    if accelerations and bearing_changes
                    else 0
                ),
                "Operational_consistency": (
                    np.mean(sog) / max(1, np.std(sog)) if np.std(sog) > 0 else 0
                ),
            }
        )

        return features

    def extract_features(self, group: pd.DataFrame) -> pd.Series:
        """
        Extract complete feature set for a single vessel trajectory

        Args:
            group: DataFrame with single vessel trajectory

        Returns:
            Series with all extracted features
        """
        # Combine all feature types
        features = {}

        # Static features (vessel characteristics)
        static_features = self.extract_static_features(group)
        features.update(static_features)

        # Movement features
        movement_features = self.extract_movement_features(group)
        features.update(movement_features)

        # Operational features
        operational_features = self.extract_operational_features(group)
        features.update(operational_features)

        return pd.Series(features)

    def extract_static_vessel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract static features per vessel (not per trajectory)

        Args:
            df: DataFrame with vessel data

        Returns:
            DataFrame with static features per vessel
        """
        logger.info("Extracting static vessel features")

        static_features = df.groupby("MMSI").first()[["Length", "Width"]].reset_index()

        # Add Draft if available
        if "Draft" in df.columns:
            draft_df = df.groupby("MMSI").first()[["Draft"]].reset_index()
            static_features = pd.merge(static_features, draft_df, on="MMSI", how="left")
        else:
            static_features["Draft"] = np.nan

        logger.info(f"Extracted static features for {len(static_features)} vessels")

        return static_features

    def extract_dynamic_features(
        self, df: pd.DataFrame, sample_trajectories: bool = None
    ) -> pd.DataFrame:
        """
        Extract dynamic features per trajectory

        Args:
            df: DataFrame with trajectory data
            sample_trajectories: Whether to sample trajectories in test mode

        Returns:
            DataFrame with dynamic features per trajectory
        """
        logger.info("Extracting dynamic trajectory features")

        sample_trajectories = (
            sample_trajectories if sample_trajectories is not None else self.config.TEST_MODE
        )

        unique_trajs = df[["MMSI", "traj_id"]].drop_duplicates()

        if sample_trajectories and not unique_trajs.empty:
            # Sample trajectories for test mode
            sample_frac = min(self.config.SAMPLE_SIZE * 10, 1.0)
            sampled_trajs = unique_trajs.sample(
                frac=sample_frac, random_state=self.config.RANDOM_STATE
            )
            df_sample = pd.merge(sampled_trajs, df, on=["MMSI", "traj_id"])

            if not df_sample.empty:
                dynamic_features = (
                    df_sample.groupby(["MMSI", "traj_id"])
                    .apply(self.extract_features)
                    .reset_index()
                )
            else:
                logger.warning("Sampled dataframe for dynamic features is empty")
                dynamic_features = pd.DataFrame(columns=["MMSI", "traj_id"])
        else:
            # Full extraction
            dynamic_features = (
                df.groupby(["MMSI", "traj_id"]).apply(self.extract_features).reset_index()
            )

        logger.info(f"Extracted dynamic features for {len(dynamic_features)} trajectories")

        return dynamic_features

    def impute_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing and infinite values in feature matrix

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with imputed values
        """
        logger.info("Imputing missing and infinite values")

        # Identify feature columns (exclude MMSI, VesselType, traj_id)
        exclude_cols = ["MMSI", "VesselType", "traj_id"]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        if not feature_cols:
            logger.warning("No feature columns found for imputation")
            return features_df

        # Convert infinite values to NaN
        X_to_impute = features_df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Check for missing values
        missing_count = X_to_impute.isnull().sum().sum()
        infinite_count = np.isinf(features_df[feature_cols]).sum().sum()

        logger.info(f"Found {missing_count} missing values and {infinite_count} infinite values")

        if missing_count > 0 or infinite_count > 0:
            # Use median imputation (robust to outliers)
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")
            features_df[feature_cols] = imputer.fit_transform(X_to_impute)

            logger.info("Imputation completed")
        else:
            logger.info("No missing or infinite values found")

        return features_df

    def create_feature_pipeline(self, df: pd.DataFrame) -> tuple:
        """
        Complete feature extraction pipeline

        Args:
            df: Input DataFrame with trajectory data

        Returns:
            Tuple of (features_df, target_labels)
        """
        logger.info("Starting feature extraction pipeline")

        # Extract static features
        static_features = self.extract_static_vessel_features(df)

        # Extract dynamic features
        dynamic_features = self.extract_dynamic_features(df)

        # Get vessel types (target labels)
        vessel_types = df.groupby("MMSI")["VesselType"].first().reset_index()

        # Merge all features
        features = pd.merge(static_features, dynamic_features, on="MMSI")
        features = pd.merge(features, vessel_types, on="MMSI")

        # Impute missing values
        features = self.impute_missing_values(features)

        # Prepare final feature matrix and targets
        exclude_cols = ["MMSI", "VesselType", "traj_id"]
        X = features.drop(columns=exclude_cols)
        y = features["VesselType"]

        logger.info(f"Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        return features, X, y


def test_feature_extraction():
    """Test feature extraction with sample data"""
    # Create sample trajectory data
    test_data = pd.DataFrame(
        {
            "MMSI": [123456789] * 5,
            "traj_id": [1] * 5,
            "Length": [200] * 5,
            "Width": [30] * 5,
            "Draft": [10] * 5,
            "LAT_smoothed": [34.05, 34.06, 34.07, 34.08, 34.09],
            "LON_smoothed": [-118.25, -118.24, -118.23, -118.22, -118.21],
            "SOG": [10, 12, 11, 9, 10],
            "COG": [45, 46, 47, 45, 44],
            "Heading": [44, 45, 46, 44, 43],
            "BaseDateTime": pd.date_range(start="2023-01-01", periods=5, freq="H"),
            "VesselType": ["30"] * 5,
        }
    )

    extractor = VesselFeatureExtractor()

    logger.info("Testing feature extraction:")
    logger.info("Input data:")
    logger.info(test_data)

    # Test single trajectory feature extraction
    features = extractor.extract_features(test_data)
    logger.info("Extracted features:")
    logger.info(features)

    # Test full pipeline
    features_df, X, y = extractor.create_feature_pipeline(test_data)
    logger.info("Pipeline results:")
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info("Sample features:")
    logger.info(X.head())

    return features_df, X, y


if __name__ == "__main__":
    test_feature_extraction()
