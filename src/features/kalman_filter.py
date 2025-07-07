"""
Kalman filter implementation for trajectory smoothing
"""
import numpy as np
import pandas as pd
from loguru import logger
from pykalman import KalmanFilter

from ..config import config


class TrajectoryKalmanFilter:
    """Kalman filter for smoothing vessel trajectories"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def apply_kalman_filter(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter to smooth a single trajectory

        Args:
            trajectory: DataFrame with LAT, LON columns

        Returns:
            DataFrame with additional LAT_smoothed, LON_smoothed columns
        """
        # Create a copy to avoid modifying original
        traj = trajectory.copy()

        # If trajectory is too short, just copy original coordinates
        if len(traj) < 2:
            traj[["LAT_smoothed", "LON_smoothed"]] = traj[["LAT", "LON"]]
            return traj

        # Extract lat/lon coordinates
        lat_lon = traj[["LAT", "LON"]].values

        try:
            # Initialize Kalman filter
            initial_state = lat_lon[0]

            kf = KalmanFilter(
                transition_matrices=np.eye(2),  # Simple position model
                observation_matrices=np.eye(2),  # Direct observation of position
                initial_state_mean=initial_state,
                observation_covariance=np.eye(2) * self.config.OBSERVATION_COVARIANCE,
                transition_covariance=np.eye(2) * self.config.TRANSITION_COVARIANCE,
            )

            # Apply smoothing
            smoothed_state_means, _ = kf.smooth(lat_lon)

            # Add smoothed coordinates to trajectory
            traj["LAT_smoothed"] = smoothed_state_means[:, 0]
            traj["LON_smoothed"] = smoothed_state_means[:, 1]

        except Exception as e:
            logger.warning(f"Kalman filter failed for trajectory: {e}")
            # Fallback to original coordinates
            traj[["LAT_smoothed", "LON_smoothed"]] = traj[["LAT", "LON"]]

        return traj

    def process_all_trajectories(self, df: pd.DataFrame, chunk_size: int = None) -> pd.DataFrame:
        """
        Apply Kalman filter to all trajectories in the dataset

        Args:
            df: DataFrame with trajectories (must have MMSI, traj_id columns)
            chunk_size: Number of trajectories to process in each chunk

        Returns:
            DataFrame with smoothed coordinates
        """
        if "traj_id" not in df.columns:
            logger.error("DataFrame must have 'traj_id' column")
            raise ValueError("DataFrame must have 'traj_id' column")

        logger.info("Applying Kalman filter to all trajectories")

        # Get unique trajectories
        unique_trajs = df[["MMSI", "traj_id"]].drop_duplicates()
        total_trajectories = len(unique_trajs)

        logger.info(f"Processing {total_trajectories} trajectories")

        # Determine chunk size
        if chunk_size is None:
            chunk_size = 50 if not self.config.TEST_MODE else 10

        # Process in chunks to manage memory
        n_chunks = min(chunk_size, total_trajectories)
        if n_chunks == 0:
            logger.warning("No trajectories to process")
            # Add smoothed columns as copies of original
            df["LAT_smoothed"] = df["LAT"]
            df["LON_smoothed"] = df["LON"]
            return df

        chunks = np.array_split(unique_trajs, n_chunks)
        smoothed_dfs = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Get data for this chunk of trajectories
            chunk_data = pd.merge(chunk, df, on=["MMSI", "traj_id"])

            if not chunk_data.empty:
                # Apply Kalman filter to each trajectory in the chunk
                smoothed_chunk = (
                    chunk_data.groupby(["MMSI", "traj_id"])
                    .apply(self.apply_kalman_filter)
                    .reset_index(drop=True)
                )

                smoothed_dfs.append(smoothed_chunk)
            else:
                logger.warning(f"Chunk {i+1} resulted in empty data after merge")

        if smoothed_dfs:
            result_df = pd.concat(smoothed_dfs, ignore_index=True)
            logger.info(f"Kalman filtering complete: {len(result_df)} records processed")
        else:
            logger.warning("No data after Kalman filter processing")
            # Fallback: add smoothed columns as copies
            result_df = df.copy()
            result_df["LAT_smoothed"] = result_df["LAT"]
            result_df["LON_smoothed"] = result_df["LON"]

        return result_df

    def validate_smoothing_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate the quality of Kalman filter smoothing

        Args:
            df: DataFrame with both original and smoothed coordinates

        Returns:
            Dictionary with quality metrics
        """
        if not all(col in df.columns for col in ["LAT", "LON", "LAT_smoothed", "LON_smoothed"]):
            logger.error("DataFrame missing required coordinate columns")
            return {}

        # Calculate differences
        lat_diff = np.abs(df["LAT"] - df["LAT_smoothed"])
        lon_diff = np.abs(df["LON"] - df["LON_smoothed"])

        # Calculate statistics
        quality_metrics = {
            "lat_mean_diff": lat_diff.mean(),
            "lat_max_diff": lat_diff.max(),
            "lat_std_diff": lat_diff.std(),
            "lon_mean_diff": lon_diff.mean(),
            "lon_max_diff": lon_diff.max(),
            "lon_std_diff": lon_diff.std(),
            "total_points": len(df),
            "smoothing_applied": (lat_diff > 0).sum(),
        }

        logger.info("Smoothing quality metrics:")
        logger.info(f"  Mean LAT difference: {quality_metrics['lat_mean_diff']:.6f}")
        logger.info(f"  Mean LON difference: {quality_metrics['lon_mean_diff']:.6f}")
        logger.info(
            f"  Points with smoothing: {quality_metrics['smoothing_applied']} "
            f"({quality_metrics['smoothing_applied']/len(df)*100:.1f}%)"
        )

        return quality_metrics


def test_kalman_filter():
    """Test the Kalman filter with sample data"""
    # Create sample trajectory
    test_traj = pd.DataFrame(
        {
            "LAT": [34.05, 34.06, 34.07, 34.08, 34.09],
            "LON": [-118.25, -118.24, -118.23, -118.22, -118.21],
            "SOG": [10.0, 10.5, 11.0, 10.8, 10.2],
            "COG": [45.0, 46.0, 47.0, 46.5, 45.5],
            "MMSI": [123456789] * 5,
            "traj_id": [1] * 5,
        }
    )

    kf = TrajectoryKalmanFilter()

    logger.info("Testing Kalman filter:")
    logger.info("Input trajectory:")
    logger.info(test_traj[["LAT", "LON"]])

    # Test single trajectory
    smoothed_single = kf.apply_kalman_filter(test_traj)
    logger.info("Single trajectory smoothing:")
    logger.info(smoothed_single[["LAT_smoothed", "LON_smoothed"]])

    # Test batch processing
    smoothed_batch = kf.process_all_trajectories(test_traj)
    logger.info("Batch processing result:")
    logger.info(smoothed_batch[["LAT_smoothed", "LON_smoothed"]])

    # Validate quality
    quality = kf.validate_smoothing_quality(smoothed_batch)
    logger.info(f"Quality metrics: {quality}")

    return smoothed_batch


if __name__ == "__main__":
    test_kalman_filter()
