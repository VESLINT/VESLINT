"""
Data preprocessing utilities for AIS data
"""
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import zscore

from ..config import config


class AISPreprocessor:
    """
    Handles preprocessing of AIS data including vessel type reduction and trajectory processing
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def reduce_vessel_types(
        self, df: pd.DataFrame, target_types: List[float] = None
    ) -> pd.DataFrame:
        """
        Reduce vessel types to predefined categories and group others as 'OTHER'

        Args:
            df: Input DataFrame
            target_types: List of vessel types to keep

        Returns:
            DataFrame with reduced vessel types
        """
        target_types = target_types or self.config.TARGET_VESSEL_TYPES

        logger.info("Reducing vessel types to target categories")
        logger.info(f"Target vessel types: {target_types}")

        initial_shape = df.shape

        # Convert VesselType to numeric
        df["VesselType"] = pd.to_numeric(df["VesselType"], errors="coerce")

        # Create reduced vessel type mapping
        df["VesselType_Reduced"] = df["VesselType"].apply(
            lambda x: str(int(x)) if pd.notna(x) and x in target_types else "OTHER"
        )

        # Replace original column
        df["VesselType"] = df["VesselType_Reduced"]
        df = df.drop(columns=["VesselType_Reduced"], errors="ignore")

        # Remove rows where VesselType became NaN
        df = df.dropna(subset=["VesselType"])

        logger.info("Vessel type distribution after reduction:")
        type_counts = df["VesselType"].value_counts()
        for vtype, count in type_counts.items():
            logger.info(f"  {vtype}: {count} ({count/len(df)*100:.1f}%)")

        logger.info(f"Shape after reduction: {df.shape} (from {initial_shape})")

        return df

    def process_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process trajectories by segmenting based on time gaps and filtering short trajectories

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with trajectory processing
        """
        logger.info("Processing trajectories")

        initial_shape = df.shape

        # Ensure datetime conversion
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

        # Sort by vessel and time
        df = df.sort_values(["MMSI", "BaseDateTime"])

        # Calculate time differences between consecutive points for each vessel
        df["time_diff"] = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds().fillna(0)

        # Create trajectory segments based on time gaps
        # A new trajectory starts when time gap exceeds threshold
        df["new_trajectory"] = (df["time_diff"] > self.config.TIME_GAP_THRESHOLD).astype(int)
        df["traj_id"] = df["new_trajectory"].cumsum()

        # Remove the helper columns
        df = df.drop(columns=["time_diff", "new_trajectory"])

        logger.info(f"Created {df['traj_id'].nunique()} trajectory segments")

        # Filter out short trajectories
        trajectory_lengths = df.groupby(["MMSI", "traj_id"]).size()
        valid_trajectories = trajectory_lengths[
            trajectory_lengths >= self.config.MIN_TRAJECTORY_LENGTH
        ].index

        df = df.set_index(["MMSI", "traj_id"]).loc[valid_trajectories].reset_index()

        final_shape = df.shape
        logger.info(f"After filtering short trajectories: {final_shape} (from {initial_shape})")

        # Log trajectory statistics
        if not df.empty:
            traj_stats = df.groupby("MMSI")["traj_id"].nunique()
            logger.info(
                f"Trajectories per vessel - Mean: {traj_stats.mean():.1f}, "
                f"Median: {traj_stats.median():.1f}, Max: {traj_stats.max()}"
            )

        return df

    def remove_outliers(self, df: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
        """
        Remove statistical outliers from the dataset

        Args:
            df: Input DataFrame
            z_threshold: Z-score threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers with z-score threshold: {z_threshold}")

        initial_count = len(df)

        # Domain-specific filtering (already done in basic validation, but double-check)
        df = df[(df["SOG"] >= self.config.MIN_SOG) & (df["SOG"] <= self.config.MAX_SOG)]
        df = df[(df["COG"] >= self.config.MIN_COG) & (df["COG"] <= self.config.MAX_COG)]
        df = df[(df["LAT"] >= -90) & (df["LAT"] <= 90)]
        df = df[(df["LON"] >= -180) & (df["LON"] <= 180)]

        # Statistical outlier removal for key numeric columns
        numeric_cols = ["SOG", "COG", "Length", "Width"]
        for col in numeric_cols:
            if col in df.columns and len(df) > 0:
                initial_col_count = len(df)
                z_scores = np.abs(zscore(df[col]))
                df = df[z_scores < z_threshold]
                removed = initial_col_count - len(df)
                if removed > 0:
                    logger.info(f"Removed {removed} outliers from {col}")

        final_count = len(df)
        removed_total = initial_count - final_count

        logger.info(
            f"Outlier removal complete: {removed_total} records removed "
            f"({removed_total/initial_count*100:.1f}%)"
        )

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")

        # Log missing value statistics
        missing_stats = df.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]

        if len(missing_cols) > 0:
            logger.info("Missing values found:")
            for col, count in missing_cols.items():
                logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            logger.info("No missing values found")
            return df

        # Handle specific columns
        if "Draft" in df.columns:
            # Draft is often missing - fill with median based on vessel type
            if df["Draft"].isnull().any():
                median_draft = df.groupby("VesselType")["Draft"].transform("median")
                df["Draft"] = df["Draft"].fillna(median_draft)

                # If still missing, fill with overall median
                if df["Draft"].isnull().any():
                    df["Draft"] = df["Draft"].fillna(df["Draft"].median())

        if "Heading" in df.columns:
            # Heading can be missing - fill with COG as approximation
            if df["Heading"].isnull().any():
                df["Heading"] = df["Heading"].fillna(df["COG"])

        # For other numeric columns, use forward fill within vessel groups
        numeric_cols = ["SOG", "COG", "LAT", "LON"]
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df.groupby("MMSI")[col].fillna(method="ffill")
                df[col] = df.groupby("MMSI")[col].fillna(method="bfill")

        # Drop any remaining rows with critical missing values
        critical_cols = ["LAT", "LON", "SOG", "COG", "VesselType"]
        before_count = len(df)
        df = df.dropna(subset=critical_cols)
        after_count = len(df)

        if before_count != after_count:
            logger.info(f"Dropped {before_count - after_count} rows with critical missing values")

        return df

    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline")

        # Step 1: Reduce vessel types
        df = self.reduce_vessel_types(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Remove outliers
        df = self.remove_outliers(df)

        # Step 4: Process trajectories
        df = self.process_trajectories(df)

        logger.info("Preprocessing pipeline complete")

        return df

    def get_preprocessing_summary(
        self, original_df: pd.DataFrame, processed_df: pd.DataFrame
    ) -> dict:
        """
        Generate summary statistics of preprocessing

        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "original_records": len(original_df),
            "processed_records": len(processed_df),
            "records_retained": len(processed_df) / len(original_df),
            "original_vessels": original_df["MMSI"].nunique(),
            "processed_vessels": processed_df["MMSI"].nunique(),
            "original_vessel_types": original_df["VesselType"].nunique(),
            "processed_vessel_types": processed_df["VesselType"].nunique(),
        }

        if "traj_id" in processed_df.columns:
            summary["total_trajectories"] = processed_df["traj_id"].nunique()
            summary["avg_trajectories_per_vessel"] = (
                processed_df.groupby("MMSI")["traj_id"].nunique().mean()
            )

        return summary
