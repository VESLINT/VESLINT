"""
Main pipeline for maritime vessel type classification
"""
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    # Try relative imports first (when running as part of package)
    from .config import config
    from .data import AISDataLoader, AISPreprocessor, setup_data_paths
    from .features import TrajectoryKalmanFilter, VesselFeatureExtractor
    from .models import StackingEnsemble
    from .utils import ModelEvaluator, ModelVisualizer
except ImportError:
    # Fall back to absolute imports (when running directly or in development)
    from src.config import config
    from src.data import AISDataLoader, AISPreprocessor, setup_data_paths
    from src.features import TrajectoryKalmanFilter, VesselFeatureExtractor
    from src.models import StackingEnsemble
    from src.utils import ModelEvaluator, ModelVisualizer

warnings.filterwarnings("ignore")


class MaritimePipeline:
    """Complete pipeline for maritime vessel type classification"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

        # Initialize components
        self.data_loader = AISDataLoader(self.config)
        self.preprocessor = AISPreprocessor(self.config)
        self.kalman_filter = TrajectoryKalmanFilter(self.config)
        self.feature_extractor = VesselFeatureExtractor(self.config)
        self.ensemble = StackingEnsemble(self.config)
        self.evaluator = ModelEvaluator()
        self.visualizer = ModelVisualizer(self.config)

        # Pipeline state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        # Results storage
        self.results = {}

    def run_smoke_tests(self) -> bool:
        """
        Run smoke tests to verify all components work

        Returns:
            True if all tests pass
        """
        logger.info("Running smoke tests")

        try:
            # Test Kalman filter
            from .features import test_kalman_filter

            test_kalman_filter()

            # Test feature extraction
            from .features import test_feature_extraction

            test_feature_extraction()

            # Test models
            from .models import test_deep_learning_models, test_traditional_models

            test_traditional_models()
            test_deep_learning_models()

            # Test metrics and visualization
            from .utils import test_metrics, test_visualization

            test_metrics()
            test_visualization()

            logger.info("All smoke tests passed!")
            return True

        except Exception as e:
            logger.error(f"Smoke tests failed: {e}")
            return False

    def load_and_prepare_data(
        self, zip_file_path: str = None, csv_file_name: str = None
    ) -> pd.DataFrame:
        """
        Load and prepare data for training

        Args:
            zip_file_path: Path to ZIP file containing AIS data
            csv_file_name: Name of CSV file in ZIP

        Returns:
            Prepared DataFrame
        """
        logger.info("Loading and preparing data")
        start_time = time.time()

        # Setup data paths
        csv_path, sample_path = setup_data_paths(zip_file_path, csv_file_name)

        # Handle test mode
        if self.config.TEST_MODE:
            if not sample_path.exists():
                logger.info("Creating sample data for test mode")
                # Extract ZIP if needed
                if not csv_path.exists():
                    if not Path(zip_file_path).exists():
                        raise FileNotFoundError(f"ZIP file not found: {zip_file_path}")
                    self.data_loader.extract_zip_file(zip_file_path, csv_path.parent)

                # Create sample
                self.data_loader.create_sample_data(csv_path, sample_path)

            data_path = sample_path
        else:
            # Extract ZIP if needed
            if not csv_path.exists():
                if not Path(zip_file_path).exists():
                    raise FileNotFoundError(f"ZIP file not found: {zip_file_path}")
                self.data_loader.extract_zip_file(zip_file_path, csv_path.parent)

            data_path = csv_path

        # Load and validate data
        df = self.data_loader.load_and_validate(data_path, self.config.TEST_MODE)

        logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data")
        start_time = time.time()

        # Get preprocessing summary
        original_df = df.copy()

        # Run preprocessing pipeline
        df = self.preprocessor.preprocess_pipeline(df)

        # Log summary
        summary = self.preprocessor.get_preprocessing_summary(original_df, df)
        logger.info("Preprocessing summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

        return df

    def apply_trajectory_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter for trajectory smoothing

        Args:
            df: DataFrame with trajectories

        Returns:
            DataFrame with smoothed coordinates
        """
        logger.info("Applying trajectory smoothing")
        start_time = time.time()

        # Apply Kalman filter
        df_smoothed = self.kalman_filter.process_all_trajectories(df)

        # Validate smoothing quality
        self.kalman_filter.validate_smoothing_quality(df_smoothed)

        logger.info(f"Trajectory smoothing completed in {time.time() - start_time:.2f} seconds")

        return df_smoothed

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Extract features from the processed data

        Args:
            df: Processed DataFrame

        Returns:
            Tuple of (features_df, X, y)
        """
        logger.info("Extracting features")
        start_time = time.time()

        # Extract features using the pipeline
        features_df, X, y = self.feature_extractor.create_feature_pipeline(df)

        logger.info(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

        return features_df, X, y

    def prepare_sequence_data(self, df: pd.DataFrame, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare sequence data for deep learning models

        Args:
            df: Original DataFrame with trajectories
            features_df: DataFrame with features

        Returns:
            Padded sequence array
        """
        logger.info("Preparing sequence data")
        start_time = time.time()

        # Get trajectory groups matching features
        feature_trajectories = features_df[["MMSI", "traj_id"]].drop_duplicates()

        # Extract sequences from original data
        sequences_list = []
        for _, row in feature_trajectories.iterrows():
            traj_data = df[(df["MMSI"] == row["MMSI"]) & (df["traj_id"] == row["traj_id"])]

            if not traj_data.empty:
                sequence = traj_data[["LAT_smoothed", "LON_smoothed", "SOG", "COG"]].values
                sequences_list.append(sequence)
            else:
                # Create dummy sequence if no data found
                sequences_list.append(np.zeros((1, 4)))

        # Pad sequences to same length
        if sequences_list:
            max_len = max(len(seq) for seq in sequences_list)
            padded_sequences = np.zeros((len(sequences_list), max_len, 4))

            for i, seq in enumerate(sequences_list):
                padded_sequences[i, : len(seq)] = seq
        else:
            padded_sequences = np.zeros((0, 1, 4))

        logger.info(f"Sequence preparation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Sequence shape: {padded_sequences.shape}")

        return padded_sequences

    def split_and_scale_data(self, X: np.ndarray, y: np.ndarray, sequences: np.ndarray) -> Tuple:
        """
        Split data into train/test and apply scaling with robust handling for small datasets
        """
        logger.info("Splitting and scaling data")
        start_time = time.time()

        # Encode string labels
        if y.dtype == "object" or isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y

        # Check class distribution for stratification
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = class_counts.min()

        # Determine if stratification is possible
        stratify_y = None
        if min_class_count >= 2:
            stratify_y = y_encoded
            logger.info(f"Using stratified split - minimum class count: {min_class_count}")
        else:
            logger.warning(f"Skipping stratification - minimum class count: {min_class_count}")
            # For very small datasets, use a different test size
            if len(X) < 20:
                test_size = max(0.1, 2 / len(X))  # At least 2 samples or 10%
                logger.info(f"Adjusting test size to {test_size:.2f} for small dataset")
            else:
                test_size = self.config.TEST_SIZE

        # Split data
        try:
            X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
                X,
                y_encoded,
                sequences,
                test_size=test_size if "test_size" in locals() else self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=stratify_y,
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
                X,
                y_encoded,
                sequences,
                test_size=test_size if "test_size" in locals() else self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
            )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Calculate class weights
        try:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}. Using uniform weights.")
            classes = np.unique(y_train)
            class_weights = {cls: 1.0 for cls in classes}

        logger.info(
            f"Data splitting and scaling completed in {time.time() - start_time:.2f} seconds"
        )
        logger.info(f"Train set: {X_train_scaled.shape[0]} samples")
        logger.info(f"Test set: {X_test_scaled.shape[0]} samples")
        logger.info(
            f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}"
        )
        logger.info(
            f"Class distribution - Test: {dict(zip(*np.unique(y_test, return_counts=True)))}"
        )

        return (X_train_scaled, X_test_scaled, y_train, y_test, seq_train, seq_test, class_weights)

    def apply_smote(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE for handling class imbalance with robust small dataset handling
        """
        logger.info("Applying SMOTE for class balancing")

        # Check if SMOTE is possible
        class_counts = pd.Series(y_train).value_counts()
        min_samples_required = min(
            self.config.SMOTE_K_NEIGHBORS + 1, 3
        )  # Reduce requirement for small datasets

        # Check if we have enough samples and multiple classes
        if len(class_counts) < 2:
            logger.warning("SMOTE skipped - only one class present")
            return X_train, y_train

        if class_counts.min() >= min_samples_required:
            try:
                # Adjust k_neighbors for small datasets
                k_neighbors = min(self.config.SMOTE_K_NEIGHBORS, class_counts.min() - 1, 5)

                smote = SMOTE(
                    sampling_strategy="auto",
                    random_state=self.config.RANDOM_STATE,
                    k_neighbors=k_neighbors,
                )

                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

                logger.info(f"SMOTE applied: {X_train.shape[0]} -> {X_resampled.shape[0]} samples")
                logger.info(f"SMOTE k_neighbors: {k_neighbors}")

                return X_resampled, y_resampled

            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using original data.")
                return X_train, y_train
        else:
            logger.warning(
                f"SMOTE skipped - minimum class size ({class_counts.min()}) "
                f"< required ({min_samples_required})"
            )
            return X_train, y_train

    def train_ensemble(
        self, X_train: np.ndarray, y_train: np.ndarray, seq_train: np.ndarray
    ) -> StackingEnsemble:
        """
        Train the stacking ensemble with proper sequence data handling
        """
        logger.info("Training ensemble model")
        start_time = time.time()

        # Check if we have sequence data
        has_sequences = seq_train is not None and len(seq_train) > 0

        if has_sequences:
            logger.info(
                f"Training with sequence data - features: {X_train.shape}, "
                f"sequences: {seq_train.shape}"
            )

            # For sequence data, we need to be careful with SMOTE
            # Option 1: Skip SMOTE when we have sequences (recommended for simplicity)
            logger.info("Skipping SMOTE due to sequence data compatibility")
            X_train_final = X_train
            y_train_final = y_train
            seq_train_final = seq_train

            # Alternative Option 2: Apply SMOTE and replicate sequences
            # Uncomment this section if you want to use SMOTE with sequences:
            """
            # Apply SMOTE to features only
            X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)

            if len(X_train_smote) > len(X_train):
                # SMOTE was applied, need to handle sequences
                logger.info(
                    f"SMOTE expanded dataset from {len(X_train)} to {len(X_train_smote)} samples"
                )

                # Create mapping for synthetic samples
                # For synthetic samples, we'll use the sequence from the nearest original sample
                seq_train_expanded = np.zeros(
                    (len(X_train_smote), seq_train.shape[1], seq_train.shape[2])
                )

                # Copy original sequences
                seq_train_expanded[: len(X_train)] = seq_train

                # For synthetic samples, replicate sequences from original samples
                # This is a simplified approach - in practice, you might want more sophisticated
                # sequence generation
                for i in range(len(X_train), len(X_train_smote)):
                    # Use sequence from a random original sample
                    orig_idx = np.random.randint(0, len(X_train))
                    seq_train_expanded[i] = seq_train[orig_idx]

                seq_train_final = seq_train_expanded
            else:
                seq_train_final = seq_train
            X_train_final = X_train_smote
            y_train_final = y_train_smote
            """

        else:
            # No sequence data, apply SMOTE normally
            logger.info("No sequence data - applying SMOTE to features")
            X_train_final, y_train_final = self.apply_smote(X_train, y_train)
            seq_train_final = None

        # Train ensemble
        self.ensemble.fit(X_train_final, y_train_final, seq_train_final)
        self.is_fitted = True

        logger.info(f"Ensemble training completed in {time.time() - start_time:.2f} seconds")
        logger.info(
            f"Final training data - Features: {X_train_final.shape}, Labels: {y_train_final.shape}"
        )
        if seq_train_final is not None:
            logger.info(f"Final sequence data: {seq_train_final.shape}")

        return self.ensemble

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray, seq_test: np.ndarray, class_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model

        Args:
            X_test: Test features
            y_test: Test labels
            seq_test: Test sequences
            class_names: Names of classes

        Returns:
            Evaluation results
        """
        logger.info("Evaluating model")
        start_time = time.time()

        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        predictions = self.ensemble.predict(X_test, seq_test)
        probabilities = self.ensemble.predict_proba(X_test, seq_test)

        # Convert back to original labels if necessary
        if hasattr(self.label_encoder, "classes_"):
            y_test_orig = self.label_encoder.inverse_transform(y_test)
            class_names = class_names or list(self.label_encoder.classes_)
        else:
            y_test_orig = y_test

        # Evaluate
        results = self.evaluator.evaluate_model(
            y_test_orig, predictions, probabilities, class_names, "StackingEnsemble"
        )

        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")

        return results

    def create_visualizations(
        self,
        y_test: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray = None,
        class_names: list = None,
    ) -> Dict[str, Any]:
        """
        Create evaluation visualizations

        Args:
            y_test: Test labels
            predictions: Predictions
            probabilities: Prediction probabilities
            class_names: Names of classes

        Returns:
            Dictionary of created plots
        """
        logger.info("Creating visualizations")

        plots = {}

        # Confusion matrix
        plots["confusion_matrix"] = self.visualizer.plot_confusion_matrix(
            y_test, predictions, class_names, "Confusion Matrix"
        )

        # Classification report
        plots["classification_report"] = self.visualizer.plot_classification_report(
            y_test, predictions, class_names, "Classification Report"
        )

        if probabilities is not None:
            # ROC curves
            plots["roc_curves"] = self.visualizer.plot_roc_curves(
                y_test, probabilities, class_names, "ROC Curves"
            )

            # Precision-Recall curves
            plots["pr_curves"] = self.visualizer.plot_precision_recall_curves(
                y_test, probabilities, class_names, "Precision-Recall Curves"
            )

        # Feature importance
        importance_df = self.ensemble.get_base_model_importance()
        if not importance_df.empty:
            plots["feature_importance"] = self.visualizer.plot_feature_importance(
                importance_df, "Base Model Importance"
            )

        # Evaluation dashboard
        plots["dashboard"] = self.visualizer.create_evaluation_dashboard(
            y_test, predictions, probabilities, class_names, "Stacking Ensemble"
        )

        logger.info("Visualization creation completed")

        return plots

    def save_model(self, model_path: str = None) -> str:
        """
        Save the trained model and preprocessing objects

        Args:
            model_path: Path to save the model

        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_path = model_path or str(self.config.MODELS_DIR / "ensemble_model.joblib")

        # Create save directory
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare save object
        save_obj = {
            "ensemble": self.ensemble,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "config": self.config,
            "results": self.results,
        }

        # Save
        joblib.dump(save_obj, model_path)

        logger.info(f"Model saved to: {model_path}")

        return model_path

    def load_model(self, model_path: str) -> "MaritimePipeline":
        """
        Load a previously trained model

        Args:
            model_path: Path to the saved model

        Returns:
            Self for method chaining
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load saved objects
        save_obj = joblib.load(model_path)

        self.ensemble = save_obj["ensemble"]
        self.scaler = save_obj["scaler"]
        self.label_encoder = save_obj["label_encoder"]
        self.results = save_obj.get("results", {})
        self.is_fitted = True

        logger.info(f"Model loaded from: {model_path}")

        return self

    def run_complete_pipeline(self, zip_file_path: str, csv_file_name: str) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation

        Args:
            zip_file_path: Path to ZIP file containing AIS data
            csv_file_name: Name of CSV file in ZIP

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()

        logger.info(
            f"Starting complete pipeline in {'TEST' if self.config.TEST_MODE else 'FULL'} mode"
        )

        # Step 1: Load and prepare data
        df = self.load_and_prepare_data(zip_file_path, csv_file_name)

        # Step 2: Preprocess data
        df = self.preprocess_data(df)

        # Step 3: Apply trajectory smoothing
        df = self.apply_trajectory_smoothing(df)

        # Step 4: Extract features
        features_df, X, y = self.extract_features(df)

        # Step 5: Prepare sequence data
        sequences = self.prepare_sequence_data(df, features_df)

        # Step 6: Split and scale data
        (
            X_train,
            X_test,
            y_train,
            y_test,
            seq_train,
            seq_test,
            class_weights,
        ) = self.split_and_scale_data(X, y, sequences)

        # Step 7: Train ensemble
        self.train_ensemble(X_train, y_train, seq_train)

        # Step 8: Evaluate model
        class_names = (
            list(self.label_encoder.classes_) if hasattr(self.label_encoder, "classes_") else None
        )

        # Make predictions first
        predictions = self.ensemble.predict(X_test, seq_test)
        probabilities = self.ensemble.predict_proba(X_test, seq_test)

        # Ensure consistent label types for evaluation
        if hasattr(self.label_encoder, "classes_"):
            # Convert both to original string labels
            y_test_str = self.label_encoder.inverse_transform(y_test)
            if isinstance(predictions[0], (int, np.integer)):
                predictions_str = self.label_encoder.inverse_transform(predictions)
            else:
                predictions_str = predictions
        else:
            y_test_str = y_test
            predictions_str = predictions

        # Evaluate with consistent label types
        evaluation_results = self.evaluator.evaluate_model(
            y_test_str, predictions_str, probabilities, class_names, "StackingEnsemble"
        )

        # Step 9: Create visualizations
        plots = self.create_visualizations(y_test_str, predictions_str, probabilities, class_names)

        # Step 10: Save model
        model_path = self.save_model()

        # Compile results
        total_time = time.time() - pipeline_start

        pipeline_results = {
            "evaluation_results": evaluation_results,
            "plots": plots,
            "model_path": model_path,
            "execution_time": total_time,
            "data_shape": df.shape,
            "features_shape": X.shape,
            "sequences_shape": sequences.shape,
            "class_names": class_names,
        }

        self.results["pipeline"] = pipeline_results

        logger.info(f"Pipeline completed successfully in {total_time/60:.2f} minutes")

        return pipeline_results


def test_pipeline():
    """Test the complete pipeline with mock data"""
    logger.info("Testing pipeline components")

    # Create pipeline instance
    pipeline = MaritimePipeline()

    # Run smoke tests
    smoke_test_passed = pipeline.run_smoke_tests()

    if smoke_test_passed:
        logger.info("Pipeline smoke tests passed!")
    else:
        logger.error("Pipeline smoke tests failed!")

    return pipeline


if __name__ == "__main__":
    test_pipeline()
