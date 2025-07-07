"""
Evaluation metrics and performance analysis utilities
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


class ModelEvaluator:
    """Comprehensive model evaluation utilities"""

    def __init__(self):
        self.results = {}

    def calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class

        Returns:
            Dictionary of basic metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        }

        return metrics

    def calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate per-class metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes

        Returns:
            DataFrame with per-class metrics
        """
        # Get classification report as dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Extract per-class metrics
        classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]

        per_class_data = []
        for class_label in classes:
            if class_label in report:
                class_metrics = report[class_label].copy()
                class_metrics["class"] = class_label
                per_class_data.append(class_metrics)

        df = pd.DataFrame(per_class_data)

        if class_names and len(class_names) == len(df):
            df["class_name"] = class_names

        return df

    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate metrics derived from confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        cm = confusion_matrix(y_true, y_pred)

        # Calculate per-class metrics from confusion matrix
        n_classes = cm.shape[0]
        class_metrics = {}

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            # Avoid division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            class_metrics[f"class_{i}"] = {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
            }

        return {"confusion_matrix": cm, "class_metrics": class_metrics}

    def calculate_multiclass_auc(
        self, y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate AUC scores for multi-class classification

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes

        Returns:
            Dictionary of AUC scores
        """
        auc_scores = {}

        try:
            # Convert labels to binary format for AUC calculation
            lb = LabelBinarizer()
            y_true_binary = lb.fit_transform(y_true)

            if y_true_binary.shape[1] == 1:
                # Binary classification case
                auc_scores["auc_binary"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class case
                # Macro-average AUC
                auc_scores["auc_macro"] = roc_auc_score(
                    y_true_binary, y_proba, average="macro", multi_class="ovr"
                )

                # Weighted-average AUC
                auc_scores["auc_weighted"] = roc_auc_score(
                    y_true_binary, y_proba, average="weighted", multi_class="ovr"
                )

                # Per-class AUC
                for i, class_label in enumerate(lb.classes_):
                    class_name = class_names[i] if class_names else f"class_{class_label}"
                    auc_scores[f"auc_{class_name}"] = roc_auc_score(
                        y_true_binary[:, i], y_proba[:, i]
                    )

        except Exception as e:
            logger.warning(f"Could not calculate AUC scores: {e}")

        return auc_scores

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        class_names: List[str] = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: Names of classes (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")

        evaluation_results = {
            "model_name": model_name,
            "n_samples": len(y_true),
            "n_classes": len(np.unique(y_true)),
        }

        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        evaluation_results["basic_metrics"] = basic_metrics

        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred, class_names)
        evaluation_results["per_class_metrics"] = per_class_metrics

        # Confusion matrix metrics
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        evaluation_results["confusion_matrix"] = cm_metrics["confusion_matrix"]
        evaluation_results["confusion_matrix_metrics"] = cm_metrics["class_metrics"]

        # AUC metrics (if probabilities provided)
        if y_proba is not None:
            auc_metrics = self.calculate_multiclass_auc(y_true, y_proba, class_names)
            evaluation_results["auc_metrics"] = auc_metrics

        # Store results
        self.results[model_name] = evaluation_results

        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"Accuracy: {basic_metrics['accuracy']:.4f}")
        logger.info(f"F1 (weighted): {basic_metrics['f1_score']:.4f}")
        logger.info(f"F1 (macro): {basic_metrics['f1_macro']:.4f}")

        return evaluation_results

    def compare_models(self, metric: str = "f1_score") -> pd.DataFrame:
        """
        Compare models based on a specific metric

        Args:
            metric: Metric to use for comparison

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No evaluation results available for comparison")
            return pd.DataFrame()

        comparison_data = []

        for model_name, results in self.results.items():
            if metric in results.get("basic_metrics", {}):
                comparison_data.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "value": results["basic_metrics"][metric],
                        "n_samples": results["n_samples"],
                        "n_classes": results["n_classes"],
                    }
                )

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            return df.sort_values("value", ascending=False)

        return pd.DataFrame()

    def get_summary_report(self) -> pd.DataFrame:
        """
        Get a summary report of all evaluated models

        Returns:
            DataFrame with summary metrics for all models
        """
        if not self.results:
            return pd.DataFrame()

        summary_data = []

        for model_name, results in self.results.items():
            basic_metrics = results.get("basic_metrics", {})
            auc_metrics = results.get("auc_metrics", {})

            summary_row = {
                "model": model_name,
                "accuracy": basic_metrics.get("accuracy", 0),
                "precision": basic_metrics.get("precision", 0),
                "recall": basic_metrics.get("recall", 0),
                "f1_weighted": basic_metrics.get("f1_score", 0),
                "f1_macro": basic_metrics.get("f1_macro", 0),
                "n_samples": results.get("n_samples", 0),
                "n_classes": results.get("n_classes", 0),
            }

            # Add AUC if available
            if "auc_macro" in auc_metrics:
                summary_row["auc_macro"] = auc_metrics["auc_macro"]
            if "auc_weighted" in auc_metrics:
                summary_row["auc_weighted"] = auc_metrics["auc_weighted"]

            summary_data.append(summary_row)

        return pd.DataFrame(summary_data)

    def print_detailed_report(self, model_name: str = None):
        """
        Print detailed evaluation report

        Args:
            model_name: Specific model to report on (if None, reports on all)
        """
        if model_name:
            models_to_report = [model_name] if model_name in self.results else []
        else:
            models_to_report = list(self.results.keys())

        for name in models_to_report:
            results = self.results[name]

            print(f"\n{'='*60}")
            print(f"DETAILED EVALUATION REPORT: {name}")
            print(f"{'='*60}")

            # Basic info
            print(f"Samples: {results['n_samples']}")
            print(f"Classes: {results['n_classes']}")

            # Basic metrics
            print("\nBASIC METRICS:")
            print("-" * 30)
            basic_metrics = results["basic_metrics"]
            for metric, value in basic_metrics.items():
                print(f"{metric.capitalize():<15}: {value:.4f}")

            # AUC metrics
            if "auc_metrics" in results:
                print("\nAUC METRICS:")
                print("-" * 30)
                for metric, value in results["auc_metrics"].items():
                    print(f"{metric.upper():<15}: {value:.4f}")

            # Confusion matrix
            print("\nCONFUSION MATRIX:")
            print("-" * 30)
            print(results["confusion_matrix"])

            # Per-class metrics
            if not results["per_class_metrics"].empty:
                print("\nPER-CLASS METRICS:")
                print("-" * 30)
                print(results["per_class_metrics"].round(4))


def test_metrics():
    """Test the evaluation metrics with sample data"""
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)

    class_names = ["Cargo", "Tanker", "Fishing"]

    # Test evaluator
    evaluator = ModelEvaluator()

    logger.info("Testing model evaluation metrics")

    # Evaluate a model
    evaluator.evaluate_model(y_true, y_pred, y_proba, class_names, "test_model")

    # Test another model for comparison
    y_pred2 = np.random.randint(0, n_classes, n_samples)
    y_proba2 = np.random.dirichlet(np.ones(n_classes), n_samples)

    evaluator.evaluate_model(y_true, y_pred2, y_proba2, class_names, "test_model_2")

    # Test comparison
    comparison = evaluator.compare_models("f1_score")
    logger.info("Model comparison:")
    logger.info(comparison)

    # Test summary
    summary = evaluator.get_summary_report()
    logger.info("Summary report:")
    logger.info(summary)

    # Print detailed report
    evaluator.print_detailed_report()

    return evaluator


if __name__ == "__main__":
    test_metrics()
