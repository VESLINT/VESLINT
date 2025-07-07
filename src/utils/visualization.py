"""
Visualization utilities for model evaluation and data analysis
"""
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import LabelBinarizer

from ..config import config

# Set style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore", category=UserWarning)


class ModelVisualizer:
    """Create visualizations for model evaluation and data analysis"""

    def __init__(self, config_obj=None, save_plots=True):
        self.config = config_obj or config
        self.save_plots = save_plots
        self.plot_dir = self.config.PLOT_DIR

        # Create plot directory
        if self.save_plots:
            Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

        # Set default figure parameters
        self.figsize_small = (8, 6)
        self.figsize_medium = (10, 8)
        self.figsize_large = (12, 10)
        self.figsize_wide = (15, 6)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            normalize: Whether to normalize the matrix

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title += " (Normalized)"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        plt.tight_layout()

        if self.save_plots:
            filename = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot: {filename}")

        return fig

    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        title: str = "Classification Report",
    ) -> plt.Figure:
        """
        Plot classification report as heatmap

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import classification_report

        # Get classification report as dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Extract metrics for each class
        classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
        metrics = ["precision", "recall", "f1-score"]

        # Create matrix for heatmap
        matrix = []
        labels = []

        for class_label in classes:
            if class_label in report:
                row = [report[class_label][metric] for metric in metrics]
                matrix.append(row)

                if class_names:
                    class_idx = int(class_label) if class_label.isdigit() else class_label
                    if isinstance(class_idx, int) and class_idx < len(class_names):
                        labels.append(class_names[class_idx])
                    else:
                        labels.append(str(class_label))
                else:
                    labels.append(str(class_label))

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            xticklabels=metrics,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Classes", fontsize=12)

        plt.tight_layout()

        if self.save_plots:
            filename = f"classification_report_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved classification report plot: {filename}")

        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = None,
        title: str = "ROC Curves",
    ) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Convert to binary format
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)

        if y_true_binary.shape[1] == 1:
            # Binary case
            y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, y_true_binary.shape[1]))

        for i, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names else f"Class {i}"

            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.8)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            filename = f"roc_curves_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curves plot: {filename}")

        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = None,
        title: str = "Precision-Recall Curves",
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multi-class classification

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Convert to binary format
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)

        if y_true_binary.shape[1] == 1:
            # Binary case
            y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        # Plot PR curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, y_true_binary.shape[1]))

        for i, color in enumerate(colors):
            precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)

            class_name = class_names[i] if class_names else f"Class {i}"

            ax.plot(
                recall,
                precision,
                color=color,
                linewidth=2,
                label=f"{class_name} (AUC = {pr_auc:.3f})",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            filename = f"pr_curves_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curves plot: {filename}")

        return fig

    def plot_feature_importance(
        self, importance_data: pd.DataFrame, title: str = "Feature Importance", top_n: int = 20
    ) -> plt.Figure:
        """
        Plot feature importance

        Args:
            importance_data: DataFrame with feature importance
            title: Plot title
            top_n: Number of top features to show

        Returns:
            Matplotlib figure
        """
        if importance_data.empty:
            logger.warning("No importance data provided")
            return None

        # Get top N features
        if "mean_importance" in importance_data.columns:
            top_features = importance_data.nlargest(top_n, "mean_importance")
            importance_col = "mean_importance"
        else:
            # Use first numeric column
            numeric_cols = importance_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found in importance data")
                return None
            importance_col = numeric_cols[0]
            top_features = importance_data.nlargest(top_n, importance_col)

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features[importance_col], color="steelblue", alpha=0.8)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"{title} (Top {top_n})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + width * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if self.save_plots:
            filename = f"feature_importance_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance plot: {filename}")

        return fig

    def plot_model_comparison(
        self,
        comparison_data: pd.DataFrame,
        metric: str = "f1_score",
        title: str = "Model Comparison",
    ) -> plt.Figure:
        """
        Plot model comparison based on a metric

        Args:
            comparison_data: DataFrame with model comparison data
            metric: Metric to compare
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if comparison_data.empty:
            logger.warning("No comparison data provided")
            return None

        fig, ax = plt.subplots(figsize=self.figsize_medium)

        # Create bar plot
        bars = ax.bar(
            comparison_data["model"],
            comparison_data["value"],
            color="lightcoral",
            alpha=0.8,
            edgecolor="darkred",
        )

        # Customize plot
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"{title} - {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        if self.save_plots:
            filename = f"model_comparison_{metric}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved model comparison plot: {filename}")

        return fig

    def plot_data_distribution(
        self, data: pd.Series, title: str = "Data Distribution"
    ) -> plt.Figure:
        """
        Plot data distribution

        Args:
            data: Data to plot
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_wide)

        # Bar plot of value counts
        value_counts = data.value_counts()
        bars = ax1.bar(
            range(len(value_counts)),
            value_counts.values,
            color="skyblue",
            alpha=0.8,
            edgecolor="navy",
        )

        ax1.set_xlabel("Categories", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title(f"{title} - Counts", fontsize=12, fontweight="bold")
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Pie chart of proportions
        ax2.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(value_counts))),
        )
        ax2.set_title(f"{title} - Proportions", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if self.save_plots:
            filename = f"data_distribution_{title.lower().replace(' ', '_')}.png"
            plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved data distribution plot: {filename}")

        return fig

    def create_evaluation_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        class_names: List[str] = None,
        model_name: str = "Model",
    ) -> plt.Figure:
        """
        Create comprehensive evaluation dashboard with robust label handling
        """
        try:
            # Determine figure layout based on available data
            if y_proba is not None:
                fig = plt.figure(figsize=(20, 12))
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            else:
                fig = plt.figure(figsize=(15, 8))
                gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

            # Confusion Matrix
            ax1 = fig.add_subplot(gs[0, 0])
            cm = confusion_matrix(y_true, y_pred)

            # Get unique labels in a consistent order
            unique_labels = sorted(list(set(list(y_true) + list(y_pred))))

            # Create label mapping for display
            if class_names:
                # Use class_names as the display names, map to unique_labels
                display_labels = []
                for label in unique_labels:
                    if label in class_names:
                        display_labels.append(label)
                    else:
                        display_labels.append(str(label))
            else:
                display_labels = [str(label) for label in unique_labels]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=display_labels,
                yticklabels=display_labels,
                ax=ax1,
            )
            ax1.set_title("Confusion Matrix", fontweight="bold")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("True")

            # Classification metrics heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            try:
                from sklearn.metrics import classification_report

                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

                # Extract classes that exist in the report
                classes = [
                    k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]
                ]
                metrics = ["precision", "recall", "f1-score"]

                matrix = []
                labels = []
                for class_label in classes:
                    if class_label in report:
                        row = [report[class_label][metric] for metric in metrics]
                        matrix.append(row)
                        labels.append(str(class_label))  # Always use string representation

                if matrix:
                    sns.heatmap(
                        matrix,
                        annot=True,
                        fmt=".3f",
                        cmap="RdYlBu_r",
                        xticklabels=metrics,
                        yticklabels=labels,
                        vmin=0,
                        vmax=1,
                        ax=ax2,
                    )
                ax2.set_title("Classification Metrics", fontweight="bold")
            except Exception as e:
                ax2.text(
                    0.5,
                    0.5,
                    f"Classification metrics unavailable\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Classification Metrics (Error)", fontweight="bold")

            if y_proba is not None:
                try:
                    # ROC Curves
                    ax3 = fig.add_subplot(gs[0, 2])
                    from sklearn.preprocessing import LabelBinarizer

                    lb = LabelBinarizer()
                    y_true_binary = lb.fit_transform(y_true)
                    if y_true_binary.shape[1] == 1:
                        y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])

                    colors = plt.cm.Set1(
                        np.linspace(0, 1, min(y_true_binary.shape[1], len(unique_labels)))
                    )
                    for i, color in enumerate(colors):
                        if i < y_true_binary.shape[1] and i < y_proba.shape[1]:
                            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            label_name = (
                                display_labels[i] if i < len(display_labels) else f"Class {i}"
                            )
                            ax3.plot(
                                fpr,
                                tpr,
                                color=color,
                                linewidth=2,
                                label=f"{label_name} (AUC={roc_auc:.3f})",
                            )

                    ax3.plot([0, 1], [0, 1], "k--", alpha=0.8)
                    ax3.set_xlabel("False Positive Rate")
                    ax3.set_ylabel("True Positive Rate")
                    ax3.set_title("ROC Curves", fontweight="bold")
                    ax3.legend(loc="lower right", fontsize=8)
                    ax3.grid(True, alpha=0.3)

                    # Precision-Recall Curves
                    ax4 = fig.add_subplot(gs[1, 0])
                    for i, color in enumerate(colors):
                        if i < y_true_binary.shape[1] and i < y_proba.shape[1]:
                            precision, recall, _ = precision_recall_curve(
                                y_true_binary[:, i], y_proba[:, i]
                            )
                            pr_auc = auc(recall, precision)
                            label_name = (
                                display_labels[i] if i < len(display_labels) else f"Class {i}"
                            )
                            ax4.plot(
                                recall,
                                precision,
                                color=color,
                                linewidth=2,
                                label=f"{label_name} (AUC={pr_auc:.3f})",
                            )

                    ax4.set_xlabel("Recall")
                    ax4.set_ylabel("Precision")
                    ax4.set_title("Precision-Recall Curves", fontweight="bold")
                    ax4.legend(loc="lower left", fontsize=8)
                    ax4.grid(True, alpha=0.3)

                    # Class distribution comparison
                    ax5 = fig.add_subplot(gs[1, 1:])

                    # Count occurrences of each unique label
                    true_counts = []
                    pred_counts = []
                    for label in unique_labels:
                        true_counts.append(np.sum(y_true == label))
                        pred_counts.append(np.sum(y_pred == label))

                    x = np.arange(len(unique_labels))
                    width = 0.35

                    ax5.bar(x - width / 2, true_counts, width, label="True", alpha=0.8)
                    ax5.bar(x + width / 2, pred_counts, width, label="Predicted", alpha=0.8)

                    ax5.set_xlabel("Classes")
                    ax5.set_ylabel("Count")
                    ax5.set_title("Class Distribution Comparison", fontweight="bold")
                    ax5.set_xticks(x)
                    ax5.set_xticklabels(display_labels, rotation=45, ha="right")
                    ax5.legend()
                    ax5.grid(True, alpha=0.3, axis="y")

                except Exception as e:
                    logger.warning(f"Error creating probability-based plots: {e}")

            fig.suptitle(
                f"Evaluation Dashboard - {model_name}", fontsize=16, fontweight="bold", y=0.98
            )

            if self.save_plots:
                filename = f"evaluation_dashboard_{model_name.lower().replace(' ', '_')}.png"
                plt.savefig(self.plot_dir / filename, dpi=300, bbox_inches="tight")
                logger.info(f"Saved evaluation dashboard: {filename}")

            return fig

        except Exception as e:
            logger.error(f"Error creating evaluation dashboard: {e}")
            # Return a simple error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                f"Dashboard creation failed:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"Evaluation Dashboard - {model_name} (Error)", fontweight="bold")
            return fig


def test_visualization():
    """Test visualization functions with sample data"""
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)

    class_names = ["Cargo", "Tanker", "Fishing"]

    # Test visualizer
    visualizer = ModelVisualizer(save_plots=False)  # Don't save during testing

    logger.info("Testing visualization functions")

    # Test individual plots
    visualizer.plot_confusion_matrix(y_true, y_pred, class_names)
    visualizer.plot_classification_report(y_true, y_pred, class_names)
    visualizer.plot_roc_curves(y_true, y_proba, class_names)
    visualizer.plot_precision_recall_curves(y_true, y_proba, class_names)

    # Test feature importance
    importance_data = pd.DataFrame(
        {"feature_1": [0.15], "feature_2": [0.12], "feature_3": [0.08], "mean_importance": [0.12]},
        index=["feature_1"],
    )

    visualizer.plot_feature_importance(importance_data)

    # Test data distribution
    data_series = pd.Series(y_true, name="vessel_type")
    visualizer.plot_data_distribution(data_series, "Vessel Type Distribution")

    # Test dashboard
    visualizer.create_evaluation_dashboard(y_true, y_pred, y_proba, class_names, "Test Model")

    plt.show()

    return visualizer


if __name__ == "__main__":
    test_visualization()
