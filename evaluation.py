"""
Evaluation metrics and visualization for clustering results.

Author: Gokul Ganesan
Project: Clustering CNN Features for Breast Cancer Prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ClusteringEvaluator:
    """Evaluate clustering performance and generate visualizations."""

    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}

    def map_clusters_to_labels(self, cluster_labels, true_labels):
        """
        Map cluster IDs to true class labels using majority voting.

        Args:
            cluster_labels (np.ndarray): Predicted cluster labels
            true_labels (np.ndarray): True class labels

        Returns:
            dict: Mapping from cluster ID to class label
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_to_class = {}

        for cluster_id in unique_clusters:
            # Get all true labels for this cluster
            mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[mask]

            # Majority vote
            if len(cluster_true_labels) > 0:
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                cluster_to_class[cluster_id] = majority_class
            else:
                cluster_to_class[cluster_id] = 0  # Default

        return cluster_to_class

    def evaluate_clustering(self, cluster_labels, true_labels, method_name='Clustering'):
        """
        Evaluate clustering performance against true labels.

        Args:
            cluster_labels (np.ndarray): Predicted cluster labels
            true_labels (np.ndarray): True class labels
            method_name (str): Name of the clustering method

        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {method_name}...")

        # Map clusters to classes
        cluster_to_class = self.map_clusters_to_labels(cluster_labels, true_labels)
        predicted_labels = np.array([cluster_to_class[c] for c in cluster_labels])

        # Calculate metrics
        metrics = {
            'method': method_name,
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predicted_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, predicted_labels, average='weighted', zero_division=0),
            'adjusted_rand_index': adjusted_rand_score(true_labels, cluster_labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, cluster_labels),
            'n_clusters': len(np.unique(cluster_labels))
        }

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(true_labels, predicted_labels)

        # Classification report
        metrics['classification_report'] = classification_report(
            true_labels, predicted_labels,
            target_names=['Benign', 'Malignant'],
            zero_division=0
        )

        self.results[method_name] = metrics

        logger.info(f"{method_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Number of clusters: {metrics['n_clusters']}")

        return metrics

    def plot_confusion_matrix(self, confusion_mat, method_name, save_path=None):
        """
        Plot confusion matrix.

        Args:
            confusion_mat (np.ndarray): Confusion matrix
            method_name (str): Name of the method
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant']
        )
        plt.title(f'Confusion Matrix - {method_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()
        plt.close()

    def plot_pca_2d(self, features, labels, cluster_labels=None, title='PCA Visualization',
                   save_path=None):
        """
        Create 2D PCA visualization.

        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): True labels
            cluster_labels (np.ndarray): Cluster assignments (optional)
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        logger.info("Generating 2D PCA visualization...")

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot by true labels
        scatter1 = axes[0].scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='coolwarm', alpha=0.6, s=30
        )
        axes[0].set_title(f'{title} - True Labels')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter1, ax=axes[0], label='Class (0=Benign, 1=Malignant)')

        # Plot by cluster labels if provided
        if cluster_labels is not None:
            scatter2 = axes[1].scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=cluster_labels, cmap='viridis', alpha=0.6, s=30
            )
            axes[1].set_title(f'{title} - Cluster Labels')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter2, ax=axes[1], label='Cluster ID')
        else:
            axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA plot saved to {save_path}")

        plt.show()
        plt.close()

    def plot_tsne_2d(self, features, labels, cluster_labels=None, title='t-SNE Visualization',
                    save_path=None, perplexity=30):
        """
        Create 2D t-SNE visualization.

        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): True labels
            cluster_labels (np.ndarray): Cluster assignments (optional)
            title (str): Plot title
            save_path (str): Path to save the plot
            perplexity (int): t-SNE perplexity parameter
        """
        logger.info("Generating 2D t-SNE visualization (this may take a while)...")

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(features)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot by true labels
        scatter1 = axes[0].scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='coolwarm', alpha=0.6, s=30
        )
        axes[0].set_title(f'{title} - True Labels')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter1, ax=axes[0], label='Class (0=Benign, 1=Malignant)')

        # Plot by cluster labels if provided
        if cluster_labels is not None:
            scatter2 = axes[1].scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=cluster_labels, cmap='viridis', alpha=0.6, s=30
            )
            axes[1].set_title(f'{title} - Cluster Labels')
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter2, ax=axes[1], label='Cluster ID')
        else:
            axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE plot saved to {save_path}")

        plt.show()
        plt.close()

    def compare_methods(self, save_path=None):
        """
        Compare multiple clustering methods.

        Args:
            save_path (str): Path to save the comparison plot
        """
        if len(self.results) == 0:
            logger.warning("No results to compare")
            return

        # Extract metrics for comparison
        methods = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score',
                          'adjusted_rand_index', 'normalized_mutual_info']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            values = [self.results[m][metric] for m in methods]
            axes[idx].bar(methods, values, color=['skyblue', 'lightcoral'][:len(methods)])
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_ylim(0, 1.0)
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        plt.suptitle('Clustering Methods Comparison', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()
        plt.close()

    def print_summary(self):
        """Print a summary of all evaluation results."""
        if len(self.results) == 0:
            logger.warning("No results to summarize")
            return

        logger.info("\n" + "=" * 80)
        logger.info("CLUSTERING EVALUATION SUMMARY")
        logger.info("=" * 80)

        for method_name, metrics in self.results.items():
            logger.info(f"\n{method_name}:")
            logger.info(f"  Accuracy:                {metrics['accuracy']:.4f}")
            logger.info(f"  Precision:               {metrics['precision']:.4f}")
            logger.info(f"  Recall:                  {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:                {metrics['f1_score']:.4f}")
            logger.info(f"  Adjusted Rand Index:     {metrics['adjusted_rand_index']:.4f}")
            logger.info(f"  Normalized Mutual Info:  {metrics['normalized_mutual_info']:.4f}")
            logger.info(f"  Number of Clusters:      {metrics['n_clusters']}")

            logger.info(f"\n  Classification Report:\n{metrics['classification_report']}")

        logger.info("=" * 80 + "\n")


def main():
    """Test the evaluation module."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200

    # Simulate features
    features = np.random.randn(n_samples, 50)

    # True labels (binary)
    true_labels = np.random.randint(0, 2, n_samples)

    # Simulate clustering results
    kmeans_labels = np.random.randint(0, 2, n_samples)
    flock_labels = np.random.randint(0, 3, n_samples)

    # Initialize evaluator
    evaluator = ClusteringEvaluator()

    # Evaluate both methods
    evaluator.evaluate_clustering(kmeans_labels, true_labels, 'KMeans')
    evaluator.evaluate_clustering(flock_labels, true_labels, 'Flock by Leader')

    # Visualizations
    evaluator.plot_pca_2d(features, true_labels, kmeans_labels, 'KMeans')
    evaluator.compare_methods()
    evaluator.print_summary()


if __name__ == '__main__':
    main()
