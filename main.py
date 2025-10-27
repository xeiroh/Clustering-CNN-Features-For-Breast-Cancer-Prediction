#!/usr/bin/env python3
"""
Main pipeline for Clustering CNN Features for Breast Cancer Prediction.

This script orchestrates the complete workflow:
1. Load BreakHis dataset
2. Fine-tune VGG16 model
3. Extract CNN features
4. Apply PCA for dimensionality reduction
5. Perform clustering (KMeans and Swarm Intelligence)
6. Evaluate and visualize results

Author: Gokul Ganesan
Project: Clustering CNN Features for Breast Cancer Prediction
"""

import os
import argparse
import numpy as np
import pickle
import logging
from datetime import datetime

from load_dataset import BreakHisDataLoader
from finetune_vgg16 import VGG16FeatureExtractor
from biological_clustering import SwarmClusterer, hyperparameter_search
from evaluation import ClusteringEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreastCancerClusteringPipeline:
    """Complete pipeline for breast cancer clustering using swarm intelligence."""

    def __init__(self, data_dir='archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
                 output_dir='results'):
        """
        Initialize the pipeline.

        Args:
            data_dir (str): Path to dataset directory
            output_dir (str): Path to save results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create output directories
        self.models_dir = os.path.join(output_dir, 'models')
        self.plots_dir = os.path.join(output_dir, 'plots', self.timestamp)
        self.features_dir = os.path.join(output_dir, 'features')

        for directory in [self.models_dir, self.plots_dir, self.features_dir]:
            os.makedirs(directory, exist_ok=True)

        # Pipeline components
        self.data_loader = None
        self.dataset = None
        self.feature_extractor = None
        self.evaluator = ClusteringEvaluator()

    def load_data(self, validation_split=0.2):
        """
        Load and preprocess the dataset.

        Args:
            validation_split (float): Fraction for validation set
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Loading Dataset")
        logger.info("=" * 80)

        self.data_loader = BreakHisDataLoader(data_dir=self.data_dir)
        self.dataset = self.data_loader.load_dataset(validation_split=validation_split)

        if self.dataset is None:
            raise ValueError("Failed to load dataset. Check data directory.")

        logger.info(f"Dataset loaded successfully!")
        logger.info(f"  Training samples: {len(self.dataset['X_train'])}")
        logger.info(f"  Validation samples: {len(self.dataset['X_val'])}")

    def train_vgg16(self, stage1_epochs=5, stage2_epochs=10):
        """
        Fine-tune VGG16 model.

        Args:
            stage1_epochs (int): Epochs for stage 1 (frozen base)
            stage2_epochs (int): Epochs for stage 2 (full fine-tuning)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Fine-tuning VGG16")
        logger.info("=" * 80)

        self.feature_extractor = VGG16FeatureExtractor(
            num_classes=self.dataset['num_classes']
        )

        # Two-stage training
        history1, history2 = self.feature_extractor.two_stage_training(
            self.dataset['X_train'], self.dataset['y_train'],
            self.dataset['X_val'], self.dataset['y_val'],
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs
        )

        # Save models
        self.feature_extractor.save_feature_extractor(
            os.path.join(self.models_dir, 'feature_extractor.h5')
        )

        logger.info("VGG16 fine-tuning completed!")

    def extract_features(self, n_pca_components=50):
        """
        Extract features and apply PCA.

        Args:
            n_pca_components (int): Number of PCA components
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Extracting Features")
        logger.info("=" * 80)

        # Extract features from all data
        logger.info("Extracting features from training set...")
        features_train = self.feature_extractor.extract_features(self.dataset['X_train'])

        logger.info("Extracting features from validation set...")
        features_val = self.feature_extractor.extract_features(self.dataset['X_val'])

        # Apply PCA
        logger.info(f"Applying PCA with {n_pca_components} components...")
        features_train_pca = self.feature_extractor.apply_pca(features_train, n_components=n_pca_components)
        features_val_pca = self.feature_extractor.pca.transform(features_val)

        # Save features and PCA
        self.feature_extractor.save_pca(os.path.join(self.models_dir, 'pca.pkl'))

        np.save(os.path.join(self.features_dir, 'features_train_pca.npy'), features_train_pca)
        np.save(os.path.join(self.features_dir, 'features_val_pca.npy'), features_val_pca)

        logger.info("Feature extraction completed!")

        return features_train_pca, features_val_pca

    def perform_clustering(self, features, labels, method='flock', **kwargs):
        """
        Perform clustering on features.

        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): True labels
            method (str): Clustering method ('kmeans' or 'flock')
            **kwargs: Additional clustering parameters

        Returns:
            np.ndarray: Cluster labels
        """
        logger.info(f"Performing {method} clustering...")

        clusterer = SwarmClusterer(method=method, **kwargs)
        cluster_labels = clusterer.fit_predict(features)

        logger.info(f"{method} clustering completed!")
        logger.info(f"  Number of clusters formed: {clusterer.n_clusters}")

        return cluster_labels

    def run_full_pipeline(self, stage1_epochs=5, stage2_epochs=10,
                         n_pca_components=50, perform_hyperparameter_search=False):
        """
        Run the complete pipeline.

        Args:
            stage1_epochs (int): Epochs for VGG16 stage 1
            stage2_epochs (int): Epochs for VGG16 stage 2
            n_pca_components (int): Number of PCA components
            perform_hyperparameter_search (bool): Whether to perform hyperparameter search
        """
        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Train VGG16
            self.train_vgg16(stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs)

            # Step 3: Extract features
            features_train_pca, features_val_pca = self.extract_features(n_pca_components)

            # Step 4: Clustering
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Clustering")
            logger.info("=" * 80)

            # Combine train and val for clustering
            all_features = np.vstack([features_train_pca, features_val_pca])
            all_labels = np.concatenate([self.dataset['y_train'], self.dataset['y_val']])

            # KMeans clustering (k=2 for benign vs malignant)
            logger.info("\n--- KMeans Clustering (k=2) ---")
            kmeans_labels_2 = self.perform_clustering(
                all_features, all_labels,
                method='kmeans', n_clusters=2, random_state=42
            )

            # KMeans clustering (k=8 for subtypes)
            logger.info("\n--- KMeans Clustering (k=8) ---")
            kmeans_labels_8 = self.perform_clustering(
                all_features, all_labels,
                method='kmeans', n_clusters=8, random_state=42
            )

            # Hyperparameter search for Flock by Leader (if requested)
            flock_params = {'max_distance': 0.1, 'min_samples': 40}
            if perform_hyperparameter_search:
                logger.info("\n--- Hyperparameter Search for Flock by Leader ---")
                param_grid = {
                    'max_distance': [0.05, 0.2],
                    'min_samples': [20, 60]
                }
                best_params = hyperparameter_search(all_features, param_grid, n_trials=10)
                flock_params = {
                    'max_distance': best_params['max_distance'],
                    'min_samples': best_params['min_samples']
                }

            # Flock by Leader clustering
            logger.info(f"\n--- Flock by Leader Clustering ---")
            logger.info(f"Parameters: max_distance={flock_params['max_distance']:.4f}, "
                       f"min_samples={flock_params['min_samples']}")
            flock_labels = self.perform_clustering(
                all_features, all_labels,
                method='flock', **flock_params, random_state=42
            )

            # Step 5: Evaluation
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: Evaluation")
            logger.info("=" * 80)

            # Evaluate KMeans k=2
            self.evaluator.evaluate_clustering(
                kmeans_labels_2, all_labels, 'KMeans (k=2)'
            )

            # Evaluate KMeans k=8
            self.evaluator.evaluate_clustering(
                kmeans_labels_8, all_labels, 'KMeans (k=8)'
            )

            # Evaluate Flock by Leader
            self.evaluator.evaluate_clustering(
                flock_labels, all_labels, 'Flock by Leader'
            )

            # Step 6: Visualizations
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: Visualizations")
            logger.info("=" * 80)

            # PCA visualizations
            self.evaluator.plot_pca_2d(
                all_features, all_labels, kmeans_labels_2,
                title='KMeans (k=2)',
                save_path=os.path.join(self.plots_dir, 'pca_kmeans_k2.png')
            )

            self.evaluator.plot_pca_2d(
                all_features, all_labels, flock_labels,
                title='Flock by Leader',
                save_path=os.path.join(self.plots_dir, 'pca_flock.png')
            )

            # t-SNE visualizations
            self.evaluator.plot_tsne_2d(
                all_features, all_labels, kmeans_labels_2,
                title='KMeans (k=2)',
                save_path=os.path.join(self.plots_dir, 'tsne_kmeans_k2.png')
            )

            self.evaluator.plot_tsne_2d(
                all_features, all_labels, flock_labels,
                title='Flock by Leader',
                save_path=os.path.join(self.plots_dir, 'tsne_flock.png')
            )

            # Confusion matrices
            kmeans_metrics = self.evaluator.results['KMeans (k=2)']
            flock_metrics = self.evaluator.results['Flock by Leader']

            self.evaluator.plot_confusion_matrix(
                kmeans_metrics['confusion_matrix'], 'KMeans (k=2)',
                save_path=os.path.join(self.plots_dir, 'confusion_kmeans_k2.png')
            )

            self.evaluator.plot_confusion_matrix(
                flock_metrics['confusion_matrix'], 'Flock by Leader',
                save_path=os.path.join(self.plots_dir, 'confusion_flock.png')
            )

            # Comparison plot
            self.evaluator.compare_methods(
                save_path=os.path.join(self.plots_dir, 'method_comparison.png')
            )

            # Print summary
            self.evaluator.print_summary()

            # Save results
            results_file = os.path.join(self.output_dir, f'results_{self.timestamp}.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump(self.evaluator.results, f)
            logger.info(f"Results saved to {results_file}")

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Results saved in: {self.output_dir}")
            logger.info(f"Plots saved in: {self.plots_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Breast Cancer Clustering using CNN Features and Swarm Intelligence'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Path to save results'
    )
    parser.add_argument(
        '--stage1-epochs',
        type=int,
        default=5,
        help='Number of epochs for VGG16 stage 1 (frozen base)'
    )
    parser.add_argument(
        '--stage2-epochs',
        type=int,
        default=10,
        help='Number of epochs for VGG16 stage 2 (full fine-tuning)'
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=50,
        help='Number of PCA components'
    )
    parser.add_argument(
        '--hyperparameter-search',
        action='store_true',
        help='Perform hyperparameter search for Flock by Leader'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = BreastCancerClusteringPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    pipeline.run_full_pipeline(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        n_pca_components=args.pca_components,
        perform_hyperparameter_search=args.hyperparameter_search
    )


if __name__ == '__main__':
    main()
