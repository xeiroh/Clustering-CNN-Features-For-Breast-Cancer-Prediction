"""
Fine-tune VGG16 model and extract features from breast cancer images.

Author: Gokul Ganesan
Project: Clustering CNN Features for Breast Cancer Prediction
"""

import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.decomposition import PCA
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VGG16FeatureExtractor:
    """Fine-tune VGG16 and extract features for clustering."""

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize the VGG16 feature extractor.

        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.feature_extractor = None
        self.pca = None

    def build_model(self, freeze_base=True):
        """
        Build VGG16 model with custom top layers.

        Args:
            freeze_base (bool): Whether to freeze base VGG16 layers

        Returns:
            Model: Keras model
        """
        # Load VGG16 with ImageNet weights
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base layers if requested
        if freeze_base:
            for layer in base_model.layers:
                layer.trainable = False
            logger.info("Base VGG16 layers frozen")
        else:
            logger.info("Base VGG16 layers unfrozen for fine-tuning")

        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # 512-D feature vector
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create full model
        model = Model(inputs=base_model.input, outputs=predictions)

        logger.info(f"Model built with {len(model.layers)} layers")
        return model

    def compile_model(self, learning_rate=0.0001):
        """
        Compile the model.

        Args:
            learning_rate (float): Learning rate for optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Model compiled with learning rate: {learning_rate}")

    def train(self, X_train, y_train, X_val, y_val, epochs=5,
             batch_size=32, model_path='models/vgg16_finetuned.h5'):
        """
        Train the model.

        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            model_path (str): Path to save the best model

        Returns:
            History: Training history
        """
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        logger.info(f"Training for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return history

    def two_stage_training(self, X_train, y_train, X_val, y_val,
                          stage1_epochs=5, stage2_epochs=10):
        """
        Two-stage training: first freeze base, then unfreeze all layers.

        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            stage1_epochs (int): Epochs for stage 1 (frozen base)
            stage2_epochs (int): Epochs for stage 2 (unfrozen)

        Returns:
            tuple: (history1, history2)
        """
        logger.info("=== Stage 1: Training with frozen base ===")
        self.model = self.build_model(freeze_base=True)
        self.compile_model(learning_rate=0.001)
        history1 = self.train(
            X_train, y_train, X_val, y_val,
            epochs=stage1_epochs,
            model_path='models/vgg16_stage1.h5'
        )

        logger.info("\n=== Stage 2: Fine-tuning entire model ===")
        # Unfreeze all layers
        for layer in self.model.layers:
            layer.trainable = True

        # Recompile with lower learning rate
        self.compile_model(learning_rate=0.0001)
        history2 = self.train(
            X_train, y_train, X_val, y_val,
            epochs=stage2_epochs,
            model_path='models/vgg16_finetuned.h5'
        )

        return history1, history2

    def build_feature_extractor(self):
        """
        Build feature extractor from the trained model.
        Extracts 512-D features from GlobalAveragePooling2D layer.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Extract features from GlobalAveragePooling2D layer
        feature_layer = None
        for layer in self.model.layers:
            if isinstance(layer, GlobalAveragePooling2D):
                feature_layer = layer
                break

        if feature_layer is None:
            raise ValueError("GlobalAveragePooling2D layer not found")

        self.feature_extractor = Model(
            inputs=self.model.input,
            outputs=feature_layer.output
        )

        logger.info(f"Feature extractor built, output shape: {self.feature_extractor.output_shape}")

    def extract_features(self, X, batch_size=32):
        """
        Extract features from images.

        Args:
            X: Input images
            batch_size (int): Batch size for extraction

        Returns:
            np.ndarray: Extracted features
        """
        if self.feature_extractor is None:
            self.build_feature_extractor()

        logger.info(f"Extracting features from {len(X)} images...")
        features = self.feature_extractor.predict(X, batch_size=batch_size, verbose=1)
        logger.info(f"Extracted features shape: {features.shape}")

        return features

    def apply_pca(self, features, n_components=50):
        """
        Apply PCA to reduce feature dimensionality.

        Args:
            features: Feature vectors
            n_components (int): Number of PCA components

        Returns:
            np.ndarray: PCA-transformed features
        """
        logger.info(f"Applying PCA with {n_components} components...")
        self.pca = PCA(n_components=n_components, random_state=42)
        features_pca = self.pca.fit_transform(features)

        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_variance:.4f}")
        logger.info(f"PCA features shape: {features_pca.shape}")

        return features_pca

    def save_feature_extractor(self, path='models/feature_extractor.h5'):
        """Save the feature extractor model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.feature_extractor is not None:
            self.feature_extractor.save(path)
            logger.info(f"Feature extractor saved to {path}")

    def save_pca(self, path='models/pca.pkl'):
        """Save the PCA transformer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.pca is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.pca, f)
            logger.info(f"PCA saved to {path}")

    def load_feature_extractor(self, path='models/feature_extractor.h5'):
        """Load a saved feature extractor."""
        from tensorflow.keras.models import load_model
        self.feature_extractor = load_model(path)
        logger.info(f"Feature extractor loaded from {path}")

    def load_pca(self, path='models/pca.pkl'):
        """Load a saved PCA transformer."""
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        logger.info(f"PCA loaded from {path}")


def main():
    """Test the VGG16 feature extractor."""
    from load_dataset import BreakHisDataLoader

    # Load data
    loader = BreakHisDataLoader()
    dataset = loader.load_dataset()

    if dataset is None:
        logger.error("Failed to load dataset")
        return

    # Initialize and train model
    extractor = VGG16FeatureExtractor(num_classes=dataset['num_classes'])

    # Two-stage training
    history1, history2 = extractor.two_stage_training(
        dataset['X_train'], dataset['y_train'],
        dataset['X_val'], dataset['y_val'],
        stage1_epochs=5,
        stage2_epochs=10
    )

    # Extract features
    features_train = extractor.extract_features(dataset['X_train'])
    features_val = extractor.extract_features(dataset['X_val'])

    # Apply PCA
    features_train_pca = extractor.apply_pca(features_train, n_components=50)
    features_val_pca = extractor.pca.transform(features_val)

    # Save models
    extractor.save_feature_extractor()
    extractor.save_pca()

    logger.info("Feature extraction pipeline completed!")


if __name__ == '__main__':
    main()
