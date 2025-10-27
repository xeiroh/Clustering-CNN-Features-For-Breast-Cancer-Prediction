"""
Load and preprocess BreakHis breast cancer histopathological image dataset.

Author: Gokul Ganesan
Project: Clustering CNN Features for Breast Cancer Prediction
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreakHisDataLoader:
    """Load and preprocess BreakHis dataset."""

    def __init__(self, data_dir='archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
                 img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Path to the dataset directory
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for data generators
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

    def load_images_from_directory(self, directory, label):
        """
        Load images from a directory with a specific label.

        Args:
            directory (str): Directory containing images
            label (int): Label for the images (0=benign, 1=malignant)

        Returns:
            tuple: (images, labels, subtype_labels, paths)
        """
        images = []
        labels = []
        subtype_labels = []
        paths = []

        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return np.array([]), np.array([]), np.array([]), []

        # Get all subdirectories (subtypes)
        subtypes = [d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))]

        for subtype_idx, subtype in enumerate(sorted(subtypes)):
            subtype_path = os.path.join(directory, subtype)

            # Walk through all subdirectories to find images
            for root, _, files in os.walk(subtype_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        try:
                            # Load and preprocess image
                            img = load_img(img_path, target_size=self.img_size)
                            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]

                            images.append(img_array)
                            labels.append(label)
                            subtype_labels.append(subtype_idx + (label * 4))  # 0-3 benign, 4-7 malignant
                            paths.append(img_path)
                        except Exception as e:
                            logger.error(f"Error loading image {img_path}: {e}")
                            continue

        return np.array(images), np.array(labels), np.array(subtype_labels), paths

    def load_dataset(self, validation_split=0.2):
        """
        Load the complete dataset with train/validation split.

        Args:
            validation_split (float): Fraction of data to use for validation

        Returns:
            dict: Dictionary containing train/val data and labels
        """
        logger.info("Loading benign images...")
        benign_images, benign_labels, benign_subtypes, benign_paths = \
            self.load_images_from_directory(self.benign_dir, label=0)

        logger.info("Loading malignant images...")
        malignant_images, malignant_labels, malignant_subtypes, malignant_paths = \
            self.load_images_from_directory(self.malignant_dir, label=1)

        # Combine datasets
        if len(benign_images) == 0 or len(malignant_images) == 0:
            logger.error("No images loaded. Please check the data directory.")
            return None

        all_images = np.concatenate([benign_images, malignant_images], axis=0)
        all_labels = np.concatenate([benign_labels, malignant_labels], axis=0)
        all_subtypes = np.concatenate([benign_subtypes, malignant_subtypes], axis=0)
        all_paths = benign_paths + malignant_paths

        logger.info(f"Loaded {len(all_images)} images total")
        logger.info(f"Benign: {len(benign_images)}, Malignant: {len(malignant_images)}")

        # Split into train and validation sets
        X_train, X_val, y_train, y_val, subtypes_train, subtypes_val, paths_train, paths_val = \
            train_test_split(all_images, all_labels, all_subtypes, all_paths,
                           test_size=validation_split, stratify=all_labels,
                           random_state=42)

        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'subtypes_train': subtypes_train,
            'subtypes_val': subtypes_val,
            'paths_train': paths_train,
            'paths_val': paths_val,
            'num_classes': 2,
            'num_subtypes': 8
        }

    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation for training.

        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels

        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )

        # No augmentation for validation
        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_generator, val_generator


def main():
    """Test the data loader."""
    loader = BreakHisDataLoader()
    dataset = loader.load_dataset()

    if dataset is not None:
        print(f"Training samples: {len(dataset['X_train'])}")
        print(f"Validation samples: {len(dataset['X_val'])}")
        print(f"Image shape: {dataset['X_train'][0].shape}")
        print(f"Number of classes: {dataset['num_classes']}")
        print(f"Number of subtypes: {dataset['num_subtypes']}")


if __name__ == '__main__':
    main()
