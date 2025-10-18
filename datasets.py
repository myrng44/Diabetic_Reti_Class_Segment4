#!/usr/bin/env python3
"""
Dataset classes for diabetic retinopathy classification and segmentation.
Supports both tasks with proper data loading and preprocessing.
"""

import os
import pandas as pd
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
import tifffile
from config import *
from preprocessing import FundusPreprocessor, PreprocessingPipeline

class ClassificationDataset(Dataset):
    """Dataset for diabetic retinopathy classification."""

    def __init__(self, image_dir, csv_file=None, image_paths=None, labels=None,
                 transform=None, preprocessor=None):
        """
        Args:
            image_dir: Directory containing images
            csv_file: CSV file with image names and labels
            image_paths: List of image paths (alternative to csv_file)
            labels: List of labels (alternative to csv_file)
            transform: Albumentations transform pipeline
            preprocessor: Image preprocessor instance
        """
        self.image_dir = image_dir
        self.transform = transform
        self.preprocessor = preprocessor or FundusPreprocessor()

        # Load data
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Handle IDRiD format: "Image name,Retinopathy grade,Risk of macular edema"
            if 'Image name' in df.columns and 'Retinopathy grade' in df.columns:
                # IDRiD format
                image_names = df['Image name'].tolist()
                labels = df['Retinopathy grade'].tolist()

                # Add .jpg extension if not present
                self.image_names = []
                for img_name in image_names:
                    if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        img_name += '.jpg'
                    self.image_names.append(img_name)
                self.labels = labels

            else:
                # Generic format: first column = image names, second column = labels
                self.image_names = df.iloc[:, 0].tolist()
                self.labels = df.iloc[:, 1].tolist()

        elif image_paths and labels:
            self.image_names = [os.path.basename(p) for p in image_paths]
            self.labels = labels
        else:
            # Auto-discover images
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            self.image_names = []
            for ext in extensions:
                self.image_names.extend([
                    os.path.basename(p) for p in glob(os.path.join(image_dir, f"*{ext}"))
                ])
            self.labels = [0] * len(self.image_names)  # Default labels

        print(f"Found {len(self.image_names)} images for classification")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Apply preprocessing
        processed_image, _ = self.preprocessor.preprocess_image(image_rgb)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=processed_image)
            image_tensor = augmented["image"]
        else:
            # Convert to tensor manually
            image_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).float() / 255.0

        return image_tensor, torch.tensor(label, dtype=torch.long), img_name


class SegmentationDataset(Dataset):
    """Dataset for diabetic retinopathy segmentation."""

    def __init__(self, image_dir, mask_dirs, transform=None, preprocessor=None):
        """
        Args:
            image_dir: Directory containing original images
            mask_dirs: Dictionary mapping lesion type to mask directory
            transform: Albumentations transform pipeline
            preprocessor: Image preprocessor instance
        """
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.preprocessor = preprocessor or FundusPreprocessor()

        # Get all image paths
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.image_names = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks for each lesion type
        masks = []
        for lesion_type in ['MA', 'HE', 'EX']:  # Microaneurysms, Hemorrhages, Exudates
            mask_path = os.path.join(self.mask_dirs[lesion_type], image_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            masks.append(mask)

        # Stack masks into a single array
        masks = np.stack(masks, axis=0)

        # Preprocess image
        if self.preprocessor:
            image, _ = self.preprocessor.preprocess_image(image)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            image = augmented['image']
            masks = augmented['masks']

        return image, masks, image_name


class MultiTaskDataset(Dataset):
    """Dataset that combines both classification and segmentation tasks."""

    def __init__(self, cls_dataset, seg_dataset, mode='classification'):
        """
        Args:
            cls_dataset: Classification dataset instance
            seg_dataset: Segmentation dataset instance
            mode: 'classification', 'segmentation', or 'both'
        """
        self.cls_dataset = cls_dataset
        self.seg_dataset = seg_dataset
        self.mode = mode

        if mode == 'classification':
            self.length = len(cls_dataset)
        elif mode == 'segmentation':
            self.length = len(seg_dataset)
        else:  # both
            self.length = min(len(cls_dataset), len(seg_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'classification':
            return self.cls_dataset[idx]
        elif self.mode == 'segmentation':
            return self.seg_dataset[idx]
        else:  # both
            cls_data = self.cls_dataset[idx % len(self.cls_dataset)]
            seg_data = self.seg_dataset[idx % len(self.seg_dataset)]
            return cls_data, seg_data


class FeatureDataset(Dataset):
    """Dataset for traditional feature-based classification."""

    def __init__(self, features, labels, feature_names=None):
        """
        Args:
            features: Feature matrix (N x D)
            labels: Labels (N,)
            feature_names: List of feature names
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.feature_names = feature_names

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ================================
# DATA SPLITTING AND CROSS-VALIDATION
# ================================

class DataSplitter:
    """Handles data splitting for training, validation, and testing."""

    def __init__(self, random_state=SEED):
        self.random_state = random_state

    def split_classification_data(self, dataset, test_size=0.2, val_size=0.1):
        """
        Split classification dataset into train/val/test.
        Args:
            dataset: ClassificationDataset instance
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
        Returns:
            train_indices, val_indices, test_indices
        """
        labels = dataset.labels
        indices = list(range(len(dataset)))

        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=labels,
            random_state=self.random_state
        )

        # Get labels for remaining data
        train_val_labels = [labels[i] for i in train_val_indices]

        # Second split: separate train and validation
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, stratify=train_val_labels,
            random_state=self.random_state
        )

        return train_indices, val_indices, test_indices

    def create_kfold_splits(self, dataset, k=K_FOLDS):
        """
        Create k-fold cross-validation splits.
        Args:
            dataset: Dataset instance
            k: Number of folds
        Returns:
            List of (train_indices, val_indices) tuples
        """
        labels = dataset.labels if hasattr(dataset, 'labels') else [0] * len(dataset)
        indices = list(range(len(dataset)))

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        splits = []

        for train_idx, val_idx in skf.split(indices, labels):
            splits.append((train_idx.tolist(), val_idx.tolist()))

        return splits


# ================================
# DATA LOADERS
# ================================

def create_data_loaders(dataset, train_indices, val_indices, test_indices=None,
                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Create data loaders for train/val/test splits."""
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    loaders = {'train': train_loader, 'val': val_loader}

    if test_indices is not None:
        test_dataset = Subset(dataset, test_indices)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        loaders['test'] = test_loader

    return loaders


def create_balanced_loader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Create a balanced data loader for imbalanced datasets."""
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # Extract labels from dataset
        labels = []
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(label)

    # Calculate class weights
    class_counts = Counter(labels)
    total_samples = len(labels)

    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)

    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )


# ================================
# DATASET ANALYSIS
# ================================

def analyze_dataset(dataset, save_path=None):
    """Analyze dataset statistics and class distribution."""
    import matplotlib.pyplot as plt
    from collections import Counter

    analysis = {}

    if hasattr(dataset, 'labels'):
        labels = dataset.labels
        class_counts = Counter(labels)
        analysis['class_distribution'] = class_counts
        analysis['total_samples'] = len(labels)
        analysis['num_classes'] = len(class_counts)

        print("Dataset Analysis:")
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Number of classes: {analysis['num_classes']}")
        print("\nClass distribution:")
        for class_idx, count in sorted(class_counts.items()):
            percentage = (count / analysis['total_samples']) * 100
            class_name = CLASSIFICATION_CLASS_NAMES[class_idx] if class_idx < len(CLASSIFICATION_CLASS_NAMES) else f"Class {class_idx}"
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

        # Plot class distribution
        if save_path:
            plt.figure(figsize=(10, 6))
            classes = [CLASSIFICATION_CLASS_NAMES[i] if i < len(CLASSIFICATION_CLASS_NAMES) else f"Class {i}"
                      for i in sorted(class_counts.keys())]
            counts = [class_counts[i] for i in sorted(class_counts.keys())]

            plt.bar(classes, counts)
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Class distribution plot saved to {save_path}")

    return analysis


def create_sample_visualization(dataset, num_samples=5, save_path=None):
    """Create visualization of sample images from dataset."""
    import matplotlib.pyplot as plt
    import random

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(indices):
        if hasattr(dataset, 'labels'):
            # Classification dataset
            image, label, filename = dataset[idx]

            # Convert tensor to numpy for display
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image

            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'{filename}')
            axes[0, i].axis('off')

            if isinstance(label, torch.Tensor):
                label = label.item()

            class_name = CLASSIFICATION_CLASS_NAMES[label] if label < len(CLASSIFICATION_CLASS_NAMES) else f"Class {label}"
            axes[1, i].text(0.5, 0.5, f'Label: {class_name}',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, i].transAxes, fontsize=12)
            axes[1, i].axis('off')
        else:
            # Segmentation dataset
            image, mask, filename = dataset[idx]

            # Convert tensor to numpy for display
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image

            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'{filename}')
            axes[0, i].axis('off')

            # Show combined mask
            if isinstance(mask, torch.Tensor):
                mask_np = mask.sum(dim=0).numpy()
            else:
                mask_np = mask.sum(axis=-1)

            axes[1, i].imshow(mask_np, cmap='hot', alpha=0.7)
            axes[1, i].set_title('Lesion Masks')
            axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test dataset creation
    from preprocessing import get_training_transforms, get_validation_transforms

    # Test classification dataset
    print("Testing classification dataset...")
    if os.path.exists(CLASSIFICATION_TRAIN_DIR):
        cls_dataset = ClassificationDataset(
            image_dir=CLASSIFICATION_TRAIN_DIR,
            csv_file=CLASSIFICATION_TRAIN_CSV,
            transform=get_validation_transforms()
        )

        analyze_dataset(cls_dataset, "class_distribution.png")
        create_sample_visualization(cls_dataset, save_path="classification_samples.png")

    # Test segmentation dataset
    print("\nTesting segmentation dataset...")
    if os.path.exists(SEGMENTATION_TRAIN_IMG_DIR):
        seg_dataset = SegmentationDataset(
            image_dir=SEGMENTATION_TRAIN_IMG_DIR,
            mask_dirs=SEGMENTATION_TRAIN_MASK_DIRS,
            transform=get_validation_transforms()
        )

        create_sample_visualization(seg_dataset, save_path="segmentation_samples.png")

    print("Dataset testing completed!")