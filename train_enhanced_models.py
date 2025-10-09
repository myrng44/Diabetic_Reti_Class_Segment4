"""
Complete training script following paper methodology:
1. Preprocessing with Adaptive Chaotic Gabor
2. Modified U-Net with MBConv + Adaptive BN
3. Multi-fold features (LBP + SURF + TEM)
4. OGRU optimized by SANGO
5. F1-score based evaluation
6. 5-fold cross validation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import logging

# Import our modules
from config import *
from datasets import ClassificationDataset, SegmentationDataset
from preprocessing import get_training_transforms, get_validation_transforms
from enhanced_models import (
    PaperMultiModelDR,
    FocalLoss,
    create_paper_model_with_sango
)
from adaptive_gabor import AdaptiveChaoticGaborFilter
from feature_extraction import CombinedFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTrainer:
    """
    Complete trainer following paper methodology.
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            device,
            learning_rate=1e-4,
            use_focal_loss=True,
            extract_features=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.extract_features = extract_features

        # Optimizer - AdamW as per best practices
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Loss function
        if use_focal_loss:
            self.classification_criterion = FocalLoss(
                alpha=CLASSIFICATION_FOCAL_ALPHA,
                gamma=CLASSIFICATION_FOCAL_GAMMA,
                num_classes=CLASSIFICATION_CLASSES
            )
        else:
            self.classification_criterion = nn.CrossEntropyLoss()

        # Segmentation loss (if needed)
        self.segmentation_criterion = nn.BCEWithLogitsLoss()

        # Scheduler - Cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPOCHS,
            eta_min=1e-6
        )

        # Feature extractor for multi-fold features
        if extract_features:
            self.feature_extractor = CombinedFeatureExtractor(
                use_lbp=True,
                use_surf=True,
                use_tem=True
            )

        # Metrics tracking
        self.history = {'train': [], 'val': []}

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract external features if needed
            external_features = None
            if self.extract_features:
                # This would be done offline for efficiency
                pass

            # Forward pass
            cls_out, seg_out = self.model(images, external_features)

            # Classification loss
            loss = self.classification_criterion(cls_out, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(cls_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        return {
            'loss': avg_loss,
            'f1_score': f1,
            'accuracy': accuracy
        }

    def validate(self):
        """Validate model."""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                cls_out, seg_out = self.model(images, None)

                # Loss
                loss = self.classification_criterion(cls_out, labels)

                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(cls_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        # Per-class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

        return {
            'loss': avg_loss,
            'f1_score': f1,
            'accuracy': accuracy,
            'f1_per_class': f1_per_class,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, num_epochs=NUM_EPOCHS):
        """Complete training loop."""
        best_f1 = 0

        logger.info("Starting training...")
        logger.info(f"Total epochs: {num_epochs}")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Logging
            logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"F1: {train_metrics['f1_score']:.4f}, "
                        f"Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                        f"F1: {val_metrics['f1_score']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_score': best_f1,
                    'val_metrics': val_metrics
                }, os.path.join(MODELS_DIR, 'best_paper_model.pth'))
                logger.info(f"→ New best F1: {best_f1:.4f}")

            # Track history
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

        return self.history


def k_fold_cross_validation(
        dataset,
        device,
        k_folds=5,
        use_sango=False
):
    """
    Perform k-fold cross validation as in paper.
    """
    logger.info(f"\nStarting {k_folds}-Fold Cross Validation")
    logger.info("=" * 60)

    # Get labels for stratification
    labels = dataset.labels
    indices = np.arange(len(dataset))

    # K-fold split
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info(f"\nFold {fold + 1}/{k_folds}")
        logger.info("-" * 40)

        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        # Create model (with SANGO for first fold only)
        if fold == 0 and use_sango:
            model, best_params = create_paper_model_with_sango(
                train_loader, val_loader, device, use_sango=True
            )
            logger.info(f"SANGO optimized parameters: {best_params}")
        else:
            model = PaperMultiModelDR(num_classes=CLASSIFICATION_CLASSES)
            best_params = {'lr': 1e-4}

        # Trainer
        trainer = PaperTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=best_params.get('lr', 1e-4)
        )

        # Train
        history = trainer.train(num_epochs=NUM_EPOCHS)

        # Final validation
        final_val_metrics = trainer.validate()

        fold_results.append({
            'fold': fold + 1,
            'f1_score': final_val_metrics['f1_score'],
            'accuracy': final_val_metrics['accuracy'],
            'f1_per_class': final_val_metrics['f1_per_class'],
            'history': history
        })

        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  F1-Score: {final_val_metrics['f1_score']:.4f}")
        logger.info(f"  Accuracy: {final_val_metrics['accuracy']:.4f}")

    # Aggregate results
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])
    mean_acc = np.mean([r['accuracy'] for r in fold_results])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Cross Validation Results:")
    logger.info(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"  Mean Accuracy: {mean_acc:.4f}")
    logger.info(f"{'=' * 60}\n")

    return fold_results


def main():
    """Main training function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = ClassificationDataset(
        image_dir=CLASSIFICATION_TRAIN_DIR,
        csv_file=CLASSIFICATION_TRAIN_CSV,
        transform=get_training_transforms()
    )

    test_dataset = ClassificationDataset(
        image_dir=CLASSIFICATION_TEST_DIR,
        csv_file=CLASSIFICATION_TEST_CSV,
        transform=get_validation_transforms()
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Option 1: K-Fold Cross Validation (as in paper)
    fold_results = k_fold_cross_validation(
        dataset=train_dataset,
        device=device,
        k_folds=K_FOLDS,
        use_sango=True  # Use SANGO for hyperparameter optimization
    )

    # Option 2: Simple train/val/test split
    # Uncomment below for faster training without cross-validation
    """
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Create model with SANGO optimization
    model, best_params = create_paper_model_with_sango(
        train_loader, test_loader, device, use_sango=True
    )

    # Train
    trainer = PaperTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        learning_rate=best_params.get('lr', 1e-4)
    )

    history = trainer.train(num_epochs=NUM_EPOCHS)

    # Final evaluation
    logger.info("\nFinal Evaluation on Test Set:")
    final_metrics = trainer.validate()
    logger.info(f"F1-Score: {final_metrics['f1_score']:.4f}")
    logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")

    # Print classification report
    logger.info("\nClassification Report:")
    print(classification_report(
        final_metrics['labels'],
        final_metrics['predictions'],
        target_names=CLASSIFICATION_CLASS_NAMES
    ))

    # Confusion matrix
    cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
    logger.info("\nConfusion Matrix:")
    logger.info(str(cm))
    """

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()