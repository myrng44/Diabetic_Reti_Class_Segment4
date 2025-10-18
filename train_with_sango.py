"""
Training module for enhanced models with SANGO optimization
Handles both classification and segmentation tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import os
from config import *
from enhanced_models import PaperMultiModelDR, FocalLoss, create_paper_model_with_sango
from datasets import SegmentationDataset, ClassificationDataset


class Trainer:
    """Training class for enhanced models - FIXED VERSION"""

    def __init__(self, model, train_loader, val_loader, device, task_type='classification'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.task_type = task_type

        # Loss functions
        self.criterion_cls = FocalLoss(num_classes=CLASSIFICATION_CLASSES)
        self.criterion_seg = nn.BCEWithLogitsLoss()

        # Optimizer and scheduler (will be set in setup_training)
        self.optimizer = None
        self.scheduler = None

    def setup_training(self, learning_rate=LEARNING_RATE, optimizer_type='adamw',
                       scheduler_type='cosine'):
        """Setup optimizer and scheduler"""

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=WEIGHT_DECAY
            )

        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=NUM_EPOCHS
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )

    def train_epoch(self):
        """Train for one epoch - FIXED to handle variable dataset outputs"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_seg_loss = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch_data in enumerate(pbar):
            # FIXED: Handle variable unpacking based on dataset type
            if len(batch_data) == 4:
                # Classification dataset with masks
                images, labels, masks, _ = batch_data
            elif len(batch_data) == 3:
                # Could be classification (image, label, filename) or segmentation
                if isinstance(batch_data[1], torch.Tensor) and batch_data[1].dim() > 0:
                    # Check if second item is label (1D) or mask (3D)
                    if batch_data[1].dim() == 1:
                        # Classification: (image, label, filename)
                        images, labels, _ = batch_data
                        masks = torch.zeros(
                            images.size(0), SEGMENTATION_CLASSES,
                            images.size(2), images.size(3)
                        )
                    else:
                        # Segmentation: (image, mask, filename)
                        images, masks, _ = batch_data
                        labels = torch.zeros(images.size(0), dtype=torch.long)
                else:
                    images, labels, _ = batch_data
                    masks = torch.zeros(
                        images.size(0), SEGMENTATION_CLASSES,
                        images.size(2), images.size(3)
                    )
            else:
                # Only 2 items
                images, labels = batch_data[:2]
                masks = torch.zeros(
                    images.size(0), SEGMENTATION_CLASSES,
                    images.size(2), images.size(3)
                )

            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            cls_out, seg_out = self.model(images)

            # Calculate losses
            loss_cls = self.criterion_cls(cls_out, labels)
            loss_seg = self.criterion_seg(seg_out, masks)

            # Combined loss
            loss = loss_cls + SEGMENTATION_BCE_WEIGHT * loss_seg

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_seg_loss += loss_seg.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'seg': f'{loss_seg.item():.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)

        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'seg_loss': avg_seg_loss
        }

    def validate_epoch(self):
        """Validate for one epoch - FIXED"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for batch_data in pbar:
                # FIXED: Handle variable unpacking
                if len(batch_data) == 4:
                    images, labels, masks, _ = batch_data
                elif len(batch_data) == 3:
                    if isinstance(batch_data[1], torch.Tensor) and batch_data[1].dim() == 1:
                        images, labels, _ = batch_data
                        masks = torch.zeros(
                            images.size(0), SEGMENTATION_CLASSES,
                            images.size(2), images.size(3)
                        )
                    else:
                        images, masks, _ = batch_data
                        labels = torch.zeros(images.size(0), dtype=torch.long)
                else:
                    images, labels = batch_data[:2]
                    masks = torch.zeros(
                        images.size(0), SEGMENTATION_CLASSES,
                        images.size(2), images.size(3)
                    )

                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                cls_out, seg_out = self.model(images)

                # Calculate losses
                loss_cls = self.criterion_cls(cls_out, labels)
                loss_seg = self.criterion_seg(seg_out, masks)
                loss = loss_cls + SEGMENTATION_BCE_WEIGHT * loss_seg

                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(cls_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)

        # Calculate accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()

        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def train(self, num_epochs=NUM_EPOCHS, save_dir=MODELS_DIR):
        """Complete training loop"""
        os.makedirs(save_dir, exist_ok=True)

        best_f1 = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log metrics
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_score'])

            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1-Score: {val_metrics['f1_score']:.4f}")

            # Save best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_score': best_f1,
                    'accuracy': val_metrics['accuracy']
                }, model_path)
                print(f"✓ Saved best model (F1: {best_f1:.4f})")

        return history


def k_fold_cross_validation(dataset, device, k_folds=K_FOLDS, use_sango=True):
    """K-fold cross validation with optional SANGO optimization"""
    print(f"\n{'=' * 70}")
    print(f"K-FOLD CROSS VALIDATION (K={k_folds})")
    print(f"{'=' * 70}\n")

    # Get labels for stratified split
    labels = dataset.labels if hasattr(dataset, 'labels') else [0] * len(dataset)
    indices = np.arange(len(dataset))

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), 1):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold}/{k_folds}")
        print(f"{'=' * 70}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

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

        # Run SANGO only on first fold
        if fold == 1 and use_sango:
            print("\n🔍 Running SANGO optimization (only on first fold)...")
            try:
                from enhanced_sango import EnhancedSANGO
                sango = EnhancedSANGO(
                    param_ranges={
                        'hidden_dim1': (128, 512),
                        'hidden_dim2': (64, 256),
                        'dropout': (0.1, 0.5),
                        'lr': (1e-5, 1e-3)
                    },
                    population_size=15,
                    max_iterations=80
                )

                def fitness_function(params):
                    try:
                        model = create_paper_model_with_sango(
                            hidden_dim1=int(params['hidden_dim1']),
                            hidden_dim2=int(params['hidden_dim2']),
                            dropout=float(params['dropout']),
                            lr=float(params['lr'])
                        ).to(device)

                        trainer = Trainer(model, train_loader, val_loader, device)
                        trainer.setup_training(learning_rate=float(params['lr']))

                        # Quick evaluation (3 epochs)
                        for _ in range(3):
                            trainer.train_epoch()
                        metrics = trainer.validate_epoch()

                        # Clean up
                        del model
                        torch.cuda.empty_cache()

                        return -metrics['f1_score']  # Negative because SANGO minimizes
                    except Exception as e:
                        print(f"Error in fitness function: {str(e)}")
                        return float('inf')

                best_params = sango.optimize(fitness_function)
                print("\nBest parameters found by SANGO:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")

                # Create model with best parameters
                model = create_paper_model_with_sango(**best_params).to(device)
                learning_rate = best_params['lr']

            except Exception as e:
                print(f"\nError during SANGO optimization: {str(e)}")
                print("Falling back to default parameters...")
                model = PaperMultiModelDR().to(device)
                learning_rate = LEARNING_RATE
        else:
            # Use best params from first fold if available
            if fold > 1 and use_sango and 'best_params' in locals():
                print(f"\n📋 Using optimized params from Fold 1")
                model = create_paper_model_with_sango(**best_params).to(device)
                learning_rate = best_params['lr']
            else:
                print(f"\n📋 Using default parameters")
                model = PaperMultiModelDR().to(device)
                learning_rate = LEARNING_RATE

        # Setup trainer
        trainer = Trainer(model, train_loader, val_loader, device)
        trainer.setup_training(learning_rate=learning_rate)

        try:
            # Train
            print(f"\n🚀 Training Fold {fold}...")
            history = trainer.train(num_epochs=NUM_EPOCHS)

            # Get final validation metrics
            final_metrics = trainer.validate_epoch()

            fold_results.append({
                'fold': fold,
                'f1_score': final_metrics['f1_score'],
                'accuracy': final_metrics['accuracy'],
                'loss': final_metrics['loss']
            })

            print(f"\n✓ Fold {fold} Complete:")
            print(f"  F1-Score: {final_metrics['f1_score']:.4f}")
            print(f"  Accuracy: {final_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            continue

        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()

    # Print summary
    if fold_results:
        print(f"\n{'=' * 70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 70}")

        avg_f1 = np.mean([r['f1_score'] for r in fold_results])
        std_f1 = np.std([r['f1_score'] for r in fold_results])
        avg_acc = np.mean([r['accuracy'] for r in fold_results])

        print(f"\nAverage F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Average Accuracy: {avg_acc:.4f}")

        print("\nPer-fold results:")
        for result in fold_results:
            print(f"  Fold {result['fold']}: F1={result['f1_score']:.4f}, "
                  f"Acc={result['accuracy']:.4f}")

    return fold_results


def train_with_cross_validation(task_type='both'):
    """Main training function with k-fold cross validation"""
    print("\n======================================================================")
    print(f"K-FOLD CROSS VALIDATION (K={K_FOLDS})")
    print("======================================================================\n")

    # Load all data
    if task_type in ['classification', 'both']:
        dataset = ClassificationDataset(
            data_dir=CLASSIFICATION_TRAIN_DIR,
            labels_file=CLASSIFICATION_TRAIN_CSV
        )
        print(f"Found {len(dataset)} images for classification")

    if task_type in ['segmentation', 'both']:
        seg_dataset = SegmentationDataset(
            image_dir=SEGMENTATION_TRAIN_IMG_DIR,
            mask_dirs=SEGMENTATION_TRAIN_MASK_DIRS
        )
        print(f"Found {len(seg_dataset)} images for segmentation")

    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # Prepare cross validation
    if task_type in ['classification', 'both']:
        labels = [label for _, label, _ in dataset]
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kfold.split(np.zeros(len(dataset)), labels))

    f1_scores = []
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n======================================================================")
        print(f"FOLD {fold}/{K_FOLDS}")
        print("======================================================================")

        # Use 90% of data for training, 10% for validation
        train_size = int(0.9 * len(train_idx))
        train_indices = train_idx[:train_size]
        val_indices = train_idx[train_size:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

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

        # Run SANGO only on first fold
        if fold == 1:
            print("\n🔍 Running SANGO optimization (only on first fold)...")
            print("Using EnhancedSANGO")
            try:
                from enhanced_sango import EnhancedSANGO
                sango = EnhancedSANGO(
                    param_ranges={
                        'hidden_dim1': (128, 512),
                        'hidden_dim2': (64, 256),
                        'dropout': (0.1, 0.5),
                        'lr': (1e-5, 1e-3)
                    },
                    population_size=15,
                    max_iterations=80
                )

                def fitness_function(params):
                    try:
                        model = create_paper_model_with_sango(
                            hidden_dim1=int(params['hidden_dim1']),
                            hidden_dim2=int(params['hidden_dim2']),
                            dropout=float(params['dropout'])
                        ).to(device)

                        trainer = Trainer(model, train_loader, val_loader, device)
                        trainer.setup_training(learning_rate=float(params['lr']))

                        # Quick evaluation (3 epochs)
                        for _ in range(3):
                            trainer.train_epoch()
                        metrics = trainer.validate_epoch()

                        # Clean up
                        del model
                        torch.cuda.empty_cache()

                        return -metrics['f1_score']  # Negative because SANGO minimizes
                    except Exception as e:
                        print(f"Error in fitness function: {str(e)}")
                        return float('inf')

                best_params = sango.optimize(fitness_function)
                print("\nBest parameters found by SANGO:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")

                # Create model with best parameters
                model = create_paper_model_with_sango(**best_params).to(device)
                learning_rate = best_params['lr']

            except Exception as e:
                print(f"\nError during SANGO optimization: {str(e)}")
                print("Falling back to default parameters...")
                model = PaperMultiModelDR().to(device)
                learning_rate = LEARNING_RATE
        else:
            # Use best params from first fold if available
            if 'best_params' in locals():
                model = create_paper_model_with_sango(**best_params).to(device)
                learning_rate = best_params['lr']
            else:
                model = PaperMultiModelDR().to(device)
                learning_rate = LEARNING_RATE

        # Setup trainer with proper error handling
        trainer = Trainer(model, train_loader, val_loader, device)
        trainer.setup_training(learning_rate=learning_rate)

        try:
            # Train
            print(f"\n🚀 Training Fold {fold}...")
            history = trainer.train(num_epochs=NUM_EPOCHS)

            # Get final validation metrics
            final_metrics = trainer.validate_epoch()

            f1_scores.append(final_metrics['f1_score'])
            accuracies.append(final_metrics['accuracy'])

            print(f"\n✓ Fold {fold} Complete:")
            print(f"  F1-Score: {final_metrics['f1_score']:.4f}")
            print(f"  Accuracy: {final_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            continue

        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()

    # Print summary
    if f1_scores:
        print(f"\n{'=' * 70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 70}")

        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_acc = np.mean(accuracies)

        print(f"\nAverage F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Average Accuracy: {avg_acc:.4f}")

        print("\nPer-fold results:")
        for fold, (f1, acc) in enumerate(zip(f1_scores, accuracies), 1):
            print(f"  Fold {fold}: F1={f1:.4f}, Acc={acc:.4f}")

    return {
        'f1_scores': f1_scores,
        'accuracies': accuracies
    }


if __name__ == "__main__":
    print("Testing training module...")

    from datasets import ClassificationDataset
    from preprocessing import get_training_transforms, get_validation_transforms

    # Test on small subset
    if os.path.exists(CLASSIFICATION_TRAIN_DIR):
        dataset = ClassificationDataset(
            image_dir=CLASSIFICATION_TRAIN_DIR,
            csv_file=CLASSIFICATION_TRAIN_CSV,
            transform=get_training_transforms()
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Test with 2 folds for quick testing
        results = k_fold_cross_validation(
            dataset=dataset,
            device=device,
            k_folds=2,
            use_sango=True  # Set to True to test SANGO
        )

        print("\n✓ Training module test completed!")