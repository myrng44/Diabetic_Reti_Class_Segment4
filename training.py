"""
Training and evaluation module for diabetic retinopathy detection.
Supports both classification and segmentation tasks with advanced features.
"""

import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # For experiment tracking (optional)

from config import *
from model import create_segmentation_model, create_classification_model, create_loss_function


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MetricsTracker:
    """Track and compute various metrics during training."""

    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, predictions, targets, loss):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()

        self.predictions.append(predictions)
        self.targets.append(targets)
        self.losses.append(loss)

    def compute_metrics(self):
        if not self.predictions:
            return {}

        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        avg_loss = np.mean(self.losses)

        metrics = {'loss': avg_loss}

        if self.task_type == 'classification':
            # Classification metrics
            if all_preds.dim() > 1:
                pred_classes = torch.argmax(all_preds, dim=1)
            else:
                pred_classes = all_preds

            accuracy = (pred_classes == all_targets).float().mean().item()
            metrics['accuracy'] = accuracy

            # Convert to numpy for sklearn metrics
            pred_np = pred_classes.numpy()
            target_np = all_targets.numpy()

            # F1 score (macro and weighted)
            metrics['f1_macro'] = f1_score(target_np, pred_np, average='macro')
            metrics['f1_weighted'] = f1_score(target_np, pred_np, average='weighted')

            # AUC (if probabilities available)
            if all_preds.dim() > 1 and all_preds.shape[1] > 2:
                try:
                    probs = torch.softmax(all_preds, dim=1).numpy()
                    auc = roc_auc_score(target_np, probs, multi_class='ovr', average='macro')
                    metrics['auc'] = auc
                except:
                    pass

        elif self.task_type == 'segmentation':
            # Segmentation metrics
            if all_preds.dim() > 2:
                pred_binary = (torch.sigmoid(all_preds) > SEGMENTATION_THRESHOLD).float()
            else:
                pred_binary = (all_preds > SEGMENTATION_THRESHOLD).float()

            # Dice coefficient
            intersection = (pred_binary * all_targets).sum()
            union = pred_binary.sum() + all_targets.sum()
            dice = (2.0 * intersection) / (union + 1e-8)
            metrics['dice'] = dice.item()

            # IoU (Jaccard Index)
            intersection = (pred_binary * all_targets).sum()
            union = pred_binary.sum() + all_targets.sum() - intersection
            iou = intersection / (union + 1e-8)
            metrics['iou'] = iou.item()

            # Per-class metrics
            if all_preds.dim() == 4:  # (N, C, H, W)
                for c in range(all_preds.shape[1]):
                    pred_c = pred_binary[:, c]
                    target_c = all_targets[:, c]

                    intersection_c = (pred_c * target_c).sum()
                    union_c = pred_c.sum() + target_c.sum()
                    dice_c = (2.0 * intersection_c) / (union_c + 1e-8)

                    class_name = SEGMENTATION_CLASS_NAMES[c] if c < len(SEGMENTATION_CLASS_NAMES) else f"class_{c}"
                    metrics[f'dice_{class_name}'] = dice_c.item()

        return metrics


class Trainer:
    """Main trainer class for both classification and segmentation."""

    def __init__(self, model, train_loader, val_loader, task_type='classification',
                 device=DEVICE, use_wandb=False, experiment_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.device = device
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name or f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Move model to device
        self.model.to(self.device)

        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = EarlyStopping()

        # Metrics tracking
        self.train_metrics = MetricsTracker(task_type)
        self.val_metrics = MetricsTracker(task_type)

        # History
        self.history = {'train': {}, 'val': {}}

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="diabetic-retinopathy",
                name=self.experiment_name,
                config={
                    'model': self.model.__class__.__name__,
                    'task_type': task_type,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'num_epochs': NUM_EPOCHS
                }
            )

    def setup_training(self, optimizer_type='adam', loss_type=None, scheduler_type='plateau'):
        """Setup optimizer, loss function, and scheduler."""

        # Setup optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                momentum=0.9
            )

        # Setup loss function
        if loss_type is None:
            if self.task_type == 'classification':
                loss_type = 'focal'
            else:
                loss_type = 'combined'

        self.criterion = create_loss_function(loss_type)

        # Setup scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
            )

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            if self.task_type == 'classification':
                images, labels, _ = batch
            else:  # segmentation
                images, masks, _ = batch
                labels = masks

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            self.train_metrics.update(outputs, labels, loss.item())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return self.train_metrics.compute_metrics()

    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)

            for batch in pbar:
                if self.task_type == 'classification':
                    images, labels, _ = batch
                else:  # segmentation
                    images, masks, _ = batch
                    labels = masks

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Update metrics
                self.val_metrics.update(outputs, labels, loss.item())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return self.val_metrics.compute_metrics()

    def train(self, num_epochs=NUM_EPOCHS, save_best=True, save_dir=MODELS_DIR):
        """Main training loop."""

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()

            # Save metrics to history
            for key, value in train_metrics.items():
                if key not in self.history['train']:
                    self.history['train'][key] = []
                self.history['train'][key].append(value)

            for key, value in val_metrics.items():
                if key not in self.history['val']:
                    self.history['val'][key] = []
                self.history['val'][key].append(value)

            # Log to wandb
            if self.use_wandb:
                log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                log_dict['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict)

            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")

            if self.task_type == 'classification':
                print(f"Train Acc: {train_metrics.get('accuracy', 0):.4f}")
                print(f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")
                print(f"Val F1: {val_metrics.get('f1_macro', 0):.4f}")
            else:
                print(f"Train Dice: {train_metrics.get('dice', 0):.4f}")
                print(f"Val Dice: {val_metrics.get('dice', 0):.4f}")
                print(f"Val IoU: {val_metrics.get('iou', 0):.4f}")

            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = os.path.join(save_dir, f"best_{self.experiment_name}.pth")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'config': {
                        'model_name': self.model.__class__.__name__,
                        'task_type': self.task_type
                    }
                }, best_model_path)

                print(f"Best model saved to {best_model_path}")

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        final_model_path = os.path.join(save_dir, f"final_{self.experiment_name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'model_name': self.model.__class__.__name__,
                'task_type': self.task_type
            }
        }, final_model_path)

        print(f"Training completed. Final model saved to {final_model_path}")

        if self.use_wandb:
            wandb.finish()

        return self.history, best_model_path


class Evaluator:
    """Evaluation class for trained models."""

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def evaluate_classification(self, test_loader, save_results=True, save_dir=RESULTS_DIR):
        """Evaluate classification model."""
        all_preds = []
        all_labels = []
        all_probs = []
        all_filenames = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images, labels, filenames = batch
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_filenames.extend(filenames)

        # Calculate metrics
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=CLASSIFICATION_CLASS_NAMES,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # AUC score
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            auc = 0.0

        results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'filenames': all_filenames
        }

        # Print results
        print(f"\nClassification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=CLASSIFICATION_CLASS_NAMES))

        # Save results
        if save_results:
            self._save_classification_results(results, save_dir)

        return results

    def evaluate_segmentation(self, test_loader, save_results=True, save_dir=RESULTS_DIR):
        """Evaluate segmentation model."""
        all_dice_scores = []
        all_iou_scores = []
        per_class_dice = {name: [] for name in SEGMENTATION_CLASS_NAMES}

        results_data = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images, masks, filenames = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                pred_masks = torch.sigmoid(outputs) > SEGMENTATION_THRESHOLD

                # Calculate batch metrics
                batch_size = images.shape[0]

                for i in range(batch_size):
                    pred_mask = pred_masks[i]  # (C, H, W)
                    true_mask = masks[i]       # (C, H, W)

                    # Overall Dice and IoU
                    intersection = (pred_mask * true_mask).sum()
                    union = pred_mask.sum() + true_mask.sum()
                    dice = (2.0 * intersection) / (union + 1e-8)

                    intersection_iou = (pred_mask * true_mask).sum()
                    union_iou = pred_mask.sum() + true_mask.sum() - intersection_iou
                    iou = intersection_iou / (union_iou + 1e-8)

                    all_dice_scores.append(dice.item())
                    all_iou_scores.append(iou.item())

                    # Per-class metrics
                    class_dice_scores = {}
                    for c, class_name in enumerate(SEGMENTATION_CLASS_NAMES):
                        pred_c = pred_mask[c]
                        true_c = true_mask[c]

                        intersection_c = (pred_c * true_c).sum()
                        union_c = pred_c.sum() + true_c.sum()
                        dice_c = (2.0 * intersection_c) / (union_c + 1e-8)

                        per_class_dice[class_name].append(dice_c.item())
                        class_dice_scores[f'dice_{class_name}'] = dice_c.item()

                    # Save result for this image
                    result_entry = {
                        'filename': filenames[i],
                        'dice': dice.item(),
                        'iou': iou.item(),
                        **class_dice_scores
                    }
                    results_data.append(result_entry)

                    # Save prediction overlay if requested
                    if save_results:
                        self._save_prediction_overlay(
                            images[i], pred_mask, true_mask,
                            filenames[i], save_dir
                        )

        # Calculate average metrics
        avg_dice = np.mean(all_dice_scores)
        avg_iou = np.mean(all_iou_scores)

        avg_class_dice = {}
        for class_name, scores in per_class_dice.items():
            avg_class_dice[f'dice_{class_name}'] = np.mean(scores)

        results = {
            'dice': avg_dice,
            'iou': avg_iou,
            'dice_scores': all_dice_scores,
            'iou_scores': all_iou_scores,
            'per_class_dice': avg_class_dice,
            'detailed_results': results_data
        }

        # Print results
        print(f"\nSegmentation Results:")
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print("\nPer-class Dice scores:")
        for class_name, score in avg_class_dice.items():
            print(f"  {class_name}: {score:.4f}")

        # Save results
        if save_results:
            self._save_segmentation_results(results, save_dir)

        return results

    def _save_classification_results(self, results, save_dir):
        """Save classification results."""
        os.makedirs(save_dir, exist_ok=True)

        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(results['confusion_matrix'],
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=CLASSIFICATION_CLASS_NAMES,
                   yticklabels=CLASSIFICATION_CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

        # Save detailed results
        results_file = os.path.join(save_dir, 'classification_results.json')
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'accuracy': results['accuracy'],
            'auc': results['auc'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'predictions': results['predictions'],
            'labels': results['labels']
        }

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save predictions CSV
        import pandas as pd
        df = pd.DataFrame({
            'filename': results['filenames'],
            'true_label': results['labels'],
            'predicted_label': results['predictions']
        })

        # Add probability columns
        for i, class_name in enumerate(CLASSIFICATION_CLASS_NAMES):
            df[f'prob_{class_name}'] = [prob[i] for prob in results['probabilities']]

        df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

        print(f"Classification results saved to {save_dir}")

    def _save_segmentation_results(self, results, save_dir):
        """Save segmentation results."""
        os.makedirs(save_dir, exist_ok=True)

        # Save detailed results CSV
        import pandas as pd
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(os.path.join(save_dir, 'segmentation_results.csv'), index=False)

        # Save summary statistics
        summary = {
            'average_dice': results['dice'],
            'average_iou': results['iou'],
            'per_class_dice': results['per_class_dice'],
            'dice_std': float(np.std(results['dice_scores'])),
            'iou_std': float(np.std(results['iou_scores']))
        }

        with open(os.path.join(save_dir, 'segmentation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Segmentation results saved to {save_dir}")

    def _save_prediction_overlay(self, image_tensor, pred_mask, true_mask, filename, save_dir):
        """Save prediction overlay visualization."""
        overlay_dir = os.path.join(save_dir, 'prediction_overlays')
        os.makedirs(overlay_dir, exist_ok=True)

        # Convert tensors to numpy
        image = image_tensor.cpu().permute(1, 2, 0).numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        pred_np = pred_mask.cpu().numpy()
        true_np = true_mask.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Prediction
        pred_combined = np.sum(pred_np, axis=0)
        axes[1].imshow(image)
        axes[1].imshow(pred_combined, alpha=0.5, cmap='hot')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # Ground truth
        true_combined = np.sum(true_np, axis=0)
        axes[2].imshow(image)
        axes[2].imshow(true_combined, alpha=0.5, cmap='hot')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

        plt.tight_layout()

        # Save
        save_path = os.path.join(overlay_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train']['loss'], label='Train Loss')
    axes[0, 0].plot(history['val']['loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy or Dice
    if 'accuracy' in history['train']:
        axes[0, 1].plot(history['train']['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val']['accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
    elif 'dice' in history['train']:
        axes[0, 1].plot(history['train']['dice'], label='Train Dice')
        axes[0, 1].plot(history['val']['dice'], label='Val Dice')
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].set_ylabel('Dice')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 or IoU
    if 'f1_macro' in history['val']:
        axes[1, 0].plot(history['val']['f1_macro'], label='Val F1 Macro')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_ylabel('F1 Score')
    elif 'iou' in history['val']:
        axes[1, 0].plot(history['val']['iou'], label='Val IoU')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].set_ylabel('IoU')

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Additional metric
    if 'auc' in history['val']:
        axes[1, 1].plot(history['val']['auc'], label='Val AUC')
        axes[1, 1].set_title('AUC Score')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Training module loaded successfully!")
    print("Use this module to train classification or segmentation models.")