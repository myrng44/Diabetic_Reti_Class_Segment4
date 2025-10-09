#!/usr/bin/env python3
"""
Simplified script to train the complete model with SANGO optimization.
This is the easiest way to use the paper implementation.
"""

import os
import sys
import torch
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all required files and modules exist."""
    logger.info("Checking requirements...")

    required_files = [
        'config.py',
        'sango.py',
        'adaptive_gabor.py',
        'enhanced_models.py',
        'train_enhanced_models.py',
        'datasets.py',
        'preprocessing.py'
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
            logger.error(f"❌ Missing: {file}")
        else:
            logger.info(f"✅ Found: {file}")

    if missing:
        logger.error(f"\nMissing files: {missing}")
        logger.error("Please ensure all paper implementation files are present.")
        return False

    # Check data
    from config import CLASSIFICATION_TRAIN_DIR, CLASSIFICATION_TRAIN_CSV

    if not os.path.exists(CLASSIFICATION_TRAIN_DIR):
        logger.error(f"❌ Training data not found: {CLASSIFICATION_TRAIN_DIR}")
        return False

    if not os.path.exists(CLASSIFICATION_TRAIN_CSV):
        logger.error(f"❌ Training labels not found: {CLASSIFICATION_TRAIN_CSV}")
        return False

    logger.info("✅ All requirements met!")
    return True


def train_with_sango_full():
    """
    Complete training pipeline with SANGO optimization.

    This will:
    1. Load and preprocess data with Adaptive Chaotic Gabor
    2. Use SANGO to find optimal hyperparameters (20-50 iterations)
    3. Train the model with best hyperparameters
    4. Perform 5-fold cross validation
    5. Evaluate on test set
    """

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE TRAINING WITH SANGO OPTIMIZATION")
    logger.info("=" * 80)

    # Import after checking requirements
    import config
    from datasets import ClassificationDataset
    from preprocessing import get_training_transforms, get_validation_transforms
    from train_enhanced_models import k_fold_cross_validation

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create directories
    config.os.makedirs(config.RESULTS_DIR, exist_ok=True)
    config.os.makedirs(config.MODELS_DIR, exist_ok=True)
    config.os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Load dataset
    logger.info("\n" + "-" * 80)
    logger.info("STEP 1: Loading Dataset")
    logger.info("-" * 80)

    train_dataset = ClassificationDataset(
        image_dir=config.CLASSIFICATION_TRAIN_DIR,
        csv_file=config.CLASSIFICATION_TRAIN_CSV,
        transform=get_training_transforms()
    )

    logger.info(f"✅ Loaded {len(train_dataset)} training samples")
    logger.info(f"   Image size: {config.TARGET_SIZE}x{config.TARGET_SIZE}")
    logger.info(f"   Classes: {config.CLASSIFICATION_CLASSES}")
    logger.info(f"   Preprocessing: CLAHE + Adaptive Chaotic Gabor")

    # Show class distribution
    from collections import Counter
    label_counts = Counter(train_dataset.labels)
    logger.info("\nClass distribution:")
    for cls_idx, count in sorted(label_counts.items()):
        cls_name = config.CLASSIFICATION_CLASS_NAMES[cls_idx] if cls_idx < len(
            config.CLASSIFICATION_CLASS_NAMES) else f"Class {cls_idx}"
        percentage = (count / len(train_dataset)) * 100
        logger.info(f"  {cls_name}: {count} ({percentage:.1f}%)")

    # Train with SANGO
    logger.info("\n" + "-" * 80)
    logger.info("STEP 2: Training with SANGO Optimization")
    logger.info("-" * 80)
    logger.info("This will take 2-4 hours depending on your hardware.")
    logger.info(f"SANGO will optimize: {list(config.SANGO_HYPERPARAMS.keys())}")
    logger.info(f"Population size: {config.SANGO_POPULATION_SIZE}")
    logger.info(f"Iterations: {config.SANGO_MAX_ITERATIONS}")
    logger.info(f"K-folds: {config.K_FOLDS}")

    # Confirm before starting
    response = input("\n⚠️  This will take several hours. Continue? (y/n): ")
    if response.lower() != 'y':
        logger.info("Training cancelled by user.")
        return

    logger.info("\n🚀 Starting training...\n")

    # Run k-fold cross validation with SANGO
    fold_results = k_fold_cross_validation(
        dataset=train_dataset,
        device=device,
        k_folds=config.K_FOLDS,
        use_sango=True
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)

    # Calculate average results
    avg_f1 = sum(r['f1_score'] for r in fold_results) / len(fold_results)
    avg_acc = sum(r['accuracy'] for r in fold_results) / len(fold_results)
    std_f1 = (sum((r['f1_score'] - avg_f1) ** 2 for r in fold_results) / len(fold_results)) ** 0.5

    logger.info(f"\nCross-Validation Results ({config.K_FOLDS} folds):")
    logger.info(f"  Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"  Average Accuracy: {avg_acc:.4f}")

    logger.info("\nPer-fold results:")
    for i, result in enumerate(fold_results):
        logger.info(f"  Fold {i + 1}: F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}")

    logger.info(f"\n📁 Results saved to: {config.RESULTS_DIR}")
    logger.info(f"📁 Models saved to: {config.MODELS_DIR}")
    logger.info(f"📁 Logs saved to: {config.LOGS_DIR}")

    return fold_results


def train_with_sango_quick():
    """
    Quick training for testing (reduced iterations).
    Useful for debugging or quick experiments.
    """

    logger.info("\n" + "=" * 80)
    logger.info("QUICK TRAINING WITH SANGO (FOR TESTING)")
    logger.info("=" * 80)

    from config import *
    from datasets import ClassificationDataset
    from preprocessing import get_training_transforms
    from train_enhanced_models import k_fold_cross_validation

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    train_dataset = ClassificationDataset(
        image_dir=CLASSIFICATION_TRAIN_DIR,
        csv_file=CLASSIFICATION_TRAIN_CSV,
        transform=get_training_transforms()
    )

    logger.info(f"Loaded {len(train_dataset)} samples")

    # Modify SANGO parameters for quick testing
    logger.info("\n⚡ Quick mode settings:")
    logger.info("  - SANGO iterations: 10 (instead of 50)")
    logger.info("  - SANGO population: 6 (instead of 10)")
    logger.info("  - Training epochs: 5 (instead of 30)")
    logger.info("  - K-folds: 3 (instead of 5)")

    # You would need to modify the function to accept these parameters
    # For now, just inform the user
    logger.warning("\n⚠️  For quick mode, edit config.py:")
    logger.warning("  SANGO_POPULATION_SIZE = 6")
    logger.warning("  SANGO_MAX_ITERATIONS = 10")
    logger.warning("  NUM_EPOCHS = 5")
    logger.warning("  K_FOLDS = 3")

    logger.info("\nTo continue with current settings, use train_with_sango_full()")


def train_without_sango():
    """
    Train paper model without SANGO optimization.
    Uses default hyperparameters - much faster but potentially lower performance.
    """

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PAPER MODEL (WITHOUT SANGO)")
    logger.info("=" * 80)

    from config import *
    from datasets import ClassificationDataset
    from preprocessing import get_training_transforms, get_validation_transforms
    from enhanced_models import PaperMultiModelDR, FocalLoss
    from training import Trainer, Evaluator
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
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

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model with default hyperparameters
    logger.info("\nCreating Paper Model with default hyperparameters:")
    logger.info("  - GRU hidden dim: 128")
    logger.info("  - GRU layers: 2")
    logger.info("  - Dropout: 0.3")
    logger.info("  - Learning rate: 1e-4")

    model = PaperMultiModelDR(
        num_classes=CLASSIFICATION_CLASSES,
        segmentation_classes=SEGMENTATION_CLASSES,
        gru_hidden_dim=128,
        gru_num_layers=2,
        gru_dropout=0.3
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        task_type='classification',
        device=device,
        use_wandb=False
    )

    trainer.setup_training(
        optimizer_type='adamw',
        loss_type='focal',
        scheduler_type='cosine'
    )

    logger.info(f"\nTraining for {NUM_EPOCHS} epochs...")

    history, best_model_path = trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir=MODELS_DIR
    )

    # Evaluate
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = Evaluator(model, device)
    results = evaluator.evaluate_classification(
        test_loader,
        save_results=True,
        save_dir=os.path.join(RESULTS_DIR, "paper_model_no_sango")
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  AUC: {results.get('auc', 0):.4f}")

    return results


def main():
    """Main entry point with menu."""

    print("=" * 80)
    print("DIABETIC RETINOPATHY DETECTION - TRAINING WITH SANGO")
    print("=" * 80)

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix issues above.")
        sys.exit(1)

    print("\nSelect training mode:\n")
    print("1. Full Training with SANGO Optimization (Recommended, ~2-4 hours)")
    print("   - Uses SANGO to find best hyperparameters")
    print("   - 5-fold cross validation")
    print("   - Best results, but slow")

    print("\n2. Quick Test with SANGO (For testing, ~30 minutes)")
    print("   - Reduced SANGO iterations")
    print("   - 3-fold cross validation")
    print("   - Good for debugging")

    print("\n3. Paper Model without SANGO (Fast, ~1 hour)")
    print("   - Uses default hyperparameters")
    print("   - No hyperparameter optimization")
    print("   - Still better than standard model")

    print("\n4. Exit")

    choice = '1'

    if choice == '1':
        train_with_sango_full()
    elif choice == '2':
        print("\n⚠️  For quick mode, please edit config.py first:")
        print("Set: SANGO_POPULATION_SIZE=6, SANGO_MAX_ITERATIONS=10, NUM_EPOCHS=5")
        response = input("Have you done this? (y/n): ")
        if response.lower() == 'y':
            train_with_sango_full()
    elif choice == '3':
        train_without_sango()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()