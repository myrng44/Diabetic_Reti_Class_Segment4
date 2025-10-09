#!/usr/bin/env python3
"""
Main pipeline for diabetic retinopathy detection.
Combines classification and segmentation with advanced features.
"""

import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime

# Import our modules
from config import *
from preprocessing import (
    get_training_transforms, get_validation_transforms, get_segmentation_transforms,
    preprocess_dataset_batch
)
from feature_extraction import CombinedFeatureExtractor, create_feature_dataset
from datasets import (
    ClassificationDataset, SegmentationDataset, DataSplitter,
    create_data_loaders, create_balanced_loader, analyze_dataset
)
from model import create_segmentation_model, create_classification_model
from training import Trainer, Evaluator, plot_training_history
from gradcam import ExplainabilityPipeline, create_explanation_dashboard

def setup_device():
    """Setup device and print system info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device

def train_classification_model(args):
    """Train classification model for DR grading."""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)

    device = setup_device()

    # Determine model type
    use_paper_model = args.use_paper_model if hasattr(args, 'use_paper_model') else False
    use_sango = args.use_sango if hasattr(args, 'use_sango') and use_paper_model else False

    # Create dataset
    print("Loading classification dataset...")

    train_dataset = ClassificationDataset(
        image_dir=args.classification_train_dir or CLASSIFICATION_TRAIN_DIR,
        csv_file=args.classification_train_csv or CLASSIFICATION_TRAIN_CSV,
        transform=get_training_transforms(),
    )

    test_dataset = ClassificationDataset(
        image_dir=args.classification_test_dir or CLASSIFICATION_TEST_DIR,
        csv_file=args.classification_test_csv or CLASSIFICATION_TEST_CSV,
        transform=get_validation_transforms(),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Analyze dataset
    analyze_dataset(train_dataset, os.path.join(RESULTS_DIR, "classification_train_analysis.png"))

    # Split dataset
    splitter = DataSplitter()

    if use_paper_model and use_sango:
        # Use SANGO with paper model
        print("\n" + "="*60)
        print("USING PAPER MODEL WITH SANGO OPTIMIZATION")
        print("="*60)

        try:
            from train_paper_model import k_fold_cross_validation

            fold_results = k_fold_cross_validation(
                dataset=train_dataset,
                device=device,
                k_folds=args.k_folds if hasattr(args, 'k_folds') else K_FOLDS,
                use_sango=True
            )
            return fold_results
        except ImportError:
            print("Error: train_paper_model.py not found. Falling back to standard model.")
            use_paper_model = False

    if use_paper_model:
        # Use paper model without SANGO
        print("Using Paper Model (without SANGO optimization)")

        try:
            from enhanced_models import PaperMultiModelDR, FocalLoss

            # Split data
            train_indices, val_indices, _ = splitter.split_classification_data(
                train_dataset, test_size=0.0, val_size=0.2
            )

            loaders = create_data_loaders(train_dataset, train_indices, val_indices)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            loaders['test'] = test_loader

            # Create paper model
            model = PaperMultiModelDR(
                num_classes=CLASSIFICATION_CLASSES,
                segmentation_classes=SEGMENTATION_CLASSES
            )

            # Create trainer with Focal Loss
            trainer = Trainer(
                model=model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                task_type='classification',
                device=device,
                use_wandb=args.use_wandb,
                experiment_name="paper_model_classification"
            )

            trainer.setup_training(
                optimizer_type='adamw',
                loss_type='focal',
                scheduler_type='cosine'
            )

            history, best_model_path = trainer.train(
                num_epochs=args.epochs or NUM_EPOCHS,
                save_dir=MODELS_DIR
            )

            # Evaluate
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            evaluator = Evaluator(model, device)
            results = evaluator.evaluate_classification(
                loaders['test'],
                save_results=True,
                save_dir=os.path.join(RESULTS_DIR, "paper_model_evaluation")
            )

            return best_model_path

        except ImportError:
            print("Error: paper_models.py not found. Falling back to standard model.")
            use_paper_model = False

    # Standard model training (original code)
    if args.use_cross_validation:
        print(f"Using {K_FOLDS}-fold cross-validation...")
        splits = splitter.create_kfold_splits(train_dataset, k=K_FOLDS)

        best_models = []
        fold_histories = []

        for fold, (train_indices, val_indices) in enumerate(splits):
            print(f"\nTraining fold {fold + 1}/{K_FOLDS}")

            loaders = create_data_loaders(train_dataset, train_indices, val_indices)

            model = create_classification_model(
                model_type="cnn_lstm",
                backbone_name=args.backbone or CLASSIFICATION_BACKBONE,
                use_attention=not args.no_attention
            )

            trainer = Trainer(
                model=model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                task_type='classification',
                device=device,
                use_wandb=args.use_wandb,
                experiment_name=f"classification_fold_{fold+1}"
            )

            trainer.setup_training(
                optimizer_type=args.optimizer or 'adam',
                loss_type=args.loss_type or 'focal',
                scheduler_type=args.scheduler or 'plateau'
            )

            history, best_model_path = trainer.train(
                num_epochs=args.epochs or NUM_EPOCHS,
                save_dir=MODELS_DIR
            )

            best_models.append(best_model_path)
            fold_histories.append(history)

        # Evaluate best model on test set
        if len(best_models) > 0:
            print("\nEvaluating best model on test set...")
            model = create_classification_model(
                model_type="cnn_lstm",
                backbone_name=args.backbone or CLASSIFICATION_BACKBONE,
                use_attention=not args.no_attention
            )
            checkpoint = torch.load(best_models[0])
            model.load_state_dict(checkpoint['model_state_dict'])

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )

            evaluator = Evaluator(model, device)
            results = evaluator.evaluate_classification(
                test_loader,
                save_results=True,
                save_dir=os.path.join(RESULTS_DIR, "classification_evaluation")
            )

        return best_models

    else:
        # Simple train/val/test split
        print("Using train/validation split...")

        # Split training data into train/val
        train_indices, val_indices, _ = splitter.split_classification_data(
            train_dataset, test_size=0.0, val_size=0.2
        )

        # Create data loaders
        loaders = create_data_loaders(train_dataset, train_indices, val_indices)

        # Add test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        loaders['test'] = test_loader

        # Create model
        model = create_classification_model(
            model_type="cnn_lstm",
            backbone_name=args.backbone or CLASSIFICATION_BACKBONE,
            use_attention=not args.no_attention
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            task_type='classification',
            device=device,
            use_wandb=args.use_wandb,
            experiment_name="classification_main"
        )

        # Setup training
        trainer.setup_training(
            optimizer_type=args.optimizer or 'adam',
            loss_type=args.loss_type or 'focal',
            scheduler_type=args.scheduler or 'plateau'
        )

        # Train
        history, best_model_path = trainer.train(
            num_epochs=args.epochs or NUM_EPOCHS,
            save_dir=MODELS_DIR
        )

        # Plot training history
        plot_training_history(history, os.path.join(
            RESULTS_DIR, "classification_training_history.png"
        ))

        # Evaluate on test set
        print("\nEvaluating on test set...")

        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        evaluator = Evaluator(model, device)
        results = evaluator.evaluate_classification(
            loaders['test'],
            save_results=True,
            save_dir=os.path.join(RESULTS_DIR, "classification_evaluation")
        )

        # Generate explanations
        if not args.no_explanations:
            print("Generating explanations...")
            explainer = ExplainabilityPipeline(model, device)

            explanations = explainer.batch_explain(
                loaders['test'],
                save_dir=os.path.join(RESULTS_DIR, "explanations", "classification"),
                max_samples=args.max_explanations or VISUALIZATION_SAMPLES
            )

            # Create dashboard
            create_explanation_dashboard(
                explanations,
                os.path.join(RESULTS_DIR, "explanations", "classification_dashboard.html")
            )

            explainer.cleanup()

        return best_model_path

def train_segmentation_model(args):
    """Train segmentation model for lesion detection."""
    print("\n" + "="*60)
    print("TRAINING SEGMENTATION MODEL")
    print("="*60)

    device = setup_device()

    # Create datasets
    print("Loading segmentation datasets...")
    train_dataset = SegmentationDataset(
        image_dir=args.seg_train_dir or SEGMENTATION_TRAIN_IMG_DIR,
        mask_dirs=SEGMENTATION_TRAIN_MASK_DIRS,
        transform=get_segmentation_transforms(is_train=True)
    )

    val_dataset = SegmentationDataset(
        image_dir=args.seg_test_dir or SEGMENTATION_TEST_IMG_DIR,
        mask_dirs=SEGMENTATION_TEST_MASK_DIRS,
        transform=get_segmentation_transforms(is_train=False)
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = create_segmentation_model(
        model_type=args.seg_model_type or "unet",
        in_channels=3,
        out_channels=SEGMENTATION_CLASSES
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_type='segmentation',
        device=device,
        use_wandb=args.use_wandb,
        experiment_name="segmentation_main"
    )

    # Setup training
    trainer.setup_training(
        optimizer_type=args.optimizer or 'adam',
        loss_type=args.loss_type or 'combined',
        scheduler_type=args.scheduler or 'plateau'
    )

    # Train
    history, best_model_path = trainer.train(
        num_epochs=args.epochs or NUM_EPOCHS,
        save_dir=MODELS_DIR
    )

    # Plot training history
    plot_training_history(history, os.path.join(RESULTS_DIR, "segmentation_training_history.png"))

    # Evaluate
    print("\nEvaluating segmentation model...")

    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate_segmentation(
        val_loader,
        save_results=True,
        save_dir=os.path.join(RESULTS_DIR, "segmentation_evaluation")
    )

    return best_model_path

def extract_traditional_features(args):
    """Extract traditional features for comparison."""
    print("\n" + "="*60)
    print("EXTRACTING TRADITIONAL FEATURES")
    print("="*60)

    from glob import glob

    # Get image paths from both train and test directories
    image_paths = []
    data_dirs = [
        args.classification_train_dir or CLASSIFICATION_TRAIN_DIR,
        args.classification_test_dir or CLASSIFICATION_TEST_DIR
    ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_paths.extend(glob(os.path.join(data_dir, f"*{ext}")))

    print(f"Found {len(image_paths)} images for feature extraction")

    # Extract features
    features, feature_names = create_feature_dataset(
        image_paths,
        output_path=os.path.join(RESULTS_DIR, "traditional_features.npz"),
        use_deep_features=args.use_deep_features
    )

    print(f"Extracted {features.shape[1]} features per image")

    # Train traditional classifier if labels available
    train_csv = args.classification_train_csv or CLASSIFICATION_TRAIN_CSV
    if train_csv and os.path.exists(train_csv):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report

        # Load labels
        df = pd.read_csv(train_csv)

        # Handle IDRiD format
        if 'Image name' in df.columns and 'Retinopathy grade' in df.columns:
            image_name_col = 'Image name'
            label_col = 'Retinopathy grade'
        else:
            image_name_col = df.columns[0]
            label_col = df.columns[1]

        image_names_csv = df[image_name_col].tolist()
        labels_csv = df[label_col].tolist()

        # Add .jpg extension if needed
        image_names_csv = [name + '.jpg' if not any(name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']) else name for name in image_names_csv]

        # Match labels to features
        image_names_features = [os.path.basename(p) for p in image_paths]

        labels = []
        matched_features = []
        for i, img_name in enumerate(image_names_features):
            if img_name in image_names_csv:
                idx = image_names_csv.index(img_name)
                labels.append(labels_csv[idx])
                matched_features.append(features[i])

        if labels:
            matched_features = np.array(matched_features)
            labels = np.array(labels)

            print(f"Matched {len(labels)} images with labels")

            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
            scores = cross_val_score(rf, matched_features, labels, cv=5, scoring='f1_macro')

            print(f"Random Forest Cross-Validation F1 Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

            # Save feature importance
            rf.fit(matched_features, labels)
            importance_dict = dict(zip(feature_names, rf.feature_importances_))

            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            print("\nTop 10 most important features:")
            for name, importance in sorted_importance[:10]:
                print(f"  {name}: {importance:.4f}")

            # Save results
            import json
            with open(os.path.join(RESULTS_DIR, "feature_importance.json"), 'w') as f:
                json.dump(dict(sorted_importance), f, indent=2)

def preprocess_images(args):
    """Preprocess all images in dataset."""
    print("\n" + "="*60)
    print("PREPROCESSING IMAGES")
    print("="*60)

    from glob import glob

    # Get image paths
    image_paths = []
    data_dirs = []

    # Classification directories
    if args.classification_train_dir:
        data_dirs.append(args.classification_train_dir)
    if args.classification_test_dir:
        data_dirs.append(args.classification_test_dir)
    if args.seg_train_dir:
        data_dirs.append(args.seg_train_dir)
    if args.seg_test_dir:
        data_dirs.append(args.seg_test_dir)

    # Add default directories if none specified
    if not data_dirs:
        data_dirs = [
            CLASSIFICATION_TRAIN_DIR,
            CLASSIFICATION_TEST_DIR,
            SEGMENTATION_TRAIN_IMG_DIR,
            SEGMENTATION_TEST_IMG_DIR
        ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_paths.extend(glob(os.path.join(data_dir, f"*{ext}")))

    image_paths = list(set(image_paths))  # Remove duplicates
    print(f"Found {len(image_paths)} images to preprocess")

    if not image_paths:
        print("No images found for preprocessing!")
        return

    # Create output directory
    output_dir = os.path.join(RESULTS_DIR, "preprocessed_images")

    # Preprocess images
    preprocess_dataset_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        use_gabor=not args.no_gabor
    )

    print(f"Preprocessed images saved to {output_dir}")
    print("\n" + "="*60)
    print("PREPROCESSING IMAGES")
    print("="*60)

    from glob import glob

    # Get image paths
    image_paths = []
    data_dirs = []

    # Classification directories
    if args.classification_train_dir:
        data_dirs.append(args.classification_train_dir)
    if args.classification_test_dir:
        data_dirs.append(args.classification_test_dir)
    if args.seg_train_dir:
        data_dirs.append(args.seg_train_dir)
    if args.seg_test_dir:
        data_dirs.append(args.seg_test_dir)

    # Add default directories if none specified
    if not data_dirs:
        data_dirs = [
            CLASSIFICATION_TRAIN_DIR,
            CLASSIFICATION_TEST_DIR,
            SEGMENTATION_TRAIN_IMG_DIR,
            SEGMENTATION_TEST_IMG_DIR
        ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_paths.extend(glob(os.path.join(data_dir, f"*{ext}")))

    image_paths = list(set(image_paths))  # Remove duplicates
    print(f"Found {len(image_paths)} images to preprocess")

    if not image_paths:
        print("No images found for preprocessing!")
        return

    # Create output directory
    output_dir = os.path.join(RESULTS_DIR, "preprocessed_images")

    # Preprocess images
    preprocess_dataset_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        use_gabor=not args.no_gabor
    )

    print(f"Preprocessed images saved to {output_dir}")

def run_complete_pipeline(args):
    """Run the complete pipeline."""
    print("\n" + "="*80)
    print("DIABETIC RETINOPATHY DETECTION - COMPLETE PIPELINE")
    print("="*80)

    results = {}

    # Step 1: Preprocess images (if requested)
    if not args.skip_preprocessing:
        preprocess_images(args)

    # Step 2: Extract traditional features (if requested)
    if args.extract_features:
        extract_traditional_features(args)

    # Step 3: Train segmentation model
    if not args.skip_segmentation:
        seg_model_path = train_segmentation_model(args)
        results['segmentation_model'] = seg_model_path

    # Step 4: Train classification model
    if not args.skip_classification:
        cls_model_path = train_classification_model(args)
        results['classification_model'] = cls_model_path

    # Step 5: Generate final report
    generate_final_report(results)

    return results

def generate_final_report(results):
    """Generate a final report of all results."""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': results,
        'configuration': {
            'target_size': TARGET_SIZE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'device': str(torch.cuda.is_available() and 'cuda' or 'cpu')
        }
    }

    # Add model summaries if available
    for model_type, model_path in results.items():
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'val_metrics' in checkpoint:
                    report[f'{model_type}_metrics'] = checkpoint['val_metrics']
            except:
                pass

    # Save report
    report_path = os.path.join(RESULTS_DIR, "final_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Final report saved to {report_path}")

    # Print summary
    print("\n" + "="*40)
    print("PIPELINE SUMMARY")
    print("="*40)
    for model_type, model_path in results.items():
        if model_path:
            print(f"✓ {model_type.replace('_', ' ').title()}: {os.path.basename(model_path)}")

    print(f"\nAll results saved to: {RESULTS_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Diabetic Retinopathy Detection Pipeline")

    # Data paths
    parser.add_argument('--classification-train-dir', type=str, help='Classification training images directory')
    parser.add_argument('--classification-test-dir', type=str, help='Classification testing images directory')
    parser.add_argument('--classification-train-csv', type=str, help='Classification training labels CSV file')
    parser.add_argument('--classification-test-csv', type=str, help='Classification testing labels CSV file')
    parser.add_argument('--seg-train-dir', type=str, help='Segmentation training images directory')
    parser.add_argument('--seg-test-dir', type=str, help='Segmentation testing images directory')

    # Model parameters
    parser.add_argument('--backbone', type=str, choices=['densenet121', 'resnet50', 'efficientnet-b0'],
                       help='Classification backbone architecture')
    parser.add_argument('--seg-model-type', type=str, choices=['unet', 'attention_unet', 'paper_unet'],
                       help='Segmentation model type')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention mechanism')

    # Paper model options
    parser.add_argument('--use-paper-model', action='store_true',
                       help='Use paper implementation (MBConv + Adaptive BN + OGRU)')
    parser.add_argument('--use-sango', action='store_true',
                       help='Use SANGO optimization for hyperparameters')
    parser.add_argument('--use-adaptive-gabor', action='store_true', default=True,
                       help='Use Adaptive Chaotic Gabor Filter from paper')
    parser.add_argument('--k-folds', type=int, default=K_FOLDS,
                       help='Number of folds for cross-validation')

    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--loss-type', type=str, help='Loss function type')

    # Pipeline options
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation', 'features', 'preprocess', 'complete'],
                       default='complete', help='Task to run')
    parser.add_argument('--use-cross-validation', action='store_true', help='Use k-fold cross-validation')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip image preprocessing')
    parser.add_argument('--skip-classification', action='store_true', help='Skip classification training')
    parser.add_argument('--skip-segmentation', action='store_true', help='Skip segmentation training')
    parser.add_argument('--extract-features', action='store_true', help='Extract traditional features')
    parser.add_argument('--use-deep-features', action='store_true', help='Include deep CNN features')
    parser.add_argument('--no-gabor', action='store_true', help='Disable Gabor filtering')
    parser.add_argument('--no-explanations', action='store_true', help='Skip generating explanations')

    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--max-explanations', type=int, help='Maximum number of explanations to generate')

    args = parser.parse_args()

    # Update global config if arguments provided
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size

    if args.learning_rate:
        global LEARNING_RATE
        LEARNING_RATE = args.learning_rate

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Run selected task
    try:
        if args.task == 'classification':
            train_classification_model(args)
        elif args.task == 'segmentation':
            train_segmentation_model(args)
        elif args.task == 'features':
            extract_traditional_features(args)
        elif args.task == 'preprocess':
            preprocess_images(args)
        elif args.task == 'complete':
            run_complete_pipeline(args)

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Models directory: {MODELS_DIR}")

if __name__ == "__main__":
    main()