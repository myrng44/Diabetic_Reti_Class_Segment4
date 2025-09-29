#!/usr/bin/env python3
"""
Easy run script for diabetic retinopathy detection with IDRiD dataset structure.
"""

import os
import sys
import subprocess


def run_command(cmd):
    """Run command and handle output."""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please ensure you're in the correct directory.")
        sys.exit(1)

    print("=" * 80)
    print("DIABETIC RETINOPATHY DETECTION - IDRiD DATASET")
    print("=" * 80)

    # Check data structure
    data_paths = {
        'Classification Train Images': 'data/B. Disease Grading/1. Original Images/a. Training Set',
        'Classification Test Images': 'data/B. Disease Grading/1. Original Images/b. Testing Set',
        'Classification Train Labels': 'data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv',
        'Classification Test Labels': 'data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv',
        'Segmentation Train Images': 'data/A. Segmentation/1. Original Images/a. Training Set',
        'Segmentation Test Images': 'data/A. Segmentation/1. Original Images/b. Testing Set'
    }

    print("\nChecking data structure...")
    missing_paths = []
    for name, path in data_paths.items():
        if os.path.exists(path):
            if path.endswith('.csv'):
                print(f"✓ {name}: Found")
            else:
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✓ {name}: Found ({count} images)")
        else:
            print(f"✗ {name}: Missing - {path}")
            missing_paths.append(path)

    if missing_paths:
        print(f"\nWarning: {len(missing_paths)} required paths are missing.")
        print("Please ensure your data follows the IDRiD dataset structure.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING OPTIONS")
    print("=" * 60)

    print("1. Complete Pipeline (Classification + Segmentation)")
    print("2. Classification Only")
    print("3. Segmentation Only")
    print("4. Preprocess Images Only")
    print("5. Extract Traditional Features")
    print("6. Quick Test (5 epochs)")

    choice = input("\nSelect option (1-6): ").strip()

    # Base command
    base_cmd = [
        sys.executable, 'main.py',
        '--classification-train-dir', 'data/B. Disease Grading/1. Original Images/a. Training Set',
        '--classification-test-dir', 'data/B. Disease Grading/1. Original Images/b. Testing Set',
        '--classification-train-csv',
        'data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv',
        '--classification-test-csv',
        'data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv',
        '--seg-train-dir', 'data/A. Segmentation/1. Original Images/a. Training Set',
        '--seg-test-dir', 'data/A. Segmentation/1. Original Images/b. Testing Set'
    ]

    if choice == '1':
        # Complete pipeline
        cmd = base_cmd + [
            '--task', 'complete',
            '--epochs', '30',
            '--backbone', 'densenet121'
        ]

        print("\nRunning complete pipeline...")
        print("This will train both classification and segmentation models.")

    elif choice == '2':
        # Classification only
        cmd = base_cmd + [
            '--task', 'classification',
            '--epochs', '25',
            '--backbone', 'densenet121',
            '--optimizer', 'adam'
        ]

        print("\nRunning classification training...")

    elif choice == '3':
        # Segmentation only
        cmd = base_cmd + [
            '--task', 'segmentation',
            '--epochs', '30',
            '--seg-model-type', 'unet'
        ]

        print("\nRunning segmentation training...")

    elif choice == '4':
        # Preprocess only
        cmd = base_cmd + [
            '--task', 'preprocess'
        ]

        print("\nPreprocessing images...")

    elif choice == '5':
        # Feature extraction
        cmd = base_cmd + [
            '--task', 'features',
            '--extract-features',
            '--use-deep-features'
        ]

        print("\nExtracting traditional features...")

    elif choice == '6':
        # Quick test
        cmd = base_cmd + [
            '--task', 'classification',
            '--epochs', '5',
            '--backbone', 'densenet121',
            '--batch-size', '4'
        ]

        print("\nRunning quick test (5 epochs)...")

    else:
        print("Invalid choice!")
        sys.exit(1)

    # Ask for additional options
    print("\nAdditional options:")

    use_cv = input("Use cross-validation? (y/n): ").strip().lower() == 'y'
    if use_cv:
        cmd.extend(['--use-cross-validation'])

    use_wandb = input("Use Weights & Biases tracking? (y/n): ").strip().lower() == 'y'
    if use_wandb:
        cmd.extend(['--use-wandb'])

    if choice in ['1', '2']:  # Classification tasks
        gen_explanations = input("Generate explanations? (y/n): ").strip().lower() == 'y'
        if not gen_explanations:
            cmd.extend(['--no-explanations'])

    # Run the command
    print(f"\nStarting training...")
    success = run_command(cmd)

    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the following directories for results:")
        print("- results/: All outputs, evaluations, and visualizations")
        print("- models/: Trained model checkpoints")
        print("- logs/: Training logs")

        if choice in ['1', '2'] and not ('--no-explanations' in cmd):
            print("- results/explanations/: Grad-CAM visualizations and explanations")
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()