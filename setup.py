#!/usr/bin/env python3
"""
Setup script to verify installation and data structure for DR detection project.
"""

import os
import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 'scikit-image',
        'scikit-learn', 'pandas', 'matplotlib', 'seaborn', 'tqdm',
        'Pillow', 'albumentations', 'scipy'
    ]

    missing_packages = []

    print("\nChecking dependencies...")
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'Pillow':
                importlib.import_module('PIL')
            elif package == 'scikit-image':
                importlib.import_module('skimage')
            elif package == 'scikit-learn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        'results',
        'models',
        'logs',
        'results/evaluations',
        'results/explanations',
        'results/preprocessed_images'
    ]

    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")


def check_data_structure():
    """Check IDRiD data structure."""
    print("\nChecking IDRiD data structure...")

    expected_structure = {
        'data/': 'Main data directory',
        'data/A. Segmentation/': 'Segmentation data',
        'data/A. Segmentation/1. Original Images/': 'Segmentation images',
        'data/A. Segmentation/1. Original Images/a. Training Set/': 'Segmentation training images',
        'data/A. Segmentation/1. Original Images/b. Testing Set/': 'Segmentation testing images',
        'data/B. Disease Grading/': 'Classification data',
        'data/B. Disease Grading/1. Original Images/': 'Classification images',
        'data/B. Disease Grading/1. Original Images/a. Training Set/': 'Classification training images',
        'data/B. Disease Grading/1. Original Images/b. Testing Set/': 'Classification testing images',
        'data/B. Disease Grading/2. Groundtruths/': 'Classification labels',
        'data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv': 'Training labels',
        'data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv': 'Testing labels'
    }

    found_paths = []
    missing_paths = []

    for path, description in expected_structure.items():
        if os.path.exists(path):
            if path.endswith('.csv'):
                print(f"✅ {description}: {path}")
            else:
                try:
                    count = len([f for f in os.listdir(path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
                    print(f"✅ {description}: {path} ({count} files)")
                except:
                    print(f"✅ {description}: {path}")
            found_paths.append(path)
        else:
            print(f"❌ {description}: {path} - Missing")
            missing_paths.append(path)

    if missing_paths:
        print(f"\n⚠️  Found {len(found_paths)}/{len(expected_structure)} expected paths")
        print("\nMissing paths:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nNote: You can still run parts of the pipeline with available data.")
        return False
    else:
        print(f"\n✅ All expected paths found!")
        return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        # Test imports
        import config
        print("✅ Config import OK")

        from preprocessing import FundusPreprocessor
        print("✅ Preprocessing import OK")

        from models import UNet, CNNLSTMClassifier
        print("✅ Models import OK")

        # Test simple model creation
        import torch
        model = UNet(in_channels=3, out_channels=4)
        test_input = torch.randn(1, 3, 512, 512)
        output = model(test_input)
        print(f"✅ U-Net test OK - Output shape: {output.shape}")

        cls_model = CNNLSTMClassifier(num_classes=5)
        cls_output = cls_model(test_input)
        print(f"✅ Classification model test OK - Output shape: {cls_output.shape}")

        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def suggest_next_steps():
    """Suggest next steps based on setup results."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE - SUGGESTED NEXT STEPS")
    print("=" * 60)

    if os.path.exists('data/B. Disease Grading/1. Original Images/a. Training Set'):
        print("📊 You have classification data available:")
        print("   python run_training.py  # Use the interactive script")
        print("   or")
        print("   python main.py --task classification --epochs 5  # Quick test")

    if os.path.exists('data/A. Segmentation/1. Original Images/a. Training Set'):
        print("\n🎯 You have segmentation data available:")
        print("   python main.py --task segmentation --epochs 10")

    print("\n🔧 Other useful commands:")
    print("   python main.py --task preprocess  # Preprocess images")
    print("   python main.py --task features --extract-features  # Extract features")
    print("   python main.py --help  # See all options")

    print("\n📁 Results will be saved to:")
    print("   - results/: All outputs and visualizations")
    print("   - models/: Trained model files")
    print("   - logs/: Training logs")


def main():
    print("=" * 60)
    print("DIABETIC RETINOPATHY DETECTION - SETUP VERIFICATION")
    print("=" * 60)

    all_checks_passed = True

    # Check Python version
    if not check_python_version():
        all_checks_passed = False

    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
        print("\n💡 Install missing dependencies with:")
        print("   pip install -r requirements.txt")

    # Check CUDA
    if not check_cuda():
        all_checks_passed = False

    # Create directories
    create_directories()

    # Check data structure
    data_ok = check_data_structure()

    # Test functionality if dependencies are OK
    if all_checks_passed:
        if not test_basic_functionality():
            all_checks_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ SETUP VERIFICATION PASSED")
        if data_ok:
            print("✅ DATA STRUCTURE OK")
            print("🚀 Ready to start training!")
        else:
            print("⚠️  SOME DATA MISSING")
            print("🔧 You can still run parts of the pipeline")
    else:
        print("❌ SETUP VERIFICATION FAILED")
        print("🔧 Please fix the issues above before proceeding")

    suggest_next_steps()


if __name__ == "__main__":
    main()