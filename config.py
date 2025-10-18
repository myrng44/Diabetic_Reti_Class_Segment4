"""
Configuration file for diabetic retinopathy classification and segmentation project.
Contains all hyperparameters, paths, and constants.
"""

import os
import torch

# ================================
# GENERAL SETTINGS
# ================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Properly check GPU availability
NUM_WORKERS = 2  # Reduced to prevent GPU memory contention
RESULTS_DIR = "results"
LOGS_DIR = "logs"
MODELS_DIR = "models"

# GPU settings
BATCH_SIZE = 4  # Reduced for better GPU memory utilization
MIXED_PRECISION = True
EMPTY_CACHE_FREQ = 1
GRADIENT_ACCUMULATION_STEPS = 2
PIN_MEMORY = True

# Training optimization
USE_AMP = True  # Enable automatic mixed precision
CUDNN_BENCHMARK = True  # Enable cuDNN benchmarking
CUDNN_DETERMINISTIC = False  # Disable for better performance

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ================================
# IMAGE PROCESSING
# ================================
TARGET_SIZE = 384  # Increased from 384
CLAHE_CLIP_LIMIT = 3.0  # Increased from 2.0
CLAHE_TILE_GRID_SIZE = (16, 16)  # Increased from (8,8)
FUNDUS_THRESHOLD = 15  # Increased from 10

# Adaptive Chaotic Gabor Filter settings
USE_ADAPTIVE_GABOR = True
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4, 0.5]  # Added more frequencies
GABOR_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150]  # Added more angles
GABOR_SIGMA = 2.5  # Increased from 2.0
GABOR_GAMMA = 0.7  # Increased from 0.5
GABOR_KERNEL_SIZE = 41  # Increased from 31

# ================================
# SEGMENTATION SETTINGS
# ================================
SEGMENTATION_CLASSES = 3  # MA, HE, EX
SEGMENTATION_CLASS_NAMES = ["MA", "HE", "EX"]
SEGMENTATION_COLORS = {
    0: (255, 0, 0),    # MA - red
    1: (0, 255, 0),    # HE - green
    2: (0, 0, 255),    # EX - blue
}

# ================================
# CLASSIFICATION SETTINGS
# ================================
CLASSIFICATION_CLASSES = 5  # 0-4 severity levels
CLASSIFICATION_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR"
]

# ================================
# TRAINING PARAMETERS
# ================================
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Memory optimization
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = True
PIN_MEMORY = True

# Cross validation
K_FOLDS = 2  # Reduced for testing
NUM_WORKERS = 2  # Reduced to prevent memory issues

# Loss weights
SEGMENTATION_BCE_WEIGHT = 0.5
CLASSIFICATION_FOCAL_ALPHA = 1.0
CLASSIFICATION_FOCAL_GAMMA = 2.0

# Memory management
GRAD_CLIP = 1.0
PIN_MEMORY = True
EMPTY_CACHE_FREQ = 1  # Empty CUDA cache every N batches

# ================================
# MODEL ARCHITECTURE
# ================================
# U-Net features
UNET_FEATURES = [64, 128, 256, 512, 1024]  # Added another layer
UNET_DROPOUT = 0.3  # Increased from 0.2

# Classification backbone
CLASSIFICATION_BACKBONE = "densenet121"  # densenet121, efficientnet-b0, resnet50
PRETRAINED = True
ATTENTION_DIM = 256
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2

# ================================
# FEATURE EXTRACTION
# ================================
# LBP parameters
LBP_RADIUS = 3
LBP_N_POINTS = 24

# SURF parameters
SURF_HESSIAN_THRESHOLD = 400

# Gabor filter parameters
GABOR_FREQUENCIES = [0.1, 0.3, 0.5]
GABOR_ANGLES = [0, 45, 90, 135]
GABOR_SIGMA_X = 2
GABOR_SIGMA_Y = 2

# ================================
# DATA PATHS (EDIT THESE TO MATCH YOUR DATA)
# ================================
# Classification data
CLASSIFICATION_TRAIN_DIR = "data/B. Disease Grading/1. Original Images/a. Training Set"
CLASSIFICATION_TEST_DIR = "data/B. Disease Grading/1. Original Images/b. Testing Set"
CLASSIFICATION_TRAIN_CSV = "data/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
CLASSIFICATION_TEST_CSV = "data/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"

# Segmentation data
SEGMENTATION_TRAIN_IMG_DIR = "data/A. Segmentation/1. Original Images/a. Training Set"
SEGMENTATION_TEST_IMG_DIR = "data/A. Segmentation/1. Original Images/b. Testing Set"

SEGMENTATION_TRAIN_MASK_DIRS = {
    "MA": "data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms_",
    "HE": "data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages_",
    "EX": "data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates_",
}

SEGMENTATION_TEST_MASK_DIRS = {
    "MA": "data/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms_",
    "HE": "data/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/2. Haemorrhages_",
    "EX": "data/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates_",
}

# ================================
# EVALUATION SETTINGS
# ================================
SEGMENTATION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.8
VISUALIZATION_SAMPLES = 5

# ================================
# AUGMENTATION SETTINGS
# ================================
AUGMENTATION_PROB = 0.5
BRIGHTNESS_CONTRAST_LIMIT = 0.2
ROTATION_LIMIT = 15
SCALE_LIMIT = 0.1
SHIFT_LIMIT = 0.1

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)