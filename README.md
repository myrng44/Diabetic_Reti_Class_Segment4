# Diabetic Retinopathy Detection Pipeline

A comprehensive deep learning pipeline for diabetic retinopathy (DR) classification and lesion segmentation with explainable AI capabilities.

## 🎯 Features

### **Stage 2: Preprocessing Pipeline**
- **Advanced Image Preprocessing**: Fundus circular masking, CLAHE enhancement, and resizing
- **Adaptive Gabor Filtering**: Texture enhancement and noise reduction
- **Data Augmentation Pipeline**: Comprehensive augmentation strategies for robust training

### **Stage 3: Image Segmentation**
- **U-Net Architecture**: Enhanced U-Net with batch normalization and dropout
- **Attention Mechanisms**: Optional attention gates for improved segmentation
- **Multi-class Lesion Segmentation**: MA (Microaneurysms), HE (Hemorrhages), EX (Hard Exudates), SE (Soft Exudates)

### **Stage 4: Feature Extraction**
- **Local Binary Patterns (LBP)**: Multi-scale texture analysis
- **SURF Features**: Scale-invariant feature detection
- **Texture Energy Measures (TEM)**: GLCM-based texture features
- **Deep Features**: CNN-extracted features for enhanced representation

### **Stage 5: Classification Models**
- **CNN+LSTM+Attention**: Advanced architecture combining spatial and temporal features
- **Multiple Backbones**: DenseNet, ResNet, EfficientNet support
- **Focal Loss**: Addresses class imbalance in DR grading

### **Stage 6: Training & Evaluation**
- **Cross-Validation**: K-fold validation for robust evaluation
- **Advanced Metrics**: IoU, DSC, F1-score, AUC with per-class analysis
- **Early Stopping**: Prevents overfitting with learning rate scheduling

### **Stage 7: Explainable AI**
- **Grad-CAM**: Gradient-based class activation mapping
- **Attention Visualization**: Visual interpretation of attention mechanisms
- **Confidence Scoring**: Monte Carlo dropout and entropy-based uncertainty
- **Interactive Dashboards**: HTML-based explanation dashboards

## 🗂️ Project Structure

```
diabetic_retinopathy_detection/
├── config.py              # Configuration and hyperparameters
├── preprocessing.py        # Image preprocessing and augmentation
├── feature_extraction.py   # Traditional feature extraction methods
├── datasets.py            # Dataset classes and data loading
├── models.py              # Neural network architectures
├── training.py            # Training and evaluation utilities
├── gradcam.py             # Explainable AI and visualization
├── main.py                # Main pipeline execution
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # Dataset directory
│   ├── classification/    # Classification images and labels
│   └── segmentation/      # Segmentation images and masks
├── results/               # Output directory
│   ├── models/           # Trained model checkpoints
│   ├── evaluations/      # Evaluation results and metrics
│   ├── explanations/     # Grad-CAM and attention visualizations
│   └── preprocessed/     # Preprocessed images
└── logs/                  # Training logs and experiment tracking
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository_url>
cd diabetic_retinopathy_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

**Classification Data Structure:**
```
data/classification/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels.csv  # Format: filename, label (0-4)
```

**Segmentation Data Structure:**
```
data/segmentation/
├── images/
│   ├── train/
│   └── test/
└── masks/
    ├── train/
    │   ├── MA/  # Microaneurysms
    │   ├── HE/  # Hemorrhages
    │   ├── EX/  # Hard Exudates
    │   └── SE/  # Soft Exudates
    └── test/
        └── ...
```

### 3. Configuration

Edit `config.py` to match your data paths and preferences:

```python
# Data paths
CLASSIFICATION_DATA_DIR = "data/classification/images"
CLASSIFICATION_CSV = "data/classification/labels.csv"
SEGMENTATION_TRAIN_IMG_DIR = "data/segmentation/images/train"
# ... other paths

# Model parameters
TARGET_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
```

### 4. Run Pipeline

**Complete Pipeline:**
```bash
python main.py --task complete
```

**Individual Tasks:**
```bash
# Preprocess images only
python main.py --task preprocess

# Train classification model only
python main.py --task classification --epochs 30

# Train segmentation model only  
python main.py --task segmentation --backbone densenet121

# Extract traditional features
python main.py --task features --extract-features --use-deep-features
```

## 🔧 Advanced Usage

### Cross-Validation Training
```bash
python main.py --task classification --use-cross-validation --epochs 50
```

### Custom Model Configuration
```bash
python main.py --task complete \
  --backbone densenet121 \
  --seg-model-type attention_unet \
  --optimizer adamw \
  --scheduler cosine \
  --batch-size 16 \
  --learning-rate 2e-4
```

### With Experiment Tracking
```bash
# Using Weights & Biases
python main.py --task complete --use-wandb
```

### Generate Explanations
```bash
python main.py --task classification --max-explanations 20
```

## 📊 Model Architectures

### Classification Model (CNN+LSTM+Attention)
- **Backbone**: DenseNet121/ResNet50/EfficientNet-B0
- **Feature Processing**: LSTM for sequence modeling
- **Attention**: Self-attention for feature enhancement
- **Output**: 5-class DR severity (0-4)

### Segmentation Model (U-Net)
- **Encoder**: Convolutional downsampling path
- **Decoder**: Transposed convolutional upsampling
- **Skip Connections**: Feature preservation
- **Output**: 4-channel lesion masks (MA, HE, EX, SE)

### Loss Functions
- **Classification**: Focal Loss (addresses class imbalance)
- **Segmentation**: Combined BCE+Dice Loss
- **Optimization**: Adam/AdamW with learning rate scheduling

## 📈 Evaluation Metrics

### Classification
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1-scores
- **AUC**: Area under ROC curve
- **Confusion Matrix**: Per-class performance analysis

### Segmentation
- **Dice Coefficient**: Overlap similarity measure
- **IoU (Jaccard)**: Intersection over Union
- **Per-Class Metrics**: Individual lesion type performance
- **Visual Overlays**: Prediction vs ground truth comparisons

## 🔍 Explainable AI Features

### Grad-CAM Visualization
```python
from gradcam import GradCAM

gradcam = GradCAM(model)
cam, prediction = gradcam.generate_cam(input_tensor)
gradcam.visualize_cam(input_tensor, cam, save_path="gradcam.png")
```

### Confidence Scoring
```python
from gradcam import ConfidenceScorer

scorer = ConfidenceScorer(model)
mean_pred, confidence, uncertainty = scorer.calculate_prediction_confidence(input_tensor)
```

### Complete Explanation Pipeline
```python
from gradcam import ExplainabilityPipeline

explainer = ExplainabilityPipeline(model)
explanation = explainer.explain_prediction(input_tensor, save_dir="explanations/")
```

## 🛠️ Preprocessing Options

### Image Preprocessing
- **Fundus Masking**: Automatic circular region detection
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Gabor Filtering**: Texture enhancement and noise reduction

### Data Augmentation
- **Geometric**: Rotation, scaling, flipping
- **Photometric**: Brightness, contrast, gamma adjustment
- **Elastic**: Deformation and distortion

## 📝 Results and Outputs

### Model Checkpoints
- `best_model.pth`: Best performing model
- `final_model.pth`: Final epoch model
- Training history and configuration saved

### Evaluation Reports
- **Classification**: Confusion matrix, classification report, ROC curves
- **Segmentation**: Dice scores, IoU metrics, visual overlays
- **JSON summaries**: Detailed metrics and statistics

### Explanations
- **Grad-CAM heatmaps**: Visual attention maps
- **Attention visualizations**: Model attention patterns
- **Confidence scores**: Prediction reliability measures
- **HTML dashboards**: Interactive result exploration

## ⚙️ Configuration Options

### Model Parameters
```python
# Classification
CLASSIFICATION_BACKBONE = "densenet121"  # resnet50, efficientnet-b0
LSTM_HIDDEN_DIM = 128
ATTENTION_DIM = 256

# Segmentation  
UNET_FEATURES = [64, 128, 256, 512]
UNET_DROPOUT = 0.2

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
```

### Loss Function Settings
```python
# Focal Loss (Classification)
CLASSIFICATION_FOCAL_ALPHA = 1.0
CLASSIFICATION_FOCAL_GAMMA = 2.0

# Combined Loss (Segmentation)
SEGMENTATION_BCE_WEIGHT = 0.5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IDRiD dataset for segmentation ground truth
- PyTorch team for the deep learning framework
- Albumentations for image augmentation
- scikit-image for traditional computer vision methods

## 📞 Contact

For questions and support, please open an issue in the repository or contact the maintainers.

---

**Note**: Ensure you have appropriate permissions and ethical approval when working with medical image data. Always follow your institution's guidelines for handling patient data.