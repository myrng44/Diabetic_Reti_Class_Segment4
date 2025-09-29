"""
Model architectures for diabetic retinopathy classification and segmentation.
Includes U-Net for segmentation and CNN+LSTM with attention for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from config import *

# ================================
# SEGMENTATION MODELS
# ================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels, out_channels, dropout_rate=UNET_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Improved U-Net architecture for lesion segmentation."""

    def __init__(self, in_channels=3, out_channels=SEGMENTATION_CLASSES,
                 features=UNET_FEATURES):
        super().__init__()
        self.features = features

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(feature * 2, feature))

        # Final classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]

        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip_connection = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder(concat_skip)

        return self.final_conv(x)


class AttentionUNet(nn.Module):
    """U-Net with attention gates."""

    def __init__(self, in_channels=3, out_channels=SEGMENTATION_CLASSES,
                 features=UNET_FEATURES):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, features)

        # Add attention gates
        self.attention_gates = nn.ModuleList()
        for i, feature in enumerate(features[:-1]):
            self.attention_gates.append(
                AttentionGate(feature, features[i+1], feature // 2)
            )

    def forward(self, x):
        # This is a simplified version - full implementation would integrate
        # attention gates into the U-Net forward pass
        return self.unet(x)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ================================
# CLASSIFICATION MODELS
# ================================

class SelfAttention(nn.Module):
    """Self-attention mechanism for feature enhancement."""

    def __init__(self, feature_dim, attention_dim=ATTENTION_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim

        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        Q = self.query(x)  # (batch_size, seq_len, attention_dim)
        K = self.key(x)    # (batch_size, seq_len, attention_dim)
        V = self.value(x)  # (batch_size, seq_len, feature_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Residual connection and layer norm
        output = self.layer_norm(x + attended)

        return output.squeeze(1) if seq_len == 1 else output


class CNNBackbone(nn.Module):
    """CNN backbone for feature extraction."""

    def __init__(self, backbone_name=CLASSIFICATION_BACKBONE, pretrained=PRETRAINED):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "densenet121":
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "efficientnet-b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        return self.backbone(x)


class CNNLSTMClassifier(nn.Module):
    """CNN + LSTM + Attention classifier for DR grading."""

    def __init__(self, num_classes=CLASSIFICATION_CLASSES,
                 backbone_name=CLASSIFICATION_BACKBONE,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 lstm_layers=LSTM_LAYERS,
                 use_attention=True):
        super().__init__()

        # CNN backbone for feature extraction
        self.cnn_backbone = CNNBackbone(backbone_name)
        cnn_feature_dim = self.cnn_backbone.feature_dim

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            lstm_hidden_dim,
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(lstm_hidden_dim * 2)
            final_dim = lstm_hidden_dim * 2
        else:
            final_dim = lstm_hidden_dim * 2

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_dim // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract CNN features
        cnn_features = self.cnn_backbone(x)  # (batch_size, feature_dim)

        # Project features
        projected_features = self.feature_projection(cnn_features)  # (batch_size, lstm_hidden_dim)

        # Add sequence dimension for LSTM
        lstm_input = projected_features.unsqueeze(1)  # (batch_size, 1, lstm_hidden_dim)

        # LSTM processing
        lstm_output, _ = self.lstm(lstm_input)  # (batch_size, 1, lstm_hidden_dim * 2)

        # Apply attention if enabled
        if self.use_attention:
            attended_output = self.attention(lstm_output)  # (batch_size, lstm_hidden_dim * 2)
        else:
            attended_output = lstm_output.squeeze(1)

        # Final classification
        logits = self.classifier(attended_output)

        return logits


class MultiTaskModel(nn.Module):
    """Multi-task model for both classification and segmentation."""

    def __init__(self, num_classes=CLASSIFICATION_CLASSES,
                 num_seg_classes=SEGMENTATION_CLASSES):
        super().__init__()

        # Shared backbone
        self.shared_backbone = CNNBackbone()

        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.shared_backbone.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Segmentation head (simplified)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.shared_backbone.feature_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, num_seg_classes, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # Extract features using shared backbone
        features = self.shared_backbone(x)

        # Classification output
        classification_logits = self.classification_head(features)

        # Segmentation output (requires feature maps, not global features)
        # This is a simplified version - in practice, you'd need intermediate features
        segmentation_logits = None  # Placeholder

        return classification_logits, segmentation_logits


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance."""

    def __init__(self, models_list, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.weights = weights if weights else [1.0] * len(models_list)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted average
        weighted_output = torch.zeros_like(outputs[0])
        total_weight = sum(self.weights)

        for output, weight in zip(outputs, self.weights):
            weighted_output += output * weight / total_weight

        return weighted_output


# ================================
# LOSS FUNCTIONS
# ================================

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""

    def __init__(self, alpha=CLASSIFICATION_FOCAL_ALPHA,
                 gamma=CLASSIFICATION_FOCAL_GAMMA,
                 num_classes=CLASSIFICATION_CLASSES):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss for segmentation."""

    def __init__(self, bce_weight=SEGMENTATION_BCE_WEIGHT):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        return self.bce_weight * bce + (1 - self.bce_weight) * dice


# ================================
# MODEL FACTORY
# ================================

def create_segmentation_model(model_type="unet", **kwargs):
    """Factory function to create segmentation models."""
    if model_type == "unet":
        return UNet(**kwargs)
    elif model_type == "attention_unet":
        return AttentionUNet(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation model type: {model_type}")


def create_classification_model(model_type="cnn_lstm", **kwargs):
    """Factory function to create classification models."""
    if model_type == "cnn_lstm":
        return CNNLSTMClassifier(**kwargs)
    elif model_type == "cnn_only":
        return CNNBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown classification model type: {model_type}")


def create_loss_function(loss_type, **kwargs):
    """Factory function to create loss functions."""
    if loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test segmentation model
    seg_model = create_segmentation_model("unet")
    seg_model.to(device)
    print(f"Segmentation model parameters: {sum(p.numel() for p in seg_model.parameters()):,}")

    # Test classification model
    cls_model = create_classification_model("cnn_lstm")
    cls_model.to(device)
    print(f"Classification model parameters: {sum(p.numel() for p in cls_model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, TARGET_SIZE, TARGET_SIZE).to(device)

    with torch.no_grad():
        seg_output = seg_model(test_input)
        cls_output = cls_model(test_input)

    print(f"Segmentation output shape: {seg_output.shape}")
    print(f"Classification output shape: {cls_output.shape}")