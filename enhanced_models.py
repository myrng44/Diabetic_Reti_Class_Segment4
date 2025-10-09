"""
Complete implementation based on paper - FIXED VERSION
- Modified U-Net with MBConv + Adaptive BN (Fixed channel matching)
- OGRU with SANGO optimization
- Multi-fold features (LBP + SURF + TEM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *


# ===================================
# 1. MBConv Block (Mobile Inverted Bottleneck)
# ===================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class AdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive Batch Normalization with embedding features.
    Based on paper equation (7) and (8).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.embedding = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        # Standard BN
        x_bn = self.bn(x)
        # Add embedding features for contrastive constraint
        return x_bn + self.embedding


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block.
    Structure: 1x1 expand -> 3x3 depthwise -> SE -> 1x1 project
    """

    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1,
                 use_se=True, dropout_rate=0.2):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                AdaptiveBatchNorm2d(hidden_dim),
                nn.SiLU()  # Swish
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            AdaptiveBatchNorm2d(hidden_dim),
            nn.SiLU()
        ])

        # SE block
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            AdaptiveBatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        if self.use_residual:
            out = self.conv(x)
            if self.dropout:
                out = self.dropout(out)
            return x + out
        else:
            return self.conv(x)


# ===================================
# 2. Modified U-Net with Efficient-Net (FIXED)
# ===================================

class ModifiedUNet(nn.Module):
    """
    Modified U-Net with MBConv blocks and Adaptive BN.
    FIXED: Proper channel matching for encoder-decoder
    """

    def __init__(self, in_channels=3, out_channels=SEGMENTATION_CLASSES,
                 base_features=32, num_stages=5):
        super().__init__()

        # Calculate feature sizes - ensure they're consistent
        features = [base_features * (2 ** i) for i in range(num_stages)]
        # features = [32, 64, 128, 256, 512] for base_features=32

        self.num_stages = num_stages

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoders.append(MBConvBlock(prev_channels, feature, stride=1))
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_channels = feature

        # Bottleneck
        bottleneck_channels = features[-1] * 2
        self.bottleneck = MBConvBlock(features[-1], bottleneck_channels)
        self.bottleneck_channels = bottleneck_channels

        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Build decoder in reverse order
        decoder_in_channels = bottleneck_channels
        for i in range(num_stages - 1, -1, -1):
            feature = features[i]

            # Upconv: reduce channels from decoder_in to feature
            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_in_channels,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )

            # Decoder block: feature (from skip) + feature (from upconv) -> feature
            self.decoders.append(
                MBConvBlock(feature * 2, feature)
            )

            # Next decoder input
            decoder_in_channels = feature

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Global pooling for classification pathway
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Encoder path
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        bottleneck_features = x

        # Decoder path
        skip_connections = skip_connections[::-1]

        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # Segmentation output
        seg_output = self.final_conv(x)

        # Feature maps for classification
        pooled_features = self.global_pool(bottleneck_features)

        return seg_output, pooled_features


# ===================================
# 3. OGRU - Optimized GRU
# ===================================

class OptimizedGRU(nn.Module):
    """
    GRU optimized by SANGO algorithm.
    Hyperparameters (hidden_dim, num_layers) are found by SANGO.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.3, num_classes=5):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features) for GRU
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # GRU forward
        gru_out, _ = self.gru(x)

        # Take last output
        last_output = gru_out[:, -1, :]

        # Classification
        output = self.fc(last_output)

        return output


# ===================================
# 4. Complete Multi-Model Architecture (FIXED)
# ===================================

class PaperMultiModelDR(nn.Module):
    """
    Complete architecture from paper combining:
    - Modified U-Net for segmentation + feature extraction
    - Multi-fold features (LBP, SURF, TEM - extracted separately)
    - DenseNet block
    - Attention mechanism
    - OGRU optimized by SANGO

    FIXED: Dynamic feature calculation based on U-Net output
    """

    def __init__(self, num_classes=CLASSIFICATION_CLASSES,
                 segmentation_classes=SEGMENTATION_CLASSES,
                 unet_base_features=32,
                 unet_num_stages=5,
                 gru_hidden_dim=128,
                 gru_num_layers=2,
                 gru_dropout=0.3):
        super().__init__()

        # 1. Modified U-Net with configurable base features
        self.unet = ModifiedUNet(
            in_channels=3,
            out_channels=segmentation_classes,
            base_features=unet_base_features,
            num_stages=unet_num_stages
        )

        # Calculate U-Net bottleneck feature dimension dynamically
        # bottleneck_channels = base_features * (2^(num_stages-1)) * 2
        unet_feature_dim = unet_base_features * (2 ** (unet_num_stages - 1)) * 2
        self.unet_feature_dim = unet_feature_dim

        # 2. DenseNet block (simplified - one dense block)
        self.dense_block = nn.Sequential(
            nn.Linear(unet_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 3. Attention mechanism (two FC layers)
        self.attention_fc1 = nn.Sequential(
            nn.Linear(unet_feature_dim, 256),
            nn.SiLU(),  # Swish
        )
        self.attention_fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

        # 4. Feature fusion
        # DenseNet output (256) + Attention output (256) = 512
        fusion_dim = 512

        # 5. OGRU (Optimized GRU)
        self.ogru = OptimizedGRU(
            input_dim=fusion_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            num_classes=num_classes
        )

    def forward(self, x, external_features=None):
        """
        Forward pass.

        Args:
            x: Input images (batch, 3, H, W)
            external_features: Optional multi-fold features (LBP+SURF+TEM)
                              Shape: (batch, feature_dim)

        Returns:
            classification_output: Class predictions
            segmentation_output: Segmentation masks
        """
        # 1. U-Net forward
        seg_output, pooled_features = self.unet(x)

        # Flatten pooled features
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # 2. DenseNet pathway
        dense_features = self.dense_block(pooled_features)

        # 3. Attention pathway
        attention_1 = self.attention_fc1(pooled_features)
        attention_weights = self.attention_fc2(attention_1)
        attention_features = attention_1 * attention_weights

        # 4. Concatenate features
        fused_features = torch.cat([dense_features, attention_features], dim=1)

        # Add external features if provided
        if external_features is not None:
            fused_features = torch.cat([fused_features, external_features], dim=1)

        # 5. OGRU classification
        classification_output = self.ogru(fused_features)

        return classification_output, seg_output


# ===================================
# 5. Focal Loss (from paper)
# ===================================

class FocalLoss(nn.Module):
    """
    Focal Loss as defined in paper equation (23).
    """

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits (batch, num_classes)
            targets: Ground truth labels (batch,)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)

        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)

        # Focal loss formula
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


# ===================================
# 6. Model factory with SANGO optimization (FIXED)
# ===================================

def create_paper_model_with_sango(
        train_loader,
        val_loader,
        device,
        use_sango=True,
        num_classes=CLASSIFICATION_CLASSES
):
    """
    Create model with SANGO-optimized hyperparameters.
    FIXED: Only optimize GRU hyperparameters, keep U-Net stable
    """

    if use_sango:
        from enhanced_sango import EnhancedSANGO, create_fitness_function_f1

        print("Running SANGO optimization for hyperparameters...")

        # FIXED: Only optimize GRU parameters, not U-Net architecture
        def model_creator(hidden_dim, num_layers, dropout):
            return PaperMultiModelDR(
                num_classes=num_classes,
                segmentation_classes=SEGMENTATION_CLASSES,
                unet_base_features=32,  # Keep stable
                unet_num_stages=5,      # Keep stable
                gru_hidden_dim=int(hidden_dim),
                gru_num_layers=int(num_layers),
                gru_dropout=dropout
            )

        fitness_fn = create_fitness_function_f1(
            model_class=model_creator,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            max_epochs=5  # Quick training for fitness evaluation
        )

        # Run SANGO - optimize only GRU hyperparameters
        sango = EnhancedSANGO(
            fitness_function=fitness_fn,
            dim=3,  # Only 3 dimensions: hidden_dim, num_layers, dropout
            population_size=8,
            max_iterations=20,
            verbose=True
        )

        best_params, best_fitness, _ = sango.optimize()

        print(f"\nSANGO found best parameters:")
        print(f"  GRU Hidden Dim: {int(best_params.get('hidden_dim', best_params.get('hidden_dim1', 128)))}")
        print(f"  GRU Num Layers: {int(best_params.get('num_layers', 2))}")
        print(f"  Dropout: {best_params.get('dropout', 0.3):.3f}")
        print(f"  Best F1-Score: {1 - best_fitness:.4f}")

        # Create final model with optimized params
        model = PaperMultiModelDR(
            num_classes=num_classes,
            segmentation_classes=SEGMENTATION_CLASSES,
            unet_base_features=32,
            unet_num_stages=5,
            gru_hidden_dim=int(best_params.get('hidden_dim', best_params.get('hidden_dim1', 128))),
            gru_num_layers=int(best_params.get('num_layers', 2)),
            gru_dropout=best_params.get('dropout', 0.3)
        )

        return model, best_params

    else:
        # Use default hyperparameters
        print("Using default hyperparameters (no SANGO optimization)")
        model = PaperMultiModelDR(
            num_classes=num_classes,
            unet_base_features=32,
            unet_num_stages=5
        )
        default_params = {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'lr': 1e-4
        }
        return model, default_params


# Test
if __name__ == "__main__":
    # Test model creation
    print("Testing model with default parameters...")
    model = PaperMultiModelDR(
        num_classes=5,
        segmentation_classes=3,
        unet_base_features=32,
        unet_num_stages=5
    )

    # Test input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    cls_out, seg_out = model(x)

    print(f"Classification output: {cls_out.shape}")  # (2, 5)
    print(f"Segmentation output: {seg_out.shape}")    # (2, 3, 224, 224)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✓ Model test passed!")