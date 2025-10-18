"""
Complete implementation with 4-parameter SANGO optimization - FIXED VERSION
- Fixed fitness function error handling
- Fixed dataset unpacking
- Added proper error logging
- Fixed imports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class AdaptiveBatchNorm2d(nn.Module):
    """Adaptive Batch Normalization with embedding features."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.embedding = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        x_bn = self.bn(x)
        return x_bn + self.embedding


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block."""

    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1,
                 use_se=True, dropout_rate=0.2):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                AdaptiveBatchNorm2d(hidden_dim),
                nn.SiLU()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            AdaptiveBatchNorm2d(hidden_dim),
            nn.SiLU()
        ])

        if use_se:
            layers.append(SEBlock(hidden_dim))

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
# 2. Modified U-Net (STABLE - No SANGO)
# ===================================

class ModifiedUNet(nn.Module):
    """
    Modified U-Net with FIXED architecture.
    This is NOT optimized by SANGO to avoid channel mismatch.
    """

    def __init__(self, in_channels=3, out_channels=SEGMENTATION_CLASSES,
                 base_features=32, num_stages=5):
        super().__init__()

        # Fixed feature progression
        features = [base_features * (2 ** i) for i in range(num_stages)]
        self.num_stages = num_stages

        # Encoder
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

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_in_channels = bottleneck_channels
        for i in range(num_stages - 1, -1, -1):
            feature = features[i]

            self.upconvs.append(
                nn.ConvTranspose2d(decoder_in_channels, feature,
                                   kernel_size=2, stride=2)
            )
            self.decoders.append(MBConvBlock(feature * 2, feature))
            decoder_in_channels = feature

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        bottleneck_features = x

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[idx]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        seg_output = self.final_conv(x)
        pooled_features = self.global_pool(bottleneck_features)

        return seg_output, pooled_features


# ===================================
# 3. OGRU - Optimized GRU
# ===================================

class OptimizedGRU(nn.Module):
    """GRU with SANGO-optimized hidden_dim"""

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

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)

        return output


# ===================================
# 4. Complete Architecture with 4 SANGO Parameters
# ===================================

class PaperMultiModelDR(nn.Module):
    """
    Complete model with 4 SANGO-optimizable parameters:
    1. densenet_hidden_dim (hidden_dim1) - DenseNet block size
    2. gru_hidden_dim (hidden_dim2) - GRU hidden size
    3. dropout - Dropout rate
    4. (lr is used in optimizer, not model architecture)
    """

    def __init__(self,
                 num_classes=CLASSIFICATION_CLASSES,
                 segmentation_classes=SEGMENTATION_CLASSES,
                 densenet_hidden_dim=256,  # SANGO param 1
                 gru_hidden_dim=128,       # SANGO param 2
                 gru_num_layers=2,
                 gru_dropout=0.3):         # SANGO param 3
        super().__init__()

        # 1. U-Net (FIXED - not optimized by SANGO)
        self.unet = ModifiedUNet(
            in_channels=3,
            out_channels=segmentation_classes,
            base_features=32,  # Fixed
            num_stages=5       # Fixed
        )

        unet_feature_dim = self.unet.bottleneck_channels

        # 2. DenseNet block with SANGO-optimized hidden dimension
        self.dense_block = nn.Sequential(
            nn.Linear(unet_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, densenet_hidden_dim),  # SANGO optimizes this
            nn.ReLU()
        )

        # 3. Attention mechanism
        self.attention_fc1 = nn.Sequential(
            nn.Linear(unet_feature_dim, densenet_hidden_dim),
            nn.SiLU(),
        )
        self.attention_fc2 = nn.Sequential(
            nn.Linear(densenet_hidden_dim, densenet_hidden_dim),
            nn.Sigmoid()
        )

        # 4. Feature fusion
        fusion_dim = densenet_hidden_dim * 2  # Dense + Attention

        # 5. OGRU with SANGO-optimized hidden_dim and dropout
        self.ogru = OptimizedGRU(
            input_dim=fusion_dim,
            hidden_dim=gru_hidden_dim,      # SANGO optimizes this
            num_layers=gru_num_layers,
            dropout=gru_dropout,             # SANGO optimizes this
            num_classes=num_classes
        )

    def forward(self, x, external_features=None):
        # U-Net
        seg_output, pooled_features = self.unet(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # DenseNet pathway
        dense_features = self.dense_block(pooled_features)

        # Attention pathway
        attention_1 = self.attention_fc1(pooled_features)
        attention_weights = self.attention_fc2(attention_1)
        attention_features = attention_1 * attention_weights

        # Fusion
        fused_features = torch.cat([dense_features, attention_features], dim=1)

        if external_features is not None:
            fused_features = torch.cat([fused_features, external_features], dim=1)

        # Classification
        classification_output = self.ogru(fused_features)

        return classification_output, seg_output


# ===================================
# 5. Focal Loss
# ===================================

class FocalLoss(nn.Module):
    """Focal Loss for classification"""

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()


# ===================================
# 6. SANGO Integration - 4 Parameters - FIXED
# ===================================

def create_paper_model_with_sango(
        train_loader,
        val_loader,
        device,
        use_sango=True,
        num_classes=CLASSIFICATION_CLASSES,
        # SANGO configuration (tunable)
        sango_pop_size: int = 12,
        sango_max_iterations: int = 60,
        sango_eval_epochs: int = 2,
        sango_train_batches: int = 20,
        sango_val_batches: int = 10,
):
    """
    Create model with SANGO optimization for 4 hyperparameters - FIXED VERSION
    """

    if use_sango:
        # Import here to check if available and set SANGO_class for later use
        try:
            from enhanced_sango import EnhancedSANGO
            SANGO_class = EnhancedSANGO
            print("Using EnhancedSANGO")
        except ImportError:
            try:
                from sango import SANGO as SANGO_impl
                SANGO_class = SANGO_impl
                print("Using original SANGO")
            except ImportError:
                print("ERROR: No SANGO implementation found!")
                print("Creating model with default parameters instead...")
                use_sango = False

    if use_sango:
        print("="*70)
        print("RUNNING SANGO OPTIMIZATION - 4 HYPERPARAMETERS")
        print("="*70)
        print("Optimizing:")
        print("  1. DenseNet Hidden Dim (hidden_dim1): [128, 512]")
        print("  2. GRU Hidden Dim (hidden_dim2): [64, 256]")
        print("  3. Dropout: [0.1, 0.5]")
        print("  4. Learning Rate (lr): [1e-5, 1e-3]")
        print("="*70)

        # Fitness function - FIXED
        def fitness_function(params):
            """
            Evaluate model with given hyperparameters - improved stability and reproducibility.
            params can be either dict or array
            """
            import random as _random
            try:
                # Handle both dict and array input
                if isinstance(params, dict):
                    densenet_hidden = int(params.get('hidden_dim1', 256))
                    gru_hidden = int(params.get('hidden_dim2', 128))
                    dropout = float(params.get('dropout', 0.3))
                    lr = float(params.get('lr', 1e-4))
                else:
                    densenet_hidden = int(params[0])
                    gru_hidden = int(params[1])
                    dropout = float(params[2])
                    lr = float(params[3])

                print(f"\nTesting: Dense={densenet_hidden}, GRU={gru_hidden}, "
                      f"Drop={dropout:.3f}, LR={lr:.6f}")

                # Reproducible seeds for each fitness evaluation (helps reduce noise)
                try:
                    torch.manual_seed(SEED)
                    np.random.seed(SEED)
                    _random.seed(SEED)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED)
                        # Keep cuDNN deterministic for reproducibility (may slow down)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                except Exception:
                    pass

                # Create model
                model = PaperMultiModelDR(
                    num_classes=num_classes,
                    segmentation_classes=SEGMENTATION_CLASSES,
                    densenet_hidden_dim=densenet_hidden,
                    gru_hidden_dim=gru_hidden,
                    gru_num_layers=2,
                    gru_dropout=dropout
                ).to(device)

                # Setup training
                criterion_cls = FocalLoss(num_classes=num_classes)
                criterion_seg = nn.BCEWithLogitsLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                # Quick training (controlled number of batches/epochs for SANGO evaluation)
                model.train()
                max_train_batches = min(sango_train_batches, len(train_loader))
                train_batches = 0

                for epoch in range(sango_eval_epochs):
                    epoch_loss = 0

                    for batch_data in train_loader:
                        # FIXED: Handle variable unpacking
                        if len(batch_data) == 4:
                            images, labels, masks, _ = batch_data
                        elif len(batch_data) == 3:
                            images, labels, masks = batch_data
                        else:
                            images, labels = batch_data[:2]
                            masks = torch.zeros(images.size(0), SEGMENTATION_CLASSES,
                                              images.size(2), images.size(3))

                        images = images.to(device)
                        labels = labels.to(device)
                        masks = masks.to(device)

                        optimizer.zero_grad()
                        cls_out, seg_out = model(images)

                        loss_cls = criterion_cls(cls_out, labels)
                        loss_seg = criterion_seg(seg_out, masks)
                        loss = loss_cls + 0.5 * loss_seg

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        train_batches += 1

                        # Limit batches for quick evaluation
                        if train_batches >= max_train_batches:
                            break

                    if train_batches >= max_train_batches:
                        break

                # Validation on a slightly larger batch set for more stable F1 estimate
                model.eval()
                all_preds = []
                all_labels = []
                val_batch_count = 0
                max_val_batches = min(sango_val_batches, len(val_loader))

                with torch.no_grad():
                    for batch_data in val_loader:
                        # FIXED: Handle variable unpacking
                        if len(batch_data) >= 2:
                            images = batch_data[0].to(device)
                            labels = batch_data[1]
                        else:
                            continue

                        cls_out, _ = model(images)
                        preds = torch.argmax(cls_out, dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.numpy())

                        val_batch_count += 1
                        if val_batch_count >= max_val_batches:
                            break

                # Calculate F1 score (use weighted average to match validation metric)
                from sklearn.metrics import f1_score as _f1
                if len(all_preds) > 0 and len(all_labels) > 0:
                    f1 = _f1(all_labels, all_preds, average='weighted', zero_division=0)
                    fitness = 1 - f1  # Minimize
                else:
                    f1 = 0.0
                    fitness = 1.0

                print(f"  → F1: {f1:.4f}, Fitness: {fitness:.4f}")

                # Cleanup
                del model
                torch.cuda.empty_cache()

                return fitness

            except Exception as e:
                print(f"  → Error in fitness function: {str(e)}")
                return 1.0  # Worst fitness

        # Initialize SANGO
        try:
            sango = SANGO_class(
                fitness_function=fitness_function,
                dim=4,
                population_size=sango_pop_size,
                max_iterations=sango_max_iterations,
                bounds={
                    'hidden_dim1': (64, 256),  # DenseNet Hidden Dim
                    'hidden_dim2': (64, 256),  # GRU Hidden Dim
                    'dropout': (0.1, 0.5),  # Dropout rate
                    'lr': (1e-5, 1e-3)
                }
            )

            print(f"\nSANGO Configuration:")
            print(f"  Population Size: {sango_pop_size}")
            print(f"  Max Iterations: {sango_max_iterations}")
            print(f"  Total Evaluations: {sango_pop_size * sango_max_iterations}")
            print()

            # Run optimization
            best_pos, best_fitness = sango.run()

            # Parse best parameters
            best_params = {
                'hidden_dim1': int(best_pos[0]),
                'hidden_dim2': int(best_pos[1]),
                'dropout': float(best_pos[2]),
                'lr': float(best_pos[3]),
                'f1_score': 1 - best_fitness
            }

        except Exception as e:
            print(f"ERROR running SANGO: {e}")
            print("Using default parameters...")
            best_params = {
                'hidden_dim1': 256,
                'hidden_dim2': 128,
                'dropout': 0.3,
                'lr': 1e-4,
                'f1_score': 0.0
            }

        print("\n" + "="*70)
        print("SANGO OPTIMIZATION COMPLETE!")
        print("="*70)
        print(f"Best Parameters Found:")
        print(f"  DenseNet Hidden Dim: {best_params['hidden_dim1']}")
        print(f"  GRU Hidden Dim: {best_params['hidden_dim2']}")
        print(f"  Dropout: {best_params['dropout']:.3f}")
        print(f"  Learning Rate: {best_params['lr']:.6f}")
        print(f"  Best F1-Score: {best_params['f1_score']:.4f}")
        print("="*70 + "\n")

        # Create final model with best parameters
        model = PaperMultiModelDR(
            num_classes=num_classes,
            segmentation_classes=SEGMENTATION_CLASSES,
            densenet_hidden_dim=best_params['hidden_dim1'],
            gru_hidden_dim=best_params['hidden_dim2'],
            gru_num_layers=2,
            gru_dropout=best_params['dropout']
        )

        return model, best_params

    else:
        # No SANGO - use default parameters
        print("Using default hyperparameters (no SANGO optimization)")

        model = PaperMultiModelDR(
            num_classes=num_classes,
            segmentation_classes=SEGMENTATION_CLASSES,
            densenet_hidden_dim=256,
            gru_hidden_dim=128,
            gru_num_layers=2,
            gru_dropout=0.3
        )

        default_params = {
            'hidden_dim1': 256,
            'hidden_dim2': 128,
            'dropout': 0.3,
            'lr': 1e-4
        }

        return model, default_params


# ===================================
# Test
# ===================================

if __name__ == "__main__":
    print("Testing PaperMultiModelDR with 4 SANGO parameters...")

    model = PaperMultiModelDR(
        num_classes=5,
        segmentation_classes=3,
        densenet_hidden_dim=256,
        gru_hidden_dim=128,
        gru_dropout=0.3
    )

    x = torch.randn(2, 3, 224, 224)
    cls_out, seg_out = model(x)

    print(f"✓ Classification output: {cls_out.shape}")
    print(f"✓ Segmentation output: {seg_out.shape}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")