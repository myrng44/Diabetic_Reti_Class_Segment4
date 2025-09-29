"""
Explainable AI module for diabetic retinopathy detection.
Includes Grad-CAM, attention visualization, and confidence scoring.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from config import *

class GradCAM:
    """Grad-CAM implementation for CNN visualization."""

    def __init__(self, model, target_layers=None):
        self.model = model
        self.target_layers = target_layers or self._get_target_layers()
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _get_target_layers(self):
        """Automatically detect target layers for Grad-CAM."""
        target_layers = []

        # For common architectures, get the last convolutional layer
        if hasattr(self.model, 'features'):
            # DenseNet, VGG-style models
            for name, module in self.model.features.named_children():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
                    target_layers.append(f'features.{name}')
        elif hasattr(self.model, 'layer4'):
            # ResNet-style models
            target_layers.append('layer4')
        elif hasattr(self.model, 'cnn_backbone'):
            # Custom models with backbone
            if hasattr(self.model.cnn_backbone, 'backbone'):
                if hasattr(self.model.cnn_backbone.backbone, 'features'):
                    target_layers.append('cnn_backbone.backbone.features')
                elif hasattr(self.model.cnn_backbone.backbone, 'layer4'):
                    target_layers.append('cnn_backbone.backbone.layer4')

        if not target_layers:
            # Fallback: find the last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layers = [name]

        print(f"Using target layers for Grad-CAM: {target_layers}")
        return target_layers

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        for target_layer in self.target_layers:
            # Navigate to the target layer
            layer = self.model
            for attr in target_layer.split('.'):
                layer = getattr(layer, attr)

            # Register hooks
            self.hooks.append(layer.register_forward_hook(forward_hook(target_layer)))
            self.hooks.append(layer.register_backward_hook(backward_hook(target_layer)))

    def generate_cam(self, input_tensor, class_idx=None, layer_name=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index (None for max prediction)
            layer_name: Specific layer to use (None for first target layer)

        Returns:
            cam: Grad-CAM heatmap
            prediction: Model prediction
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # Get prediction
        if output.dim() == 2 and output.shape[1] > 1:
            prediction = F.softmax(output, dim=1)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
        else:
            prediction = torch.sigmoid(output)
            class_idx = 0

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx] if output.dim() > 1 else output[0]
        class_score.backward()

        # Get target layer
        target_layer = layer_name or self.target_layers[0]

        # Get gradients and activations
        gradients = self.gradients[target_layer]  # (1, C, H, W)
        activations = self.activations[target_layer]  # (1, C, H, W)

        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Generate CAM
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # (H, W)

        # Apply ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), prediction.cpu().numpy()

    def visualize_cam(self, input_tensor, cam, original_image=None,
                     class_idx=None, save_path=None, alpha=0.4):
        """
        Visualize Grad-CAM heatmap overlaid on original image.

        Args:
            input_tensor: Original input tensor
            cam: Grad-CAM heatmap
            original_image: Original image (before preprocessing)
            class_idx: Predicted class index
            save_path: Path to save visualization
            alpha: Overlay transparency

        Returns:
            visualization: Combined visualization
        """
        # Convert input tensor to image
        if original_image is None:
            if input_tensor.dim() == 4:
                img = input_tensor.squeeze(0)
            else:
                img = input_tensor

            # Denormalize if needed
            img = img.permute(1, 2, 0).cpu().numpy()
            if img.min() < 0:  # Likely normalized
                # Reverse ImageNet normalization
                mean = np.array(IMAGENET_MEAN)
                std = np.array(IMAGENET_STD)
                img = img * std + mean

            img = np.clip(img, 0, 1)
            original_image = (img * 255).astype(np.uint8)

        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

        # Create heatmap
        heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)

        # Overlay heatmap on original image
        visualization = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(visualization)
        title = f'Grad-CAM Overlay'
        if class_idx is not None and class_idx < len(CLASSIFICATION_CLASS_NAMES):
            title += f'\nPredicted: {CLASSIFICATION_CLASS_NAMES[class_idx]}'
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")

        plt.show()

        return visualization

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


class AttentionVisualizer:
    """Visualizer for attention mechanisms in models."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []

        # Register hooks for attention layers
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register hooks for attention layers."""
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Some attention modules return (output, attention_weights)
                    if len(output) > 1:
                        self.attention_maps[name] = output[1].detach()
                    else:
                        self.attention_maps[name] = output[0].detach()
                else:
                    self.attention_maps[name] = output.detach()
            return hook

        # Find attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or isinstance(module, torch.nn.MultiheadAttention):
                self.hooks.append(module.register_forward_hook(attention_hook(name)))

    def visualize_attention(self, input_tensor, save_path=None):
        """Visualize attention maps."""
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)

        if not self.attention_maps:
            print("No attention maps found. Model may not have attention mechanisms.")
            return

        num_maps = len(self.attention_maps)
        fig, axes = plt.subplots(1, num_maps + 1, figsize=(5 * (num_maps + 1), 5))

        if num_maps == 0:
            axes = [axes]

        # Show original image
        img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if img.min() < 0:
            mean = np.array(IMAGENET_MEAN)
            std = np.array(IMAGENET_STD)
            img = img * std + mean
        img = np.clip(img, 0, 1)

        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Show attention maps
        for idx, (name, attention_map) in enumerate(self.attention_maps.items()):
            if attention_map.dim() > 2:
                # Average over heads/channels if needed
                if attention_map.dim() == 4:  # (batch, heads, seq, seq)
                    attention_map = attention_map.mean(dim=1)  # Average over heads
                if attention_map.dim() == 3:  # (batch, seq, seq) or similar
                    attention_map = attention_map[0]  # Take first batch

                # If it's a sequence attention, reshape to spatial
                if attention_map.shape[0] == attention_map.shape[1]:  # Square matrix
                    # Try to reshape to spatial dimensions
                    seq_len = attention_map.shape[0]
                    spatial_size = int(np.sqrt(seq_len))
                    if spatial_size * spatial_size == seq_len:
                        attention_map = attention_map.view(spatial_size, spatial_size)

            # Resize to match input image size
            if attention_map.dim() == 2:
                attention_np = attention_map.cpu().numpy()
                attention_resized = cv2.resize(
                    attention_np,
                    (input_tensor.shape[3], input_tensor.shape[2])
                )

                axes[idx + 1].imshow(attention_resized, cmap='hot')
                axes[idx + 1].set_title(f'Attention: {name}')
                axes[idx + 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")

        plt.show()

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


class ConfidenceScorer:
    """Generate confidence scores for predictions."""

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def calculate_prediction_confidence(self, input_tensor, num_samples=10):
        """
        Calculate prediction confidence using Monte Carlo Dropout.

        Args:
            input_tensor: Input tensor
            num_samples: Number of MC samples

        Returns:
            mean_prediction: Mean prediction
            confidence: Prediction confidence (1 - std)
            uncertainty: Prediction uncertainty (std)
        """
        # Enable dropout for inference
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout or type(m) == torch.nn.Dropout2d:
                m.train()

        self.model.apply(enable_dropout)

        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                output = self.model(input_tensor)

                if output.dim() > 1 and output.shape[1] > 1:
                    # Classification
                    probs = F.softmax(output, dim=1)
                else:
                    # Binary or regression
                    probs = torch.sigmoid(output)

                predictions.append(probs.cpu().numpy())

        # Calculate statistics
        predictions = np.array(predictions)  # (num_samples, batch_size, num_classes)

        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        # Confidence as 1 - normalized std
        if mean_prediction.ndim > 1:
            confidence = 1 - (std_prediction / (mean_prediction + 1e-8))
            confidence = np.mean(confidence, axis=-1)  # Average over classes
        else:
            confidence = 1 - std_prediction

        # Set model back to eval mode
        self.model.eval()

        return mean_prediction, confidence, std_prediction

    def calculate_entropy_confidence(self, input_tensor):
        """Calculate confidence based on prediction entropy."""
        with torch.no_grad():
            output = self.model(input_tensor)

            if output.dim() > 1 and output.shape[1] > 1:
                probs = F.softmax(output, dim=1)

                # Calculate entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                # Normalize entropy to [0, 1]
                max_entropy = np.log(output.shape[1])
                normalized_entropy = entropy / max_entropy

                # Confidence = 1 - normalized_entropy
                confidence = 1 - normalized_entropy

                return probs.cpu().numpy(), confidence.cpu().numpy()
            else:
                probs = torch.sigmoid(output)
                # For binary case, calculate entropy differently
                entropy = -(probs * torch.log(probs + 1e-8) +
                          (1 - probs) * torch.log(1 - probs + 1e-8))
                confidence = 1 - entropy / np.log(2)  # Normalize by log(2)

                return probs.cpu().numpy(), confidence.cpu().numpy()


class ExplainabilityPipeline:
    """Complete explainability pipeline combining all techniques."""

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device

        # Initialize components
        self.gradcam = GradCAM(model)
        self.attention_viz = AttentionVisualizer(model)
        self.confidence_scorer = ConfidenceScorer(model, device)

    def explain_prediction(self, input_tensor, original_image=None,
                          save_dir=None, filename=None):
        """
        Complete explanation of a single prediction.

        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original image before preprocessing
            save_dir: Directory to save explanations
            filename: Base filename for saving

        Returns:
            explanation_dict: Dictionary containing all explanations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        explanation = {}

        # 1. Basic prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))

            if output.dim() > 1 and output.shape[1] > 1:
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0, pred_class].item()

                explanation['prediction'] = {
                    'class': pred_class,
                    'class_name': CLASSIFICATION_CLASS_NAMES[pred_class] if pred_class < len(CLASSIFICATION_CLASS_NAMES) else f"Class {pred_class}",
                    'probability': pred_prob,
                    'all_probabilities': probs.cpu().numpy().tolist()
                }
            else:
                prob = torch.sigmoid(output).item()
                explanation['prediction'] = {
                    'probability': prob,
                    'binary_prediction': 1 if prob > 0.5 else 0
                }

        # 2. Grad-CAM visualization
        try:
            cam, _ = self.gradcam.generate_cam(input_tensor.to(self.device))
            explanation['gradcam'] = cam

            if save_dir and filename:
                gradcam_path = os.path.join(save_dir, f"{filename}_gradcam.png")
                self.gradcam.visualize_cam(
                    input_tensor, cam, original_image,
                    explanation['prediction'].get('class'),
                    save_path=gradcam_path
                )
                explanation['gradcam_path'] = gradcam_path
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            explanation['gradcam'] = None

        # 3. Attention visualization (if applicable)
        try:
            if save_dir and filename:
                attention_path = os.path.join(save_dir, f"{filename}_attention.png")
                self.attention_viz.visualize_attention(
                    input_tensor.to(self.device),
                    save_path=attention_path
                )
                explanation['attention_path'] = attention_path
        except Exception as e:
            print(f"Attention visualization failed: {e}")

        # 4. Confidence scoring
        try:
            # Monte Carlo confidence
            mean_pred, mc_confidence, uncertainty = self.confidence_scorer.calculate_prediction_confidence(
                input_tensor.to(self.device)
            )

            # Entropy confidence
            entropy_probs, entropy_confidence = self.confidence_scorer.calculate_entropy_confidence(
                input_tensor.to(self.device)
            )

            explanation['confidence'] = {
                'monte_carlo_confidence': float(mc_confidence[0]) if mc_confidence.ndim > 0 else float(mc_confidence),
                'entropy_confidence': float(entropy_confidence[0]) if entropy_confidence.ndim > 0 else float(entropy_confidence),
                'prediction_uncertainty': float(uncertainty.std()) if hasattr(uncertainty, 'std') else float(uncertainty),
                'reliable': float(mc_confidence[0] if mc_confidence.ndim > 0 else mc_confidence) > CONFIDENCE_THRESHOLD
            }
        except Exception as e:
            print(f"Confidence calculation failed: {e}")
            explanation['confidence'] = None

        # 5. Feature importance (simplified)
        try:
            # Calculate gradient-based feature importance
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor.to(self.device))

            if output.dim() > 1:
                target_output = output[0, explanation['prediction']['class']]
            else:
                target_output = output[0]

            self.model.zero_grad()
            target_output.backward()

            gradients = input_tensor.grad.abs().mean(dim=(0, 2, 3))  # Average over spatial dims
            feature_importance = gradients.cpu().numpy()

            explanation['feature_importance'] = {
                'channel_importance': feature_importance.tolist(),
                'most_important_channel': int(np.argmax(feature_importance))
            }

        except Exception as e:
            print(f"Feature importance calculation failed: {e}")
            explanation['feature_importance'] = None

        return explanation

    def batch_explain(self, dataloader, save_dir, max_samples=None):
        """
        Generate explanations for a batch of samples.

        Args:
            dataloader: DataLoader with samples to explain
            save_dir: Directory to save explanations
            max_samples: Maximum number of samples to process

        Returns:
            List of explanations
        """
        os.makedirs(save_dir, exist_ok=True)
        explanations = []

        count = 0
        for batch in tqdm(dataloader, desc="Generating explanations"):
            if len(batch) == 3:
                images, labels, filenames = batch
            else:
                images, labels = batch
                filenames = [f"sample_{count + i}" for i in range(len(images))]

            for i in range(images.shape[0]):
                if max_samples and count >= max_samples:
                    break

                image_tensor = images[i:i+1]  # Keep batch dimension
                filename = os.path.splitext(filenames[i])[0]

                explanation = self.explain_prediction(
                    image_tensor,
                    save_dir=save_dir,
                    filename=filename
                )

                explanation['filename'] = filenames[i]
                explanation['true_label'] = labels[i].item() if hasattr(labels[i], 'item') else labels[i]

                explanations.append(explanation)
                count += 1

                if max_samples and count >= max_samples:
                    break

        # Save batch explanations summary
        summary_path = os.path.join(save_dir, "explanations_summary.json")

        # Prepare summary for JSON serialization
        json_explanations = []
        for exp in explanations:
            json_exp = exp.copy()
            # Convert numpy arrays to lists
            if 'gradcam' in json_exp and json_exp['gradcam'] is not None:
                json_exp['gradcam'] = "saved_as_image"  # Don't serialize large arrays
            json_explanations.append(json_exp)

        import json
        with open(summary_path, 'w') as f:
            json.dump(json_explanations, f, indent=2)

        print(f"Generated {len(explanations)} explanations. Summary saved to {summary_path}")

        return explanations

    def cleanup(self):
        """Clean up all components."""
        self.gradcam.cleanup()
        self.attention_viz.cleanup()


def create_explanation_dashboard(explanations, save_path):
    """Create an HTML dashboard with explanations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DR Detection Explanations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .explanation { border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 5px; }
            .prediction { background-color: #e7f3ff; padding: 10px; border-radius: 3px; margin: 10px 0; }
            .confidence { background-color: #f0f8f0; padding: 10px; border-radius: 3px; margin: 10px 0; }
            .reliable { color: green; font-weight: bold; }
            .unreliable { color: red; font-weight: bold; }
            .images { display: flex; gap: 10px; flex-wrap: wrap; }
            .images img { max-width: 300px; height: auto; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Diabetic Retinopathy Detection - Explanations Dashboard</h1>
    """

    for i, exp in enumerate(explanations):
        html_content += f"""
        <div class="explanation">
            <h3>Sample {i+1}: {exp.get('filename', 'Unknown')}</h3>
            
            <div class="prediction">
                <strong>Prediction:</strong> {exp['prediction'].get('class_name', 'N/A')}<br>
                <strong>Probability:</strong> {exp['prediction'].get('probability', 0):.3f}<br>
                <strong>True Label:</strong> {exp.get('true_label', 'Unknown')}
            </div>
            
            <div class="confidence">
                <strong>Confidence Analysis:</strong><br>
        """

        if exp.get('confidence'):
            conf = exp['confidence']
            reliability_class = "reliable" if conf.get('reliable', False) else "unreliable"
            html_content += f"""
                MC Confidence: {conf.get('monte_carlo_confidence', 0):.3f}<br>
                Entropy Confidence: {conf.get('entropy_confidence', 0):.3f}<br>
                <span class="{reliability_class}">
                    {'Reliable' if conf.get('reliable', False) else 'Unreliable'} Prediction
                </span>
            """

        html_content += """
            </div>
            
            <div class="images">
        """

        # Add explanation images if available
        if 'gradcam_path' in exp:
            html_content += f'<img src="{os.path.basename(exp["gradcam_path"])}" alt="Grad-CAM">'

        if 'attention_path' in exp:
            html_content += f'<img src="{os.path.basename(exp["attention_path"])}" alt="Attention">'

        html_content += """
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(save_path, 'w') as f:
        f.write(html_content)

    print(f"Explanation dashboard saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Explainable AI module loaded successfully!")
    print("Use ExplainabilityPipeline to generate explanations for your models.")