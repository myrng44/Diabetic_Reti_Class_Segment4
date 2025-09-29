"""
Feature extraction module for diabetic retinopathy detection.
Includes LBP, SURF, and texture energy measures (TEM).
"""

import cv2
import numpy as np
from skimage import feature
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from config import *

class LBPExtractor:
    """Local Binary Pattern feature extractor."""

    def __init__(self, radius=LBP_RADIUS, n_points=LBP_N_POINTS, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method

    def extract_features(self, image):
        """
        Extract LBP features from image.
        Args:
            image: Input image (RGB or grayscale)
        Returns:
            features: LBP histogram features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Compute LBP
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method=self.method)

        # Calculate histogram
        n_bins = self.n_points + 2 if self.method == 'uniform' else 2**self.n_points
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Normalize histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)

        return hist, lbp

    def extract_multiscale_features(self, image, scales=[1, 2, 3]):
        """Extract LBP features at multiple scales."""
        features_list = []

        for scale in scales:
            radius = self.radius * scale
            n_points = self.n_points

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            lbp = local_binary_pattern(gray, n_points, radius, method=self.method)

            n_bins = n_points + 2 if self.method == 'uniform' else 2**n_points
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)

            features_list.append(hist)

        return np.concatenate(features_list)


class SURFExtractor:
    """SURF feature detector and descriptor extractor."""

    def __init__(self, hessian_threshold=SURF_HESSIAN_THRESHOLD):
        self.hessian_threshold = hessian_threshold
        # Try to create SURF detector
        try:
            self.surf = cv2.xfeatures2d.SURF_create(self.hessian_threshold)
        except AttributeError:
            # Fallback to ORB if SURF not available
            print("SURF not available, using ORB instead")
            self.surf = cv2.ORB_create(nfeatures=500)

    def extract_features(self, image):
        """
        Extract SURF keypoints and descriptors.
        Args:
            image: Input image
        Returns:
            keypoints: List of keypoints
            descriptors: Feature descriptors
            feature_vector: Aggregated feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.surf.detectAndCompute(gray, None)

        if descriptors is None:
            # Return zeros if no features found
            return keypoints, None, np.zeros(128)

        # Aggregate descriptors using bag-of-words approach
        # Simple approach: use mean and std of descriptors
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        feature_vector = np.concatenate([mean_desc, std_desc])

        return keypoints, descriptors, feature_vector

    def create_bow_features(self, descriptors_list, vocab_size=100):
        """
        Create bag-of-words features from list of descriptors.
        Args:
            descriptors_list: List of descriptor arrays
            vocab_size: Size of vocabulary
        Returns:
            bow_features: Bag-of-words feature vectors
        """
        from sklearn.cluster import KMeans

        # Concatenate all descriptors
        all_descriptors = np.vstack([desc for desc in descriptors_list if desc is not None])

        # Create vocabulary using K-means
        kmeans = KMeans(n_clusters=vocab_size, random_state=SEED)
        kmeans.fit(all_descriptors)

        # Create BoW features for each image
        bow_features = []
        for descriptors in descriptors_list:
            if descriptors is None:
                bow_features.append(np.zeros(vocab_size))
            else:
                # Assign each descriptor to closest cluster
                labels = kmeans.predict(descriptors)
                hist, _ = np.histogram(labels, bins=vocab_size, range=(0, vocab_size))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-8)
                bow_features.append(hist)

        return np.array(bow_features), kmeans


class TextureEnergyMeasure:
    """Texture Energy Measure (TEM) using Gray Level Co-occurrence Matrix."""

    def __init__(self, distances=[1, 2, 3], angles=[0, 45, 90, 135]):
        self.distances = distances
        self.angles = np.deg2rad(angles)

    def extract_features(self, image):
        """
        Extract texture features using GLCM.
        Args:
            image: Input grayscale image
        Returns:
            features: Texture feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Quantize image to reduce computation
        gray = (gray // 32).astype(np.uint8)  # 8 levels

        features = []

        for distance in self.distances:
            # Compute GLCM for all angles
            glcm = graycomatrix(
                gray,
                distances=[distance],
                angles=self.angles,
                levels=8,
                symmetric=True,
                normed=True
            )

            # Extract Haralick features
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()

            features.extend([
                np.mean(contrast), np.std(contrast),
                np.mean(dissimilarity), np.std(dissimilarity),
                np.mean(homogeneity), np.std(homogeneity),
                np.mean(energy), np.std(energy),
                np.mean(correlation), np.std(correlation)
            ])

        return np.array(features)

    def extract_advanced_features(self, image):
        """Extract additional texture features."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        features = []

        # Statistical features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            entropy(np.histogram(gray, bins=256)[0] + 1e-8),
        ])

        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude)
        ])

        return np.array(features)


class CombinedFeatureExtractor:
    """Combines all feature extraction methods."""

    def __init__(self, use_lbp=True, use_surf=True, use_tem=True):
        self.use_lbp = use_lbp
        self.use_surf = use_surf
        self.use_tem = use_tem

        if use_lbp:
            self.lbp_extractor = LBPExtractor()
        if use_surf:
            self.surf_extractor = SURFExtractor()
        if use_tem:
            self.tem_extractor = TextureEnergyMeasure()

    def extract_features(self, image):
        """
        Extract combined features from image.
        Args:
            image: Input image
        Returns:
            feature_vector: Combined feature vector
            feature_dict: Dictionary of individual features
        """
        feature_dict = {}
        feature_vector = []

        if self.use_lbp:
            lbp_features, lbp_image = self.lbp_extractor.extract_features(image)
            feature_dict['lbp'] = lbp_features
            feature_dict['lbp_image'] = lbp_image
            feature_vector.append(lbp_features)

        if self.use_surf:
            keypoints, descriptors, surf_features = self.surf_extractor.extract_features(image)
            feature_dict['surf_keypoints'] = keypoints
            feature_dict['surf_descriptors'] = descriptors
            feature_dict['surf_features'] = surf_features
            feature_vector.append(surf_features)

        if self.use_tem:
            tem_features = self.tem_extractor.extract_features(image)
            tem_advanced = self.tem_extractor.extract_advanced_features(image)
            combined_tem = np.concatenate([tem_features, tem_advanced])
            feature_dict['tem'] = combined_tem
            feature_vector.append(combined_tem)

        # Combine all features
        if feature_vector:
            combined_features = np.concatenate(feature_vector)
        else:
            combined_features = np.array([])

        return combined_features, feature_dict

    def extract_multiscale_features(self, image, scales=[0.5, 1.0, 1.5]):
        """Extract features at multiple scales."""
        all_features = []

        original_size = image.shape[:2]

        for scale in scales:
            # Resize image
            new_size = (int(original_size[1] * scale), int(original_size[0] * scale))
            if scale != 1.0:
                scaled_image = cv2.resize(image, new_size)
            else:
                scaled_image = image.copy()

            # Extract features
            features, _ = self.extract_features(scaled_image)
            all_features.append(features)

        return np.concatenate(all_features)


class DeepFeatureExtractor:
    """Extract deep features using pre-trained CNN."""

    def __init__(self, model_name='resnet50', layer_name='avgpool'):
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.eval()
        self.model.to(self.device)

        # Register hook to extract features
        self.features = None
        self.hook = self._register_hook(layer_name)

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _register_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.features = output.detach()

        # Find the layer and register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module.register_forward_hook(hook_fn)

        raise ValueError(f"Layer {layer_name} not found in model")

    def extract_features(self, image):
        """
        Extract deep features from image.
        Args:
            image: Input image (RGB numpy array)
        Returns:
            features: Deep feature vector
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            input_tensor = image.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Get features and flatten
        features = self.features.cpu().numpy().flatten()
        return features


def create_feature_dataset(image_paths, output_path=None, use_deep_features=False):
    """
    Create feature dataset from list of images.
    Args:
        image_paths: List of image file paths
        output_path: Path to save features (optional)
        use_deep_features: Whether to include deep CNN features
    Returns:
        features: Array of feature vectors
        feature_names: List of feature names
    """
    from tqdm import tqdm

    extractor = CombinedFeatureExtractor()
    if use_deep_features:
        deep_extractor = DeepFeatureExtractor()

    all_features = []

    for img_path in tqdm(image_paths, desc="Extracting features"):
        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Extract traditional features
        features, _ = extractor.extract_features(img_rgb)

        # Extract deep features if requested
        if use_deep_features:
            deep_features = deep_extractor.extract_features(img_rgb)
            features = np.concatenate([features, deep_features])

        all_features.append(features)

    features_array = np.array(all_features)

    # Create feature names
    feature_names = []
    if extractor.use_lbp:
        feature_names.extend([f"lbp_{i}" for i in range(LBP_N_POINTS + 2)])
    if extractor.use_surf:
        feature_names.extend([f"surf_mean_{i}" for i in range(64)])
        feature_names.extend([f"surf_std_{i}" for i in range(64)])
    if extractor.use_tem:
        n_tem_features = len(extractor.tem_extractor.extract_features(img_rgb))
        n_advanced_features = len(extractor.tem_extractor.extract_advanced_features(img_rgb))
        feature_names.extend([f"tem_{i}" for i in range(n_tem_features)])
        feature_names.extend([f"tem_advanced_{i}" for i in range(n_advanced_features)])

    if use_deep_features:
        feature_names.extend([f"deep_{i}" for i in range(deep_extractor.feature_dim)])

    # Save if requested
    if output_path:
        np.savez(output_path,
                features=features_array,
                feature_names=feature_names,
                image_paths=image_paths)
        print(f"Features saved to {output_path}")

    return features_array, feature_names


def visualize_features(image, feature_dict, save_path=None):
    """
    Visualize extracted features.
    Args:
        image: Original image
        feature_dict: Dictionary of features from CombinedFeatureExtractor
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # LBP image
    if 'lbp_image' in feature_dict:
        axes[1].imshow(feature_dict['lbp_image'], cmap='gray')
        axes[1].set_title('LBP Pattern')
        axes[1].axis('off')

    # LBP histogram
    if 'lbp' in feature_dict:
        axes[2].bar(range(len(feature_dict['lbp'])), feature_dict['lbp'])
        axes[2].set_title('LBP Histogram')
        axes[2].set_xlabel('Pattern')
        axes[2].set_ylabel('Frequency')

    # SURF keypoints
    if 'surf_keypoints' in feature_dict and feature_dict['surf_keypoints']:
        img_with_keypoints = image.copy()
        for kp in feature_dict['surf_keypoints']:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(img_with_keypoints, (x, y), 3, (255, 0, 0), -1)
        axes[3].imshow(img_with_keypoints)
        axes[3].set_title(f'SURF Keypoints ({len(feature_dict["surf_keypoints"])})')
        axes[3].axis('off')

    # Texture features visualization
    if 'tem' in feature_dict:
        axes[4].plot(feature_dict['tem'])
        axes[4].set_title('Texture Features')
        axes[4].set_xlabel('Feature Index')
        axes[4].set_ylabel('Value')

    # Combined features
    if len(feature_dict) > 0:
        # Get all features except images
        numeric_features = []
        for key, value in feature_dict.items():
            if key not in ['lbp_image', 'surf_keypoints', 'surf_descriptors']:
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    numeric_features.extend(value.tolist())

        if numeric_features:
            axes[5].plot(numeric_features)
            axes[5].set_title('All Combined Features')
            axes[5].set_xlabel('Feature Index')
            axes[5].set_ylabel('Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        img = cv2.imread(input_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract features
            extractor = CombinedFeatureExtractor()
            features, feature_dict = extractor.extract_features(img_rgb)

            print(f"Extracted {len(features)} features")
            print(f"Feature vector shape: {features.shape}")

            # Visualize
            visualize_features(img_rgb, feature_dict, "feature_visualization.png")

        else:
            print(f"Could not read image: {input_path}")
    else:
        print("Usage: python feature_extraction.py <image_path>")