"""
Image preprocessing module for diabetic retinopathy detection.
Includes CLAHE, fundus masking, Gabor filtering, and augmentation pipelines.
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import filters
from config import *

# Import Adaptive Chaotic Gabor if enabled
if USE_ADAPTIVE_GABOR:
    try:
        from adaptive_gabor import AdaptiveChaoticGaborFilter
    except ImportError:
        print("Warning: adaptive_gabor.py not found. Using standard preprocessing.")
        USE_ADAPTIVE_GABOR = False

class FundusPreprocessor:
    """Main preprocessing class for fundus images."""

    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
        self.augmentation = self.get_augmentation_pipeline()

    def apply_clahe_rgb(self, image):
        """Enhanced CLAHE application with additional color space processing."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE on L channel with adaptive clip limit
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        cl = clahe.apply(l)

        # Apply additional contrast enhancement
        cl = cv2.normalize(cl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Merge channels and convert back to RGB
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        # Apply additional color correction
        enhanced = cv2.addWeighted(enhanced, 1.2, image, 0.0, 0)
        return enhanced

    def get_augmentation_pipeline(self):
        """Enhanced augmentation pipeline with more transformations."""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3
                ),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                fill_value=0,
                p=0.2
            ),
            ToTensorV2()
        ])

    def fundus_circle_mask(self, image, thr=FUNDUS_THRESHOLD):
        """
        Create a circular mask of the fundus region.
        Returns mask (H,W) dtype uint8 {0,1}.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray, dtype=np.uint8)

        largest = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(largest)

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
        return mask

    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline."""
        # Create fundus mask with improved thresholding
        circle_mask = self.fundus_circle_mask(image)

        # Apply mask
        img_masked = image.copy()
        img_masked[circle_mask == 0] = 0

        # Apply enhanced CLAHE
        img_clahe = self.apply_clahe_rgb(img_masked)

        # Apply bilateral filter for noise reduction while preserving edges
        img_filtered = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply unsharp masking for detail enhancement
        gaussian = cv2.GaussianBlur(img_filtered, (0, 0), 2.0)
        img_enhanced = cv2.addWeighted(img_filtered, 1.5, gaussian, -0.5, 0)

        # Resize with better interpolation
        img_resized = cv2.resize(
            img_enhanced,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LANCZOS4
        )

        return img_resized, circle_mask


class GaborFilter:
    """Adaptive Gabor filter for texture enhancement and noise reduction."""

    def __init__(self, frequencies=GABOR_FREQUENCIES, angles=GABOR_ANGLES):
        self.frequencies = frequencies
        self.angles = np.deg2rad(angles)

    def apply_gabor_bank(self, image):
        """
        Apply bank of Gabor filters with different frequencies and orientations.
        Args:
            image: Grayscale image
        Returns:
            filtered_image: Enhanced image
            responses: List of filter responses
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        responses = []

        for freq in self.frequencies:
            for angle in self.angles:
                # Apply Gabor filter
                real, _ = filters.gabor(
                    gray,
                    frequency=freq,
                    theta=angle,
                    sigma_x=GABOR_SIGMA_X,
                    sigma_y=GABOR_SIGMA_Y
                )
                responses.append(real)

        # Combine responses (take maximum response)
        combined = np.maximum.reduce(responses)

        # Normalize to 0-255
        combined = ((combined - combined.min()) /
                    (combined.max() - combined.min()) * 255).astype(np.uint8)

        return combined, responses

    def enhance_contrast(self, image):
        """Apply contrast enhancement after Gabor filtering."""
        # Apply histogram equalization
        if len(image.shape) == 3:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            enhanced = cv2.merge((l_eq, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            enhanced = cv2.equalizeHist(image)

        return enhanced

    def denoise(self, image, kernel_size=5):
        """Apply denoising using bilateral filter."""
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, kernel_size, 80, 80)
        else:
            denoised = cv2.bilateralFilter(image, kernel_size, 80, 80)
        return denoised


class AdaptiveGaborFilter:
    """Enhanced Adaptive Gabor filter with dynamic parameter adjustment."""

    def __init__(self, frequencies=GABOR_FREQUENCIES, angles=GABOR_ANGLES):
        self.frequencies = frequencies
        self.angles = np.deg2rad(angles)
        self.sigma = GABOR_SIGMA
        self.gamma = GABOR_GAMMA
        self.kernel_size = GABOR_KERNEL_SIZE

    def create_gabor_kernel(self, frequency, angle):
        """Create Gabor kernel with dynamic parameters."""
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        # Calculate kernel size based on frequency
        kernel_size = min(self.kernel_size,
                         int(2 * np.ceil(max(sigma_x, sigma_y) * 3)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma_x,
            angle,
            frequency,
            self.gamma,
            0,
            ktype=cv2.CV_32F
        )

    def apply_gabor_bank(self, image):
        """Apply enhanced Gabor filter bank with response normalization."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Normalize input image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        responses = []
        for freq in self.frequencies:
            for angle in self.angles:
                kernel = self.create_gabor_kernel(freq, angle)
                response = cv2.filter2D(gray, cv2.CV_32F, kernel)
                responses.append(response)

        # Combine responses using maximum response
        combined = np.maximum.reduce(responses)

        # Normalize and enhance contrast
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
        combined = combined.astype(np.uint8)

        return combined, responses


def get_training_transforms(target_size=TARGET_SIZE):
    """Get augmentation pipeline for training."""
    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=AUGMENTATION_PROB),
        A.VerticalFlip(p=AUGMENTATION_PROB),
        A.Rotate(limit=ROTATION_LIMIT, p=AUGMENTATION_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=BRIGHTNESS_CONTRAST_LIMIT,
            contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
            p=AUGMENTATION_PROB
        ),
        A.ShiftScaleRotate(
            shift_limit=SHIFT_LIMIT,
            scale_limit=SCALE_LIMIT,
            rotate_limit=ROTATION_LIMIT,
            p=AUGMENTATION_PROB
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=AUGMENTATION_PROB),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_validation_transforms(target_size=TARGET_SIZE):
    """Get transforms for validation/testing (no augmentation)."""
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_segmentation_transforms(target_size=TARGET_SIZE, is_train=True):
    """Get transforms for segmentation task (includes mask transforms)."""
    if is_train:
        return A.Compose([
            A.Resize(target_size, target_size),
            A.HorizontalFlip(p=AUGMENTATION_PROB),
            A.VerticalFlip(p=AUGMENTATION_PROB),
            A.Rotate(limit=ROTATION_LIMIT, p=AUGMENTATION_PROB),
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_CONTRAST_LIMIT,
                contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
                p=AUGMENTATION_PROB
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(target_size, target_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


class PreprocessingPipeline:
    """Complete preprocessing pipeline combining all techniques."""

    def __init__(self, use_gabor=USE_ADAPTIVE_GABOR, use_adaptive_gabor=True,
                 save_intermediates=False):
        self.fundus_preprocessor = FundusPreprocessor()

        # Choose between standard and adaptive chaotic Gabor
        if use_gabor and use_adaptive_gabor and USE_ADAPTIVE_GABOR:
            try:
                from adaptive_gabor import AdaptiveChaoticGaborFilter
                self.gabor_filter = AdaptiveChaoticGaborFilter(
                    frequencies=GABOR_FREQUENCIES,
                    angles=GABOR_ANGLES,
                    sigma=GABOR_SIGMA,
                    gamma=GABOR_GAMMA,
                    ksize=GABOR_KERNEL_SIZE
                )
                print("Using Adaptive Chaotic Gabor Filter (Paper method)")
            except ImportError:
                self.gabor_filter = GaborFilter() if use_gabor else None
                print("Using standard Gabor Filter")
        elif use_gabor:
            self.gabor_filter = GaborFilter()
        else:
            self.gabor_filter = None

        self.save_intermediates = save_intermediates

    def process_image(self, image, output_dir=None, filename=None):
        """
        Complete preprocessing pipeline.
        Args:
            image: Input RGB image
            output_dir: Directory to save intermediate results
            filename: Base filename for saving
        Returns:
            processed_image: Final processed image
            intermediate_results: Dict of intermediate processing steps
        """
        intermediate_results = {}

        # Step 1: Basic preprocessing (CLAHE + masking)
        processed_img, circle_mask = self.fundus_preprocessor.preprocess_image(image)
        intermediate_results['clahe'] = processed_img
        intermediate_results['mask'] = circle_mask

        # Step 2: Gabor filtering (optional)
        if self.gabor_filter:
            if hasattr(self.gabor_filter, 'process_image'):
                # Adaptive Chaotic Gabor (Paper method)
                gabor_enhanced, gabor_result, responses = self.gabor_filter.process_image(
                    processed_img,
                    apply_enhancement=True,
                    apply_denoising=True
                )
                intermediate_results['gabor'] = gabor_result
                intermediate_results['gabor_responses'] = responses[:3]  # Save first 3
                processed_img = gabor_enhanced
            else:
                # Standard Gabor
                gabor_enhanced, gabor_responses = self.gabor_filter.apply_gabor_bank(processed_img)
                if len(gabor_enhanced.shape) == 2:
                    gabor_enhanced = cv2.cvtColor(gabor_enhanced, cv2.COLOR_GRAY2RGB)
                gabor_enhanced[circle_mask == 0] = 0
                denoised = self.gabor_filter.denoise(gabor_enhanced)
                intermediate_results['gabor'] = gabor_enhanced
                intermediate_results['denoised'] = denoised
                processed_img = denoised

            # Save intermediate results if requested
        if self.save_intermediates and output_dir and filename:
            self._save_intermediates(intermediate_results, output_dir, filename)

        return processed_img, intermediate_results

    def _save_intermediates(self, results, output_dir, filename):
        """Save intermediate processing results."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]

        for step_name, img in results.items():
            if step_name == 'mask':
                # Save mask as grayscale
                save_path = os.path.join(output_dir, f"{base_name}_{step_name}.png")
                cv2.imwrite(save_path, img * 255)
            else:
                # Save image
                save_path = os.path.join(output_dir, f"{base_name}_{step_name}.png")
                if len(img.shape) == 3:
                    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(save_path, img)


def preprocess_dataset_batch(image_paths, output_dir, use_gabor=True):
    """
    Batch preprocess a list of images.
    Args:
        image_paths: List of image file paths
        output_dir: Directory to save processed images
        use_gabor: Whether to apply Gabor filtering
    """
    import os
    from tqdm import tqdm

    pipeline = PreprocessingPipeline(use_gabor=use_gabor, save_intermediates=True)
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing images"):
        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Process
        filename = os.path.basename(img_path)
        processed_img, _ = pipeline.process_image(
            img_rgb,
            output_dir=output_dir,
            filename=filename
        )

        # Save final processed image
        final_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(final_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        img = cv2.imread(input_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pipeline = PreprocessingPipeline(use_gabor=True, save_intermediates=True)
            processed, intermediates = pipeline.process_image(
                img_rgb,
                output_dir="preprocessed_samples",
                filename="test.jpg"
            )

            print("Preprocessing completed. Check 'preprocessed_samples' directory.")
        else:
            print(f"Could not read image: {input_path}")
    else:
        print("Usage: python preprocessing.py <image_path>")