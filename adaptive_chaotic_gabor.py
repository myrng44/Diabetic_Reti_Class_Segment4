"""
Adaptive Chaotic Gabor Filter với Chebyshev Chaotic Map
Based on paper equation (2) and (3)
"""

import cv2
import numpy as np
from typing import List, Tuple


class AdaptiveChaoticGaborFilter:
    """
    Adaptive Gabor Filter enhanced with Chebyshev chaotic map.

    From paper:
    g(x,y; λ, θ, ψ, σ, γ) = exp(-(x'^2 + γ^2*y'^2)/(2σ^2)) * exp(i(2πx'/λ + ψ)) + d_{t+1}

    where d_{t+1} = cos(0.5 * cos^{-1}(d_t)) is the Chebyshev chaotic map
    """

    def __init__(
            self,
            frequencies: List[float] = [0.1, 0.3, 0.5],
            angles: List[int] = [0, 45, 90, 135],
            sigma: float = 2.0,
            gamma: float = 0.5,
            psi: float = 0,
            ksize: int = 31
    ):
        """
        Initialize Adaptive Chaotic Gabor Filter.

        Args:
            frequencies: Wavelengths (lambda) for Gabor kernels
            angles: Orientations in degrees
            sigma: Standard deviation of Gaussian envelope
            gamma: Spatial aspect ratio
            psi: Phase offset
            ksize: Kernel size (must be odd)
        """
        self.frequencies = frequencies
        self.angles = [np.deg2rad(angle) for angle in angles]
        self.sigma = sigma
        self.gamma = gamma
        self.psi = psi
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1

        # Initialize Chebyshev chaotic sequence
        self.chaotic_sequence = self._generate_chebyshev_sequence(
            len(frequencies) * len(self.angles)
        )

    def _generate_chebyshev_sequence(self, length: int, d0: float = 0.7) -> np.ndarray:
        """
        Generate Chebyshev chaotic map sequence.

        d_{t+1} = cos(0.5 * arccos(d_t))

        Args:
            length: Number of values to generate
            d0: Initial value in range (-1, 1)

        Returns:
            Chaotic sequence
        """
        sequence = np.zeros(length)
        d_t = d0

        for i in range(length):
            # Chebyshev map iteration
            d_t = np.cos(0.5 * np.arccos(np.clip(d_t, -1, 1)))
            sequence[i] = d_t

        return sequence

    def create_gabor_kernel(
            self,
            wavelength: float,
            theta: float,
            chaotic_value: float
    ) -> np.ndarray:
        """
        Create single Gabor kernel with chaotic enhancement.

        Args:
            wavelength: Wavelength (lambda) of sinusoidal factor
            theta: Orientation of normal to parallel stripes
            chaotic_value: Value from Chebyshev map

        Returns:
            Complex Gabor kernel
        """
        # Create kernel coordinates
        kernel_half = self.ksize // 2
        y, x = np.meshgrid(
            np.arange(-kernel_half, kernel_half + 1),
            np.arange(-kernel_half, kernel_half + 1)
        )

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Gaussian envelope
        gaussian = np.exp(
            -(x_theta ** 2 + self.gamma ** 2 * y_theta ** 2) / (2 * self.sigma ** 2)
        )

        # Sinusoidal carrier with phase
        sinusoid = np.exp(1j * (2 * np.pi * x_theta / wavelength + self.psi))

        # Apply chaotic enhancement (paper equation 2)
        # Add chaotic value to enhance adaptivity
        gabor_kernel = gaussian * sinusoid + chaotic_value

        return gabor_kernel

    def apply_filter_bank(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply bank of adaptive chaotic Gabor filters.

        Args:
            image: Grayscale image

        Returns:
            enhanced_image: Combined filtered result
            responses: List of individual filter responses
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0

        responses = []
        chaotic_idx = 0

        # Apply filters with different frequencies and orientations
        for wavelength in self.frequencies:
            for theta in self.angles:
                # Get chaotic value for this filter
                chaotic_value = self.chaotic_sequence[chaotic_idx]
                chaotic_idx += 1

                # Create kernel
                kernel = self.create_gabor_kernel(wavelength, theta, chaotic_value)

                # Apply filter (use real part)
                filtered = cv2.filter2D(gray, cv2.CV_32F, np.real(kernel))

                responses.append(filtered)

        # Combine responses - take maximum magnitude
        combined = np.maximum.reduce([np.abs(r) for r in responses])

        # Normalize to [0, 255]
        combined = ((combined - combined.min()) /
                    (combined.max() - combined.min() + 1e-8) * 255).astype(np.uint8)

        return combined, responses

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for contrast enhancement."""
        if len(image.shape) == 3:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Equalize L channel
            l_eq = cv2.equalizeHist(l)

            # Merge and convert back
            enhanced = cv2.merge((l_eq, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            enhanced = cv2.equalizeHist(image)

        return enhanced

    def denoise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving denoising."""
        denoised = cv2.bilateralFilter(image, kernel_size, 75, 75)
        return denoised

    def process_image(
            self,
            image: np.ndarray,
            apply_enhancement: bool = True,
            apply_denoising: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Complete adaptive chaotic Gabor filtering pipeline.

        Args:
            image: Input RGB or grayscale image
            apply_enhancement: Apply contrast enhancement
            apply_denoising: Apply denoising

        Returns:
            final_image: Processed image
            gabor_result: Gabor filtered result
            responses: Individual filter responses
        """
        # Apply Gabor filter bank
        gabor_result, responses = self.apply_filter_bank(image)

        # Convert to RGB if needed
        if len(gabor_result.shape) == 2:
            gabor_rgb = cv2.cvtColor(gabor_result, cv2.COLOR_GRAY2RGB)
        else:
            gabor_rgb = gabor_result.copy()

        final_image = gabor_rgb

        # Optional enhancement
        if apply_enhancement:
            final_image = self.enhance_contrast(final_image)

        # Optional denoising
        if apply_denoising:
            final_image = self.denoise(final_image)

        return final_image, gabor_result, responses


def compare_gabor_filters(image: np.ndarray) -> dict:
    """
    Compare conventional Gabor vs Adaptive Chaotic Gabor.

    Returns comparison results and visualizations.
    """
    # Adaptive Chaotic Gabor
    adaptive_filter = AdaptiveChaoticGaborFilter()
    adaptive_result, _, _ = adaptive_filter.process_image(image)

    # Conventional Gabor (without chaotic enhancement)
    conventional_filter = AdaptiveChaoticGaborFilter()
    conventional_filter.chaotic_sequence = np.zeros_like(conventional_filter.chaotic_sequence)
    conventional_result, _, _ = conventional_filter.process_image(image)

    # Calculate metrics
    def calculate_metrics(img1, img2):
        # Contrast enhancement ratio
        contrast_1 = img1.std()
        contrast_2 = img2.std()

        # Edge preservation (gradient magnitude)
        grad_1 = np.abs(cv2.Sobel(img1, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(img1, cv2.CV_64F, 0, 1))
        grad_2 = np.abs(cv2.Sobel(img2, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(img2, cv2.CV_64F, 0, 1))

        return {
            'contrast': contrast_1 / (contrast_2 + 1e-8),
            'edge_magnitude': grad_1.mean() / (grad_2.mean() + 1e-8)
        }

    metrics = calculate_metrics(adaptive_result, conventional_result)

    return {
        'adaptive_result': adaptive_result,
        'conventional_result': conventional_result,
        'metrics': metrics
    }


if __name__ == "__main__":
    # Test với ảnh mẫu
    import matplotlib.pyplot as plt

    # Tạo ảnh test
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    test_image = cv2.GaussianBlur(test_image, (5, 5), 0)

    # Apply filter
    agf = AdaptiveChaoticGaborFilter()
    result, gabor, responses = agf.process_image(test_image)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gabor, cmap='gray')
    axes[0, 1].set_title('Gabor Result')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result)
    axes[0, 2].set_title('Final Enhanced')
    axes[0, 2].axis('off')

    # Show some individual responses
    for i in range(3):
        if i < len(responses):
            axes[1, i].imshow(responses[i], cmap='gray')
            axes[1, i].set_title(f'Response {i + 1}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('gabor_test.png', dpi=150)
    print("Test completed! Saved to gabor_test.png")