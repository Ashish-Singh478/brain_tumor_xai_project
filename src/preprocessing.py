"""MRI preprocessing operations described in the reference paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable parameters whose values are not specified by the paper."""

    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    colormap: int = cv2.COLORMAP_JET
    crop_padding: int = 0


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array and fail explicitly on invalid input."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_brain_region(image: np.ndarray, padding: int = 0) -> np.ndarray:
    """Crop to the extreme points of the largest foreground contour."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if width <= 1 or height <= 1:
        return image

    image_height, image_width = image.shape[:2]
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return image[top:bottom, left:right]


def preprocess_image(
    image: np.ndarray,
    image_size: int,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Apply crop, bilateral denoising, colormap, and resize in paper order."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    config = config or PreprocessingConfig()
    cropped = crop_brain_region(image, config.crop_padding)
    denoised = cv2.bilateralFilter(
        cropped,
        config.bilateral_diameter,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )
    grayscale = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    colored = cv2.applyColorMap(grayscale, config.colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.resize(colored, (image_size, image_size), interpolation=cv2.INTER_AREA)

