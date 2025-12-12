"""
Image Preprocessor
==================

Load and preprocess images for training.
"""

import numpy as np
from PIL import Image
from typing import Tuple


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Load image and preprocess for model input.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    target_size : tuple
        (height, width) for resizing
    normalize : bool
        Whether to normalize to [0, 1]
        
    Returns
    -------
    np.ndarray
        Preprocessed image array (H, W, 3)
    """
    img = Image.open(image_path)
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize
    if normalize:
        img_array = img_array / 255.0
    
    return img_array
