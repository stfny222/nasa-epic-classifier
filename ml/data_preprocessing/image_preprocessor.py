"""
Image Preprocessor
==================

Resize, normalize, and augment images for training.
"""

import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Dict


# Define geographic label columns - continents only (ignoring ocean/arctic/antarctic)
GEOGRAPHIC_LABEL_COLUMNS = [
    'visible_north_america', 'visible_south_america', 'visible_europe',
    'visible_africa', 'visible_asia', 'visible_oceania'
]


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


def encode_geographic_labels(row: pd.Series) -> np.ndarray:
    """
    Encode geographic labels as binary array.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing geographic label columns
        
    Returns
    -------
    np.ndarray
        Binary array of shape (11,) with 0/1 for each region
    """
    labels = np.zeros(len(GEOGRAPHIC_LABEL_COLUMNS), dtype=np.float32)
    
    for i, col in enumerate(GEOGRAPHIC_LABEL_COLUMNS):
        if col in row and row[col]:
            labels[i] = 1.0
    
    return labels
