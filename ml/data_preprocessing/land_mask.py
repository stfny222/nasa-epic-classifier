"""
Land/Water/Cloud Masking for EPIC Images

Applies weighted masking to focus model on landmasses while de-emphasizing:
- White pixels (clouds)
- Blue pixels (water/ocean)
"""

import numpy as np
import cv2
from PIL import Image


def create_land_mask(image_array, 
                     cloud_threshold=200,
                     water_threshold=150,
                     land_weight=1.0,
                     cloud_weight=0.1,
                     water_weight=0.2):
    """
    Create a land/water/cloud mask for EPIC satellite imagery.
    
    Args:
        image_array: numpy array of shape (H, W, 3) with values 0-255
        cloud_threshold: RGB value above which pixels are considered clouds (white)
        water_threshold: Blue channel value above which pixels are considered water
        land_weight: Weight for land pixels (default 1.0)
        cloud_weight: Weight for cloud pixels (default 0.1)
        water_weight: Weight for water pixels (default 0.2)
    
    Returns:
        mask: float array of shape (H, W) with weights for each pixel
              - 1.0 for land (dark colors)
              - cloud_weight for clouds (white/bright)
              - water_weight for water (blue-heavy)
    """
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    image_array = image_array.astype(np.float32)
    H, W = image_array.shape[:2]
    
    # Extract RGB channels
    R = image_array[:, :, 0]
    G = image_array[:, :, 1]
    B = image_array[:, :, 2]
    
    # Initialize mask with land weight
    mask = np.ones((H, W), dtype=np.float32) * land_weight
    
    # Identify clouds: high values across all channels (white)
    # Clouds typically have R≈G≈B and all > threshold
    cloud_mask = (R > cloud_threshold) & (G > cloud_threshold) & (B > cloud_threshold)
    mask[cloud_mask] = cloud_weight
    
    # Identify water: high blue relative to red (blue-heavy)
    # Also apply water_threshold on blue channel
    # Water typically has B > R and B > G
    water_mask = (B > R + 30) & (B > G + 30) & (B > water_threshold)
    mask[water_mask] = water_weight
    
    # Normalize to [0, 1] range
    mask = mask / np.max([land_weight, cloud_weight, water_weight])
    
    return mask


def apply_land_mask(image_array, mask_weight=0.5):
    """
    Apply land mask by adjusting pixel intensity.
    
    This reduces the intensity of cloud/water pixels while preserving land features.
    
    Args:
        image_array: numpy array of shape (H, W, 3)
        mask_weight: How strongly to apply the mask (0=no effect, 1=maximum effect)
    
    Returns:
        masked_image: Image with cloud/water pixels dampened
    """
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    mask = create_land_mask(image_array)
    
    # Apply mask by scaling pixel intensities
    # Land pixels (mask=1) are unaffected
    # Cloud/water pixels are darkened
    masked = image_array.astype(np.float32).copy()
    
    for c in range(3):
        masked[:, :, c] = masked[:, :, c] * (1 - mask_weight * (1 - mask))
    
    return np.clip(masked, 0, 255).astype(np.uint8)


def create_focus_mask_tensor(image_array, device='cpu'):
    """
    Create a spatial attention mask that can be used in loss weighting.
    
    This creates a mask where:
    - Land pixels get weight 1.0 (full importance)
    - Water pixels get weight 0.3 (reduced importance)
    - Cloud pixels get weight 0.1 (minimal importance)
    
    Args:
        image_array: numpy array or PIL Image
        device: torch device for the output tensor
    
    Returns:
        mask: torch tensor of shape (H, W) with attention weights
    """
    import torch
    
    mask = create_land_mask(image_array, 
                           cloud_weight=0.1,
                           water_weight=0.3,
                           land_weight=1.0)
    
    return torch.from_numpy(mask).float().to(device)


# Test/visualization functions
def visualize_mask(image_path_or_array):
    """
    Visualize the land/water/cloud mask for debugging.
    
    Args:
        image_path_or_array: Path to image file or numpy array
    
    Returns:
        mask visualization (for notebook display)
    """
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array).convert('RGB')
        image_array = np.array(image)
    else:
        image_array = image_path_or_array
    
    mask = create_land_mask(image_array)
    
    # Create colored visualization
    # Red = land (weight 1.0)
    # Yellow = water (weight 0.3)
    # Cyan = clouds (weight 0.1)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Normalize mask to [0, 1]
    mask_norm = mask / mask.max()
    
    # Use mask values to color
    vis[:, :, 0] = (mask_norm * 255).astype(np.uint8)  # Red channel
    vis[:, :, 1] = ((1 - mask_norm) * 255).astype(np.uint8)  # Green channel
    vis[:, :, 2] = ((1 - mask_norm) * 255).astype(np.uint8)  # Blue channel
    
    return Image.fromarray(vis)


if __name__ == "__main__":
    # Test with a sample image
    import matplotlib.pyplot as plt
    
    # Create synthetic test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[20:40, 20:40] = [100, 150, 50]  # Green land
    test_img[50:70, 50:70] = [50, 100, 200]  # Blue water
    test_img[80:90, 80:90] = [240, 240, 245]  # White clouds
    
    mask = create_land_mask(test_img)
    masked = apply_land_mask(test_img, mask_weight=0.8)
    
    print(f"Mask range: {mask.min():.3f} - {mask.max():.3f}")
    print(f"Original dtype: {test_img.dtype}, Masked dtype: {masked.dtype}")
    print(f"Land region (should be ~1.0): {mask[25, 25]:.3f}")
    print(f"Water region (should be ~0.2): {mask[60, 60]:.3f}")
    print(f"Cloud region (should be ~0.1): {mask[85, 85]:.3f}")
