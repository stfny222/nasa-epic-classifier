"""
Multi-Scale CNN for Geographic Classification
==============================================

CNN with parallel branches using different kernel sizes to capture
continents at different scales (large: N.America, Africa vs small: Oceania).
"""

import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for geographic classification.
    
    Uses parallel convolutional branches with different kernel sizes
    to capture features at multiple scales, then concatenates them.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_labels=6, dropout_rate=0.5):
        """
        Parameters
        ----------
        input_shape : tuple
            (height, width, channels)
        num_labels : int
            Number of binary labels (6 continents)
        dropout_rate : float
            Dropout probability
        """
        super(MultiScaleCNN, self).__init__()
        
        # Initial conv layer (shared across all branches)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        
        # Multi-scale branches (parallel processing)
        # Branch 1: Small features (3x3 kernels) - good for small continents like Oceania
        self.branch_small = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # Branch 2: Medium features (5x5 kernels) - good for medium continents
        self.branch_medium = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # Branch 3: Large features (7x7 kernels) - good for large continents like Africa, Asia
        self.branch_large = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # Combine branches (64 + 64 + 64 = 192 channels)
        # Further processing on concatenated features
        self.combined_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, 3, H, W)
            
        Returns
        -------
        torch.Tensor
            Logits (batch, num_labels)
        """
        # Initial shared processing
        x = self.initial_conv(x)  # (batch, 32, 112, 112)
        
        # Parallel multi-scale branches
        small = self.branch_small(x)   # (batch, 64, 56, 56)
        medium = self.branch_medium(x) # (batch, 64, 56, 56)
        large = self.branch_large(x)   # (batch, 64, 56, 56)
        
        # Concatenate along channel dimension
        combined = torch.cat([small, medium, large], dim=1)  # (batch, 192, 56, 56)
        
        # Further processing
        x = self.combined_conv(combined)  # (batch, 256, 14, 14)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_labels)
        
        return logits


def build_multiscale_cnn(input_shape=(224, 224, 3), num_labels=6, dropout_rate=0.5):
    """
    Build and return a multi-scale CNN model.
    
    Parameters
    ----------
    input_shape : tuple
        Input image shape (H, W, C)
    num_labels : int
        Number of binary labels (6 continents)
    dropout_rate : float
        Dropout rate
        
    Returns
    -------
    MultiScaleCNN
        Instantiated model
        
    Notes
    -----
    Architecture rationale:
    - Parallel branches capture features at different scales
    - Small kernels (3x3): Fine details, edges, small regions (Oceania, Europe)
    - Medium kernels (5x5): Mid-scale patterns (South America)
    - Large kernels (7x7): Large-scale structures (Africa, Asia, N.America)
    - Concatenating all scales gives model access to multi-resolution features
    
    Expected performance:
    - Better than baseline CNN (0.45) due to multi-scale processing
    - Possibly similar to ResNet (0.83) but without pre-training advantage
    - Target F1: 0.55-0.65
    """
    model = MultiScaleCNN(
        input_shape=input_shape,
        num_labels=num_labels,
        dropout_rate=dropout_rate
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nMulti-Scale CNN Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: 3 parallel branches (3x3, 5x5, 7x7 kernels)")
    print(f"  Feature fusion: Concatenation → combined processing")
    
    return model
