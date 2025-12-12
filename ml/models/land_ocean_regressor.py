"""Land-Ocean Regressor Model

ResNet18-based regression model for predicting the percentage of land visible
in satellite images from DSCOVR/EPIC.

Output: Continuous percentage (0-100%) representing visible land area.
"""

import torch
import torch.nn as nn
from torchvision import models


class LandPercentageRegressor(nn.Module):
    """ResNet18 adapted for land percentage regression (0-100).
    
    Architecture:
    - ResNet18 backbone (pretrained on ImageNet)
    - Custom regression head: Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→1)
    - Output: Single continuous value (0-100%) clamped to valid range
    
    Input: 224×224 RGB images with ImageNet normalization
    Output: Land percentage (0-100)
    """
    
    def __init__(self, pretrained=True):
        """Initialize the regressor.
        
        Args:
            pretrained: Whether to load ImageNet weights (default: True)
        """
        super().__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the final FC layer for regression (outputs single value)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Single output: land percentage (0-100)
        )
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, 1) with raw predictions
        """
        return self.resnet(x)
