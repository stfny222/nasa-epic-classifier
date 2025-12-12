"""
ResNet Transfer Learning for Geographic Classification
=======================================================

Pre-trained ResNet18/34 fine-tuned for continent visibility prediction.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetGeographic(nn.Module):
    """
    ResNet-based model for multi-label geographic classification.
    
    Uses pre-trained ResNet backbone with custom classification head.
    """
    
    def __init__(self, num_labels=6, backbone='resnet18', pretrained=True, freeze_early_layers=True):
        """
        Parameters
        ----------
        num_labels : int
            Number of binary labels (6 continents)
        backbone : str
            ResNet variant: 'resnet18', 'resnet34', 'resnet50'
        pretrained : bool
            Use ImageNet pre-trained weights
        freeze_early_layers : bool
            Freeze layers 1-3, only train layer4 + head
        """
        super(ResNetGeographic, self).__init__()
        
        # Load pre-trained ResNet
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze early layers (conv1, bn1, layer1, layer2, layer3)
        if freeze_early_layers:
            for name, param in self.resnet.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
                    param.requires_grad = False
        
        # Remove original FC layer
        self.resnet.fc = nn.Identity()
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        
        self.backbone_name = backbone
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
        # Extract features with ResNet
        features = self.resnet(x)  # (batch, feature_dim)
        
        # Classification head
        logits = self.head(features)  # (batch, num_labels)
        
        return logits
    
    def unfreeze_all_layers(self):
        """Unfreeze all ResNet layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True


def build_resnet_geographic(
    num_labels=6, 
    backbone='resnet18', 
    pretrained=True, 
    freeze_early_layers=True
):
    """
    Build and return a ResNet-based geographic classifier.
    
    Parameters
    ----------
    num_labels : int
        Number of binary labels (6 continents)
    backbone : str
        ResNet variant: 'resnet18' (11M params), 'resnet34' (21M), 'resnet50' (25M)
    pretrained : bool
        Use ImageNet pre-trained weights (recommended)
    freeze_early_layers : bool
        Freeze early layers to prevent overfitting on small dataset
        
    Returns
    -------
    ResNetGeographic
        Instantiated model
        
    Notes
    -----
    Training strategy:
    1. Start with freeze_early_layers=True, train 20-30 epochs
    2. Optionally unfreeze all layers and fine-tune with lower LR (0.00001)
    
    Expected improvement over baseline CNN:
    - Baseline F1: 0.45
    - ResNet F1: 0.50-0.60 (pre-trained features help with small dataset)
    """
    model = ResNetGeographic(
        num_labels=num_labels,
        backbone=backbone,
        pretrained=pretrained,
        freeze_early_layers=freeze_early_layers
    )
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nResNet Geographic Model ({backbone}):")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")
    
    return model
