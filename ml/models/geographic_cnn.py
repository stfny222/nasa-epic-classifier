"""
Geographic CNN for Continent Classification
============================================

CNN for predicting visible continents (6 binary labels).
"""

import torch
import torch.nn as nn


class GeographicCNN(nn.Module):
    """
    CNN for multi-label geographic classification.
    
    Predicts which continents are visible in Earth observation images.
    Uses sigmoid activation for independent binary predictions.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_labels=6, dropout_rate=0.5):
        """
        Parameters
        ----------
        input_shape : tuple
            (height, width, channels) - note: PyTorch uses CHW internally
        num_labels : int
            Number of binary labels (default 6 for continents)
        dropout_rate : float
            Dropout probability
        """
        super(GeographicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer - sigmoid for multi-label binary classification
        self.fc_output = nn.Linear(512, num_labels)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, 3, height, width)
            
        Returns
        -------
        torch.Tensor
            Logits for each label (batch, num_labels)
        """
        # Convolutional feature extraction
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output (logits, will apply sigmoid in loss function)
        x = self.fc_output(x)
        
        return x


def build_geographic_cnn(input_shape=(224, 224, 3), num_labels=6, dropout_rate=0.5):
    """
    Build and return a geographic CNN model.
    
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
    GeographicCNN
        Instantiated model
    """
    model = GeographicCNN(input_shape, num_labels, dropout_rate)
    return model
