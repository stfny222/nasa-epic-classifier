"""
Loss Functions
==============

Loss functions for training (PyTorch).
"""

import torch
import torch.nn as nn

class GeographicBCELoss(nn.Module):
    """
    Binary cross-entropy loss for multi-label geographic classification.
    
    Each of the 6 continent labels is an independent binary classification task.
    """
    
    def __init__(self, pos_weight=None):
        """
        Parameters
        ----------
        pos_weight : torch.Tensor, optional
            Weight for positive class per label (shape: [num_labels])
            Useful for imbalanced labels
        """
        super(GeographicBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, outputs, targets):
        """
        Compute binary cross-entropy loss.
        
        Parameters
        ----------
        outputs : torch.Tensor
            Model logits (batch, num_labels)
        targets : torch.Tensor
            Ground truth binary labels (batch, num_labels)
            
        Returns
        -------
        tuple
            (total_loss, loss_dict)
        """
        loss = self.criterion(outputs, targets)
        
        loss_dict = {
            'geographic': loss.item()
        }
        
        return loss, loss_dict
