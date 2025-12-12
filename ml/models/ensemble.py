"""
Ensemble Model - ResNet18 + Multi-Scale CNN
============================================

Combines predictions from ResNet18 (transfer learning) and 
Multi-Scale CNN using weighted averaging.
"""

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for geographic classification.
    
    Combines predictions from different architectures using weighted averaging.
    """
    
    def __init__(self, models, weights=None):
        """
        Parameters
        ----------
        models : list of nn.Module
            List of trained models to ensemble
        weights : list of float, optional
            Weight for each model (should sum to 1.0)
            If None, uses equal weights
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        print(f"\nEnsemble Model:")
        print(f"  Number of models: {len(models)}")
        print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    
    def forward(self, x):
        """
        Forward pass - average predictions from all models.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, 3, H, W)
            
        Returns
        -------
        torch.Tensor
            Averaged logits (batch, num_labels)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        # Stack and apply weights
        stacked = torch.stack(predictions, dim=0)  # (num_models, batch, num_labels)
        weights_expanded = self.weights.view(-1, 1, 1)  # (num_models, 1, 1)
        
        # Weighted average
        ensemble_pred = (stacked * weights_expanded).sum(dim=0)  # (batch, num_labels)
        
        return ensemble_pred


def build_ensemble(model_configs, num_labels=6):
    """
    Build ensemble from trained model checkpoints.
    
    Parameters
    ----------
    model_configs : list of dict
        Each dict contains:
        - 'type': 'resnet18' or 'multiscale'
        - 'checkpoint_path': path to .pt file
        - 'weight': ensemble weight (optional)
    num_labels : int
        Number of output labels
        
    Returns
    -------
    EnsembleModel
        Ensemble model ready for inference
        
    Example
    -------
    >>> configs = [
    ...     {
    ...         'type': 'resnet18',
    ...         'checkpoint_path': 'experiments/resnet_transfer/outputs/model_best.pt',
    ...         'weight': 0.6  # Higher weight for better performer
    ...     },
    ...     {
    ...         'type': 'multiscale',
    ...         'checkpoint_path': 'experiments/multiscale_cnn/outputs/model_best.pt',
    ...         'weight': 0.4
    ...     }
    ... ]
    >>> ensemble = build_ensemble(configs, num_labels=6)
    """
    from models.resnet_geographic import build_resnet_geographic
    from models.multiscale_cnn import build_multiscale_cnn
    
    models = []
    weights = []
    
    print("Building ensemble from trained models...")
    
    for i, config in enumerate(model_configs):
        print(f"\n[{i+1}/{len(model_configs)}] Loading {config['type']}...")
        
        # Build model architecture
        if config['type'] == 'resnet18':
            model = build_resnet_geographic(
                num_labels=num_labels,
                backbone='resnet18',
                pretrained=False,  # Will load trained weights
                freeze_early_layers=True
            )
        elif config['type'] == 'multiscale':
            model = build_multiscale_cnn(
                num_labels=num_labels
            )
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Load trained weights
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()  # Set to evaluation mode
        
        # Freeze all parameters (ensemble is for inference only)
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
        weights.append(config.get('weight', 1.0))
        
        print(f"  ✓ Loaded checkpoint: {config['checkpoint_path']}")
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights=weights)
    ensemble.eval()
    
    return ensemble


def evaluate_ensemble_with_strategies(models_dict, X_test, y_test, label_names=None):
    """
    Evaluate ensemble with different weighting strategies.
    
    Parameters
    ----------
    models_dict : dict
        {model_name: {'model': model, 'f1': f1_score}}
    X_test : np.ndarray
        Test images
    y_test : np.ndarray
        Test labels
    label_names : list, optional
        Label names
        
    Returns
    -------
    dict
        Results for each ensemble strategy
    """
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Convert to torch
    X_torch = torch.from_numpy(X_test).permute(0, 3, 1, 2)
    
    # Get predictions from each model
    all_preds = {}
    for name, info in models_dict.items():
        model = info['model']
        model.eval()
        with torch.no_grad():
            outputs = model(X_torch)
            probs = torch.sigmoid(outputs).numpy()
        all_preds[name] = probs
    
    results = {}
    
    # Strategy 1: Equal weights
    equal_avg = np.mean([probs for probs in all_preds.values()], axis=0)
    preds_equal = (equal_avg > 0.35).astype(float)
    
    results['equal_weights'] = {
        'f1_macro': f1_score(y_test, preds_equal, average='macro', zero_division=0),
        'precision_macro': precision_score(y_test, preds_equal, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, preds_equal, average='macro', zero_division=0),
    }
    
    # Strategy 2: F1-weighted (better models get higher weight)
    f1_weights = np.array([info['f1'] for info in models_dict.values()])
    f1_weights = f1_weights / f1_weights.sum()
    
    weighted_avg = sum(probs * w for probs, w in zip(all_preds.values(), f1_weights))
    preds_weighted = (weighted_avg > 0.35).astype(float)
    
    results['f1_weighted'] = {
        'f1_macro': f1_score(y_test, preds_weighted, average='macro', zero_division=0),
        'precision_macro': precision_score(y_test, preds_weighted, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, preds_weighted, average='macro', zero_division=0),
        'weights': {name: float(w) for name, w in zip(models_dict.keys(), f1_weights)}
    }
    
    # Strategy 3: Max voting (take prediction if any model says yes with high confidence)
    max_probs = np.maximum.reduce([probs for probs in all_preds.values()])
    preds_max = (max_probs > 0.4).astype(float)
    
    results['max_voting'] = {
        'f1_macro': f1_score(y_test, preds_max, average='macro', zero_division=0),
        'precision_macro': precision_score(y_test, preds_max, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, preds_max, average='macro', zero_division=0),
    }
    
    return results
