"""
Ensemble Experiment
===================

Evaluate ensemble of ResNet18 + Multi-Scale CNN with different strategies.
"""

import sys
import os
import yaml
import json
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_preprocessing import load_preprocessed_epic_data
from data_preprocessing.image_preprocessor import GEOGRAPHIC_LABEL_COLUMNS
from models.resnet_geographic import build_resnet_geographic
from models.multiscale_cnn import build_multiscale_cnn
from training.geographic_trainer import evaluate_geographic_model


def main():
    """Main experiment function."""
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("EPIC ENSEMBLE EXPERIMENT (ResNet18 + Multi-Scale CNN)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(json.dumps(config, indent=2))
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/2] Loading preprocessed data...")
    
    dates = config['data'].get('dates', None)
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = \
        load_preprocessed_epic_data(
            dates=dates,
            collection=config['data']['collection'],
            image_type=config['data']['image_type'],
            target_size=tuple(config['data']['target_size']),
            limit_per_date=config['data'].get('limit_per_date'),
            force_recompute=False
        )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Test: {len(y_test['geographic'])} samples")
    
    # Load individual models
    print("\n[2/2] Loading trained models...")
    
    models_info = {}
    
    for i, model_config in enumerate(config['ensemble']['models']):
        print(f"\n[{i+1}/{len(config['ensemble']['models'])}] Loading {model_config['type']}...")
        
        # Resolve checkpoint path
        checkpoint_path = Path(__file__).parent / model_config['checkpoint_path']
        
        # Build model
        if model_config['type'] == 'resnet18':
            model = build_resnet_geographic(
                num_labels=len(GEOGRAPHIC_LABEL_COLUMNS),
                backbone='resnet18',
                pretrained=False,
                freeze_early_layers=True
            )
        elif model_config['type'] == 'multiscale':
            model = build_multiscale_cnn(
                num_labels=len(GEOGRAPHIC_LABEL_COLUMNS)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Load individual metrics
        metrics_path = checkpoint_path.parent / "test_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        models_info[model_config['type']] = {
            'model': model,
            'f1': metrics['f1_macro'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'weight': model_config.get('weight', 1.0)
        }
        
        print(f"  ✓ Loaded: F1={metrics['f1_macro']:.4f}")
    
    # Evaluate ensemble with different strategies
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION - Multiple Strategies")
    print("=" * 70)
    
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
    
    X_torch = torch.from_numpy(X_test).permute(0, 3, 1, 2)
    y_true = y_test['geographic']
    threshold = config['evaluation']['threshold']
    
    # Get predictions from each model
    all_probs = {}
    for name, info in models_info.items():
        model = info['model']
        with torch.no_grad():
            outputs = model(X_torch)
            probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs[name] = probs
    
    results = {}
    
    # Strategy 1: Equal weights
    print(f"\n1. Equal Weights (0.5 each)")
    equal_avg = np.mean([probs for probs in all_probs.values()], axis=0)
    preds_equal = (equal_avg > threshold).astype(float)
    
    results['equal_weights'] = {
        'accuracy': accuracy_score(y_true.flatten(), preds_equal.flatten()),
        'hamming_loss': hamming_loss(y_true, preds_equal),
        'f1_macro': f1_score(y_true, preds_equal, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, preds_equal, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, preds_equal, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, preds_equal, average='micro', zero_division=0),
        'mean_prob': float(equal_avg.mean()),
        'weights': {name: 0.5 for name in models_info.keys()}
    }
    
    print(f"  F1 (macro):  {results['equal_weights']['f1_macro']:.4f}")
    print(f"  Precision:   {results['equal_weights']['precision_macro']:.4f}")
    print(f"  Recall:      {results['equal_weights']['recall_macro']:.4f}")
    
    # Strategy 2: F1-weighted (performance-based)
    print(f"\n2. F1-Weighted (based on individual performance)")
    f1_scores = [info['f1'] for info in models_info.values()]
    f1_weights = np.array(f1_scores) / sum(f1_scores)
    
    weighted_avg = sum(probs * w for probs, w in zip(all_probs.values(), f1_weights))
    preds_weighted = (weighted_avg > threshold).astype(float)
    
    results['f1_weighted'] = {
        'accuracy': accuracy_score(y_true.flatten(), preds_weighted.flatten()),
        'hamming_loss': hamming_loss(y_true, preds_weighted),
        'f1_macro': f1_score(y_true, preds_weighted, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, preds_weighted, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, preds_weighted, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, preds_weighted, average='micro', zero_division=0),
        'mean_prob': float(weighted_avg.mean()),
        'weights': {name: float(w) for name, w in zip(models_info.keys(), f1_weights)}
    }
    
    print(f"  Weights: {results['f1_weighted']['weights']}")
    print(f"  F1 (macro):  {results['f1_weighted']['f1_macro']:.4f}")
    print(f"  Precision:   {results['f1_weighted']['precision_macro']:.4f}")
    print(f"  Recall:      {results['f1_weighted']['recall_macro']:.4f}")
    
    # Strategy 3: Manual config weights
    print(f"\n3. Config Weights (from config.yaml)")
    config_weights = [info['weight'] for info in models_info.values()]
    config_weights = np.array(config_weights) / sum(config_weights)
    
    config_avg = sum(probs * w for probs, w in zip(all_probs.values(), config_weights))
    preds_config = (config_avg > threshold).astype(float)
    
    results['config_weights'] = {
        'accuracy': accuracy_score(y_true.flatten(), preds_config.flatten()),
        'hamming_loss': hamming_loss(y_true, preds_config),
        'f1_macro': f1_score(y_true, preds_config, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, preds_config, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, preds_config, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, preds_config, average='micro', zero_division=0),
        'mean_prob': float(config_avg.mean()),
        'weights': {name: float(w) for name, w in zip(models_info.keys(), config_weights)}
    }
    
    print(f"  Weights: {results['config_weights']['weights']}")
    print(f"  F1 (macro):  {results['config_weights']['f1_macro']:.4f}")
    print(f"  Precision:   {results['config_weights']['precision_macro']:.4f}")
    print(f"  Recall:      {results['config_weights']['recall_macro']:.4f}")
    
    # Save results
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON - All Models")
    print("=" * 70)
    print(f"\nModel                        F1 (macro)   Precision   Recall")
    print("-" * 70)
    
    # Individual models
    for name, info in models_info.items():
        print(f"{name:27s}  {info['f1']:.4f}       {info['precision']:.4f}      {info['recall']:.4f}")
    
    print("-" * 70)
    
    # Ensemble strategies
    for strategy_name, strategy_results in results.items():
        display_name = strategy_name.replace('_', ' ').title()
        print(f"{display_name:27s}  {strategy_results['f1_macro']:.4f}       {strategy_results['precision_macro']:.4f}      {strategy_results['recall_macro']:.4f}")
    
    # Find best
    best_strategy = max(results.items(), key=lambda x: x[1]['f1_macro'])
    print("\n" + "=" * 70)
    print(f"🏆 Best Strategy: {best_strategy[0].replace('_', ' ').title()}")
    print(f"   F1 (macro) = {best_strategy[1]['f1_macro']:.4f}")
    print("=" * 70)
    
    print(f"\n✓ Ensemble evaluation complete!")
    print(f"  Results saved to: {output_dir}/ensemble_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
