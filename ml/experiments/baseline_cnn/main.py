"""
Baseline CNN Experiment
========================

Train and evaluate a simple CNN for multi-label classification of EPIC imagery.
"""

import sys
import os
import yaml
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_preprocessing import load_preprocessed_epic_data
from data_preprocessing.image_preprocessor import GEOGRAPHIC_LABEL_COLUMNS
from models.geographic_cnn import build_geographic_cnn
from training.geographic_trainer import train_geographic_model, evaluate_geographic_model


def main():
    """Main experiment function."""
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("EPIC BASELINE CNN EXPERIMENT")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(json.dumps(config, indent=2))
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("\n[1/3] Loading and preprocessing data...")
    
    # Get dates from config
    dates = config['data'].get('dates', None)
    if dates:
        print(f"Using {len(dates)} dates from config (seasonal sampling):")
        for d in dates:
            print(f"  - {d}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = \
        load_preprocessed_epic_data(
            dates=dates,
            collection=config['data']['collection'],
            image_type=config['data']['image_type'],
            target_size=tuple(config['data']['target_size']),
            limit_per_date=config['data'].get('limit_per_date'),
            force_recompute=False
        )
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {len(y_train['geographic'])} samples")
    print(f"  Val:   {len(y_val['geographic'])} samples")
    print(f"  Test:  {len(y_test['geographic'])} samples")
    
    # Geographic label statistics
    print(f"\n  Geographic labels (% positive per region):")
    for i, label_name in enumerate(GEOGRAPHIC_LABEL_COLUMNS):
        pct = (y_train['geographic'][:, i].sum() / len(y_train['geographic'])) * 100
        region = label_name.replace('visible_', '').replace('_', ' ').title()
        print(f"    {region:<20s}: {pct:5.1f}%")
    
    # Build model
    print("\n[2/3] Building model...")
    num_labels = len(GEOGRAPHIC_LABEL_COLUMNS)  # 6 binary labels
    model = build_geographic_cnn(
        input_shape=tuple(config['model']['input_shape']),
        num_labels=num_labels,
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Save model architecture
    with open(output_dir / "model_architecture.txt", "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[3/3] Training model...")
    history = train_geographic_model(
        model,
        X_train, y_train['geographic'],
        X_val, y_val['geographic'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        output_dir=output_dir,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    label_names = [label.replace('visible_', '').replace('_', ' ').title() 
                   for label in GEOGRAPHIC_LABEL_COLUMNS]
    
    results = evaluate_geographic_model(
        model, 
        X_test, 
        y_test['geographic'], 
        batch_size=config['training']['batch_size'],
        label_names=label_names
    )
    
    # Save results
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f"\n✓ Experiment complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  - model_best.pt: Best model checkpoint")
    print(f"  - training_history.csv: Training curves")
    print(f"  - test_metrics.json: Test set metrics")
    
    return results


if __name__ == "__main__":
    results = main()
