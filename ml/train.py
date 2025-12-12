"""
Land-Ocean Regressor Training Script

Trains a ResNet18 model to predict the percentage of land visible in satellite images.
Saves trained model and metrics to ml/models/ for deployment.

Usage:
    python ml/train.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 16,
    'learning_rate': 0.0005,
    'num_epochs': 30,
    'early_stopping_patience': 5,
    'weight_decay': 1e-4,
    'random_state': 42,
}

# Import transforms from shared module (ensures consistency with inference)
from ml.preprocessing import TRAIN_TRANSFORM, VAL_TRANSFORM


class LandOceanDataset(Dataset):
    """PyTorch Dataset for land percentage regression."""
    
    def __init__(self, image_paths, land_percentages, transform=None):
        self.image_paths = image_paths
        self.land_percentages = land_percentages
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        land_pct = self.land_percentages[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(land_pct, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(land_pct, dtype=torch.float32)


class LandPercentageRegressor(nn.Module):
    """ResNet18 adapted for land percentage regression (0-100)."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.resnet(x)


def load_data(dates=None):
    """Load and prepare EPIC satellite data."""
    print("Loading EPIC satellite data...")
    
    from ml.data_preprocessing.pipeline import load_preprocessed_epic_data
    from ml.api_client import list_available_dates
    from ml.data_preprocessing.geographic_labels import compute_geographic_labels
    
    # Get available dates
    if dates is None:
        all_dates = list_available_dates(collection="natural")
        dates = all_dates[-20:] if len(all_dates) >= 20 else all_dates
    
    print(f"Using {len(dates)} dates: {dates[0]} to {dates[-1]}")
    
    # Load preprocessed data
    X_train_arr, X_val_arr, X_test_arr, y_train, y_val, y_test, train_df, val_df, test_df = load_preprocessed_epic_data(
        dates=dates,
        force_recompute=False
    )
    
    # Combine all dataframes
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Ensure land_percentage exists
    if 'land_percentage' not in df.columns:
        print("Computing land_percentage from coordinates...")
        land_pcts = []
        for idx, row in df.iterrows():
            try:
                geo_labels = compute_geographic_labels(row['lat'], row['lon'])
                land_pcts.append(geo_labels.get('land_percentage', 50))
            except:
                land_pcts.append(50)
        df['land_percentage'] = land_pcts
    
    df['land_percentage'] = df['land_percentage'].astype(float)
    
    # Filter clean data
    df_clean = df[df['image_path'].notna() & df['land_percentage'].notna()].copy()
    
    print(f"\nDataset Statistics:")
    print(f"  Total rows: {len(df_clean)}")
    print(f"  Land % - Mean: {df_clean['land_percentage'].mean():.1f}%, "
          f"Std: {df_clean['land_percentage'].std():.1f}%")
    print(f"  Land % - Range: [{df_clean['land_percentage'].min():.1f}%, "
          f"{df_clean['land_percentage'].max():.1f}%]")
    
    return df_clean


def prepare_dataloaders(df_clean):
    """Create train/val/test dataloaders."""
    X = df_clean['image_path'].values
    y = df_clean['land_percentage'].values
    
    # Split: 80% train, 20% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG['random_state']
    )
    
    # Split: temp 50-50 for val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=CONFIG['random_state']
    )
    
    print(f"\nData Split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create datasets and dataloaders
    train_dataset = LandOceanDataset(X_train, y_train, transform=TRAIN_TRANSFORM)
    val_dataset = LandOceanDataset(X_val, y_val, transform=VAL_TRANSFORM)
    test_dataset = LandOceanDataset(X_test, y_test, transform=VAL_TRANSFORM)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, (X_test, y_test)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(device), targets.to(device).unsqueeze(1)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mae = torch.mean(torch.abs(outputs - targets)).item()
        total_mae += mae
    
    return total_loss / len(loader), total_mae / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating"):
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - targets)).item()
            total_mae += mae
    
    return total_loss / len(loader), total_mae / len(loader)


def evaluate_test_set(model, loader, device):
    """Evaluate on test set and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.clamp(outputs, 0, 100)
            
            all_preds.extend(outputs.cpu().squeeze().numpy())
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_targets)


def train_model(train_loader, val_loader):
    """Train the model with early stopping."""
    device = CONFIG['device']
    
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    
    # Create model
    model = LandPercentageRegressor(pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Use project-relative path for checkpoint
    checkpoint_path = Path(__file__).parent / 'models' / '.best_model_checkpoint.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Loss: MSE")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Early Stopping: {CONFIG['early_stopping_patience']} epochs")
    print(f"\nStarting training...\n")
    
    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, MAE={val_mae:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
        print()
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print("✓ Training complete!")
    
    return model, history


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss (MSE)', marker='o', markersize=4)
    axes[0].plot(history['val_loss'], label='Val Loss (MSE)', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_mae'], label='Train MAE', marker='o', markersize=4)
    axes[1].plot(history['val_mae'], label='Val MAE', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved training history to {output_dir / 'training_history.png'}")
    plt.close()


def plot_test_results(test_preds, test_targets, mae, rmse, r2, output_dir):
    """Plot test set evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Predictions vs Ground Truth
    axes[0, 0].scatter(test_targets, test_preds, alpha=0.6, s=100)
    axes[0, 0].plot([0, 100], [0, 100], 'r--', label='Perfect Prediction', linewidth=2)
    axes[0, 0].set_xlabel('Ground Truth Land %')
    axes[0, 0].set_ylabel('Predicted Land %')
    axes[0, 0].set_title('Predictions vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 100])
    axes[0, 0].set_ylim([0, 100])
    
    # Residuals plot
    residuals = test_preds - test_targets
    axes[0, 1].scatter(test_targets, residuals, alpha=0.6, s=100)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Ground Truth Land %')
    axes[0, 1].set_ylabel('Prediction Error (%)')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 0].hist(np.abs(residuals), bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Absolute Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Metrics bar chart
    metrics_names = ['MAE', 'RMSE', 'R²']
    metrics_values = [mae / 100, rmse / 100, r2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Regression Metrics')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, metric_name in zip(bars, metrics_names):
        if metric_name == 'MAE':
            label = f'{mae:.1f}%'
        elif metric_name == 'RMSE':
            label = f'{rmse:.1f}%'
        else:
            label = f'{r2:.3f}'
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       label, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_results.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved test results to {output_dir / 'test_results.png'}")
    plt.close()


def save_model(model, metrics, output_dir):
    """Save trained model and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_package = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'architecture': 'ResNet18',
            'task': 'regression',
            'output_range': [0, 100],
            'input_size': 224,
            'pretrained': True
        },
        'metrics': metrics,
        'training_config': {
            'batch_size': CONFIG['batch_size'],
            'learning_rate': CONFIG['learning_rate'],
            'optimizer': 'Adam',
            'loss_function': 'MSE'
        }
    }
    
    model_file = output_dir / 'land_ocean_regressor.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✓ Model saved to: {model_file}")
    print(f"  File size: {model_file.stat().st_size / 1e6:.1f} MB")
    
    metrics_file = output_dir / 'regression_metrics.json'
    metrics_dict = {
        'mae': float(metrics['test_mae']),
        'rmse': float(metrics['test_rmse']),
        'mse': float(metrics['test_mse']),
        'r2_score': float(metrics['test_r2']),
        'output_range': [0, 100]
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_file}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("EPIC Land-Ocean Regressor - Training Script")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(__file__).parent / 'models'
    
    # Load data
    df_clean = load_data()
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader, (X_test, y_test) = prepare_dataloaders(df_clean)
    
    # Train model
    model, history = train_model(train_loader, val_loader)
    
    # Evaluate on test set
    device = CONFIG['device']
    test_preds, test_targets = evaluate_test_set(model, test_loader, device)
    
    # Calculate metrics
    mse = mean_squared_error(test_targets, test_preds)
    mae = mean_absolute_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, test_preds)
    
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (REGRESSION)")
    print("=" * 70)
    print(f"Mean Absolute Error (MAE):     {mae:.2f}%")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}%")
    print(f"Mean Squared Error (MSE):      {mse:.4f}")
    print(f"R² Score:                      {r2:.4f}")
    
    errors = np.abs(test_preds - test_targets)
    print(f"\nPrediction Error Statistics:")
    print(f"  Mean error: {errors.mean():.2f}%")
    print(f"  Median error: {np.median(errors):.2f}%")
    print(f"  Max error: {errors.max():.2f}%")
    print(f"  Min error: {errors.min():.2f}%")
    
    # Save model and metrics
    metrics = {
        'test_mae': mae,
        'test_rmse': rmse,
        'test_mse': mse,
        'test_r2': r2
    }
    save_model(model, metrics, output_dir)
    
    # Generate plots
    plot_training_history(history, output_dir)
    plot_test_results(test_preds, test_targets, mae, rmse, r2, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {output_dir}")
    print(f"  - land_ocean_regressor.pkl")
    print(f"  - regression_metrics.json")
    print(f"  - training_history.png")
    print(f"  - test_results.png")


if __name__ == '__main__':
    main()
