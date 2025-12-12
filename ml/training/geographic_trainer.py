"""
Geographic Trainer
==================

Training loop for multi-label geographic classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from .losses import GeographicBCELoss


class GeographicDataset(Dataset):
    """PyTorch Dataset for geographic labels."""
    
    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : np.ndarray
            Images (N, H, W, C) in HWC format
        y : np.ndarray
            Binary labels (N, num_labels)
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert HWC to CHW for PyTorch
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1)
        label = torch.from_numpy(self.y[idx])
        return image, label


def train_geographic_model(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    early_stopping_patience=10,
    output_dir=None,
    verbose=1
):
    """
    Train multi-label geographic classification model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    X_train, X_val : np.ndarray
        Training and validation images
    y_train, y_val : np.ndarray
        Training and validation labels (binary, shape: [N, num_labels])
    epochs : int
        Maximum epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    early_stopping_patience : int
        Patience for early stopping
    output_dir : Path
        Directory to save checkpoints
    verbose : int
        Verbosity level
        
    Returns
    -------
    dict
        Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create datasets
    train_dataset = GeographicDataset(X_train, y_train)
    val_dataset = GeographicDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Compute pos_weight to handle class imbalance
    # pos_weight = (num_negative / num_positive) per label
    pos_counts = y_train.sum(axis=0)  # shape: (num_labels,)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.from_numpy(neg_counts / (pos_counts + 1e-5)).float().to(device)
    
    print(f"\nClass balance (pos_weight per label):")
    for i, w in enumerate(pos_weight):
        print(f"  Label {i}: {w:.2f}")
    
    # Setup loss and optimizer
    criterion = GeographicBCELoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("=" * 70)
    print("Training Multi-Label Geographic CNN (PyTorch)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)
    print()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss, loss_dict = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy (threshold at 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
            
            pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if output_dir:
                checkpoint_path = Path(output_dir) / "model_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return history


def evaluate_geographic_model(model, X_test, y_test, batch_size=16, label_names=None, threshold=0.35):
    """
    Evaluate model on test set.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test : np.ndarray
        Test images
    y_test : np.ndarray
        Test labels (binary)
    batch_size : int
        Batch size
    label_names : list, optional
        Names of the labels for display
    threshold : float
        Decision threshold for binary predictions (default: 0.35)
        Note: 0.35 works better than 0.5 due to model's conservative predictions
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
    
    if label_names is None:
        label_names = [
            'North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania'
        ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_dataset = GeographicDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    
    # Overall metrics
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'mean_predicted_prob': float(y_prob.mean()),
    }
    
    # Per-label metrics
    per_label_metrics = {}
    for i, name in enumerate(label_names):
        per_label_metrics[name] = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'support': int(y_true[:, i].sum()),
        }
    
    metrics['per_label'] = per_label_metrics
    
    # Print results
    print("\n" + "=" * 70)
    print("Test Set Evaluation - Geographic Labels")
    print("=" * 70)
    print(f"\nDecision threshold: {threshold:.2f}")
    print(f"Mean predicted probability: {metrics['mean_predicted_prob']:.4f}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy (exact match): {metrics['accuracy']:.4f}")
    print(f"  Hamming Loss:           {metrics['hamming_loss']:.4f}")
    print(f"  Precision (macro):      {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):         {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):             {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):             {metrics['f1_micro']:.4f}")
    
    print(f"\nPer-Label Performance:")
    print(f"{'Label':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 70)
    for name, scores in per_label_metrics.items():
        print(f"{name:<20s} {scores['precision']:>10.4f} {scores['recall']:>10.4f} "
              f"{scores['f1']:>10.4f} {scores['support']:>10d}")
    
    return metrics
