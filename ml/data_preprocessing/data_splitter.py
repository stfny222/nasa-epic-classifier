"""
Data Splitter
=============

Split data into train/val/test with stratification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def split_data(
    catalog_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify_column: str = "cloud_class"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split catalog into train/val/test sets with stratification.
    
    Parameters
    ----------
    catalog_df : pd.DataFrame
        Complete labeled catalog
    train_ratio : float
        Fraction for training (default 0.70)
    val_ratio : float
        Fraction for validation (default 0.15)
    test_ratio : float
        Fraction for testing (default 0.15)
    random_state : int
        Random seed for reproducibility
    stratify_column : str
        Column to use for stratification
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        catalog_df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=catalog_df[stratify_column] if stratify_column in catalog_df.columns else None
    )
    
    # Second split: val vs test
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df[stratify_column] if stratify_column in temp_df.columns else None
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(catalog_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(catalog_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(catalog_df)*100:.1f}%)")
    
    if stratify_column in catalog_df.columns:
        print(f"\nClass distribution ({stratify_column}):")
        print("Train:")
        print(train_df[stratify_column].value_counts(normalize=True))
        print("\nVal:")
        print(val_df[stratify_column].value_counts(normalize=True))
        print("\nTest:")
        print(test_df[stratify_column].value_counts(normalize=True))
    
    return train_df, val_df, test_df
