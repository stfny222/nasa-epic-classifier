"""
Complete Preprocessing Pipeline
================================

High-level orchestration with caching.
"""

import pathlib
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from ml.api_client import download_images
from .catalog_builder import build_catalog
from .label_generator import compute_catalog_geographic_labels
from .data_splitter import split_data
from .image_preprocessor import load_and_preprocess_image


def load_preprocessed_epic_data(
    dates: Optional[List[str]] = None,
    collection: str = "natural",
    image_type: str = "png",
    target_size: Tuple[int, int] = (224, 224),
    limit_per_date: Optional[int] = None,
    cache_dir: pathlib.Path = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: download, label, split, and preprocess EPIC data.
    
    Parameters
    ----------
    dates : List[str], optional
        Dates to load. If None, uses a default recent date.
    collection : str
        EPIC collection ('natural', 'enhanced')
    image_type : str
        Image type ('png', 'jpg', 'thumbs')
    target_size : tuple
        (height, width) for resizing images
    limit_per_date : int, optional
        Limit images per date (for testing)
    cache_dir : Path, optional
        Cache directory (default: processed_data/)
    force_recompute : bool
        Whether to skip cache and recompute everything
        
    Returns
    -------
    tuple
        X_train, X_val, X_test : np.ndarray - Image arrays
        y_train, y_val, y_test : dict - Label dictionaries
        train_df, val_df, test_df : pd.DataFrame - Catalog dataframes
    """
    if cache_dir is None:
        # Get ml/ directory relative to this file
        ml_dir = pathlib.Path(__file__).parent.parent
        cache_dir = ml_dir / "processed_data"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits_cache = cache_dir / "splits"
    splits_cache.mkdir(exist_ok=True)
    
    # Cache paths - include number of dates in key to avoid loading wrong cached data
    # When dates=None, it will be resolved to 1 date below, so we use that in the key
    num_dates = len(dates) if dates else 1
    limit_str = f"_limit{limit_per_date}" if limit_per_date else ""
    cache_file = splits_cache / f"preprocessed_{collection}_{image_type}_{target_size[0]}_dates{num_dates}{limit_str}.pkl"
    
    # Try loading from cache
    if not force_recompute and cache_file.exists():
        print(f"Loading preprocessed data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"✓ Loaded from cache")
        return cached
    
    print("=" * 70)
    print("EPIC Data Preprocessing Pipeline")
    print("=" * 70)
    
    # Step 1: Download images
    print("\n[1/5] Downloading images...")
    if dates is None:
        from ml.api_client import list_available_dates
        all_dates = list_available_dates(collection)
        # Use last available date by default
        dates = [all_dates[-1]]
        print(f"Using default date: {dates[0]}")
    
    for date in dates:
        download_images(
            date, 
            collection=collection,
            image_type=image_type,
            limit=limit_per_date,
            overwrite=False
        )
    
    # Step 2: Build catalog
    print("\n[2/5] Building catalog...")
    ml_dir = pathlib.Path(__file__).parent.parent
    catalog = build_catalog(
        dates=dates,
        collection=collection,
        image_type=image_type,
        cache_dir=ml_dir / "data_epic"
    )
    
    if len(catalog) == 0:
        raise ValueError("No images found in catalog!")
    
    # Step 3: Compute labels
    print("\n[3/5] Computing geographic labels...")
    catalog = compute_catalog_geographic_labels(catalog, overwrite=False)
    
    # Step 4: Split data
    print("\n[4/5] Splitting data...")
    train_df, val_df, test_df = split_data(catalog, stratify_column="cloud_class")
    
    # Step 5: Load and preprocess images
    print("\n[5/5] Loading and preprocessing images...")
    
    def load_split(df: pd.DataFrame):
        from PIL import Image
        import logging
        
        X = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                img_array = load_and_preprocess_image(
                    row["image_path"],
                    target_size=target_size,
                    normalize=True
                )
                X.append(img_array)
            except Exception as e:
                skipped_count += 1
                logging.warning(f"Failed to load image {row['image_path']}: {e}")
                continue
        
        if skipped_count > 0:
            print(f"  ⚠ Skipped {skipped_count} images due to errors")
        
        X = np.array(X, dtype=np.float32)
        y = {}
        return X, y
    
    print("  Loading train set...")
    X_train, y_train = load_split(train_df)
    
    print("  Loading val set...")
    X_val, y_val = load_split(val_df)
    
    print("  Loading test set...")
    X_test, y_test = load_split(test_df)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Cache results
    print(f"\nSaving to cache: {cache_file}")
    result = (X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    
    return result
