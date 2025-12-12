"""
Label Generator
===============

Compute ground truth labels for regression from image coordinates.
Uses haversine distance calculation to determine land percentage visible
from DSCOVR's Earth-Sun L1 viewing geometry.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .geographic_labels import compute_geographic_labels


def compute_catalog_geographic_labels(
    catalog_df: pd.DataFrame,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Compute ground truth land percentage labels for each image.
    
    Uses haversine distance from image centroid to continent centers
    to estimate land percentage visible from DSCOVR viewing geometry
    (~60° hemisphere visible from Earth-Sun L1 point).
    
    Parameters
    ----------
    catalog_df : pd.DataFrame
        Catalog with 'lat', 'lon' columns (from EPIC API metadata)
    overwrite : bool
        Whether to recompute existing labels
        
    Returns
    -------
    pd.DataFrame
        Catalog with added regression ground truth columns:
        - land_percentage: float (0-100) - primary target variable
        - ocean_percentage: float (0-100)
    """
    # Check if labels already exist
    if not overwrite and "land_percentage" in catalog_df.columns:
        print("Geographic labels already computed. Set overwrite=True to recompute.")
        return catalog_df
    
    # Columns to compute (ground truth for regression)
    label_keys = ['land_percentage', 'ocean_percentage']
    labels = {key: [] for key in label_keys}
    
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Computing geographic labels"):
        lat = row.get("lat")
        lon = row.get("lon")
        
        try:
            if lat is not None and lon is not None:
                geo_labels = compute_geographic_labels(lat, lon)
                for key in label_keys:
                    labels[key].append(geo_labels[key])
            else:
                # Missing coordinates - use NaN
                for key in label_keys:
                    labels[key].append(np.nan)
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            for key in label_keys:
                labels[key].append(np.nan)
    
    result_df = catalog_df.copy()
    for col, values in labels.items():
        result_df[col] = values
    
    print(f"\n✓ Geographic labels computed for {len(result_df)} images")
    print(f"\nGround truth statistics:")
    print(f"  Land percentage - Mean: {result_df['land_percentage'].mean():.1f}%, Std: {result_df['land_percentage'].std():.1f}%")
    print(f"  Range: [{result_df['land_percentage'].min():.1f}%, {result_df['land_percentage'].max():.1f}%]")
    
    return result_df
