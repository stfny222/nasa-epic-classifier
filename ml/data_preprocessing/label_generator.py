"""
Label Generator
===============

Compute geographic continent labels from image coordinates.
Uses deterministic haversine distance calculation to determine which continents
are visible from DSCOVR's Earth-Sun L1 viewing geometry.
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
    Compute geographic continent labels for each image in catalog.
    
    Uses deterministic haversine distance from image centroid to continent centers
    to determine which continents are visible from the DSCOVR viewing geometry
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
        Catalog with added continent visibility columns:
        - visible_north_america, visible_south_america, visible_europe
        - visible_africa, visible_asia, visible_oceania
        - hemisphere, land_percentage, ocean_percentage, land_ocean_class
    """
    # Check if labels already exist
    if not overwrite and "visible_north_america" in catalog_df.columns:
        print("Geographic labels already computed. Set overwrite=True to recompute.")
        return catalog_df
    
    # Continent label column names (only what the models use)
    continent_keys = [
        'visible_north_america', 'visible_south_america', 'visible_europe',
        'visible_africa', 'visible_asia', 'visible_oceania'
    ]
    
    # Metadata columns to include
    metadata_keys = ['hemisphere', 'land_percentage', 'ocean_percentage', 'land_ocean_class']
    
    all_keys = continent_keys + metadata_keys
    labels = {key: [] for key in all_keys}
    
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Computing geographic labels"):
        lat = row.get("lat")
        lon = row.get("lon")
        
        try:
            if lat is not None and lon is not None:
                geo_labels = compute_geographic_labels(lat, lon)
                # Only extract continent and metadata labels
                for key in all_keys:
                    labels[key].append(geo_labels[key])
            else:
                # Missing coordinates
                for key in continent_keys:
                    labels[key].append(False)
                for key in metadata_keys:
                    if key == 'hemisphere':
                        labels[key].append('Unknown')
                    elif key in ['land_percentage', 'ocean_percentage']:
                        labels[key].append(np.nan)
                    elif key == 'land_ocean_class':
                        labels[key].append('Unknown')
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            for key in continent_keys:
                labels[key].append(False)
            for key in metadata_keys:
                if key == 'hemisphere':
                    labels[key].append('Unknown')
                elif key in ['land_percentage', 'ocean_percentage']:
                    labels[key].append(np.nan)
                elif key == 'land_ocean_class':
                    labels[key].append('Unknown')
    
    result_df = catalog_df.copy()
    for col, values in labels.items():
        result_df[col] = values
    
    print(f"\n✓ Geographic labels computed for {len(result_df)} images")
    print(f"\nGeographic label distribution:")
    print(f"  Hemisphere:")
    print(f"    {result_df['hemisphere'].value_counts().to_dict()}")
    print(f"\n  Land/Ocean classification:")
    print(f"    {result_df['land_ocean_class'].value_counts().to_dict()}")
    print(f"\n  Continent visibility (% of images):")
    for col in continent_keys:
        pct = result_df[col].sum() / len(result_df) * 100
        region = col.replace('visible_', '').replace('_', ' ').title()
        print(f"    {region:20s}: {pct:5.1f}%")
    
    return result_df
