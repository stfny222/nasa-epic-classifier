"""
Catalog Builder
===============

Build master catalog from downloaded images and metadata.
"""

import pathlib
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

from ml.api_client import get_metadata_for_date


def normalize_metadata(metadata: List[dict], date_str: str) -> pd.DataFrame:
    """Convert raw API metadata to normalized DataFrame."""
    # Handle empty metadata
    if not metadata:
        return pd.DataFrame(columns=["image", "date", "caption", "lat", "lon", "date_str", "hour"])
    
    rows = []
    for item in metadata:
        row = {
            "image": item.get("image"),
            "date": item.get("date"),
            "caption": item.get("caption"),
            "lat": (item.get("centroid_coordinates") or {}).get("lat"),
            "lon": (item.get("centroid_coordinates") or {}).get("lon"),
            "date_str": date_str,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    
    return df


def build_catalog(
    dates: List[str],
    collection: str = "natural",
    image_type: str = "png",
    cache_dir: pathlib.Path = None
) -> pd.DataFrame:
    """
    Build catalog from downloaded images.
    
    Parameters
    ----------
    dates : List[str]
        Dates to include in catalog
    collection : str
        EPIC collection name
    image_type : str
        Image type (png, jpg, thumbs)
    cache_dir : Path
        Base directory for data
        
    Returns
    -------
    pd.DataFrame
        Catalog with image paths and metadata
    """
    if cache_dir is None:
        # Get ml/ directory relative to this file
        ml_dir = pathlib.Path(__file__).parent.parent
        cache_dir = ml_dir / "data_epic"
    
    all_metadata = []
    
    for date_str in tqdm(dates, desc="Building catalog"):
        metadata = get_metadata_for_date(date_str, collection)
        df = normalize_metadata(metadata, date_str)
        
        # Add image paths
        image_dir = cache_dir / "images" / collection / date_str.replace("-", "/") / image_type
        ext = "png" if image_type == "png" else "jpg"
        df["image_path"] = df["image"].apply(lambda x: str(image_dir / f"{x}.{ext}"))
        
        # Filter to only existing images
        df["exists"] = df["image_path"].apply(lambda x: pathlib.Path(x).exists())
        df = df[df["exists"]].drop(columns=["exists"], errors="ignore")
        
        all_metadata.append(df)
    
    if not all_metadata:
        return pd.DataFrame()
    
    catalog = pd.concat(all_metadata, ignore_index=True)
    
    print(f"Built catalog: {len(catalog)} images across {len(dates)} dates")
    
    return catalog
