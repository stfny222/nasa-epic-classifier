"""
EPIC API Client
===============

Handle all API interactions with NASA's EPIC API.
"""

import pathlib
import requests
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

BASE_API = "https://epic.gsfc.nasa.gov/api"
BASE_ARCHIVE = "https://epic.gsfc.nasa.gov/archive"


def api_get(path_parts: List[str], params: Optional[Dict] = None, timeout: int = 60) -> dict:
    """GET JSON from EPIC API."""
    params = params or {}
    url = "/".join([BASE_API.strip("/")] + [str(x).strip("/") for x in path_parts])
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_available_dates(collection: str = "natural") -> List[str]:
    """Return sorted list of available dates with EPIC imagery."""
    try:
        dates = api_get([collection, "available"])
    except requests.HTTPError:
        dates = api_get([collection, "all"])
    
    if dates and isinstance(dates[0], dict) and "date" in dates[0]:
        dates = [d["date"] for d in dates]
    
    return sorted(dates)


def get_metadata_for_date(date_str: str, collection: str = "natural") -> List[Dict]:
    """Fetch metadata for all images on a specific date."""
    return api_get([collection, "date", date_str])


def build_image_url(
    image_name: str, 
    date_str: str, 
    collection: str = "natural", 
    image_type: str = "png"
) -> str:
    """Build archive URL for an EPIC image."""
    y, m, d = date_str.split("-")
    ext = "png" if image_type == "png" else "jpg"
    return f"{BASE_ARCHIVE}/{collection}/{y}/{m}/{d}/{image_type}/{image_name}.{ext}"


def download_images(
    date_str: str,
    collection: str = "natural",
    image_type: str = "png",
    limit: Optional[int] = None,
    output_dir: Optional[pathlib.Path] = None,
    overwrite: bool = False
) -> Tuple[List[pathlib.Path], List[Dict]]:
    """
    Download EPIC images for a specific date.
    
    Returns
    -------
    tuple
        (list of downloaded paths, metadata list)
    """
    metadata = get_metadata_for_date(date_str, collection)
    
    if output_dir is None:
        # Get ml/ directory relative to this file
        ml_dir = pathlib.Path(__file__).parent.parent
        output_dir = (
            ml_dir / "data_epic" / "images" / collection 
            / date_str.replace("-", "/") / image_type
        )
    
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    desc = f"Downloading {collection}/{date_str}/{image_type}"
    
    for i, item in enumerate(tqdm(metadata, desc=desc)):
        if limit is not None and i >= limit:
            break
        
        image_name = item["image"]
        url = build_image_url(image_name, date_str, collection, image_type)
        ext = "png" if image_type == "png" else "jpg"
        dest = output_dir / f"{image_name}.{ext}"
        
        if dest.exists() and not overwrite:
            paths.append(dest)
            continue
        
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            paths.append(dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            if dest.exists():
                dest.unlink()
    
    return paths, metadata
