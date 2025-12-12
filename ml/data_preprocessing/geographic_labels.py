"""
Geographic Label Generator
===========================

Generate labels based on geographic position and visible regions.
Uses centroid coordinates to determine which continents/oceans are visible.
"""

import numpy as np
from typing import Dict, Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth (in degrees).
    
    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of first point (degrees)
    lat2, lon2 : float
        Coordinates of second point (degrees)
        
    Returns
    -------
    float
        Angular distance in degrees
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Convert back to degrees
    return np.degrees(c)


def get_visible_continents(lat: float, lon: float, view_radius: float = 60.0) -> Dict[str, bool]:
    """
    Determine which continents are visible from a given viewpoint.
    
    DSCOVR at L1 point (~1.5 million km from Earth) sees approximately half
    the Earth's surface (~60° radius from centroid).
    
    Parameters
    ----------
    lat : float
        Latitude of image centroid (-90 to 90)
    lon : float
        Longitude of image centroid (-180 to 180)
    view_radius : float
        Angular radius of visible area (degrees), default 60°
        
    Returns
    -------
    dict
        Boolean flags for each continent:
        - north_america
        - south_america
        - europe
        - africa
        - asia
        - oceania
    """
    # Define approximate centers of major continents
    # Coordinates chosen to represent the "heart" of each region
    continents = {
        'north_america': (45.0, -100.0),    # Central USA/Canada
        'south_america': (-15.0, -60.0),    # Central Brazil
        'europe': (50.0, 10.0),              # Central Europe
        'africa': (0.0, 20.0),               # Central Africa
        'asia': (30.0, 100.0),               # Central/East Asia
        'oceania': (-25.0, 135.0),           # Australia
    }
    
    visible = {}
    
    for continent_name, (region_lat, region_lon) in continents.items():
        # Calculate angular distance from centroid to continent center
        distance = haversine_distance(lat, lon, region_lat, region_lon)
        
        # Continent is visible if within view radius
        # Use slightly larger radius for large continents to catch edges
        if continent_name in ['asia', 'africa', 'north_america']:
            # Large continents
            visible[continent_name] = distance < (view_radius + 10.0)
        else:
            # Standard radius for other regions
            visible[continent_name] = distance < view_radius
    
    return visible


def compute_hemisphere_label(lat: float, lon: float) -> str:
    """
    Determine primary hemisphere view.
    
    Parameters
    ----------
    lat : float
        Latitude of centroid
    lon : float
        Longitude of centroid
        
    Returns
    -------
    str
        One of: 'Americas', 'Europe_Africa', 'Asia_Pacific', 'Polar'
    """
    if abs(lat) > 70:
        return 'Polar'
    elif -130 < lon < -30:
        return 'Americas'
    elif -30 <= lon < 60:
        return 'Europe_Africa'
    else:
        return 'Asia_Pacific'


def compute_land_ocean_percentage(lat: float, lon: float) -> Tuple[float, float]:
    """
    Estimate land vs ocean percentage in view based on centroid.
    
    This is a rough approximation based on known Earth geography.
    More accurate than pixel-based methods for cloudy images.
    
    Parameters
    ----------
    lat : float
        Latitude of centroid
    lon : float
        Longitude of centroid
        
    Returns
    -------
    tuple
        (land_percentage, ocean_percentage) both 0-100
    """
    visible = get_visible_continents(lat, lon)
    
    # Rough estimates of land coverage when each continent is visible
    # Based on actual Earth geography
    land_contribution = 0.0
    
    # Continents contribute land
    if visible['north_america']:
        land_contribution += 25.0
    if visible['south_america']:
        land_contribution += 20.0
    if visible['europe']:
        land_contribution += 15.0
    if visible['africa']:
        land_contribution += 30.0
    if visible['asia']:
        land_contribution += 35.0
    if visible['oceania']:
        land_contribution += 10.0
    
    # Cap at 100% and estimate ocean as remainder
    land_pct = min(100.0, land_contribution)
    ocean_pct = 100.0 - land_pct
    
    return land_pct, ocean_pct


def compute_geographic_labels(lat: float, lon: float) -> Dict:
    """
    Compute geographic labels for an image based on coordinates.
    
    Computes continent visibility (6 binary labels) and metadata (4 features)
    using deterministic haversine distance from image centroid to continent centers.
    
    Parameters
    ----------
    lat : float
        Latitude of image centroid
    lon : float
        Longitude of image centroid
        
    Returns
    -------
    dict
        Dictionary containing:
        - visible_* : bool flags for each continent (6 labels)
        - hemisphere : str (categorical: Americas/Europe_Africa/Asia_Pacific/Polar)
        - land_percentage : float (0-100)
        - ocean_percentage : float (0-100)
        - land_ocean_class : str (Mostly Land / Balanced / Mostly Ocean)
    """
    visible = get_visible_continents(lat, lon)
    hemisphere = compute_hemisphere_label(lat, lon)
    land_pct, ocean_pct = compute_land_ocean_percentage(lat, lon)
    
    # Classify land/ocean dominance
    if land_pct > 60:
        land_ocean_class = "Mostly Land"
    elif land_pct < 30:
        land_ocean_class = "Mostly Ocean"
    else:
        land_ocean_class = "Balanced"
    
    # Combine all labels
    labels = {
        # Binary continent visibility (6 labels)
        'visible_north_america': visible['north_america'],
        'visible_south_america': visible['south_america'],
        'visible_europe': visible['europe'],
        'visible_africa': visible['africa'],
        'visible_asia': visible['asia'],
        'visible_oceania': visible['oceania'],
        
        # Categorical/continuous metadata
        'hemisphere': hemisphere,
        'land_percentage': land_pct,
        'ocean_percentage': ocean_pct,
        'land_ocean_class': land_ocean_class,
    }
    
    return labels


if __name__ == "__main__":
    # Test cases
    test_cases = [
        (0.0, -100.0, "Americas view"),
        (20.0, 0.0, "Europe/Africa view"),
        (0.0, 120.0, "Asia/Pacific view"),
        (45.0, -75.0, "North America centered"),
        (-20.0, -50.0, "South America/Atlantic"),
    ]
    
    print("Geographic Label Generator - Test Cases")
    print("=" * 70)
    
    for lat, lon, description in test_cases:
        print(f"\n{description} (lat={lat}, lon={lon})")
        print("-" * 70)
        labels = compute_geographic_labels(lat, lon)
        
        # Print visible continents
        visible_continents = [k.replace('visible_', '') for k, v in labels.items() 
                             if k.startswith('visible_') and v]
        print(f"Visible continents: {', '.join(visible_continents) if visible_continents else 'None'}")
        print(f"Hemisphere: {labels['hemisphere']}")
        print(f"Land: {labels['land_percentage']:.1f}%, Ocean: {labels['ocean_percentage']:.1f}%")
        print(f"Classification: {labels['land_ocean_class']}")
