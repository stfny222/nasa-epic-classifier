"""
Geographic Label Generator
===========================

Compute ground truth land percentage from satellite image coordinates.
Uses haversine distance from centroid to continent centers.
"""

import numpy as np
from typing import Dict


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
    
    DSCOVR at L1 point sees approximately half Earth's surface (~60° radius).
    
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
        distance = haversine_distance(lat, lon, region_lat, region_lon)
        # Add buffer for large continents
        radius = (view_radius + 10.0) if continent_name in ['asia', 'africa', 'north_america'] else view_radius
        visible[continent_name] = distance < radius
    
    return visible


def compute_land_ocean_percentage(lat: float, lon: float) -> float:
    """
    Estimate land percentage in view based on centroid coordinates.
    
    Uses visible continents and their approximate land contributions.
    This is ground truth for regression training.
    
    Parameters
    ----------
    lat : float
        Latitude of centroid
    lon : float
        Longitude of centroid
        
    Returns
    -------
    float
        Land percentage (0-100)
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
    
    return min(100.0, land_contribution)


def compute_geographic_labels(lat: float, lon: float) -> Dict:
    """
    Compute ground truth labels for training regression model.
    
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
        - land_percentage: float (0-100) - ground truth for regression
    """
    land_pct = compute_land_ocean_percentage(lat, lon)
    
    return {
        'land_percentage': land_pct,
        'ocean_percentage': 100.0 - land_pct,
    }
