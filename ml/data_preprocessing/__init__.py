"""
Data Preprocessing Package
===========================

Complete pipeline for loading, labeling, and preprocessing EPIC imagery.
"""

from .api_client import (
    list_available_dates,
    get_metadata_for_date,
    download_images,
    build_image_url
)

from .label_generator import compute_catalog_geographic_labels

from .data_splitter import split_data

from .pipeline import load_preprocessed_epic_data

__all__ = [
    'list_available_dates',
    'get_metadata_for_date',
    'download_images',
    'build_image_url',
    'compute_catalog_geographic_labels',
    'split_data',
    'load_preprocessed_epic_data',
]
