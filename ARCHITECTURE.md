# Architecture & Technical Design

## Task Overview

**Objective:** Predict the percentage of land visible in satellite images (0-100%)

**Ground Truth Source:** Geographic coordinates (lat/lon) from NASA EPIC metadata
- Compute haversine distance from centroid to continent centers
- Estimate land percentage based on visible continents
- Returns value 0-100 representing estimated land coverage

**Model Output:** Single continuous value (regression)
- Input: 224×224 RGB satellite image
- Output: Land percentage (0-100%)

## Model Architecture

### ResNet18 Transfer Learning

```
ImageNet Pre-trained ResNet18
  ├── Layers 0-3: FROZEN (universal visual features)
  ├── Layer 4: FINE-TUNED (domain-specific features)
  └── Custom Head:
      ├── AdaptiveAvgPool2D → (512,)
      ├── FC(512→256) + ReLU + Dropout(0.3)
      └── FC(256→1) → Regression output (0-100%)
```

**Why This Works:**
- **11.3M total parameters, 5M trainable** prevents overfitting on 141 images
- **ImageNet features** capture Earth patterns (clouds, continents, water)
- **Layer 4 fine-tuning** adapts pre-trained features to geographic estimation
- **Single output** directly predicts land percentage

**Performance:**
- MAE: 5.48% | RMSE: 8.32% | R²: 0.8296

## Data Pipeline

1. **Download:** NASA EPIC API → images via `ml/data_preprocessing/api_client.py`
2. **Label:** Geographic coordinates → land percentage via `ml/data_preprocessing/geographic_labels.py`
3. **Split:** 70/15/15 train/val/test (stratified) via `ml/data_preprocessing/data_splitter.py`
4. **Preprocess:** Normalize per ImageNet via `ml/preprocessing.py` (shared transforms)
5. **Train:** ResNet18 regression with early stopping via `ml/train.py`

## Shared Code Architecture

### Transform Pipeline (`ml/preprocessing.py`)

**Single source of truth** for all image transformations:

```python
# Used by training (with augmentation)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Used by inference (no augmentation)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Benefits:**
- Guarantees training and inference use identical normalization
- Changes propagate everywhere automatically
- Prevents drift bugs

**Used by:**
- `ml/train.py` - Training pipeline
- `app/model_utils.py` - Inference wrapper

### API Client (`ml/api_client.py`)

**Single implementation** used by both ML and app:

```python
list_available_dates(collection='natural')        # List EPIC image dates
get_metadata_for_date(date_str, collection)      # Fetch image metadata
build_image_url(image_name, date_str, ...)       # Construct image URLs
download_images(date_str, collection, limit)     # Batch download
```

**Used by:**
- `ml/train.py` - Download training data
- `ml/data_preprocessing/pipeline.py` - Build catalogs
- `app/app.py` - `/api/random` endpoint

### Ground Truth Computation (`ml/data_preprocessing/geographic_labels.py`)

**Shared by training and inference:**

```python
compute_geographic_labels(lat, lon) → {'land_percentage': 0-100, 'ocean_percentage': 0-100}
```

Used in:
- Training - generating regression labels
- App - displaying ground truth for comparison

## Module Organization

```
ml/
├── preprocessing.py              ← Shared transforms (training + inference)
├── api_client.py                 ← Shared API client (all modules)
├── train.py                      ← Training script
├── models/
│   ├── __init__.py
│   └── land_ocean_regressor.py   ← Model architecture
└── data_preprocessing/
    ├── __init__.py
    ├── geographic_labels.py      ← Shared ground truth
    ├── image_preprocessor.py     ← Image loading
    ├── pipeline.py               ← Data pipeline
    ├── label_generator.py        ← Label computation
    ├── data_splitter.py          ← Train/val/test split
    └── catalog_builder.py        ← Catalog construction

app/
├── app.py                        ← Flask application
├── model_utils.py               ← Inference wrapper
└── templates/                   ← HTML templates
```

## Design Decisions

### Why Regression vs Classification?

Initially considered multi-label classification (predict 6 visible continents), but chose regression because:

| Aspect | Regression | Classification |
|--------|-----------|-----------------|
| **Output** | Single % value (0-100) | 6 binary labels |
| **Ground Truth** | Coordinates → continuous % | Coordinates → discrete continents |
| **Model Size** | 1 output neuron | 6 output neurons |
| **Metric** | MAE: 5.92% | F1: ~0.83 |
| **Interpretability** | Direct % useful for applications | Region identification |
| **Loss Function** | MSE (simpler) | BCE (class imbalance) |

**Conclusion:** Regression provides more actionable output (land/cloud coverage percentage)

### Why Transfer Learning?

ResNet18 pre-trained from ImageNet vs training from scratch:

| Aspect | Transfer | From Scratch |
|--------|----------|-------------|
| **Parameters** | 11.3M (5M trainable) | 11.3M (all trainable) |
| **Convergence** | 25 epochs | 50+ epochs |
| **Final MAE** | 5.92% | Would be ~12% |
| **Dataset** | 141 images | 141 images |
| **Overfitting Risk** | Low | High |

**ImageNet features** capture fundamental Earth patterns (water, clouds, continents), eliminating need to learn from scratch

---

See RESULTS.md for metrics and SETUP.md for usage.
