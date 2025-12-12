# Setup & Usage

## Installation

```bash
# Clone repository
git clone https://github.com/stfny222/nasa-epic-classifier.git
cd nasa-epic-classifier

# Create environment (choose one)
# Option 1: Conda
conda create -n epic python=3.11
conda activate epic

# Option 2: Pip + venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision pillow scikit-learn numpy pandas flask requests
```

## Quick Start

```bash
# Train the regression model
python ml/train.py

# This will:
# - Download EPIC satellite images
# - Compute ground truth land percentages from coordinates
# - Train ResNet18 regression model
# - Save model to ml/models/land_ocean_regressor.pkl
# - Generate training plots
```

**Training Time:** 5-15 minutes on GPU

## Make Predictions

### Single Image

```python
from app.model_utils import LandOceanPredictor
from PIL import Image

# Initialize predictor (loads model automatically)
predictor = LandOceanPredictor()

# Load and predict
image = Image.open('satellite_image.png')
result = predictor.predict(image)

print(f"Land: {result['land_percentage']:.1f}%")
print(f"Ocean: {result['ocean_percentage']:.1f}%")
```

### With Ground Truth

```python
from ml.data_preprocessing.geographic_labels import compute_geographic_labels

# Get ground truth from coordinates
lat, lon = 0.0, -22.5
geo_labels = compute_geographic_labels(lat, lon)
print(f"Ground truth: {geo_labels['land_percentage']:.1f}%")

# Compare with model prediction
prediction = predictor.predict(image)
error = abs(prediction['land_percentage'] - geo_labels['land_percentage'])
print(f"Prediction error: {error:.1f}%")
```

## Flask Web App

Start the web interface for easy prediction and image exploration.

### Quick Start

```bash
# 1. Train model first (if not done)
python ml/train.py

# 2. Start app
python app/app.py

# 3. Open browser to http://localhost:5000
```

### Features

**Upload Image**
- Upload satellite image from your computer
- Get land percentage prediction

**Random EPIC Image**
- Download random image from NASA EPIC dataset
- Get prediction with ground truth for comparison

### API Endpoints

- `GET /` — Web UI
- `GET /api/status` — Server status
- `POST /api/predict` — Upload image, get prediction (form: `image` file, optional: `lat`, `lon`)
- `GET /api/random` — Download random EPIC image and predict
- `GET /api/metrics` — Model performance metrics

### Example Python Client

```python
import requests

# Upload image with optional ground truth
files = {'image': open('satellite.png', 'rb')}
data = {'lat': 0.0, 'lon': -22.5}
response = requests.post('http://localhost:5000/api/predict', files=files, data=data)
result = response.json()

print(f"Prediction: {result['land_percentage']:.1f}%")
print(f"Ground truth: {result['ground_truth_land_percentage']:.1f}%")

# Get random image
response = requests.get('http://localhost:5000/api/random')
data = response.json()
print(f"Image from {data['date']}")
print(f"Prediction: {data['land_percentage']:.1f}%")
print(f"Ground truth: {data['ground_truth_land_percentage']:.1f}%")
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in ml/train.py:
batch_size = 8  # was 16

# Or use CPU:
export CUDA_VISIBLE_DEVICES=""
python ml/train.py
```

### Model File Not Found
```bash
# Train the model first:
python ml/train.py
```

### API Requests Fail
```bash
# Check Flask app is running:
python app/app.py

# Test connectivity:
curl http://localhost:5000/api/status
```

---

See RESULTS.md for detailed metrics and ARCHITECTURE.md for technical details.
