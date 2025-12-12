# NASA EPIC Classifier

Predict land/ocean percentage in satellite images using deep learning.

## Quick Results

- **Model:** ResNet18 Transfer Learning (Regression)
- **Dataset:** 253 NASA EPIC satellite images  
- **Task:** Predict percentage of land visible (0-100%)
- **MAE:** 5.48% | **R²:** 0.8296

## Quick Start

```bash
# Setup
conda create -n epic python=3.11
conda activate epic
pip install torch torchvision pillow flask requests numpy pandas scikit-learn

# Train model (5-15 min on GPU)
python ml/train.py

# Start web app
python app/app.py

# Open http://localhost:5000
```

## How It Works

1. **Ground Truth:** Extract lat/lon from satellite metadata
   - Compute distance to continent centers (haversine)
   - Estimate land percentage based on visible continents
   - Output: 0-100% land coverage

2. **Model:** ResNet18 fine-tuned from ImageNet
   - Pre-trained backbone → trainable layer 4 → regression head
   - Predicts land percentage from image pixels
   - Optimized for small datasets

3. **Web App:** Flask interface for prediction
   - Upload images or get random EPIC images
   - Compare predictions with ground truth
   - REST API for integration

## Documentation

- **[SETUP.md](SETUP.md)** — Installation, training, API usage
- **[RESULTS.md](RESULTS.md)** — Detailed metrics and analysis
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Model design and pipeline

---

**Best Score:** MAE 5.48% | **License:** MIT
