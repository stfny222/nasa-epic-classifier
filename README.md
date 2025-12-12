# NASA EPIC Classifier

Classify satellite images by visible continents using deep learning.

## Quick Results

- **Best Model:** ResNet18 Transfer Learning (F1=0.83)
- **Dataset:** 141 NASA EPIC satellite images  
- **Task:** Predict which continents are visible

## Quick Start

```bash
# Setup
conda create -n epic python=3.11
conda activate epic
pip install torch torchvision pillow scikit-learn pyyaml numpy pandas

# Train best model (5-15 min on GPU)
python ml/experiments/resnet_transfer/main.py
```

## Model Comparison

| Model | F1 | Accuracy |
|-------|-----|----------|
| Baseline CNN | 0.45 | 0.30 |
| Multi-Scale | 0.74 | 0.83 |
| **ResNet18** | **0.83** | **0.90** |
| Ensemble | 0.80 | 0.88 |

## How It Works

1. **Labels:** Extract lat/lon from satellite metadata
   - Compute distance to continent centers (haversine)
   - Label visible if within 2000 km
   - Output: 6 binary continent labels

2. **Model:** ResNet18 fine-tuned from ImageNet
   - Pre-trained backbone + trainable layer 4 + head
   - Threshold = 0.35 (tuned for this dataset)

## Documentation

- **[SETUP.md](SETUP.md)** — Installation, training, inference
- **[RESULTS.md](RESULTS.md)** — Detailed metrics and analysis
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Model designs and algorithms

---

License: MIT | Best F1: 0.83
