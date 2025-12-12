# Architecture & Technical Design

## Geographic Label Generation

**Algorithm:** Haversine distance from image centroid to continent centers

```
For each satellite image:
  1. Extract lat/lon from NASA EPIC metadata
  2. Compute haversine distance to 6 continent centers
  3. If distance ≤ 2000 km → image shows that continent
  4. Generate 6 binary labels [NA, SA, EU, AF, AS, OC]
```

**Continent Centers:**
- North America: 47.0°N, 97.0°W
- South America: 10.0°S, 60.0°W
- Europe: 50.0°N, 10.0°E
- Africa: 6.0°N, 20.0°E
- Asia: 45.0°N, 100.0°E
- Oceania: 22.0°S, 147.0°E

## Model Architectures

### ResNet18 Transfer Learning (BEST - F1=0.83)

**Architecture:**
```
ImageNet Pre-trained ResNet18
  ├── Layers 0-3: FROZEN (universal visual features)
  ├── Layer 4: FINE-TUNED (domain-specific features)
  └── Custom Head:
      ├── AdaptiveAvgPool2D
      ├── FC(256) + ReLU + Dropout(0.3)
      ├── FC(6) + Sigmoid
```

**Why This Works:**
- **11.3M total parameters, 5M trainable** prevents overfitting
- **ImageNet features** capture Earth patterns (clouds, continents, water)
- **Layer 4 fine-tuning** specializes to geographic classification
- **Lower LR (0.0001)** prevents catastrophic forgetting

**Performance:** F1=0.83 (best model)

### Baseline CNN (F1=0.45)

**Architecture:**
```
Input (299×299×3)
  ↓ Conv(32, 3×3) + ReLU + BatchNorm
  ↓ MaxPool(2×2)
  ↓ Conv(64, 3×3) + ReLU + BatchNorm
  ↓ MaxPool(2×2)
  ↓ Conv(128, 3×3) + ReLU + BatchNorm
  ↓ MaxPool(2×2)
  ↓ Flatten
  ↓ FC(256) + ReLU + Dropout(0.5)
  ↓ FC(6) + Sigmoid
```

**Why Lower Performance:** 520K parameters too small for 141 images without pre-training

### Multi-Scale CNN (F1=0.74)

**Architecture:**
```
Input (299×299×3)
  ├─ Branch 3×3 ─┐
  ├─ Branch 5×5 ─┼─ Concatenate ─ FC(256) ─ FC(6) + Sigmoid
  └─ Branch 7×7 ─┘
```

**Why Good but Not Best:** 1.06M parameters helps, but lacks ImageNet pre-training

### Ensemble (F1=0.80)

**Strategy:** Average predictions from ResNet18 + Multi-Scale with weights

**Result:** Underperforms best single model (ResNet18 at 0.83)
- Strong model's confident predictions diluted by weaker model

## Training Configuration

**Loss Function:** Binary Cross-Entropy with pos_weight for class imbalance

```
Loss = -w_pos * y * log(p) - (1-y) * log(1-p)
```

**Optimizer:** Adam
- ResNet18: lr=0.0001 (fine-tuning)
- Others: lr=0.001 (training from scratch)

**Early Stopping:** Monitor val_loss, patience=5

**Threshold:** 0.35 (tuned)
- Standard 0.5 produces F1=0.0 (too conservative)
- 0.35 optimal for model's probability distribution

## Data Pipeline

1. **Download:** NASA EPIC API → 100-300 images
2. **Label:** Geographic coordinates → 6 binary labels
3. **Split:** 70/15/15 train/val/test (stratified)
4. **Preprocess:** Normalize per ImageNet (resize, center crop)
5. **Cache:** Save to disk for fast training

## Performance Comparison

**ResNet18 Advantage:**
- Pre-trained on 1M images vs from-scratch on 141
- Faster convergence (25 vs 50 epochs)
- Better final metrics (+84% F1)
- Efficient use of limited data

**Key Takeaway:** Transfer learning essential for small datasets

---

See RESULTS.md for detailed metrics and SETUP.md for usage.
