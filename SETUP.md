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
pip install torch torchvision pillow scikit-learn pyyaml numpy pandas
```

## Quick Start

```bash
# Download data and train best model
python ml/experiments/resnet_transfer/main.py

# Output: model saved to ml/models/checkpoints/resnet18_best.pt
# Test F1: 0.8316
```

**Training Time:** 5-15 minutes on GPU (NVIDIA Tesla T4 or better)

## Make Predictions

### Single Image

```python
import torch
from torchvision import transforms
from PIL import Image
from ml.models.resnet_transfer import ResNetTransferLearner

# Load model
model = ResNetTransferLearner()
model.load_state_dict(torch.load('ml/models/checkpoints/resnet18_best.pt'))
model.eval()

# Load image
image = Image.open('satellite_image.png')
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
img_tensor = preprocess(image).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)[0]
    predictions = (probs > 0.35).int()

continents = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']
visible = [continents[i] for i, p in enumerate(predictions) if p]
print(f"Visible: {visible}")
```

### Batch Processing

```python
from torch.utils.data import DataLoader
from ml.data_preprocessing.image_preprocessor import EpicDataset

# Create dataset
dataset = EpicDataset(image_dir='path/to/images')
loader = DataLoader(dataset, batch_size=16)

# Predict on all
all_preds = []
with torch.no_grad():
    for images in loader:
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.35).int()
        all_preds.append(preds.cpu())

predictions = torch.cat(all_preds, dim=0)
torch.save(predictions, 'predictions.pt')
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config.yaml:
batch_size: 8  # was 16

# Or use CPU:
export CUDA_VISIBLE_DEVICES=""
python ml/experiments/resnet_transfer/main.py
```

### Model Predicting All Zeros (F1=0.0)
```bash
# Check threshold in main.py:
predictions = (logits > 0.35).int()  # NOT > 0.5

# Verify data loading:
python -c "
from ml.data_preprocessing.image_preprocessor import load_epic_data
X_train, _, _, y_train, _, _ = load_epic_data()
print(f'Positive rate: {y_train.mean():.1%}')  # Should be 30-45%
"
```

### FileNotFoundError: data_epic not found
```bash
# Download data first:
cd ml/data_preprocessing
python catalog_builder.py

# Or create directories:
mkdir -p ml/data_epic ml/processed_data
```

### Training Loss Not Decreasing
```bash
# Reduce learning rate in config.yaml:
learning_rate: 0.00005  # was 0.0001

# Check data normalization (ImageNet standard):
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

## Running Experiments

```bash
# Baseline CNN
python ml/experiments/baseline_cnn/main.py

# Multi-Scale CNN
python ml/experiments/multiscale_cnn/main.py

# Ensemble
python ml/experiments/ensemble/main.py

# Results saved to ml/experiments/{name}/outputs/
```

## Configuration

Edit `ml/experiments/resnet_transfer/config.yaml`:

```yaml
num_epochs: 30
batch_size: 16
learning_rate: 0.0001
threshold: 0.35
```

---

See RESULTS.md for detailed metrics and ARCHITECTURE.md for technical details.
