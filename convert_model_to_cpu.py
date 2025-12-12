"""Convert CUDA model to CPU model."""
import torch
from pathlib import Path

model_path = Path('ml/models/land_ocean_regressor.pkl')

# Load with CPU mapping
print(f"Loading model from {model_path}...")
model_package = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# Save back to CPU
print(f"Saving CPU-compatible model...")
torch.save(model_package, model_path)

print(f"✓ Model converted to CPU format")
