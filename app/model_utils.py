"""Model utilities for land percentage prediction."""
import pickle
import torch
from PIL import Image
from pathlib import Path
from ml.models import LandPercentageRegressor
from ml.preprocessing import INFERENCE_TRANSFORM


class LandOceanPredictor:
    """Land vs Ocean percentage predictor for satellite images."""
    
    def __init__(self, model_path=None):
        """Initialize land ocean predictor.
        
        Args:
            model_path: Path to trained model pickle file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = INFERENCE_TRANSFORM
        
        # Load model
        if model_path is None:
            model_path = Path(__file__).parent.parent / 'ml' / 'models' / 'land_ocean_regressor.pkl'
        
        self.model_path = Path(model_path)
        self.model = None
        self.metrics = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load model from pickle file."""
        try:
            # Use torch.load with map_location and weights_only=False for pickle compatibility
            model_package = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model architecture
            self.model = LandPercentageRegressor(pretrained=False)
            self.model.load_state_dict(model_package['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store metrics
            self.metrics = model_package.get('metrics', {})
            
            print(f"✓ Land-Ocean predictor loaded from {self.model_path}")
            print(f"  Test MAE: {self.metrics.get('test_mae', 'N/A'):.2f}%")
            print(f"  Test R²: {self.metrics.get('test_r2', 'N/A'):.4f}")
        
        except FileNotFoundError:
            print(f"⚠ Model file not found at {self.model_path}")
            self.model = None
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            self.model = None
    
    def predict(self, image_input):
        """Predict land percentage in an image.
        
        Args:
            image_input: Either PIL Image or file path
            
        Returns:
            dict with prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'land_percentage': None,
                'ocean_percentage': None
            }
        
        # Load image if needed
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        # Transform and predict
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                land_pct = torch.clamp(output, 0, 100).squeeze().item()
            
            ocean_pct = 100 - land_pct
            
            return {
                'land_percentage': round(land_pct, 2),
                'ocean_percentage': round(ocean_pct, 2),
                'success': True
            }
        
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'land_percentage': None,
                'ocean_percentage': None
            }
