"""Flask application for EPIC Land-Ocean Regressor server."""
import os
import sys
import random
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local modules
from app.model_utils import LandOceanPredictor
from ml.data_preprocessing.geographic_labels import compute_geographic_labels
from ml.api_client import list_available_dates, get_metadata_for_date, build_image_url


def download_random_epic_image():
    """Download a random EPIC image and metadata.
    
    Returns:
        tuple: (image_data_bytes, image_info_dict) or (None, None) if error
    """
    try:
        # Get recent available dates
        dates = list_available_dates(collection='natural')
        if not dates:
            return None, None
        
        # Pick random date from recent ones
        random_date = random.choice(dates[:30])  # Limit to last 30 dates
        
        # Fetch metadata for that date
        metadata = get_metadata_for_date(random_date, collection='natural')
        if not metadata:
            return None, None
        
        # Pick random image from that date
        image_info = random.choice(metadata)
        image_name = image_info['image']
        
        # Download image
        image_url = build_image_url(image_name, random_date, collection='natural', image_type='png')
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        return response.content, {
            'date': random_date,
            'image_name': image_name,
            'lat': float(image_info.get('centroid_coordinates', {}).get('lat', 0)),
            'lon': float(image_info.get('centroid_coordinates', {}).get('lon', 0))
        }
    
    except Exception as e:
        print(f"Error downloading random image: {e}")
        return None, None


def create_app(config=None):
    """Create Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Create upload folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize land-ocean predictor
    try:
        app.predictor = LandOceanPredictor()
        print(f"✓ Land-Ocean predictor ready")
    except Exception as e:
        print(f"⚠ Could not initialize predictor: {e}")
        app.predictor = None
    
    # Routes
    @app.route('/')
    def index():
        """Serve home page."""
        return render_template('index.html')
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Analyze land percentage in uploaded image.
        
        Expected: multipart/form-data with 'image' file and optional 'lat', 'lon'
        Returns: JSON with land percentage prediction
        """
        if not app.predictor:
            return jsonify({'error': 'Predictor not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        try:
            # Read image
            image = Image.open(file.stream).convert('RGB')
            
            # Predict
            result = app.predictor.predict(image)
            
            if 'error' in result:
                return jsonify(result), 500
            
            # Get optional ground truth from coordinates
            lat = request.form.get('lat', type=float)
            lon = request.form.get('lon', type=float)
            
            ground_truth = None
            if lat is not None and lon is not None:
                try:
                    geo_labels = compute_geographic_labels(lat, lon)
                    ground_truth = geo_labels.get('land_percentage', None)
                except:
                    pass
            
            return jsonify({
                'success': True,
                'land_percentage': result['land_percentage'],
                'ocean_percentage': result['ocean_percentage'],
                'ground_truth_land_percentage': ground_truth
            })
        
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/random', methods=['GET'])
    def get_random():
        """Download and analyze a random EPIC image.
        
        Returns: JSON with image (base64), metadata, prediction, and ground truth
        """
        if not app.predictor:
            return jsonify({'error': 'Predictor not loaded'}), 500
        
        try:
            # Download random image
            image_data, image_info = download_random_epic_image()
            
            if image_data is None:
                return jsonify({'error': 'Could not download image'}), 500
            
            # Load image and predict
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            result = app.predictor.predict(image)
            
            if 'error' in result:
                return jsonify(result), 500
            
            # Get ground truth from coordinates
            lat = image_info.get('lat', 0)
            lon = image_info.get('lon', 0)
            
            ground_truth = None
            try:
                geo_labels = compute_geographic_labels(lat, lon)
                ground_truth = geo_labels.get('land_percentage', None)
            except:
                pass
            
            # Encode image as base64 for frontend
            buffered = io.BytesIO()
            image.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'date': image_info.get('date'),
                'location': {
                    'lat': lat,
                    'lon': lon
                },
                'land_percentage': result['land_percentage'],
                'ocean_percentage': result['ocean_percentage'],
                'ground_truth_land_percentage': ground_truth
            })
        
        except Exception as e:
            return jsonify({'error': f'Failed to get random image: {str(e)}'}), 500
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """Check server status."""
        return jsonify({
            'status': 'ok',
            'service': 'EPIC Land-Ocean Regressor',
            'predictor_loaded': app.predictor is not None
        })
    
    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        """Get model metrics."""
        if not app.predictor:
            return jsonify({'error': 'Predictor not loaded'}), 500
        
        return jsonify({
            'success': True,
            'metrics': app.predictor.metrics
        })
    
    @app.route('/static/training_history.png')
    def serve_training_history():
        """Serve training history visualization."""
        try:
            model_dir = Path(__file__).parent.parent / 'ml' / 'models'
            img_path = model_dir / 'training_history.png'
            if not img_path.exists():
                return jsonify({'error': 'Image not found'}), 404
            return send_file(str(img_path), mimetype='image/png')
        except Exception as e:
            return jsonify({'error': f'Failed to serve image: {str(e)}'}), 500
    
    @app.route('/static/test_results.png')
    def serve_test_results():
        """Serve test results visualization."""
        try:
            model_dir = Path(__file__).parent.parent / 'ml' / 'models'
            img_path = model_dir / 'test_results.png'
            if not img_path.exists():
                return jsonify({'error': 'Image not found'}), 404
            return send_file(str(img_path), mimetype='image/png')
        except Exception as e:
            return jsonify({'error': f'Failed to serve image: {str(e)}'}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
