#!/usr/bin/env python3
"""
Property Price Prediction Backend API
Flask-based backend to support the frontend property price prediction application
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

class PropertyPricePredictor:
    """Property price prediction model and data management."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_info = {}
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one."""
        model_path = 'models/property_predictor.pkl'
        scaler_path = 'models/scaler.pkl'
        encoders_path = 'models/encoders.pkl'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                
                logger.info("Loaded existing trained model")
                self.model_info = {
                    'status': 'loaded',
                    'last_trained': datetime.now().isoformat(),
                    'model_type': 'Random Forest'
                }
            else:
                self.train_new_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.train_new_model()
    
    def train_new_model(self):
        """Train a new property price prediction model."""
        logger.info("Training new property price prediction model...")
        
        # Generate synthetic training data
        data = self.generate_training_data(5000)
        
        # Preprocess data
        X, y = self.preprocess_data(data)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Save model
        self.save_model()
        
        self.model_info = {
            'status': 'trained',
            'last_trained': datetime.now().isoformat(),
            'model_type': 'Random Forest',
            'training_samples': len(data),
            'features': len(self.feature_names)
        }
        
        logger.info("Model training completed successfully")
    
    def generate_training_data(self, n_samples=5000):
        """Generate synthetic property data for training."""
        np.random.seed(42)
        
        # Property types and their base prices
        property_types = ['apartment', 'villa', 'townhouse', 'commercial', 'penthouse']
        property_base_prices = {
            'apartment': 150000,
            'villa': 300000,
            'townhouse': 250000,
            'commercial': 400000,
            'penthouse': 500000
        }
        
        # Generate data
        data = []
        for _ in range(n_samples):
            property_type = np.random.choice(property_types)
            base_price = property_base_prices[property_type]
            
            # Generate features
            square_feet = np.random.normal(2000, 800)
            bedrooms = np.random.randint(1, 6)
            bathrooms = np.random.randint(1, 4)
            year_built = np.random.randint(1980, 2024)
            age = 2024 - year_built
            
            # Location features
            location_quality = np.random.choice(['low', 'medium', 'high'])
            distance_to_city = np.random.uniform(0, 30)
            crime_rate = np.random.uniform(0.01, 0.15)
            
            # Amenities
            has_parking = np.random.choice([0, 1])
            has_pool = np.random.choice([0, 1])
            has_garden = np.random.choice([0, 1])
            has_gym = np.random.choice([0, 1])
            has_security = np.random.choice([0, 1])
            
            # Calculate price with realistic relationships
            price = (
                base_price +
                square_feet * 80 +  # $80 per sq ft
                bedrooms * 20000 +   # $20k per bedroom
                bathrooms * 15000 +  # $15k per bathroom
                -age * 1500 +        # $1.5k depreciation per year
                (location_quality == 'high') * 100000 +
                (location_quality == 'medium') * 50000 +
                -distance_to_city * 5000 +  # Closer = more expensive
                -crime_rate * 500000 +      # Lower crime = higher price
                has_parking * 25000 +
                has_pool * 35000 +
                has_garden * 20000 +
                has_gym * 15000 +
                has_security * 30000 +
                np.random.normal(0, 30000)  # Random noise
            )
            
            price = max(price, 50000)  # Minimum price
            
            data.append({
                'property_type': property_type,
                'square_feet': square_feet,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built,
                'age': age,
                'location_quality': location_quality,
                'distance_to_city': distance_to_city,
                'crime_rate': crime_rate,
                'has_parking': has_parking,
                'has_pool': has_pool,
                'has_garden': has_garden,
                'has_gym': has_gym,
                'has_security': has_security,
                'price': price
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data):
        """Preprocess data for machine learning."""
        # Handle categorical variables
        categorical_cols = ['property_type', 'location_quality']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'price']
        self.feature_names = feature_cols
        
        X = data[feature_cols]
        y = data['price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def save_model(self):
        """Save the trained model and preprocessing objects."""
        try:
            with open('models/property_predictor.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            with open('models/encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def predict_price(self, features):
        """Make price prediction for given features."""
        try:
            # Preprocess input features
            features_df = pd.DataFrame([features])
            
            # Encode categorical variables
            for col, le in self.label_encoders.items():
                if col in features_df.columns:
                    features_df[col] = le.transform(features_df[col])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Reorder columns to match training data
            features_df = features_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence (using model's prediction variance)
            predictions = []
            for _ in range(10):
                pred = self.model.predict(features_scaled)[0]
                predictions.append(pred)
            
            confidence = 1 - (np.std(predictions) / np.mean(predictions))
            confidence = max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
            
            return {
                'predicted_price': float(prediction),
                'confidence': float(confidence),
                'price_range': {
                    'low': float(prediction * 0.85),
                    'high': float(prediction * 1.15)
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Error making prediction: {str(e)}")

# Initialize predictor
predictor = PropertyPricePredictor()

# Sample data for frontend
SAMPLE_LOCATIONS = [
    "Downtown", "Midtown", "Uptown", "Suburbs", "Rural Area",
    "Beachfront", "Mountain View", "City Center", "Residential District"
]

PROPERTY_TYPES = [
    "apartment", "villa", "townhouse", "commercial", "penthouse"
]

@app.route('/')
def home():
    """Home endpoint - returns basic API info."""
    return jsonify({
        'message': 'Property Price Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'API information',
            'GET /api/health': 'Health check',
            'GET /api/model-info': 'Model information',
            'GET /api/sample-data': 'Sample data for frontend',
            'POST /api/predict': 'Make price prediction',
            'GET /api/features': 'Available features',
            'POST /api/retrain': 'Retrain model (admin)'
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': predictor.model_info.get('status', 'unknown')
    })

@app.route('/api/model-info')
def model_info():
    """Get information about the trained model."""
    return jsonify({
        'model_info': predictor.model_info,
        'feature_count': len(predictor.feature_names),
        'features': predictor.feature_names
    })

@app.route('/api/sample-data')
def sample_data():
    """Get sample data for frontend forms."""
    return jsonify({
        'locations': SAMPLE_LOCATIONS,
        'property_types': PROPERTY_TYPES,
        'amenities': [
            'parking', 'pool', 'garden', 'gym', 'security',
            'elevator', 'balcony', 'fireplace', 'central_heating', 'air_conditioning'
        ],
        'sample_property': {
            'property_type': 'apartment',
            'square_feet': 1500,
            'bedrooms': 2,
            'bathrooms': 2,
            'year_built': 2010,
            'location_quality': 'medium',
            'distance_to_city': 10,
            'has_parking': 1,
            'has_pool': 0,
            'has_garden': 1,
            'has_gym': 1,
            'has_security': 1
        }
    })

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_price():
    """Make a property price prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['property_type', 'square_feet', 'bedrooms', 'bathrooms']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Validate data types and ranges
        validation_errors = []
        
        if not isinstance(data['square_feet'], (int, float)) or data['square_feet'] <= 0:
            validation_errors.append('square_feet must be a positive number')
        
        if not isinstance(data['bedrooms'], int) or data['bedrooms'] < 1 or data['bedrooms'] > 10:
            validation_errors.append('bedrooms must be an integer between 1 and 10')
        
        if not isinstance(data['bathrooms'], (int, float)) or data['bathrooms'] < 0.5 or data['bathrooms'] > 10:
            validation_errors.append('bathrooms must be a number between 0.5 and 10')
        
        if data['property_type'] not in PROPERTY_TYPES:
            validation_errors.append(f'property_type must be one of: {PROPERTY_TYPES}')
        
        if validation_errors:
            return jsonify({'error': 'Validation errors', 'details': validation_errors}), 400
        
        # Set default values for optional fields
        features = {
            'property_type': data['property_type'],
            'square_feet': float(data['square_feet']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': float(data['bathrooms']),
            'year_built': data.get('year_built', 2020),
            'age': 2024 - data.get('year_built', 2020),
            'location_quality': data.get('location_quality', 'medium'),
            'distance_to_city': data.get('distance_to_city', 15.0),
            'crime_rate': data.get('crime_rate', 0.05),
            'has_parking': data.get('has_parking', 0),
            'has_pool': data.get('has_pool', 0),
            'has_garden': data.get('has_garden', 0),
            'has_gym': data.get('has_gym', 0),
            'has_balcony': data.get('has_balcony', 0),
            'has_elevator': data.get('has_elevator', 0),
            'has_fireplace': data.get('has_fireplace', 0),
            'has_security': data.get('has_security', 0)
        }
        
        # Make prediction
        result = predictor.predict_price(features)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['input_features'] = features
        
        # Calculate additional insights
        avg_price_per_sqft = result['predicted_price'] / features['square_feet']
        result['insights'] = {
            'price_per_sqft': round(avg_price_per_sqft, 2),
            'price_category': _categorize_price(result['predicted_price']),
            'comparison': _get_price_comparison(result['predicted_price'], features)
        }
        
        logger.info(f"Prediction made successfully: ${result['predicted_price']:,.2f}")
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def _categorize_price(price):
    """Categorize price into ranges."""
    if price < 100000:
        return 'Budget'
    elif price < 250000:
        return 'Affordable'
    elif price < 500000:
        return 'Mid-range'
    elif price < 1000000:
        return 'Luxury'
    else:
        return 'Ultra-luxury'

def _get_price_comparison(predicted_price, features):
    """Get price comparison insights."""
    # This would typically come from a database of recent sales
    # For now, return estimated comparisons
    property_type = features['property_type']
    base_comparisons = {
        'apartment': 200000,
        'villa': 400000,
        'townhouse': 300000,
        'commercial': 500000,
        'penthouse': 600000
    }
    
    avg_type_price = base_comparisons.get(property_type, 300000)
    
    return {
        'vs_property_type_avg': round(((predicted_price - avg_type_price) / avg_type_price) * 100, 1),
        'vs_market_avg': round(((predicted_price - 350000) / 350000) * 100, 1),
        'market_position': 'above' if predicted_price > avg_type_price else 'below'
    }

@app.route('/api/features')
def get_features():
    """Get available features and their descriptions."""
    feature_descriptions = {
        'property_type': 'Type of property (apartment, villa, etc.)',
        'square_feet': 'Property area in square feet',
        'bedrooms': 'Number of bedrooms',
        'bathrooms': 'Number of bathrooms',
        'year_built': 'Year the property was built',
        'location_quality': 'Quality of location (low, medium, high)',
        'distance_to_city': 'Distance to city center in miles',
        'crime_rate': 'Local crime rate (0-1 scale)',
        'has_parking': 'Whether property has parking (0 or 1)',
        'has_pool': 'Whether property has a pool (0 or 1)',
        'has_garden': 'Whether property has a garden (0 or 1)',
        'has_gym': 'Whether property has a gym (0 or 1)',
        'has_security': 'Whether property has security (0 or 1)'
    }
    
    return jsonify({
        'features': feature_descriptions,
        'required_features': ['property_type', 'square_feet', 'bedrooms', 'bathrooms'],
        'optional_features': [f for f in feature_descriptions.keys() if f not in ['property_type', 'square_feet', 'bedrooms', 'bathrooms']]
    })

@app.route('/api/retrain', methods=['POST'])
@limiter.limit("1 per hour")
def retrain_model():
    """Retrain the model (admin endpoint)."""
    try:
        # In production, add authentication here
        logger.info("Model retraining requested")
        
        # Retrain model
        predictor.train_new_model()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'timestamp': datetime.now().isoformat(),
            'model_info': predictor.model_info
        })
        
    except Exception as e:
        logger.error(f"Model retraining error: {e}")
        return jsonify({'error': 'Failed to retrain model'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("🏠 Property Price Prediction API Starting...")
    print(f"Model status: {predictor.model_info.get('status', 'unknown')}")
    print("API endpoints available at:")
    print("  - GET  /: API information")
    print("  - GET  /api/health: Health check")
    print("  - GET  /api/model-info: Model information")
    print("  - GET  /api/sample-data: Sample data")
    print("  - POST /api/predict: Make prediction")
    print("  - GET  /api/features: Available features")
    print("  - POST /api/retrain: Retrain model")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
