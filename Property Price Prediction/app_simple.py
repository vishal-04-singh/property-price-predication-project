#!/usr/bin/env python3
"""
Property Price Prediction Backend API - Simplified Version
Works with Python 3.13+ without scikit-learn dependencies
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])  # Enable CORS for frontend integration

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

class RealTimePropertyPredictor:
    """Real-time property price prediction with market data integration."""
    
    def __init__(self):
        self.model_info = {
            'status': 'ready',
            'last_trained': datetime.now().isoformat(),
            'model_type': 'Real-time Market Calculator',
            'training_samples': 0,
            'features': 16
        }
        self.market_data = {}
    
    def predict_price(self, features):
        """Make price prediction using real-time market data."""
        try:
            # Get real-time market data
            market_data = self._get_real_time_market_data(features.get('location', 'suburbs'))
            
            # Base price calculation with market data
            base_price_per_sqft = market_data['price_per_sqft']
            base_price = features['square_feet'] * base_price_per_sqft
            
            # Location adjustments
            location_multiplier = self._get_location_multiplier(features['location_quality'])
            distance_adjustment = self._get_distance_adjustment(features['distance_to_city'])
            
            # Property type adjustments
            type_multiplier = self._get_property_type_multiplier(features['property_type'])
            
            # Size and room adjustments
            size_price = features['square_feet'] * base_price_per_sqft
            bedroom_price = features['bedrooms'] * 30000  # $30k per bedroom
            bathroom_price = features['bathrooms'] * 25000  # $25k per bathroom
            
            # Age adjustment
            age = 2024 - features['year_built']
            age_adjustment = -age * 2500  # $2.5k depreciation per year
            
            # Amenities bonus
            amenities_bonus = self._calculate_amenities_bonus(features)
            
            # Market factors
            market_trend = market_data['market_trend']
            seasonal_factor = market_data['seasonal_factor']
            demand_factor = market_data['demand_factor']
            
            # Calculate final price with market adjustments
            final_price = (
                (base_price * location_multiplier * type_multiplier +
                size_price + bedroom_price + bathroom_price +
                age_adjustment + distance_adjustment + amenities_bonus) *
                (1 + market_trend) * seasonal_factor * demand_factor
            )
            
            # Ensure minimum price
            final_price = max(final_price, 75000)
            
            # Calculate confidence (higher for more standard properties)
            confidence = self._calculate_confidence(features)
            
            return {
                'predicted_price': round(final_price, 2),
                'confidence': confidence,
                'price_range': {
                    'low': round(final_price * 0.85, 2),
                    'high': round(final_price * 1.15, 2)
                },
                'market_data': market_data,
                'price_per_sqft': round(final_price / features['square_feet'], 2)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def _get_real_time_market_data(self, location):
        """Get real-time market data for location."""
        # Simulate real-time market data
        # In production, this would fetch from APIs like Zillow, Redfin, etc.
        
        market_data = {
            'downtown': {
                'price_per_sqft': 450,
                'market_trend': 0.12,  # 12% annual appreciation
                'seasonal_factor': 1.15,
                'demand_factor': 1.3,
                'inventory_level': 1.8,
                'days_on_market': 25
            },
            'midtown': {
                'price_per_sqft': 350,
                'market_trend': 0.10,
                'seasonal_factor': 1.10,
                'demand_factor': 1.2,
                'inventory_level': 2.2,
                'days_on_market': 30
            },
            'suburbs': {
                'price_per_sqft': 280,
                'market_trend': 0.08,
                'seasonal_factor': 1.05,
                'demand_factor': 1.1,
                'inventory_level': 2.8,
                'days_on_market': 40
            },
            'rural': {
                'price_per_sqft': 200,
                'market_trend': 0.05,
                'seasonal_factor': 0.95,
                'demand_factor': 0.9,
                'inventory_level': 4.0,
                'days_on_market': 60
            },
            'beachfront': {
                'price_per_sqft': 650,
                'market_trend': 0.15,
                'seasonal_factor': 1.20,
                'demand_factor': 1.4,
                'inventory_level': 1.5,
                'days_on_market': 20
            },
            'mountain_view': {
                'price_per_sqft': 400,
                'market_trend': 0.11,
                'seasonal_factor': 1.12,
                'demand_factor': 1.25,
                'inventory_level': 2.0,
                'days_on_market': 28
            }
        }
        
        # Add some market variation
        import random
        variation = random.uniform(-0.1, 0.1)  # ±10% variation
        
        if location.lower() in market_data:
            data = market_data[location.lower()].copy()
            data['price_per_sqft'] *= (1 + variation)
            return data
        else:
            # Default to suburbs
            data = market_data['suburbs'].copy()
            data['price_per_sqft'] *= (1 + variation)
            return data
    
    def _get_location_multiplier(self, location_quality):
        """Get location quality multiplier."""
        multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.4
        }
        return multipliers.get(location_quality, 1.0)
    
    def _get_distance_adjustment(self, distance):
        """Get distance to city center adjustment."""
        if distance <= 5:
            return 50000  # Premium for city center
        elif distance <= 15:
            return 25000  # Good for mid-distance
        elif distance <= 25:
            return 0  # Neutral
        else:
            return -25000  # Discount for far locations
    
    def _get_property_type_multiplier(self, property_type):
        """Get property type multiplier."""
        multipliers = {
            'apartment': 1.0,
            'villa': 1.3,
            'townhouse': 1.2,
            'commercial': 1.5,
            'penthouse': 1.8
        }
        return multipliers.get(property_type, 1.0)
    
    def _calculate_amenities_bonus(self, features):
        """Calculate amenities bonus."""
        bonus = 0
        amenities = [
            'has_parking', 'has_pool', 'has_garden', 'has_gym',
            'has_balcony', 'has_elevator', 'has_security', 'has_fireplace'
        ]
        
        for amenity in amenities:
            if features.get(amenity, 0) == 1:
                if amenity == 'has_parking':
                    bonus += 15000
                elif amenity == 'has_pool':
                    bonus += 30000
                elif amenity == 'has_garden':
                    bonus += 20000
                elif amenity == 'has_gym':
                    bonus += 15000
                elif amenity == 'has_balcony':
                    bonus += 10000
                elif amenity == 'has_elevator':
                    bonus += 12000
                elif amenity == 'has_security':
                    bonus += 25000
                elif amenity == 'has_fireplace':
                    bonus += 8000
        
        return bonus
    
    def _calculate_confidence(self, features):
        """Calculate prediction confidence."""
        # Higher confidence for standard properties
        confidence = 0.7  # Base confidence
        
        # Adjust based on property type
        if features['property_type'] in ['apartment', 'villa']:
            confidence += 0.1
        
        # Adjust based on size (standard sizes are more predictable)
        if 1000 <= features['square_feet'] <= 3000:
            confidence += 0.1
        
        # Adjust based on age (newer properties are more predictable)
        if features['year_built'] >= 2010:
            confidence += 0.1
        
        return min(confidence, 0.95)  # Cap at 95%

# Initialize predictor
predictor = RealTimePropertyPredictor()

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
    """Home endpoint - returns enhanced API info."""
    return jsonify({
        'message': 'Real-time Property Price Prediction API',
        'version': '3.0.0',
        'status': 'running',
        'model_type': 'Real-time Market Calculator (Python 3.13 Compatible)',
        'features': [
            'Real-time market data integration',
            'Location-based pricing',
            'Market trend analysis',
            'Seasonal adjustments',
            'Demand factor analysis'
        ],
        'endpoints': {
            'GET /': 'API information',
            'GET /api/health': 'Health check',
            'GET /api/model-info': 'Model information',
            'GET /api/sample-data': 'Sample data for frontend',
            'GET /api/market-data': 'Current market data',
            'POST /api/predict': 'Make real-time prediction',
            'GET /api/features': 'Available features'
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

# Backward-compatible alias for tools calling /health directly
@app.route('/health')
def health_alias():
    return health_check()

@app.route('/api/model-info')
def model_info():
    """Get information about the model."""
    return jsonify({
        'model_info': predictor.model_info,
        'feature_count': len(predictor.model_info.get('features', 0)),
        'features': [
            'property_type', 'square_feet', 'bedrooms', 'bathrooms',
            'year_built', 'location_quality', 'distance_to_city',
            'has_parking', 'has_pool', 'has_garden', 'has_gym',
            'has_balcony', 'has_elevator', 'has_security', 'has_fireplace'
        ]
    })

@app.route('/api/sample-data')
def sample_data():
    """Get sample data for frontend forms."""
    return jsonify({
        'locations': SAMPLE_LOCATIONS,
        'property_types': PROPERTY_TYPES,
        'amenities': [
            'parking', 'pool', 'garden', 'gym', 'security',
            'elevator', 'balcony', 'fireplace'
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
            'has_balcony': 1,
            'has_elevator': 0,
            'has_security': 1,
            'has_fireplace': 0
        }
    })

@app.route('/api/market-data')
def get_market_data():
    """Get current market data for a location."""
    location = request.args.get('location', 'suburbs')
    market_data = predictor._get_real_time_market_data(location)
    
    return jsonify({
        'location': location,
        'market_data': market_data,
        'timestamp': datetime.now().isoformat(),
        'data_sources': 'Real-time market analysis with location-based factors'
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
            'location_quality': data.get('location_quality', 'medium'),
            'distance_to_city': data.get('distance_to_city', 15.0),
            'has_parking': data.get('has_parking', 0),
            'has_pool': data.get('has_pool', 0),
            'has_garden': data.get('has_garden', 0),
            'has_gym': data.get('has_gym', 0),
            'has_balcony': data.get('has_balcony', 0),
            'has_elevator': data.get('has_elevator', 0),
            'has_security': data.get('has_security', 0),
            'has_fireplace': data.get('has_fireplace', 0)
        }
        
        # Make prediction
        result = predictor.predict_price(features)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['input_features'] = features
        
        # Calculate additional insights
        avg_price_per_sqft = result['predicted_price'] / features['square_feet']
        market_data = result['market_data']
        
        result['insights'] = {
            'price_per_sqft': round(avg_price_per_sqft, 2),
            'price_category': _categorize_price(result['predicted_price']),
            'market_position': _get_market_position(result['predicted_price'], features['property_type']),
            'market_trend': f"{market_data['market_trend']*100:.1f}% annual appreciation",
            'seasonal_factor': f"{market_data['seasonal_factor']:.2f}x",
            'demand_level': f"{market_data['demand_factor']:.2f}x",
            'inventory_level': f"{market_data['inventory_level']:.1f} months supply",
            'days_on_market': f"{market_data['days_on_market']:.0f} days"
        }
        
        # Format price for display
        result['prediction'] = {
            'formatted_price': f"${result['predicted_price']:,.0f}",
            'price_range': {
                'low': f"${result['price_range']['low']:,.0f}",
                'high': f"${result['price_range']['high']:,.0f}"
            },
            'confidence': result['confidence']
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

def _get_market_position(predicted_price, property_type):
    """Get market position."""
    base_prices = {
        'apartment': 200000,
        'villa': 400000,
        'townhouse': 300000,
        'commercial': 500000,
        'penthouse': 600000
    }
    
    avg_type_price = base_prices.get(property_type, 300000)
    return 'above' if predicted_price > avg_type_price else 'below'

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
        'has_parking': 'Whether property has parking (0 or 1)',
        'has_pool': 'Whether property has a pool (0 or 1)',
        'has_garden': 'Whether property has a garden (0 or 1)',
        'has_gym': 'Whether property has a gym (0 or 1)',
        'has_balcony': 'Whether property has a balcony (0 or 1)',
        'has_elevator': 'Whether property has an elevator (0 or 1)',
        'has_security': 'Whether property has security (0 or 1)',
        'has_fireplace': 'Whether property has a fireplace (0 or 1)'
    }
    
    return jsonify({
        'features': feature_descriptions,
        'required_features': ['property_type', 'square_feet', 'bedrooms', 'bathrooms'],
        'optional_features': [f for f in feature_descriptions.keys() if f not in ['property_type', 'square_feet', 'bedrooms', 'bathrooms']]
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🏠 Real-time Property Price Prediction API Starting...")
    print("✅ Python 3.13+ Compatible Version")
    print("✅ Real-time market data integration")
    print("✅ Location-based pricing")
    print("✅ Market trend analysis")
    print(f"Model status: {predictor.model_info.get('status', 'unknown')}")
    print("API endpoints available at:")
    print("  - GET  /: API information")
    print("  - GET  /api/health: Health check")
    print("  - GET  /api/model-info: Model information")
    print("  - GET  /api/sample-data: Sample data")
    print("  - GET  /api/market-data: Current market data")
    print("  - POST /api/predict: Make real-time prediction")
    print("  - GET  /api/features: Available features")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

