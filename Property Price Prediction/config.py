"""
Configuration file for Property Price Prediction API
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # API settings
    API_TITLE = 'Property Price Prediction API'
    API_VERSION = '1.0.0'
    API_DESCRIPTION = 'AI-powered property price prediction service'
    
    # Model settings
    MODEL_DIR = 'models'
    MODEL_FILENAME = 'property_predictor.pkl'
    SCALER_FILENAME = 'scaler.pkl'
    ENCODERS_FILENAME = 'encoders.pkl'
    
    # Training settings
    TRAINING_SAMPLES = 5000
    MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Rate limiting
    RATE_LIMITS = {
        'default': '200 per day, 50 per hour',
        'prediction': '10 per minute',
        'retrain': '1 per hour'
    }
    
    # CORS settings
    CORS_ORIGINS = [
        'http://localhost:3000',  # React dev server
        'http://localhost:8080',  # Vue dev server
        'http://127.0.0.1:5500',  # Live server
        'http://localhost:5500',  # Live server
        '*'  # Allow all origins in development
    ]
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Property data settings
    PROPERTY_TYPES = [
        'apartment', 'villa', 'townhouse', 'commercial', 'penthouse'
    ]
    
    LOCATION_QUALITIES = ['low', 'medium', 'high']
    
    AMENITIES = [
        'parking', 'pool', 'garden', 'gym', 'security',
        'elevator', 'balcony', 'fireplace', 'central_heating', 'air_conditioning'
    ]
    
    # Validation rules
    VALIDATION_RULES = {
        'square_feet': {'min': 100, 'max': 10000},
        'bedrooms': {'min': 1, 'max': 10},
        'bathrooms': {'min': 0.5, 'max': 10},
        'year_built': {'min': 1900, 'max': 2024},
        'distance_to_city': {'min': 0, 'max': 50}
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    CORS_ORIGINS = ['*']

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    CORS_ORIGINS = [
        'https://yourdomain.com',
        'https://www.yourdomain.com'
    ]

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    TRAINING_SAMPLES = 100

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}



