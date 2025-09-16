#!/usr/bin/env python3
"""
Property Price Prediction Streamlit Web App
A user-friendly web interface for predicting property prices using the trained ML model.
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🏠 Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitPropertyPredictor:
    """Streamlit wrapper for property price prediction model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects from disk."""
        model_path = 'models/property_predictor.pkl'
        scaler_path = 'models/scaler.pkl'
        encoders_path = 'models/encoders.pkl'
        
        try:
            if not os.path.exists(model_path):
                st.error("❌ Model file not found! Please ensure the model is trained first.")
                st.stop()
            
            # Load model components
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            # Set feature names based on the model (must match training data)
            self.feature_names = [
                'property_type', 'square_feet', 'bedrooms', 'bathrooms', 
                'year_built', 'age', 'location_quality', 'distance_to_city',
                'crime_rate', 'has_parking', 'has_pool', 'has_garden',
                'has_gym', 'has_security'
            ]
            
            self.model_info = {
                'status': 'loaded',
                'model_type': 'Random Forest Regressor',
                'features': len(self.feature_names)
            }
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
    
    def predict_price(self, features):
        """Make a property price prediction."""
        try:
            # Prepare feature vector
            feature_vector = []
            
            for feature_name in self.feature_names:
                if feature_name in self.label_encoders:
                    # Handle categorical features
                    try:
                        encoded_value = self.label_encoders[feature_name].transform([features[feature_name]])[0]
                        feature_vector.append(encoded_value)
                    except ValueError:
                        # Handle unseen categories
                        feature_vector.append(0)
                else:
                    # Handle numerical features
                    feature_vector.append(features[feature_name])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            predicted_price = self.model.predict(feature_vector_scaled)[0]
            
            return {
                'predicted_price': max(0, predicted_price),  # Ensure non-negative price
                'success': True
            }
            
        except Exception as e:
            return {
                'predicted_price': 0,
                'success': False,
                'error': str(e)
            }

def main():
    """Main Streamlit application."""
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitPropertyPredictor()
    
    predictor = st.session_state.predictor
    
    # App title and description
    st.title("🏠 Property Price Prediction")
    st.markdown("""
    Welcome to the **Property Price Predictor**! This application uses machine learning 
    to estimate property prices based on various features like location, size, and amenities.
    
    Fill in the property details below to get an instant price prediction.
    """)
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        if predictor.model_info:
            st.success(f"Status: {predictor.model_info['status'].title()}")
            st.info(f"Model Type: {predictor.model_info['model_type']}")
            st.info(f"Features: {predictor.model_info['features']}")
        
        st.header("📋 Instructions")
        st.markdown("""
        1. **Fill in the property details** in the main form
        2. **Click 'Predict Price'** to get the estimate
        3. **Review the results** and insights
        4. **Adjust parameters** to see how they affect the price
        """)
    
    # Main form
    st.header("🏡 Property Details")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        
        property_type = st.selectbox(
            "Property Type",
            ["apartment", "villa", "townhouse", "commercial", "penthouse"],
            help="Select the type of property"
        )
        
        square_feet = st.number_input(
            "Square Feet",
            min_value=100,
            max_value=10000,
            value=2000,
            step=50,
            help="Total area of the property in square feet"
        )
        
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Total number of bedrooms"
        )
        
        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Total number of bathrooms"
        )
        
        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2024,
            value=2010,
            step=1,
            help="Year when the property was built"
        )
    
    with col2:
        st.subheader("Location & Features")
        
        location_quality = st.selectbox(
            "Location Quality",
            ["low", "medium", "high"],
            index=1,
            help="Overall quality/desirability of the location"
        )
        
        distance_to_city = st.number_input(
            "Distance to City Center (km)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            help="Distance to the nearest city center"
        )
        
        crime_rate = st.slider(
            "Crime Rate",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Local crime rate (0.0 = very safe, 0.2 = high crime)"
        )
    
    # Amenities section
    st.subheader("🏊 Amenities")
    
    # Create columns for amenities checkboxes
    col3, col4 = st.columns(2)
    
    with col3:
        has_parking = st.checkbox("🚗 Parking", help="Has dedicated parking space")
        has_pool = st.checkbox("🏊 Swimming Pool", help="Has swimming pool")
        has_garden = st.checkbox("🌳 Garden", help="Has garden or yard")
    
    with col4:
        has_gym = st.checkbox("🏋️ Gym/Fitness Center", help="Has gym or fitness facilities")
        has_security = st.checkbox("🔒 Security System", help="Has security system or guards")
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        
        # Prepare features dictionary
        features = {
            'property_type': property_type,
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'year_built': year_built,
            'age': 2024 - year_built,
            'location_quality': location_quality,
            'distance_to_city': distance_to_city,
            'crime_rate': crime_rate,
            'has_parking': 1 if has_parking else 0,
            'has_pool': 1 if has_pool else 0,
            'has_garden': 1 if has_garden else 0,
            'has_gym': 1 if has_gym else 0,
            'has_security': 1 if has_security else 0
        }
        
        # Make prediction
        with st.spinner("🤖 Calculating property price..."):
            result = predictor.predict_price(features)
        
        # Display results
        if result['success']:
            predicted_price = result['predicted_price']
            
            # Create results columns
            st.markdown("---")
            st.header("💰 Prediction Results")
            
            # Main price display
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #4CAF50, #45a049);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            ">
                <h2 style="color: white; margin: 0;">Estimated Property Price</h2>
                <h1 style="color: white; margin: 10px 0; font-size: 3em;">${predicted_price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col6, col7, col8 = st.columns(3)
            
            with col6:
                price_per_sqft = predicted_price / square_feet
                st.metric(
                    "Price per Sq. Ft.",
                    f"${price_per_sqft:.0f}",
                    help="Estimated price per square foot"
                )
            
            with col7:
                # Calculate price category
                if predicted_price < 300000:
                    category = "Budget-Friendly"
                    color = "🟢"
                elif predicted_price < 600000:
                    category = "Mid-Range"
                    color = "🟡"
                else:
                    category = "Luxury"
                    color = "🔴"
                
                st.metric(
                    "Price Category",
                    f"{color} {category}",
                    help="Price category based on estimated value"
                )
            
            with col8:
                # Show property age
                property_age = 2024 - year_built
                st.metric(
                    "Property Age",
                    f"{property_age} years",
                    help="Age of the property"
                )
            
            # Feature impact insights
            st.subheader("📈 Key Factors")
            insights = []
            
            # Location impact
            if location_quality == "high":
                insights.append("🌟 High-quality location adds significant value")
            elif location_quality == "low":
                insights.append("📍 Location quality may reduce property value")
            
            # Size impact
            if square_feet > 3000:
                insights.append("🏠 Large property size increases value")
            elif square_feet < 1000:
                insights.append("🏠 Compact size may limit value")
            
            # Amenities impact
            amenity_count = sum([has_parking, has_pool, has_garden, has_gym, has_security])
            if amenity_count >= 4:
                insights.append("⭐ Excellent amenities boost property value")
            elif amenity_count <= 1:
                insights.append("🔧 Limited amenities may affect pricing")
            
            # Age impact
            if property_age <= 5:
                insights.append("🆕 New construction commands premium pricing")
            elif property_age >= 30:
                insights.append("🏗️ Older property may have lower market value")
            
            for insight in insights:
                st.info(insight)
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.json(features)
        
        else:
            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏠 Property Price Predictor | Built with Streamlit & Machine Learning</p>
        <p><small>Predictions are estimates based on historical data and may not reflect actual market values.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()