#!/usr/bin/env python3
"""
Property Price Prediction - Basic Machine Learning Program
This program demonstrates fundamental ML concepts including:
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Cross-validation
- Hyperparameter tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class PropertyPricePredictor:
    """Main class for property price prediction using machine learning."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_importance = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic property data for demonstration."""
        np.random.seed(42)
        
        # Generate realistic property features
        data = {
            'square_feet': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'location_quality': np.random.choice(['low', 'medium', 'high'], n_samples),
            'has_garage': np.random.choice([0, 1], n_samples),
            'has_pool': np.random.choice([0, 1], n_samples),
            'distance_to_city_center': np.random.uniform(0, 30, n_samples)
        }
        
        # Create target variable (price) with some realistic relationships
        base_price = 200000
        price = (
            base_price +
            data['square_feet'] * 100 +  # $100 per sq ft
            data['bedrooms'] * 25000 +   # $25k per bedroom
            data['bathrooms'] * 15000 +  # $15k per bathroom
            -data['age'] * 2000 +        # $2k depreciation per year
            (data['location_quality'] == 'high') * 50000 +
            (data['location_quality'] == 'medium') * 25000 +
            data['has_garage'] * 15000 +
            data['has_pool'] * 25000 +
            -data['distance_to_city_center'] * 3000 +  # Closer = more expensive
            np.random.normal(0, 20000, n_samples)  # Random noise
        )
        
        data['price'] = np.maximum(price, 50000)  # Minimum price $50k
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data):
        """Preprocess the data for machine learning."""
        # Handle categorical variables
        categorical_cols = ['location_quality']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'price']
        X = data[feature_cols]
        y = data['price']
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple machine learning models."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print()
        
        self.models = results
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[best_model_name]['model']
        
        # Get feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning on the best model."""
        if not self.best_model:
            print("No best model found. Train models first.")
            return
        
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid based on model type
        if isinstance(self.best_model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(self.best_model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            print("Hyperparameter tuning not implemented for this model type.")
            return
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            self.best_model.__class__(**{k: v for k, v in self.best_model.get_params().items() 
                                       if k in param_grid}),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        X_scaled = self.scaler.transform(X)
        grid_search.fit(X_scaled, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
    
    def plot_results(self, X_test, y_test):
        """Plot model performance and feature importance."""
        if not self.best_model:
            print("No best model found. Train models first.")
            return
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Machine Learning Model Performance Analysis', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Price')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        bars = axes[1, 0].bar(model_names, r2_scores)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color bars based on performance
        for bar, score in zip(bars, r2_scores):
            if score == max(r2_scores):
                bar.set_color('green')
            else:
                bar.set_color('lightblue')
        
        # 4. Feature importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Feature Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def make_prediction(self, features):
        """Make a prediction for new property features."""
        if not self.best_model:
            print("No best model found. Train models first.")
            return None
        
        # Preprocess features
        features_df = pd.DataFrame([features])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in features_df.columns:
                features_df[col] = le.transform(features_df[col])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        
        return prediction

def main():
    """Main function to run the machine learning program."""
    print("🏠 Property Price Prediction - Machine Learning Program")
    print("=" * 60)
    
    # Initialize predictor
    predictor = PropertyPricePredictor()
    
    # Generate sample data
    print("📊 Generating sample property data...")
    data = predictor.generate_sample_data(1000)
    print(f"Generated {len(data)} property records")
    print(f"Data shape: {data.shape}")
    print()
    
    # Display sample data
    print("📋 Sample data:")
    print(data.head())
    print()
    
    # Data statistics
    print("📈 Data statistics:")
    print(data.describe())
    print()
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    X, y = predictor.preprocess_data(data)
    print(f"Features: {X.shape[1]}")
    print(f"Target range: ${y.min():,.2f} - ${y.max():,.2f}")
    print()
    
    # Train models
    print("🤖 Training machine learning models...")
    X_test, y_test = predictor.train_models(X, y)
    
    # Hyperparameter tuning
    print("⚙️ Performing hyperparameter tuning...")
    predictor.hyperparameter_tuning(X, y)
    
    # Plot results
    print("📊 Generating performance plots...")
    predictor.plot_results(X_test, y_test)
    
    # Example prediction
    print("🔮 Making example prediction...")
    example_features = {
        'square_feet': 2500,
        'bedrooms': 3,
        'bathrooms': 2,
        'age': 10,
        'location_quality': 'medium',
        'has_garage': 1,
        'has_pool': 0,
        'distance_to_city_center': 15
    }
    
    predicted_price = predictor.make_prediction(example_features)
    if predicted_price:
        print(f"Example property features: {example_features}")
        print(f"Predicted price: ${predicted_price:,.2f}")
    
    print("\n✅ Machine learning program completed successfully!")
    print("\nKey concepts demonstrated:")
    print("- Data generation and preprocessing")
    print("- Multiple ML algorithms (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)")
    print("- Model evaluation and comparison")
    print("- Cross-validation")
    print("- Hyperparameter tuning")
    print("- Feature importance analysis")
    print("- Visualization of results")

if __name__ == "__main__":
    main()
