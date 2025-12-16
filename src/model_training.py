"""
Model training module for PLA/PHB predictive modeling
"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def train_tensile_model(X, y, save_path="models/tensile_model.pkl"):
    """Train model for tensile strength prediction"""
    # Linear regression works best for tensile strength
    model = LinearRegression()
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Tensile model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, save_path)
    
    return model

def train_impact_model(X, y, save_path="models/impact_model.pkl"):
    """Train model for impact toughness prediction"""
    # Random forest works well for impact toughness
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=3,
        random_state=42
    )
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Impact model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, save_path)
    
    return model

def train_degradation_model(X, y, save_path="models/degradation_model.pkl"):
    """Train model for degradation prediction"""
    # Gradient boosting works best for degradation
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Degradation model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, save_path)
    
    return model

def train_all_models():
    """Train all models and save them"""
    # Load and prepare data
    from .data_processing import load_experimental_data, augment_data_with_physics, prepare_features_for_prediction
    
    df = load_experimental_data()
    df_augmented = augment_data_with_physics(df)
    
    # Train tensile model
    X_tensile, y_tensile, _ = prepare_features_for_prediction(df_augmented, 'tensile')
    tensile_model = train_tensile_model(X_tensile, y_tensile)
    
    # Train impact model
    X_impact, y_impact, _ = prepare_features_for_prediction(df_augmented, 'impact')
    impact_model = train_impact_model(X_impact, y_impact)
    
    # Train degradation model
    X_degradation, y_degradation, _ = prepare_features_for_prediction(df_augmented, 'degradation')
    degradation_model = train_degradation_model(X_degradation, y_degradation)
    
    print("All models trained and saved successfully")
    
    return tensile_model, impact_model, degradation_model