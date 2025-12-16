"""
Prediction module for PLA/PHB properties
"""

import joblib
import numpy as np
import pandas as pd

class PLAPHBPredictor:
    """Main class for predicting PLA/PHB blend properties"""
    
    def __init__(self, model_dir="models/"):
        """Initialize predictor with trained models"""
        try:
            self.tensile_model = joblib.load(f"{model_dir}/tensile_model.pkl")
            self.impact_model = joblib.load(f"{model_dir}/impact_model.pkl")
            self.degradation_model = joblib.load(f"{model_dir}/degradation_model.pkl")
        except FileNotFoundError:
            print("Model files not found. Please run model training first.")
            self.tensile_model = None
            self.impact_model = None
            self.degradation_model = None
    
    def predict_tensile_strength(self, phb_content, tg, crystallinity, relaxation_time):
        """Predict tensile strength for given parameters"""
        if self.tensile_model is None:
            raise ValueError("Tensile model not loaded")
        
        features = np.array([[phb_content, tg, crystallinity, relaxation_time]])
        return float(self.tensile_model.predict(features)[0])
    
    def predict_impact_toughness(self, phb_content, tg, crystallinity, relaxation_time):
        """Predict impact toughness for given parameters"""
        if self.impact_model is None:
            raise ValueError("Impact model not loaded")
        
        features = np.array([[phb_content, tg, crystallinity, relaxation_time]])
        return float(self.impact_model.predict(features)[0])
    
    def predict_degradation_rate(self, phb_content, tg, crystallinity, ph=7.4):
        """Predict degradation rate for given parameters"""
        if self.degradation_model is None:
            raise ValueError("Degradation model not loaded")
        
        features = np.array([[phb_content, tg, crystallinity, ph]])
        return float(self.degradation_model.predict(features)[0])
    
    def predict(self, phb_content, tg, crystallinity, relaxation_time, ph=7.4):
        """Predict all properties for a given formulation"""
        predictions = {
            'phb_content': phb_content,
            'tensile_strength_MPa': self.predict_tensile_strength(phb_content, tg, crystallinity, relaxation_time),
            'impact_toughness_kJ_m2': self.predict_impact_toughness(phb_content, tg, crystallinity, relaxation_time),
            'degradation_rate_%': self.predict_degradation_rate(phb_content, tg, crystallinity, ph)
        }
        
        return predictions
    
    def batch_predict(self, formulations_df):
        """Predict properties for multiple formulations"""
        results = []
        
        for _, row in formulations_df.iterrows():
            pred = self.predict(
                phb_content=row['phb_content'],
                tg=row.get('tg', 55.0),
                crystallinity=row.get('crystallinity', 20.0),
                relaxation_time=row.get('relaxation_time', 0.1),
                ph=row.get('ph', 7.4)
            )
            results.append(pred)
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, property_type='tensile'):
        """Get feature importance for a specific property"""
        if property_type == 'tensile' and hasattr(self.tensile_model, 'feature_importances_'):
            return self.tensile_model.feature_importances_
        elif property_type == 'impact' and hasattr(self.impact_model, 'feature_importances_'):
            return self.impact_model.feature_importances_
        elif property_type == 'degradation' and hasattr(self.degradation_model, 'feature_importances_'):
            return self.degradation_model.feature_importances_
        else:
            return None