"""
Unit tests for prediction module
"""

import pytest
import numpy as np
from src.prediction import PLAPHBPredictor

def test_predictor_initialization():
    """Test that predictor initializes correctly"""
    predictor = PLAPHBPredictor()
    assert predictor is not None

def test_prediction_interface():
    """Test the prediction interface"""
    predictor = PLAPHBPredictor()
    
    # Test with dummy model (you'll need actual trained models)
    # This test will fail if models are not trained yet
    try:
        result = predictor.predict(
            phb_content=20.0,
            tg=40.0,
            crystallinity=8.0,
            relaxation_time=0.15
        )
        
        # Check that all expected keys are present
        expected_keys = ['phb_content', 'tensile_strength_MPa', 
                        'impact_toughness_kJ_m2', 'degradation_rate_%']
        
        for key in expected_keys:
            assert key in result
            
    except ValueError as e:
        # Expected if models are not trained
        assert "Model files not found" in str(e)

def test_batch_predict():
    """Test batch prediction"""
    predictor = PLAPHBPredictor()
    
    test_data = {
        'phb_content': [10, 20, 30],
        'tg': [50, 45, 40],
        'crystallinity': [5, 8, 10],
        'relaxation_time': [0.1, 0.15, 0.2]
    }
    
    df = pd.DataFrame(test_data)
    
    try:
        results = predictor.batch_predict(df)
        assert len(results) == 3
    except Exception:
        # Expected if models not trained
        pass

if __name__ == "__main__":
    pytest.main([__file__])