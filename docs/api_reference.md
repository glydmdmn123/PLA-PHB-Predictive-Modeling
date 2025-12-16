# API Reference

## PLAPHBPredictor Class

Main class for predicting PLA/PHB blend properties.

### Methods

#### `__init__(model_dir="models/")`
Initialize predictor with trained models.

**Parameters:**
- `model_dir` (str): Directory containing model files

#### `predict(phb_content, tg, crystallinity, relaxation_time, ph=7.4)`
Predict all properties for a given formulation.

**Parameters:**
- `phb_content` (float): PHB weight percentage (0-40)
- `tg` (float): Glass transition temperature (°C)
- `crystallinity` (float): Crystallinity percentage (%)
- `relaxation_time` (float): Relaxation time (s)
- `ph` (float, optional): Environmental pH, default 7.4

**Returns:**
- dict: Dictionary with predicted properties

#### `predict_tensile_strength(phb_content, tg, crystallinity, relaxation_time)`
Predict tensile strength only.

#### `predict_impact_toughness(phb_content, tg, crystallinity, relaxation_time)`
Predict impact toughness only.

#### `predict_degradation_rate(phb_content, tg, crystallinity, ph=7.4)`
Predict degradation rate only.

#### `batch_predict(formulations_df)`
Predict properties for multiple formulations.

**Parameters:**
- `formulations_df` (pd.DataFrame): DataFrame with formulation parameters

**Returns:**
- pd.DataFrame: DataFrame with predictions

## ReverseDesignOptimizer Class

Optimizer for reverse design of formulations.

### Methods

#### `__init__(predictor)`
Initialize optimizer with a predictor instance.

#### `find_optimal_formulation(constraints, bounds=None)`
Find optimal formulation meeting all constraints.

**Parameters:**
- `constraints` (dict): Dictionary of property constraints
- `bounds` (list): Parameter bounds, default [(0,40), (30,60), (0,40), (0.05,2.0)]

**Returns:**
- dict: Optimal formulation and predictions

#### `generate_pareto_front(objectives, bounds=None, n_points=50)`
Generate Pareto front for multi-objective optimization.