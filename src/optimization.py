"""
Optimization module for reverse design of PLA/PHB blends
"""

import numpy as np
from scipy.optimize import differential_evolution

class ReverseDesignOptimizer:
    """Optimizer for finding formulations meeting specific requirements"""
    
    def __init__(self, predictor):
        """Initialize with a predictor instance"""
        self.predictor = predictor
    
    def find_optimal_formulation(self, constraints, bounds=None):
        """
        Find optimal formulation meeting all constraints
        
        Parameters:
        -----------
        constraints : dict
            Dictionary of constraints, e.g.:
            {
                'tensile_min': 50.0,  # Minimum tensile strength (MPa)
                'tensile_max': 70.0,  # Maximum tensile strength (MPa)
                'impact_min': 0.25,   # Minimum impact toughness (kJ/m²)
                'degradation_max': 8.0, # Maximum degradation rate (%)
                'ph': 3.0             # pH for degradation constraint
            }
        bounds : list of tuples, optional
            Bounds for each parameter [(min, max), ...]
            Default: [(0, 40), (30, 60), (0, 40), (0.05, 2.0)]
        
        Returns:
        --------
        dict : Optimal formulation and predicted properties
        """
        if bounds is None:
            # Default bounds: [PHB, Tg, crystallinity, relaxation_time]
            bounds = [(0, 40), (30, 60), (0, 40), (0.05, 2.0)]
        
        # Define optimization objective function
        def objective(x):
            # x = [phb_content, tg, crystallinity, relaxation_time]
            phb, tg, crystallinity, relaxation_time = x
            
            try:
                # Predict properties
                tensile = self.predictor.predict_tensile_strength(phb, tg, crystallinity, relaxation_time)
                impact = self.predictor.predict_impact_toughness(phb, tg, crystallinity, relaxation_time)
                degradation = self.predictor.predict_degradation_rate(phb, tg, crystallinity, 
                                                                      constraints.get('ph', 7.4))
                
                # Calculate constraint violations
                penalty = 0
                
                if 'tensile_min' in constraints and tensile < constraints['tensile_min']:
                    penalty += 100 * (constraints['tensile_min'] - tensile)
                if 'tensile_max' in constraints and tensile > constraints['tensile_max']:
                    penalty += 100 * (tensile - constraints['tensile_max'])
                if 'impact_min' in constraints and impact < constraints['impact_min']:
                    penalty += 100 * (constraints['impact_min'] - impact)
                if 'degradation_max' in constraints and degradation > constraints['degradation_max']:
                    penalty += 100 * (degradation - constraints['degradation_max'])
                
                # Maximize tensile and impact, minimize degradation
                score = tensile/100 + impact - degradation/10 - penalty
                
                return -score  # Negative because we minimize
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1e6  # Large penalty for errors
        
        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42
        )
        
        # Extract optimal solution
        optimal_x = result.x
        phb_opt, tg_opt, crystallinity_opt, relaxation_time_opt = optimal_x
        
        # Get predictions for optimal formulation
        predictions = self.predictor.predict(
            phb_content=phb_opt,
            tg=tg_opt,
            crystallinity=crystallinity_opt,
            relaxation_time=relaxation_time_opt,
            ph=constraints.get('ph', 7.4)
        )
        
        # Calculate comprehensive score
        predictions['comprehensive_score'] = -result.fun
        
        return predictions
    
    def generate_pareto_front(self, objectives, bounds=None, n_points=50):
        """
        Generate Pareto front for multi-objective optimization
        
        Parameters:
        -----------
        objectives : list of str
            Objectives to optimize, e.g. ['tensile', 'impact', 'degradation']
        bounds : list of tuples, optional
            Parameter bounds
        n_points : int
            Number of points on Pareto front
        
        Returns:
        --------
        list : List of optimal solutions on Pareto front
        """
        # This is a simplified implementation
        # In practice, you might want to use NSGA-II or similar multi-objective algorithms
        
        solutions = []
        
        for _ in range(n_points):
            # Generate random weights for objectives
            weights = np.random.dirichlet(np.ones(len(objectives)))
            
            # Define weighted objective
            def weighted_objective(x):
                phb, tg, crystallinity, relaxation_time = x
                
                try:
                    tensile = self.predictor.predict_tensile_strength(phb, tg, crystallinity, relaxation_time)
                    impact = self.predictor.predict_impact_toughness(phb, tg, crystallinity, relaxation_time)
                    degradation = self.predictor.predict_degradation_rate(phb, tg, crystallinity, 7.4)
                    
                    # Weighted sum of objectives
                    score = 0
                    if 'tensile' in objectives:
                        idx = objectives.index('tensile')
                        score += weights[idx] * tensile/100
                    if 'impact' in objectives:
                        idx = objectives.index('impact')
                        score += weights[idx] * impact
                    if 'degradation' in objectives:
                        idx = objectives.index('degradation')
                        score += weights[idx] * (1 - degradation/10)
                    
                    return -score
                    
                except Exception:
                    return 1e6
            
            # Optimize with current weights
            if bounds is None:
                bounds = [(0, 40), (30, 60), (0, 40), (0.05, 2.0)]
            
            result = differential_evolution(
                weighted_objective,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42
            )
            
            if result.success:
                solutions.append({
                    'formulation': result.x,
                    'weights': weights,
                    'score': -result.fun
                })
        
        return solutions