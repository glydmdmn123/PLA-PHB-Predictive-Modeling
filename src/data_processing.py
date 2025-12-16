"""
Data processing module for PLA/PHB predictive modeling
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

def load_experimental_data(filepath="data/experimental_data.csv"):
    """Load experimental data from CSV file"""
    df = pd.read_csv(filepath)
    return df

def augment_data_with_physics(df, n_augment=50):
    """Augment data using physics-constrained interpolation"""
    augmented_rows = []
    
    for i in range(len(df) - 1):
        phb_start = df.iloc[i]['PHB_wt%']
        phb_end = df.iloc[i + 1]['PHB_wt%']
        
        # Generate intermediate PHB values
        phb_values = np.linspace(phb_start, phb_end, n_augment // (len(df) - 1) + 2)[1:-1]
        
        for phb in phb_values:
            new_row = {'PHB_wt%': phb}
            
            for col in df.columns:
                if col != 'PHB_wt%':
                    # Linear interpolation between data points
                    f = interp1d(df['PHB_wt%'], df[col], kind='linear', fill_value='extrapolate')
                    base_value = float(f(phb))
                    
                    # Add small noise based on measurement uncertainty
                    noise = np.random.normal(0, max(abs(base_value) * 0.02, 0.1))
                    new_row[col] = base_value + noise
            
            augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return combined_df

def prepare_features_for_prediction(df, target_property):
    """Prepare features for model training/prediction"""
    if target_property == 'degradation':
        # For degradation, we need to create pH variations
        features = []
        targets = []
        
        for _, row in df.iterrows():
            # Acidic condition
            features.append([
                row['PHB_wt%'],
                row['Tg_C'],
                row['Crystallinity_%'],
                3.0  # pH for acidic
            ])
            targets.append(row['Weight_loss_pH3_60d_%'])
            
            # Neutral condition
            features.append([
                row['PHB_wt%'],
                row['Tg_C'],
                row['Crystallinity_%'],
                7.4  # pH for neutral
            ])
            targets.append(row['Weight_loss_pH7_60d_%'])
        
        feature_names = ['PHB_wt%', 'Tg_C', 'Crystallinity_%', 'pH']
    
    else:
        # For mechanical properties
        feature_names = ['PHB_wt%', 'Tg_C', 'Crystallinity_%', 'Relaxation_time_s']
        features = df[feature_names].values
        targets = df[f'{target_property}_MPa' if 'strength' in target_property else 
                    f'{target_property}_kJ_m2'].values
    
    return np.array(features), np.array(targets), feature_names