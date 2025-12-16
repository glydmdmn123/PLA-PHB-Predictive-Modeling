# PLA/PHB Predictive Modeling Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains predictive models for mechanical properties and degradation behavior of PLA/PHB blends, developed as part of the PhD thesis:

**"Fracture and Degradation Behavior of PLA/PHB Blends: From PHB Ratio Effects to Multi-Environment Analysis with Experimental-Data-Driven Framework for Biomedical Scaffolds"**

*PhD Thesis, Aalborg University, 2027*

## Features

- **Predictive Models**: Estimate tensile strength, impact toughness, and degradation rate based on material composition and properties
- **Reverse Design**: Find optimal formulations meeting specific clinical requirements
- **Feature Analysis**: Identify key material parameters controlling performance
- **Uncertainty Quantification**: Provide prediction intervals for reliability assessment

## Installation

```bash
# Clone the repository
git clone https://github.com/glydmdmn/PLA-PHB-Predictive-Modeling.git
cd PLA-PHB-Predictive-Modeling

# Install dependencies
pip install -r requirements.txt

# Install as a package (optional)
pip install -e .

## Quick Start
from src.prediction import PLAPHBPredictor

# Initialize predictor
predictor = PLAPHBPredictor()

# Predict properties for a formulation
properties = predictor.predict(
    phb_content=20.0,  # PHB percentage
    tg=40.0,           # Glass transition temperature (°C)
    crystallinity=8.0, # Crystallinity (%)
    relaxation_time=0.15,  # Relaxation time (s)
    ph=7.4            # Environmental pH
)

print(properties)

##Project Structure
PLA-PHB-Predictive-Modeling/
├── data/              # Experimental and augmented datasets
├── models/            # Trained model files (.pkl)
├── src/               # Source code modules
├── notebooks/         # Example notebooks
├── tests/            # Unit tests
└── docs/             # Documentation

##Citation
Gao, L. (2027). Fracture and Degradation Behavior of PLA/PHB Blends: From PHB Ratio Effects to Multi-Environment Analysis with Experimental-Data-Driven Framework for Biomedical Scaffolds [PhD Thesis]. 
Aalborg University.