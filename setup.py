from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pla-phb-predictor",
    version="1.0.0",
    author="Luyao Gao",
    author_email="glydmdmn@gmail.com",
    description="Predictive models for PLA/PHB blend properties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PLA-PHB-Predictive-Modeling",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "data/*.csv"],
    },
)