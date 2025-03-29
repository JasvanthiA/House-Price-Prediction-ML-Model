# House Price Prediction Project

This project performs data analysis on housing data, including data preprocessing, exploratory data analysis, and machine learning model implementations for price prediction.

## Project Structure

```
Excel_data_project/
│
├── data/               # Data directory
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
│
├── src/               # Source code
│   ├── data_loader.py           # Data loading utilities
│   ├── data_preprocessing.py     # Data preprocessing operations
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── linear_model.py          # Linear Regression model implementation
│   ├── polynomial_model.py      # Polynomial Regression model
│   ├── elastic_net_polynomial.py # Elastic Net with polynomial features
│   ├── random_forest_model.py   # Random Forest Regressor implementation
│   ├── xgboost_model.py        # XGBoost Regressor implementation
│   └── model_comparison.py      # Model comparison and evaluation
│
├── reports/           # Generated analysis
│   ├── figures/       # Generated plots and visualizations
│   └── text/         # Generated text reports
│
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your raw data file (Dataset 1.xlsx) in the `data/raw/` directory.

2. Run the EDA script:
```bash
python src/eda.py
```

3. Run the preprocessing script:
```bash
python src/data_preprocessing.py
```

4. Train and evaluate models:
   - Linear Regression: `python src/linear_model.py`
   - Polynomial Regression: `python src/polynomial_model.py`
   - Elastic Net with Polynomial Features: `python src/elastic_net_polynomial.py`
   - Random Forest: `python src/random_forest_model.py`
   - XGBoost: `python src/xgboost_model.py`

5. Compare model performances:
```bash
python src/model_comparison.py
```

6. Check the results in:
- Processed data: `data/processed/cleaned_dataset.xlsx`
- EDA reports: `reports/text/eda_summary.txt`
- Model results: `reports/text/model_results.txt`
- Visualizations: `reports/figures/`

## Models Implemented

1. **Linear Regression**: Basic linear regression model with feature selection
2. **Polynomial Regression**: Enhanced linear model with polynomial features
3. **Elastic Net with Polynomial Features**: Regularized regression combining L1 and L2 penalties
4. **Random Forest**: Ensemble learning using random forest algorithm
5. **XGBoost**: Gradient boosting implementation for improved accuracy

Each model implementation includes:
- Data preprocessing
- Feature selection/engineering
- Model training
- Cross-validation
- Performance evaluation
- Results visualization