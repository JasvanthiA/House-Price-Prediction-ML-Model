# Setup Instructions

This document provides detailed instructions for setting up the House Price Prediction ML Model project on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/JasvanthiA/House-Price-Prediction-ML-Model.git
cd House-Price-Prediction-ML-Model
```

2. **Create a Virtual Environment**
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure Setup

1. **Create Required Directories**
The project uses the following directory structure:
```
House-Price-Prediction-ML-Model/
├── data/
│   ├── raw/         # Store your raw dataset here
│   └── processed/   # Processed datasets will be saved here
├── reports/
│   ├── figures/     # Generated visualizations
│   └── text/        # Generated reports and metrics
└── src/            # Source code
```

2. **Data Setup**
- Place your raw housing dataset in the `data/raw/` directory
- The processed data will be automatically saved in `data/processed/`

## Model Configuration

1. **Environment Variables**
If needed, create a `.env` file in the root directory with any required environment variables:
```
# Example .env file
MODEL_OUTPUT_PATH=reports/
DATA_PATH=data/raw/housing_data.csv
```

2. **Model Parameters**
Model parameters can be adjusted in the respective model files in the `src/` directory.

## Running the Models

1. **Data Preprocessing**
```bash
python src/data_preprocessing.py
```

2. **Training Models**
```bash
# Run individual models
python src/polynomial_model.py
python src/elastic_net_polynomial.py
python src/random_forest_model.py
python src/xgboost_model.py

# Run model comparison
python src/model_comparison.py
```

## Troubleshooting

Common issues and their solutions:

1. **Package Installation Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

2. **Memory Issues**
- Reduce batch size in model parameters
- Use smaller dataset for testing
- Close other memory-intensive applications

3. **CUDA/GPU Issues**
- Ensure CUDA toolkit is installed for GPU support
- Check GPU compatibility with installed packages
- Try running on CPU if GPU issues persist

## Additional Resources

- [Python Installation Guide](https://www.python.org/downloads/)
- [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [Virtual Environment Tutorial](https://docs.python.org/3/tutorial/venv.html)

## Support

If you encounter any issues:
1. Check the [Issues](https://github.com/JasvanthiA/House-Price-Prediction-ML-Model/issues) page
2. Create a new issue with detailed information about your problem
3. Refer to CONTRIBUTING.md for guidelines on reporting issues