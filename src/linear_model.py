import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
import os
from datetime import datetime

def remove_outliers(df, y, numerical_cols, threshold=1.5):
    """
    Remove outliers using IQR method while preserving all columns
    
    Args:
        df: Full DataFrame with all columns
        y: Target variable
        numerical_cols: List of numerical column names to check for outliers
        threshold: IQR multiplier for outlier detection
    """
    X_num = df[numerical_cols]
    Q1 = X_num.quantile(0.25)
    Q3 = X_num.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((X_num < (Q1 - threshold * IQR)) | (X_num > (Q3 + threshold * IQR))).any(axis=1)
    return df[outlier_mask], y[outlier_mask]

def main():
    # Create output file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('reports', 'text', f'ml_model_results_{timestamp}.txt')
    
    # Create a list to store output lines
    output_lines = []
    
    # Read the cleaned dataset
    data_path = os.path.join('data', 'processed', 'cleaned_dataset.xlsx')
    output_lines.append(f"Attempting to read file from: {data_path}")
    
    try:
        df = pd.read_excel(data_path)
        output_lines.append("\nDataset loaded successfully!")
        output_lines.append("\nColumns in the dataset:")
        output_lines.append(str(df.columns.tolist()))
        
        # Try to identify the target column (house value column)
        possible_target_columns = [col for col in df.columns if 'house' in col.lower() and 'value' in col.lower()]
        if not possible_target_columns:
            possible_target_columns = [col for col in df.columns if 'price' in col.lower()]
        
        if not possible_target_columns:
            raise ValueError("Could not find the target column (house value/price column). Available columns: " + ", ".join(df.columns))
        
        target_column = possible_target_columns[0]
        output_lines.append(f"\nUsing '{target_column}' as the target variable")
        
        # Separate features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        output_lines.append("\nFeature columns:")
        output_lines.append(str(X.columns.tolist()))
        
        # Get numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = ['ocean_proximity'] if 'ocean_proximity' in X.columns else []
        
        output_lines.append("\nNumerical columns: " + str(list(numerical_cols)))
        output_lines.append("Categorical columns: " + str(categorical_cols))
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ] if categorical_cols else [('num', StandardScaler(), numerical_cols)]
        )
        
        # Split the data with stratification
        if categorical_cols:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=X[categorical_cols[0]]
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        output_lines.append(f"\nTraining set size: {X_train.shape[0]}")
        output_lines.append(f"Test set size: {X_test.shape[0]}")
        
        # Remove outliers from training data (keeping all columns)
        X_train_clean, y_train_clean = remove_outliers(
            X_train, y_train, numerical_cols
        )
        
        output_lines.append(f"Training set size after outlier removal: {X_train_clean.shape[0]}")
        
        # Fit preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train_clean)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names = list(numerical_cols)
        if categorical_cols:
            feature_names.extend([
                f"{col}_{val}" for col, vals in 
                zip(categorical_cols, preprocessor.named_transformers_['cat'].categories_) 
                for val in vals[1:]
            ])
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_processed, y_train_clean)
        
        # Make predictions
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        metrics = {
            'Training Data': {
                'R-squared': r2_score(y_train_clean, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train_clean, y_train_pred)),
                'MAE': mean_absolute_error(y_train_clean, y_train_pred),
                'MSE': mean_squared_error(y_train_clean, y_train_pred)
            },
            'Test Data': {
                'R-squared': r2_score(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred)
            }
        }
        
        # Print and save metrics
        output_lines.append("\nModel Performance Metrics:")
        output_lines.append("-" * 50)
        for dataset, metric_values in metrics.items():
            output_lines.append(f"\n{dataset}:")
            for metric_name, value in metric_values.items():
                output_lines.append(f"{metric_name}: {value:.4f}")
        
        # Print and save feature importance
        output_lines.append("\nFeature Importance:")
        output_lines.append("-" * 50)
        for feature, coef in zip(feature_names, model.coef_):
            output_lines.append(f"{feature}: {coef:.4f}")
            
        # Write all output to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nResults have been saved to: {output_file}")
            
    except Exception as e:
        error_message = f"\nError occurred: {str(e)}"
        output_lines.append(error_message)
        # Write error message to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(error_message)
        print(f"Error log has been saved to: {output_file}")
        raise

if __name__ == "__main__":
    main()