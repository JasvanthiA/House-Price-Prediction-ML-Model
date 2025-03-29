import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import os
from datetime import datetime
from scipy.stats import randint, uniform, loguniform

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
    output_file = os.path.join('reports', 'text', f'xgboost_results_{timestamp}.txt')
    
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
        
        # Drop specified columns
        columns_to_drop = ['total_bedrooms', 'households']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        output_lines.append(f"\nDropped columns: {columns_to_drop}")
        
        # Separate features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        output_lines.append("\nFeature columns after dropping:")
        output_lines.append(str(X.columns.tolist()))
        
        # Get numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = ['ocean_proximity'] if 'ocean_proximity' in X.columns else []
        
        # Convert categorical columns to category type
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        
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
        
        # Further split training data into train and validation for early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        output_lines.append(f"\nTraining set size: {X_train_final.shape[0]}")
        output_lines.append(f"Validation set size: {X_val.shape[0]}")
        output_lines.append(f"Test set size: {X_test.shape[0]}")
        
        # Remove outliers from training data (keeping all columns)
        X_train_clean, y_train_clean = remove_outliers(
            X_train_final, y_train_final, numerical_cols
        )
        
        output_lines.append(f"Training set size after outlier removal: {X_train_clean.shape[0]}")
        
        # Fit the preprocessor on the full training data
        preprocessor.fit(X_train_clean)
        
        # Transform the data
        X_train_processed = preprocessor.transform(X_train_clean)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names = list(numerical_cols)
        if categorical_cols:
            # Get the fitted OneHotEncoder
            cat_encoder = preprocessor.named_transformers_['cat']
            feature_names.extend([
                f"{col}_{val}" for col, vals in 
                zip(categorical_cols, cat_encoder.categories_) 
                for val in vals[1:]
            ])
        
        # Create XGBoost model with early stopping and categorical feature support
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=15,
            eval_metric='rmse',
            # Set the specific parameters that worked well
            reg_lambda=5,
            reg_alpha=0.1,
            n_estimators=200,
            min_child_weight=10,
            max_depth=5,
            learning_rate=0.1,
            gamma=1
        )
        
        # Since we're using fixed parameters, we don't need RandomizedSearchCV
        # Just fit the model directly
        output_lines.append("\nUsing fixed parameters:")
        output_lines.append("reg_lambda: 5")
        output_lines.append("reg_alpha: 0.1")
        output_lines.append("n_estimators: 200")
        output_lines.append("min_child_weight: 10")
        output_lines.append("max_depth: 5")
        output_lines.append("learning_rate: 0.1")
        output_lines.append("gamma: 1")
        
        # Fit the model
        xgb_model.fit(
            X_train_processed, y_train_clean,
            eval_set=[(X_val_processed, y_val)],
            verbose=False
        )
        
        # Get validation score
        val_pred = xgb_model.predict(X_val_processed)
        val_score = r2_score(y_val, val_pred)
        output_lines.append(f"\nValidation R-squared score: {val_score:.4f}")
        
        # Make predictions with model
        y_train_pred = xgb_model.predict(X_train_processed)
        y_test_pred = xgb_model.predict(X_test_processed)
        
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
        output_lines.append("\nFeature Importance (Top 10 features):")
        output_lines.append("-" * 50)
        
        # Get feature importance from the XGBoost model
        feature_importance = dict(zip(feature_names, xgb_model.feature_importances_))
        
        # Sort by importance and get top 10
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for feature, importance in top_features:
            output_lines.append(f"{feature}: {importance:.4f}")
            
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