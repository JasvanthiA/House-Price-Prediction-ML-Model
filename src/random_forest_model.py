import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from datetime import datetime
from scipy.stats import randint, uniform

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
    output_file = os.path.join('reports', 'text', f'random_forest_results_{timestamp}.txt')
    
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
        
        # Create Random Forest model
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', rf)
        ])
        
        # Define parameter distributions for RandomizedSearchCV
        param_distributions = {
            'regressor__n_estimators': randint(100, 1000),
            'regressor__max_depth': randint(10, 100),
            'regressor__min_samples_split': randint(2, 20),
            'regressor__min_samples_leaf': randint(1, 10),
            'regressor__max_features': uniform(0.1, 0.9)
        }
        
        # Create RandomizedSearchCV object
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        output_lines.append("\nStarting RandomizedSearchCV with the following parameters:")
        output_lines.append(f"Number of iterations: 20")
        output_lines.append(f"Cross-validation folds: 5")
        output_lines.append("\nParameter ranges:")
        for param, dist in param_distributions.items():
            output_lines.append(f"{param}: {dist}")
        
        # Fit the RandomizedSearchCV
        random_search.fit(X_train_clean, y_train_clean)
        
        # Get best parameters and score
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        output_lines.append("\nBest parameters found:")
        for param, value in best_params.items():
            output_lines.append(f"{param}: {value}")
        output_lines.append(f"Best cross-validation R-squared score: {best_score:.4f}")
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Make predictions with best model
        y_train_pred = best_model.predict(X_train_clean)
        y_test_pred = best_model.predict(X_test)
        
        # Get feature names after preprocessing
        feature_names = list(numerical_cols)
        if categorical_cols:
            # Get the fitted OneHotEncoder
            cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
            feature_names.extend([
                f"{col}_{val}" for col, vals in 
                zip(categorical_cols, cat_encoder.categories_) 
                for val in vals[1:]
            ])
        
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
        
        # Get feature importance from the Random Forest model
        feature_importance = dict(zip(feature_names, best_model.named_steps['regressor'].feature_importances_))
        
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