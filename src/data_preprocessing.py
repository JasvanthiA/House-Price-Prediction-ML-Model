import pandas as pd
import numpy as np
from data_loader import read_excel_file, get_raw_data_path, get_processed_data_path

def preprocess_data(df):
    """
    Preprocess the dataset by:
    1. Imputing missing values in total_bedrooms
    2. Dropping total_rooms column
    3. Applying log transformation to numerical columns (while keeping original column names)
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_cleaned = df.copy()
    
    # 1. Impute missing values in total_bedrooms with median
    df_cleaned['total_bedrooms'] = df_cleaned['total_bedrooms'].fillna(df_cleaned['total_bedrooms'].median())
    
    # 2. Drop total_rooms column
    df_cleaned = df_cleaned.drop('total_rooms', axis=1)
    
    # 3. Apply log transformation to numerical columns (keeping original names)
    numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    
    # Add small constant to avoid log(0) for columns that might have zeros
    for col in numerical_cols:
        if col != 'longitude' and col != 'latitude':  # Skip coordinates as they can be negative
            min_val = df_cleaned[col].min()
            if min_val <= 0:
                df_cleaned[col] = df_cleaned[col] + abs(min_val) + 1
            df_cleaned[col] = np.log1p(df_cleaned[col])
            # No longer renaming columns after log transformation
    
    return df_cleaned

def save_preprocessed_data(df, output_file):
    """
    Save the preprocessed DataFrame to an Excel file
    
    Args:
        df (pandas.DataFrame): Preprocessed DataFrame
        output_file (str): Name of the output Excel file
    """
    try:
        output_path = get_processed_data_path(output_file)
        df.to_excel(output_path, index=False)
        print(f"\nPreprocessed data saved successfully to {output_path}")
        print("\nShape of preprocessed data:", df.shape)
        print("\nColumns in preprocessed data:", df.columns.tolist())
    except Exception as e:
        print(f"Error saving preprocessed data: {str(e)}")

if __name__ == "__main__":
    # Read the data
    input_file = "Dataset 1.xlsx"
    input_path = get_raw_data_path(input_file)
    df = read_excel_file(input_path)
    
    if df is not None:
        # Perform preprocessing
        df_cleaned = preprocess_data(df)
        
        # Save preprocessed data
        output_file = "cleaned_dataset.xlsx"
        save_preprocessed_data(df_cleaned, output_file)