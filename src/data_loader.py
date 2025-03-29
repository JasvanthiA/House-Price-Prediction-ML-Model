import pandas as pd
import os

def read_excel_file(file_path):
    """
    Read an Excel file and return its contents as a pandas DataFrame
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Contents of the Excel file
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

def get_raw_data_path(filename):
    """
    Get the full path for a file in the raw data directory
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Full path to the file
    """
    return os.path.join('data', 'raw', filename)

def get_processed_data_path(filename):
    """
    Get the full path for a file in the processed data directory
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Full path to the file
    """
    return os.path.join('data', 'processed', filename)