import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import read_excel_file, get_raw_data_path

def perform_eda(df):
    """
    Perform exploratory data analysis on the DataFrame
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    """
    # Create reports directory if it doesn't exist
    os.makedirs('reports/text', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Basic information about the dataset
    print("\n=== Basic Information ===")
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print("\nNumerical Columns Summary:\n", df.describe())
    
    # Save basic statistics to a text file
    with open(os.path.join('reports', 'text', 'eda_summary.txt'), 'w') as f:
        f.write("=== Basic Information ===\n")
        f.write(f"\nDataset Shape: {df.shape}\n")
        f.write(f"\nColumns: {df.columns.tolist()}\n")
        f.write("\nData Types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing Values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\n=== Basic Statistics ===\n")
        f.write("\nNumerical Columns Summary:\n")
        f.write(str(df.describe()))

def plot_numerical_distributions(df):
    """
    Create distribution plots for numerical columns
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join('reports', 'figures', f'distribution_{col}.png'))
        plt.close()

def plot_correlation_matrix(df):
    """
    Create correlation matrix heatmap for numerical columns
    
    Args:
        df (pandas.DataFrame): Input DataFrame
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join('reports', 'figures', 'correlation_matrix.png'))
        plt.close()

if __name__ == "__main__":
    # Read the data
    input_file = "Dataset 1.xlsx"
    input_path = get_raw_data_path(input_file)
    df = read_excel_file(input_path)
    
    if df is not None:
        # Perform EDA
        perform_eda(df)
        
        # Create visualizations
        plot_numerical_distributions(df)
        plot_correlation_matrix(df)
        
        print("\nEDA completed! Check the generated files in reports directory:")
        print("- reports/text/eda_summary.txt: Contains basic information and statistics")
        print("- reports/figures/distribution_*.png: Distribution plots for numerical columns")
        print("- reports/figures/correlation_matrix.png: Correlation matrix heatmap")