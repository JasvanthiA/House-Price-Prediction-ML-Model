import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def extract_r2_scores():
    # Dictionary to store model names and their R-squared scores
    model_scores = {}
    
    # Read results from each model file
    results_dir = 'reports/text'
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(results_dir, filename)
            
            # Determine model name from filename
            if 'ml_model' in filename.lower():
                model_name = 'Linear Regression'
            elif 'polynomial' in filename.lower():
                model_name = 'Polynomial Regression'
            elif 'elastic' in filename.lower():
                model_name = 'Elastic Net Polynomial'
            elif 'random_forest' in filename.lower():
                model_name = 'Random Forest'
            elif 'xgboost' in filename.lower():
                model_name = 'XGBoost'
            else:
                continue
            
            # Read the file and extract R-squared score
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Look for test data R-squared score
                if 'Test Data' in content:
                    test_section = content.split('Test Data')[1].split('\n')
                    for line in test_section:
                        if 'R-squared' in line:
                            r2_score = float(line.split(':')[1].strip())
                            model_scores[model_name] = r2_score
                            break
    
    return model_scores

def create_comparison_chart(model_scores):
    # Create figure with a specific size
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    # Create bar plot with custom colors
    bars = plt.bar(models, scores)
    
    # Customize the plot
    plt.title('Model Comparison: R-squared Scores', fontsize=14, pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('R-squared Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create reports/figures directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'reports/figures/model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart has been saved to: reports/figures/model_comparison_{timestamp}.png")

def main():
    # Extract R-squared scores from all model results
    model_scores = extract_r2_scores()
    
    if not model_scores:
        print("No model results found. Please run the models first.")
        return
    
    # Create and save the comparison chart
    create_comparison_chart(model_scores)
    
    # Print the scores
    print("\nR-squared Scores for each model:")
    for model, score in model_scores.items():
        print(f"{model}: {score:.3f}")

if __name__ == "__main__":
    main()