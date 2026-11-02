import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU # Needed for loading the model

# --- CONFIGURATION ---
DATA_PATH = 'fpl_data_23_25.csv' # Raw data for EDA
MODEL_PATH = 'best_gru_model.keras'
SCALER_PATH = 'scaler.pkl'
FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET = 'total_points'

# --- SET STYLE ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def plot_correlation_heatmap(df):
    """Plot 1: Which features matter?"""
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Select only numeric columns for correlation
    corr_cols = FEATURES
    corr_matrix = df[corr_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, center=0, linewidths=.5)
    
    plt.title('Feature Correlation Matrix', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('viz_1_correlations.png', dpi=300)
    print("-> Saved viz_1_correlations.png")

def plot_points_distribution(df):
    """Plot 2: What are we trying to predict?"""
    print("Generating Points Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Filter for active players only to make the plot readable
    active_df = df[df['minutes'] > 0]
    
    sns.histplot(active_df['total_points'], bins=30, kde=True, color='teal')
    
    plt.title('Distribution of FPL Points (Active Players)', fontsize=16, weight='bold')
    plt.xlabel('Total Points Scored')
    plt.ylabel('Frequency')
    plt.axvline(active_df['total_points'].mean(), color='red', linestyle='--', label=f"Mean: {active_df['total_points'].mean():.2f}")
    plt.legend()
    plt.savefig('viz_2_distribution.png', dpi=300)
    print("-> Saved viz_2_distribution.png")

def plot_experiment_results():
    """Plot 3 & 4: Experiment Comparisons (Hardcoded from your results)"""
    print("Generating Experiment Charts...")
    
    # --- EXPERIMENT A: SEQUENCE LENGTH ---
    lengths = ['3 Weeks', '5 Weeks', '10 Weeks']
    # Your real results: 2.05, 2.09, 2.29
    mae_scores = [2.05, 2.09, 2.29] 
    
    plt.figure(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c'] # Green (Best), Blue, Red (Worst)
    bars = plt.bar(lengths, mae_scores, color=colors)
    
    plt.title('Impact of History Length on Error (MAE)', fontsize=16, weight='bold')
    plt.ylabel('Mean Absolute Error (Lower is Better)')
    plt.ylim(1.8, 2.4) # Zoom in to show difference
    
    # Add labels on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', weight='bold')
        
    plt.savefig('viz_3_experiment_length.png', dpi=300)
    print("-> Saved viz_3_experiment_length.png")

    # --- EXPERIMENT B: ARCHITECTURE ---
    models = ['GRU', 'LSTM', 'Bi-LSTM']

    arch_scores = [0.0638, 0.0659, 0.0681]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, arch_scores, color=['#2ecc71', '#f1c40f', '#e74c3c'])
    
    plt.title('Model Architecture Performance (Scaled Error)', fontsize=16, weight='bold')
    plt.ylabel('Scaled MAE (Lower is Better)')
    plt.ylim(0.060, 0.070) # Zoom in
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0002, round(yval, 4), ha='center', va='bottom', weight='bold')

    plt.savefig('viz_4_experiment_model.png', dpi=300)
    print("-> Saved viz_4_experiment_model.png")

def plot_predictions_vs_actuals():
    """Plot 5: Scatter Plot of Truth"""

    try:
        print("Generating Prediction Scatter Plot...")
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        scaler = joblib.load(SCALER_PATH)
        

        model = load_model(MODEL_PATH)
        preds_scaled = model.predict(X_test, verbose=0).flatten()
        
        # Unscale
        dummy_pred = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy_actual = np.zeros((len(y_test), len(FEATURES)))
        target_idx = FEATURES.index(TARGET)
        
        dummy_pred[:, target_idx] = preds_scaled
        dummy_actual[:, target_idx] = y_test
        
        real_preds = scaler.inverse_transform(dummy_pred)[:, target_idx]
        real_actuals = scaler.inverse_transform(dummy_actual)[:, target_idx]
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(real_actuals, real_preds, alpha=0.3, color='purple', s=10)
        
        # Perfect prediction line
        max_val = max(real_actuals.max(), real_preds.max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        plt.title('Predicted vs Actual Points (Test Set)', fontsize=16, weight='bold')
        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.legend()
        plt.xlim(0, 25) 
        plt.ylim(0, 25)
        plt.savefig('viz_5_pred_vs_actual.png', dpi=300)
        print("-> Saved viz_5_pred_vs_actual.png")
        
    except Exception as e:
        print(f"Skipping Plot 5 due to missing data files: {e}")

def main():
    # Load raw data for EDA
    df = pd.read_csv(DATA_PATH)
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].astype(int)
        
    plot_correlation_heatmap(df)
    plot_points_distribution(df)
    plot_experiment_results()
    plot_predictions_vs_actuals()
    print("\nAll visualizations created successfully!")

if __name__ == "__main__":
    main()