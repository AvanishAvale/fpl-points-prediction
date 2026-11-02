import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU

# --- CONFIGURATION ---
DATA_PATH = 'fpl_data_23_25.csv'
MODEL_PATH = 'best_gru_model.keras' 
SCALER_PATH = 'scaler.pkl'
FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET = 'total_points'

sns.set_style("whitegrid")

def plot_residuals(y_true, y_pred):
    """
    Viz 6: Error Residuals
    Shows if the model systematically over/under predicts.
    """
    print("Generating Residuals Plot...")
    residuals = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.axvline(np.mean(residuals), color='black', linestyle='-', label=f'Mean Error: {np.mean(residuals):.2f}')
    
    plt.title('Error Distribution (Residuals)', fontsize=16, weight='bold')
    plt.xlabel('Prediction Error (Points)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(-10, 10)
    
    plt.savefig('viz_6_residuals.png', dpi=300)
    print("-> Saved viz_6_residuals.png")

def plot_player_case_study(df, player_name, model, scaler):
    """
    Viz 7: The "Star Player" Tracker (Now with Averages!)
    """
    print(f"Generating Case Study for {player_name}...")
    
    # Filter for the player
    player_df = df[df['name'].str.contains(player_name, case=False)].copy()
    player_df = player_df.sort_values('kickoff_time')
    
    if len(player_df) < 5:
        print(f"Not enough data for {player_name}")
        return

    # Scale data
    scaled_features = scaler.transform(player_df[FEATURES])
    
    dates = []
    actuals = []
    predictions = []
    SEQ_LEN = 3 
    
    # Walk through the season
    for i in range(len(scaled_features) - SEQ_LEN):
        seq = scaled_features[i:i+SEQ_LEN].reshape(1, SEQ_LEN, len(FEATURES))
        actual_scaled = scaled_features[i+SEQ_LEN][FEATURES.index(TARGET)]
        pred_scaled = model.predict(seq, verbose=0)[0][0]
        
        # Unscale
        dummy_pred = np.zeros((1, len(FEATURES)))
        dummy_act = np.zeros((1, len(FEATURES)))
        target_idx = FEATURES.index(TARGET)
        
        dummy_pred[0, target_idx] = pred_scaled
        dummy_act[0, target_idx] = actual_scaled
        
        pred_real = scaler.inverse_transform(dummy_pred)[0, target_idx]
        act_real = scaler.inverse_transform(dummy_act)[0, target_idx]
        
        dates.append(player_df.iloc[i+SEQ_LEN]['kickoff_time'])
        predictions.append(pred_real)
        actuals.append(act_real)

    avg_actual = np.mean(actuals)
    avg_pred = np.mean(predictions)
    total_actual = np.sum(actuals)
    total_pred = np.sum(predictions)

    print(f"\n--- STATS FOR {player_name.upper()} ---")
    print(f"Average Actual Points:    {avg_actual:.2f}")
    print(f"Average Predicted Points: {avg_pred:.2f}")
    print(f"Total Actual Points:      {total_actual:.2f}")
    print(f"Total Predicted Points:   {total_pred:.2f}")
    print(f"-----------------------------------------\n")
        
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(dates)), actuals, marker='o', label=f'Actual (Avg: {avg_actual:.1f})', color='black', alpha=0.5)
    plt.plot(range(len(dates)), predictions, marker='x', label=f'AI Prediction (Avg: {avg_pred:.1f})', color='#e74c3c', linewidth=2)
    
    plt.title(f'Case Study: {player_name} (2024-25 Season)', fontsize=16, weight='bold')
    plt.xlabel('Gameweeks')
    plt.ylabel('Points')
    plt.legend()
    plt.savefig('viz_7_case_study.png', dpi=300)
    print("-> Saved viz_7_case_study.png")

def plot_feature_scatter(df):
    """
    Viz 8: Feature Impact (ICT Index vs Points)
    Justifies why we use advanced stats.
    """
    print("Generating Feature Scatter Plot...")
    
    plt.figure(figsize=(10, 6))
    
    sample = df.sample(n=2000, random_state=42)
    
    sns.scatterplot(data=sample, x='ict_index', y='total_points', alpha=0.5, hue='total_points', palette='viridis', legend=False)
    sns.regplot(data=sample, x='ict_index', y='total_points', scatter=False, color='red', line_kws={'label': 'Trend Line'})
    
    plt.title('Why "ICT Index" Matters: Strong Correlation with Points', fontsize=16, weight='bold')
    plt.xlabel('ICT Index (Influence, Creativity, Threat)')
    plt.ylabel('Total FPL Points')
    plt.legend()
    
    plt.savefig('viz_8_feature_impact.png', dpi=300)
    print("-> Saved viz_8_feature_impact.png")

def main():
    # Load everything
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure was_home is int
        if 'was_home' in df.columns:
            df['was_home'] = df['was_home'].astype(int)
            
        scaler = joblib.load(SCALER_PATH)
        model = load_model(MODEL_PATH)
        
        # Load Test Data for Residuals
        y_test = np.load('y_test.npy')
        X_test = np.load('X_test.npy')
        
        # Generate Predictions for Residuals
        print("Running batch predictions...")
        preds_scaled = model.predict(X_test, verbose=0).flatten()
        
        # Unscale residuals
        dummy_pred = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy_act = np.zeros((len(y_test), len(FEATURES)))
        target_idx = FEATURES.index(TARGET)
        dummy_pred[:, target_idx] = preds_scaled
        dummy_act[:, target_idx] = y_test
        
        y_pred_real = scaler.inverse_transform(dummy_pred)[:, target_idx]
        y_true_real = scaler.inverse_transform(dummy_act)[:, target_idx]
        
        # --- GENERATE PLOTS ---
        plot_residuals(y_true_real, y_pred_real)
    
        plot_player_case_study(df, "Haaland", model, scaler)
        
        plot_feature_scatter(df)
        
        print("\nAll extra visualizations created!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the training scripts first!")

if __name__ == "__main__":
    main()