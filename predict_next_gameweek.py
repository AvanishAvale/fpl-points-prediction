import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


DATA_PATH = 'fpl_data_23_25.csv'
MODEL_PATH = 'best_fpl_model.keras'
SCALER_PATH = 'scaler.pkl'
SEQUENCE_LENGTH = 5

FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]

def main():
    print("Loading resources...")
    df = pd.read_csv(DATA_PATH)
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    
    print("Processing latest player stats...")
    
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].astype(int)
        
    df[FEATURES] = scaler.transform(df[FEATURES])
    
    player_groups = df.groupby('name')
    
    prediction_list = []
    
    for player_name, group in player_groups:
        # Sort by date
        group = group.sort_values('kickoff_time')
        
        if len(group) < SEQUENCE_LENGTH:
            continue
            
        last_5_games = group[FEATURES].values[-SEQUENCE_LENGTH:]
        
        input_seq = last_5_games.reshape(1, SEQUENCE_LENGTH, len(FEATURES))
        
        predicted_scaled = model.predict(input_seq, verbose=0)
        
        prediction_list.append({
            'name': player_name,
            'predicted_scaled': predicted_scaled[0][0]
        })
        
    print(f"\nGenerated predictions for {len(prediction_list)} players.")
    
    pred_df = pd.DataFrame(prediction_list)
    dummy_matrix = np.zeros((len(pred_df), len(FEATURES)))
    
    target_index = FEATURES.index('total_points')
    dummy_matrix[:, target_index] = pred_df['predicted_scaled'].values
    
    real_values = scaler.inverse_transform(dummy_matrix)

    # Extract the real predicted points
    pred_df['predicted_points'] = real_values[:, target_index]
    
   
    print("\n=============================================")
    print("   🤖 FPL AI: PREDICTED DREAM TEAM (Next GW)")
    print("=============================================")
    
    # Sort by highest predicted points
    top_players = pred_df.sort_values(by='predicted_points', ascending=False).head(20)
    
    print(f"{'Player':<25} | {'Predicted Points':<15}")
    print("-" * 45)
    
    for _, row in top_players.iterrows():
        print(f"{row['name']:<25} | {row['predicted_points']:.2f}")

if __name__ == "__main__":
    main()