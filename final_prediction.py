import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input  # <--- Using GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION (The Winning Combination) ---
DATA_PATH = 'fpl_data_23_25.csv'
SEQUENCE_LENGTH = 3  # Winner of Experiment 1
FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET = 'total_points'

def create_sequences(df, seq_length, scaler):
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])
    
    sequences = []
    targets = []
    player_groups = df_scaled.groupby('name')
    
    for _, group in player_groups:
        group = group.sort_values('kickoff_time')
        data = group[FEATURES].values
        target = group[TARGET].values
        
        if len(data) < seq_length + 1:
            continue
            
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(target[i+seq_length])
            
    return np.array(sequences), np.array(targets)

def build_winning_model(input_shape):
    """
    The Champion Model: GRU
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False), # GRU beat LSTM in our tests
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].astype(int)

    # 1. SETUP SCALER
    print("Fitting Scaler...")
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES])
    
    # 2. PREPARE TRAINING DATA
    print(f"Preparing sequences (Length: {SEQUENCE_LENGTH})...")
    X, y = create_sequences(df, SEQUENCE_LENGTH, scaler)
    
    # 3. TRAIN FINAL MODEL
    print("Training the Champion GRU Model...")
    model = build_winning_model((SEQUENCE_LENGTH, len(FEATURES)))
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)
    print("Model trained.")

    # 4. PREDICT NEXT GAMEWEEK
    print("\nGenerating predictions for the NEXT match...")
    
    latest_player_data = []
    player_groups = df.groupby('name')
    
    for player_name, group in player_groups:
        group = group.sort_values('kickoff_time')
        
        if len(group) < SEQUENCE_LENGTH:
            continue
            
        # Get last 3 games raw data
        last_games = group.tail(SEQUENCE_LENGTH).copy()
        last_games[FEATURES] = scaler.transform(last_games[FEATURES])
        
        # Reshape for GRU (1, 3, 14)
        input_seq = last_games[FEATURES].values.reshape(1, SEQUENCE_LENGTH, len(FEATURES))
        
        # Predict
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        
        latest_player_data.append({
            'name': player_name,
            'pred_scaled': pred_scaled
        })

    # 5. INVERSE TRANSFORM PREDICTIONS
    print("Refining results...")
    results_df = pd.DataFrame(latest_player_data)
    
    dummy = np.zeros((len(results_df), len(FEATURES)))
    target_idx = FEATURES.index(TARGET)
    dummy[:, target_idx] = results_df['pred_scaled'].values
    
    real_points = scaler.inverse_transform(dummy)[:, target_idx]
    results_df['predicted_points'] = real_points
    
    # 6. SHOW THE DREAM TEAM
    print("\n\n=======================================================")
    print(f"   🔮 FPL AI DREAM TEAM (Model: GRU | Lookback: {SEQUENCE_LENGTH}w)")
    print("=======================================================")
    
    active_players = results_df[results_df['predicted_points'] > 0.5]
    top_20 = active_players.sort_values('predicted_points', ascending=False).head(20)
    
    print(f"{'Rank':<5} | {'Player Name':<25} | {'Predicted Points'}")
    print("-" * 50)
    
    for i, (idx, row) in enumerate(top_20.iterrows()):
        print(f"{i+1:<5} | {row['name']:<25} | {row['predicted_points']:.2f}")

if __name__ == "__main__":
    main()