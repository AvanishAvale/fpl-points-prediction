import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
DATA_PATH = 'fpl_data_23_25.csv'
# The lengths we want to test
SEQUENCE_LENGTHS = [3, 5, 10]
FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET = 'total_points'

def create_sequences(df, seq_length, scaler):
    """
    Custom sequence generator for the experiment loop.
    """
    # 1. Scale Data (Fit on whole DF for consistency in this experiment)
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])
    
    sequences = []
    targets = []
    
    # 2. Group by Player
    player_groups = df_scaled.groupby('name')
    
    for player_name, group in player_groups:
        group = group.sort_values('kickoff_time')
        data_values = group[FEATURES].values
        target_values = group[TARGET].values
        
        if len(data_values) < seq_length + 1:
            continue
            
        for i in range(len(data_values) - seq_length):
            sequences.append(data_values[i:i+seq_length])
            targets.append(target_values[i+seq_length])
            
    return np.array(sequences), np.array(targets)

def build_model(seq_length, n_features):
    """
    Builds the standard LSTM with a dynamic input shape.
    """
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Initial Preprocessing (Encoding, Sorting)
    if 'was_home' in df.columns:
        df['was_home'] = df['was_home'].astype(int)
    
    # Fit Scaler ONCE on the entire dataset to ensure fair comparison
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES])

    # Split Data
    train_df = df[df['season'] == '2023-24']
    test_df = df[df['season'] == '2024-25']
    
    results = {}

    # --- EXPERIMENT LOOP ---
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n\n========================================")
        print(f"   TESTING SEQUENCE LENGTH: {seq_len}")
        print(f"========================================")
        
        # 1. Create Sequences for this length
        print(f"Generating {seq_len}-week sequences...")
        X_train, y_train = create_sequences(train_df, seq_len, scaler)
        X_test, y_test = create_sequences(test_df, seq_len, scaler)
        
        print(f"Training Samples: {len(X_train)}")
        
        # 2. Build Model
        model = build_model(seq_len, len(FEATURES))
        
        # 3. Train
        early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30, # Slightly fewer epochs for speed
            batch_size=32,
            callbacks=[early_stop],
            verbose=0 # Silent training, we only care about the result
        )
        
        # 4. Evaluate (Scaled MAE)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"-> Resulting MAE (Scaled): {mae:.5f}")
        
        # 5. Calculate REAL MAE (Un-scaled)
        # We predict on test set to get the real error
        preds_scaled = model.predict(X_test, verbose=0)
        
        # Inverse Transform Trick
        dummy_pred = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy_actual = np.zeros((len(y_test), len(FEATURES)))
        target_idx = FEATURES.index(TARGET)
        
        dummy_pred[:, target_idx] = preds_scaled.flatten()
        dummy_actual[:, target_idx] = y_test.flatten()
        
        real_preds = scaler.inverse_transform(dummy_pred)[:, target_idx]
        real_actuals = scaler.inverse_transform(dummy_actual)[:, target_idx]
        
        real_mae = np.mean(np.abs(real_preds - real_actuals))
        print(f"-> Resulting MAE (Real Points): {real_mae:.4f}")
        
        results[seq_len] = real_mae

    # --- FINAL COMPARISON ---
    print("\n\n========================================")
    print("   EXPERIMENT RESULTS SUMMARY")
    print("========================================")
    print(f"{'Seq Length':<12} | {'Real MAE (Points)':<15}")
    print("-" * 35)
    
    best_len = None
    best_score = float('inf')
    
    for length, score in results.items():
        print(f"{length:<12} | {score:.4f}")
        if score < best_score:
            best_score = score
            best_len = length
            
    print("-" * 35)
    print(f"WINNER: {best_len} Gameweeks Lookback")

    # Plot
    plt.bar(results.keys(), results.values(), color=['red', 'blue', 'green'])
    plt.xlabel('Sequence Length (Weeks)')
    plt.ylabel('Mean Absolute Error (Lower is Better)')
    plt.title('Impact of History Length on Accuracy')
    plt.xticks(SEQUENCE_LENGTHS)
    plt.savefig('experiment_results.png')
    print("Graph saved to experiment_results.png")

if __name__ == "__main__":
    main()