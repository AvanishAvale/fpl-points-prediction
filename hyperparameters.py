import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
X_TRAIN_PATH = 'X_train.npy'
Y_TRAIN_PATH = 'y_train.npy'
X_TEST_PATH = 'X_test.npy'
Y_TEST_PATH = 'y_test.npy'

# The Grid of Hyperparameters to test
DROPOUT_RATES = [0.1, 0.2, 0.3]
GRU_UNITS = [32, 64, 128]
LEARNING_RATES = [0.001, 0.0001]

def build_model(input_shape, units, dropout, lr):
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

def main():
    print("Loading data...")
    try:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
    except FileNotFoundError:
        print("Data not found! Run preprocessing.py first.")
        return

    input_shape = (X_train.shape[1], X_train.shape[2])
    
    results = []
    
    print(f"\nStarting Grid Search on {len(DROPOUT_RATES) * len(GRU_UNITS) * len(LEARNING_RATES)} combinations...")
    
    best_mae = float('inf')
    best_config = None

    # --- GRID SEARCH LOOP ---
    for dr in DROPOUT_RATES:
        for units in GRU_UNITS:
            for lr in LEARNING_RATES:
                config_name = f"Units={units} | Drop={dr} | LR={lr}"
                print(f"\nTesting: {config_name}")
                
                model = build_model(input_shape, units, dr, lr)
                
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20, # Shorter epochs for speed
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0 
                )
                
                loss, mae = model.evaluate(X_test, y_test, verbose=0)
                print(f"   -> Result MAE: {mae:.5f}")
                
                results.append({
                    'Units': units,
                    'Dropout': dr,
                    'LR': lr,
                    'MAE': mae
                })
                
                if mae < best_mae:
                    best_mae = mae
                    best_config = config_name

    # --- REPORTING ---
    print("\n\n========================================")
    print("   HYPERPARAMETER RESULTS")
    print("========================================")
    
    results_df = pd.DataFrame(results).sort_values(by='MAE')
    print(results_df)
    
    print(f"\n🏆 WINNER: {best_config}")
    print(f"   With Scaled MAE: {best_mae:.5f}")
    
    results_df.to_csv('hyperparameter_results.csv', index=False)
    print("Results saved to hyperparameter_results.csv")
    
    plt.figure(figsize=(10, 6))
    results_df['Config'] = results_df.apply(lambda x: f"U:{int(x.Units)} D:{x.Dropout} L:{x.LR}", axis=1)
    
    plt.barh(results_df['Config'].head(10), results_df['MAE'].head(10), color='teal')
    plt.xlabel('Scaled MAE (Lower is Better)')
    plt.title('Top 10 Hyperparameter Configurations')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('viz_9_hyperparameters.png')
    print("Graph saved to viz_9_hyperparameters.png")

if __name__ == "__main__":
    main()