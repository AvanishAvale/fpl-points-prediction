import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = 'best_fpl_model.keras'
SCALER_PATH = 'scaler.pkl'
X_TEST_PATH = 'X_test.npy'
Y_TEST_PATH = 'y_test.npy'

FEATURES = [
    'minutes', 'total_points', 'goals_scored', 'assists', 
    'clean_sheets', 'goals_conceded', 'saves', 'bonus', 
    'influence', 'creativity', 'threat', 'ict_index', 'value', 'was_home'
]
TARGET_INDEX = FEATURES.index('total_points') # Find which column is the target

def main():
    print("Loading resources...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("Making predictions...")
    # The model outputs a single number (the scaled points)
    predictions_scaled = model.predict(X_test)

    
    dummy_pred = np.zeros((len(predictions_scaled), len(FEATURES)))
    dummy_actual = np.zeros((len(y_test), len(FEATURES)))
    
    # 2. Put our predictions into the 'total_points' column (index 1)
    dummy_pred[:, TARGET_INDEX] = predictions_scaled.flatten()
    dummy_actual[:, TARGET_INDEX] = y_test.flatten()
    
    # 3. Inverse transform the whole matrix
    pred_inverse = scaler.inverse_transform(dummy_pred)
    actual_inverse = scaler.inverse_transform(dummy_actual)
    
    # 4. Extract just the 'total_points' column
    real_predictions = pred_inverse[:, TARGET_INDEX]
    real_actuals = actual_inverse[:, TARGET_INDEX]

    # --- EVALUATION ---
    print("\n--- REAL WORLD RESULTS ---")
    
    # Calculate Real MAE
    mae = np.mean(np.abs(real_predictions - real_actuals))
    print(f"Real Mean Absolute Error: {mae:.2f} points")
    print("(This means on average, the prediction is off by this many points)")

    # Show some examples
    print("\nExample Predictions vs Reality:")
    print(f"{'Predicted':<12} | {'Actual':<10} | {'Error':<10}")
    print("-" * 35)
    
    for i in range(10): # Show first 10 examples
        pred = real_predictions[i]
        act = real_actuals[i]
        print(f"{pred:<12.2f} | {act:<10.2f} | {abs(pred-act):<10.2f}")

if __name__ == "__main__":
    main()