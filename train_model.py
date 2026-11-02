import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
X_TRAIN_PATH = 'X_train.npy'
Y_TRAIN_PATH = 'y_train.npy'
X_TEST_PATH = 'X_test.npy'
Y_TEST_PATH = 'y_test.npy'

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def build_model(input_shape):
    """
    Defines the LSTM architecture.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        
        LSTM(64, return_sequences=False),
        
    
        Dropout(0.2),
   
        Dense(32, activation='relu'),
  
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def main():
    # 1. Load Data
    print("Loading data...")
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples.")

    # 2. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (5, 14)
    model = build_model(input_shape)
    model.summary()

    # 3. Define Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Save the best version of the model
    checkpoint = ModelCheckpoint('best_fpl_model.keras', monitor='val_loss', save_best_only=True)

    # 4. Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # 5. Evaluate
    print("\nEvaluating on Test Set (2024-25 Season)...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f" -> On average, the model is off by {mae:.4f} points.")

    # 6. Plot Training History
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.savefig('training_history.png')
    print("Training graph saved to training_history.png")

if __name__ == "__main__":
    main()