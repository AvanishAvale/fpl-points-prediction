import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
X_TRAIN_PATH = 'X_train.npy'
Y_TRAIN_PATH = 'y_train.npy'
X_TEST_PATH = 'X_test.npy'
Y_TEST_PATH = 'y_test.npy'

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 

def build_bidirectional_model(input_shape):
    """
    Experiment: Bidirectional LSTM
    """
    model = Sequential([
        Input(shape=input_shape),
        

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        
        # Second LSTM layer to deepen understanding
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def main():
    print("Loading data for Experiment 2 (Bidirectional)...")
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_bidirectional_model(input_shape)
    
    print("\n--- Model Architecture ---")
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_bidirectional_model.keras', monitor='val_loss', save_best_only=True)

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("\nEvaluating Bidirectional Model...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"New MAE: {mae:.4f}")
    
    # Save comparison graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Bidirectional LSTM Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('bidirectional_history.png')

if __name__ == "__main__":
    main()