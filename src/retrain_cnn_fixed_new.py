import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os

# Stability for Raspberry Pi
os.environ['OMP_NUM_THREADS'] = '2'

X_PATH = "../data/train_ready/X.npy"
Y_PATH = "../data/train_ready/y.npy"
MODEL_OUT = "../models/cnn/cnn_v3_multi_final.keras"

def train():
    if not os.path.exists(X_PATH):
        print(f"Error: {X_PATH} not found!")
        return

    X = np.load(X_PATH).reshape(-1, 21, 3)
    y = np.load(Y_PATH)

    # Shuffle
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    num_classes = len(np.unique(y))
    print(f"🚀 Retraining SkySign (Multi-User) on {num_classes} classes...")

    model = models.Sequential([
        Input(shape=(21, 3)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Increased epochs slightly for the expanded dataset
    model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2, verbose=1)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"✅ Saved final multi-user model to {MODEL_OUT}")

if __name__ == "__main__":
    train()
