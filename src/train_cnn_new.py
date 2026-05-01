import numpy as np
import os
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── STABILITY HEADER: PREVENT PI CRASHES ──────────────────────────────────────
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Prevent TF from hogging all RAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# MATCHED TO prepare_dataset_v3.py
SIGNS = [
    "ALLERGIC", "BATHROOM", "CALL", "EMERGENCY", "FOOD", "HELP",
    "LANDING", "PAIN", "REPEAT", "SEATBELT", "STOP", "THANK",
    "TAKE", "WATER", "YES"
]

DATA_X_PATH = "../data/train_ready/X.npy"
DATA_Y_PATH = "../data/train_ready/y.npy"
MODEL_DIR   = "../models/cnn"
RANDOM_SEED = 42
EPOCHS      = 80
BATCH_SIZE  = 32

def build_cnn(n_classes):
    # Input is (21 joints, 3 coordinates)
    inputs = keras.Input(shape=(21, 3), name="landmarks")
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x) 

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading SkySign Multi-User v3 dataset...")
    X = np.load(DATA_X_PATH)
    y = np.load(DATA_Y_PATH)

    # RESHAPE: (Samples, 63) -> (Samples, 21, 3)
    X_cnn = X.reshape(-1, 21, 3)
    num_classes = len(np.unique(y))

    # Convert y to One-Hot for Categorical Crossentropy
    y_cat = keras.utils.to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_cat, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"CNN Input Shape: {X_cnn.shape[1:]}")
    print(f"Training on {num_classes} classes...")

    model = build_cnn(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True
    )

    print("\n🚀 Starting CNN Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    model_path = os.path.join(MODEL_DIR, "cnn_v3_multi.keras")
    model.save(model_path)

    print(f"\n✓ CNN Training Complete. Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
