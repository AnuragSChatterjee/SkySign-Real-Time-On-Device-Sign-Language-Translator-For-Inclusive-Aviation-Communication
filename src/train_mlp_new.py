"""
SkySign — train_mlp_new.py
Trains a lightweight MLP classifier on MediaPipe hand landmark features.
Optimized for multi-user v3 dataset.
"""

import numpy as np
import os
import json
import time
import pickle
import matplotlib
matplotlib.use('Agg')  # Headless safe for Raspberry Pi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# MATCHED EXACTLY TO prepare_dataset_v3.py
SIGNS = [
    "ALLERGIC", "BATHROOM", "CALL", "EMERGENCY", "FOOD", "HELP",
    "LANDING", "PAIN", "REPEAT", "SEATBELT", "STOP", "THANK",
    "TAKE", "WATER", "YES"
]

DATA_X_PATH = "../data/train_ready/X.npy"
DATA_Y_PATH = "../data/train_ready/y.npy"
MODEL_DIR   = "../models/mlp"
RESULTS_DIR = "../results"
RANDOM_SEED = 42

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5)
    plt.title('MLP Confusion Matrix — SkySign Multi-User v3', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")

def measure_inference_latency(model, X_test, n_runs=100):
    """Measure per-sample inference latency in ms."""
    for _ in range(10): # Warmup
        model.predict(X_test[:1])
    latencies = []
    for i in range(min(n_runs, len(X_test))):
        t0 = time.perf_counter()
        model.predict(X_test[i:i+1])
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": float(np.mean(latencies)),
        "p95_ms":  float(np.percentile(latencies, 95)),
        "max_ms":  float(np.max(latencies)),
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("="*55)
    print("  SkySign — MLP Training (Multi-User) [v3 Data]")
    print("="*55)

    # ── 1. LOAD DATA ──
    print("\nLoading dataset from train_ready...")
    if not os.path.exists(DATA_X_PATH):
        print(f"ERROR: {DATA_X_PATH} not found. Run prepare_dataset_v3.py first.")
        return

    X = np.load(DATA_X_PATH)
    y = np.load(DATA_Y_PATH)

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # Map label names based on classes present in the data
    label_names = [SIGNS[i] for i in unique_classes]

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features.")
    print(f"Detected {num_classes} unique classes.")

    # ── 2. SPLIT DATA (70/15/15) ──
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED,
        stratify=y_temp)

    print(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 3. TRAIN MLP ──
    # Increased complexity slightly to handle multi-user variance
    print(f"\nTraining MLP Classifier (63 → 256 → 128 → 64 → {num_classes})...")

    t_start = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=5e-4,           # Slightly higher regularization for generalization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_SEED,
        verbose=False
    )

    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t_start
    print(f"  Training complete in {train_time:.1f}s ({model.n_iter_} iterations)")

    # ── 4. EVALUATE ──
    print("\nEvaluating...")
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1  = f1_score(y_test, y_pred_test, average='macro')

    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test F1 Score: {test_f1:.4f}")

    # ── 5. LATENCY & SIZE ──
    latency = measure_inference_latency(model, X_test)
    print(f"  Inference Latency: {latency['mean_ms']:.3f}ms (Mean)")

    import io
    buf = io.BytesIO()
    pickle.dump(model, buf)
    model_size_kb = buf.tell() / 1024
    print(f"  Model size: {model_size_kb:.1f} KB")

    # ── 6. SAVE ARTIFACTS ──
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, label_names, os.path.join(RESULTS_DIR, "mlp_v3_multi_cm.png"))

    model_path = os.path.join(MODEL_DIR, "mlp_model_v3_multi.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metrics for comparison
    metrics = {
        "model": "MLP_v3_Multi",
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "mean_latency_ms": round(latency['mean_ms'], 3),
        "model_size_kb": round(model_size_kb, 1),
        "classes_trained": label_names
    }
    with open(os.path.join(RESULTS_DIR, "mlp_v3_multi_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()
