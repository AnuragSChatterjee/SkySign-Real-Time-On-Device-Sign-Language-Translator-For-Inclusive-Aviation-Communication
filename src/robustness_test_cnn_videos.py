import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score

# --- UPDATED CONFIG ---
X_TEST = "../data/train_ready/X.npy"
Y_TEST = "../data/train_ready/y.npy"
MODEL_PATH = "../models/cnn/cnn_v3_multi_int8.tflite"

def run_robustness():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    # Load a subset for testing
    print("Loading test data...")
    X = np.load(X_TEST).reshape(-1, 21, 3)
    y = np.load(Y_TEST)

    # Use 1000 samples for a more statistically significant test
    indices = np.random.permutation(len(X))[:1000]
    X_test, y_test = X[indices], y[indices]

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    
    # Handle quantization parameters safely
    quant = in_det.get('quantization_parameters', {})
    s = quant.get('scales', [1.0])[0]
    zp = quant.get('zero_points', [0])[0]

    # Testing different types of environmental "stress"
    # Noise (jitter) and Shift (hand placement variation)
    scenarios = [
        {"name": "Clean", "sigma": 0.0, "shift": 0.0},
        {"name": "Light Jitter", "sigma": 0.01, "shift": 0.0},
        {"name": "Heavy Jitter", "sigma": 0.03, "shift": 0.0},
        {"name": "Slight Shift", "sigma": 0.0, "shift": 0.05},
        {"name": "Combined Stress", "sigma": 0.02, "shift": 0.05},
    ]

    print(f"\n{'Scenario':<18} | {'Noise (σ)':<10} | {'Shift':<8} | {'Accuracy'}")
    print("-" * 55)

    for sc in scenarios:
        preds = []
        sigma = sc["sigma"]
        shift = sc["shift"]

        for i in range(len(X_test)):
            sample = X_test[i:i+1].copy()
            
            # 1. Apply Gaussian Noise (Jitter)
            if sigma > 0:
                sample += np.random.normal(0, sigma, sample.shape)
            
            # 2. Apply Spatial Shift (Translation)
            if shift > 0:
                sample += shift
            
            # 3. INT8 Quantization
            inp = (sample / s + zp).astype(np.int8)

            interpreter.set_tensor(in_det['index'], inp)
            interpreter.invoke()
            output = interpreter.get_tensor(out_det['index'])
            preds.append(np.argmax(output))

        acc = accuracy_score(y_test, preds)
        status = "✅" if acc > 0.90 else "⚠️"
        print(f"{sc['name']:<18} | {sigma:<10} | {shift:<8} | {status} {acc*100:>6.1f}%")

if __name__ == "__main__":
    run_robustness()
