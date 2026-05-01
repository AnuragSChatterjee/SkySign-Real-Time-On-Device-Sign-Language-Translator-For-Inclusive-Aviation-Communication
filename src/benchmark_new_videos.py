import numpy as np
import os
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score

X_TEST = "../data/train_ready/X.npy"
Y_TEST = "../data/train_ready/y.npy"

# UPDATED TO MULTI-USER FILENAMES
MODELS = {
    "CNN_Full": "../models/cnn/cnn_v3_multi_final.keras",
    "TFLite_INT8": "../models/cnn/cnn_v3_multi_int8.tflite"
}

def bench_tflite(path, X, y):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    
    # Extract quantization parameters
    quant = in_det.get('quantization_parameters', {})
    s = quant.get('scales', [1.0])[0]
    zp = quant.get('zero_points', [0])[0]

    preds, lats = [], []
    for i in range(len(X)):
        inp = X[i:i+1].astype(np.float32)
        if in_det['dtype'] == np.int8:
            inp = (inp / s + zp).astype(np.int8)

        start = time.perf_counter()
        interpreter.set_tensor(in_det['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])
        lats.append((time.perf_counter() - start) * 1000)
        preds.append(np.argmax(out))

    return accuracy_score(y, preds), np.mean(lats)

def main():
    if not os.path.exists(X_TEST):
        print("Data not found!")
        return
        
    X = np.load(X_TEST).reshape(-1, 21, 3)
    y = np.load(Y_TEST)

    # Use 1000 random samples for benchmark
    idx = np.random.permutation(len(X))[:1000]
    X_sub, y_sub = X[idx], y[idx]

    print(f"\n{'Model Type':<15} | {'Accuracy':<10} | {'Latency (ms)':<12}")
    print("-" * 45)

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"{name:<15} | {'MISSING':<10} | {'N/A'}")
            continue
            
        if path.endswith(".keras"):
            m = tf.keras.models.load_model(path)
            # Warmup
            _ = m.predict(X_sub[:10], verbose=0)
            t0 = time.perf_counter()
            res = m.predict(X_sub, verbose=0)
            lat = ((time.perf_counter() - t0) / 1000) * 1000
            acc = accuracy_score(y_sub, np.argmax(res, axis=1))
        else:
            acc, lat = bench_tflite(path, X_sub, y_sub)

        print(f"{name:<15} | {acc*100:>8.1f}% | {lat:>8.4f}")

if __name__ == "__main__":
    main()
