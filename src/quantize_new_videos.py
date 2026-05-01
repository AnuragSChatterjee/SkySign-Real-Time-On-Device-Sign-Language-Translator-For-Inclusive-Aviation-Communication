import numpy as np
import tensorflow as tf
import os

# UPDATED PATHS
MODEL_IN = "../models/cnn/cnn_v3_multi_final.keras"
MODEL_OUT = "../models/cnn/cnn_v3_multi_int8.tflite"
X_PATH = "../data/train_ready/X.npy"

def main():
    if not os.path.exists(MODEL_IN):
        print(f"❌ Error: {MODEL_IN} not found! Run training first.")
        return

    print(f"Loading {MODEL_IN} for quantization...")
    model = tf.keras.models.load_model(MODEL_IN)
    X = np.load(X_PATH).reshape(-1, 21, 3)

    # Shuffle for representative data
    np.random.shuffle(X)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for INT8 calibration
    def representative_data_gen():
        for i in range(500):
            # Ensure shape is (1, 21, 3)
            yield [X[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting to INT8 TFLite... (this may take a minute on the Pi)")
    tflite_model = converter.convert()
    
    with open(MODEL_OUT, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ SUCCESS: Saved INT8 model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
