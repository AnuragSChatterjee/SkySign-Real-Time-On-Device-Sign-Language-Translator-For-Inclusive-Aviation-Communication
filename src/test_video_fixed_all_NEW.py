import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite
import os
from collections import Counter

# UPDATED DIRECTORIES
DATA_ROOT = "../data/raw_videos"
SUB_DIRS = ["Anurag_Chatterjee_Datasets", "Pramod_G_Datasets1", "Pramod_G_Datasets_2"]
MODEL_PATH = "../models/cnn/cnn_v3_multi_int8.tflite"

SIGNS = [
    "ALLERGIC", "BATHROOM", "CALL", "EMERGENCY", "FOOD", "HELP",
    "LANDING", "PAIN", "REPEAT", "SEATBELT", "STOP", "THANK",
    "TAKE", "WATER", "YES"
]

# Mapping inconsistent filenames to our 15 classes
NAME_MAP = {
    "THANK_YOU": "THANK",
    "TAKEOFF": "TAKE",
    "TAKE_OFF": "TAKE",
    "LAND": "LANDING",
    "THIRST": "WATER"
}

def normalize_landmarks(landmarks):
    coords = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    coords -= coords[0] 
    max_val = np.max(np.abs(coords))
    if max_val > 1e-6: coords /= max_val
    return coords

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model {MODEL_PATH} not found!")
        return

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    
    quant = in_det.get('quantization_parameters', {})
    s, zp = quant.get('scales', [1.0])[0], quant.get('zero_points', [0])[0]

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )
    
    results = {sign: {'hits': 0, 'total': 0} for sign in SIGNS}

    for sub in SUB_DIRS:
        folder_path = os.path.join(DATA_ROOT, sub)
        if not os.path.exists(folder_path): continue
        
        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
        print(f"\n📂 Processing {sub} ({len(video_files)} videos)...")

        for vid in video_files:
            # 1. Identify the ground truth sign from filename
            gt = None
            fn_upper = vid.upper().replace(".MP4", "")
            
            # Check for direct matches or mapped variations
            for s_name in SIGNS:
                if s_name in fn_upper:
                    gt = s_name
                    break
            
            if not gt:
                for alias, real_name in NAME_MAP.items():
                    if alias in fn_upper:
                        gt = real_name
                        break
            
            if not gt:
                print(f"⏩ Skipping {vid}: No match.")
                continue

            cap = cv2.VideoCapture(os.path.join(folder_path, vid))
            window = [] 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    norm = normalize_landmarks(res.multi_hand_landmarks[0].landmark).reshape(1, 21, 3)
                    inp = (norm / s + zp).astype(np.int8) if in_det['dtype'] == np.int8 else norm

                    interpreter.set_tensor(in_det['index'], inp)
                    interpreter.invoke()
                    pred_idx = np.argmax(interpreter.get_tensor(out_det['index'])[0])

                    window.append(SIGNS[pred_idx])
                    if len(window) > 10: window.pop(0)

                    final_pred = Counter(window).most_common(1)[0][0]
                    results[gt]['total'] += 1
                    if final_pred == gt: results[gt]['hits'] += 1
            
            cap.release()
            print(f"✅ Validated: {gt} from {vid}")

    # Final Report
    print("\n" + "="*40)
    print(f"{'SIGN':<15} | {'ACCURACY (%)'}")
    print("-" * 40)
    total_h, total_s = 0, 0
    for sign in SIGNS:
        stat = results[sign]
        acc = (stat['hits']/stat['total']*100) if stat['total'] > 0 else 0
        print(f"{sign:<15} | {acc:>8.2f}%")
        total_h += stat['hits']; total_s += stat['total']
    
    overall = (total_h/total_s*100) if total_s > 0 else 0
    print("-" * 40)
    print(f"{'OVERALL':<15} | {overall:>8.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
