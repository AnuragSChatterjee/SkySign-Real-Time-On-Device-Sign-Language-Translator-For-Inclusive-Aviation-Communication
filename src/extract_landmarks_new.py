import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def get_landmarks(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return None

# --- CONFIG ---
RAW_VIDEOS_BASE = os.path.expanduser('~/skysign/skysign_v3_multi/data/raw_videos')
OUTPUT_PATH = os.path.expanduser('~/skysign/skysign_v3_multi/data/processed')
# Folders to process
DATASETS = ["Anurag_Chatterjee_Datasets", "Pramod_G_Datasets1", "Pramod_G_Datasets_2"]

# Mapping to standardize Pramod's filenames to your 15 SIGNS
SIGN_MAP = {
    "THIRST": "WATER",
    "LAND": "LANDING",
    "TAKEOFF": "TAKE",
    "THANK_YOU": "THANK",
    "THANK_YOU2": "THANK"
}

os.makedirs(OUTPUT_PATH, exist_ok=True)
session_id = datetime.now().strftime("%Y%m%d_%H%M")

for dataset_folder in DATASETS:
    user_name = "Anurag" if "Anurag" in dataset_folder else "Pramod"
    video_dir = os.path.join(RAW_VIDEOS_BASE, dataset_folder)
    
    if not os.path.exists(video_dir):
        print(f"Skipping {dataset_folder}, path not found.")
        continue

    print(f"\n📂 Processing Dataset: {dataset_folder}")
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        # Extract base sign name (remove user prefix, numbers, and extension)
        raw_name = video_file.split('.')[0].split('_')
        # Handle cases like "Pramod_Allergic2" or "Anurag_Stop"
        sign_name = raw_name[1].upper() if len(raw_name) > 1 else raw_name[0].upper()
        
        # Clean specific trailing numbers/strings
        for suffix in ["2", "_RETAKE", "OFF"]:
            sign_name = sign_name.replace(suffix, "")
        
        # Apply mapping
        sign_name = SIGN_MAP.get(sign_name, sign_name)

        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        processed_data = []

        print(f"  -> {video_file} (as {sign_name})...", end=" ", flush=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            landmarks = get_landmarks(frame)
            if landmarks is not None:
                processed_data.append(landmarks)

        cap.release()

        if processed_data:
            output_filename = f"{user_name}_{sign_name}_{session_id}.npy"
            save_path = os.path.join(OUTPUT_PATH, output_filename)
            np.save(save_path, np.array(processed_data))
            print(f"DONE ({len(processed_data)} frames)")
        else:
            print("FAILED (No hands)")

print(f"\n✨ Extraction complete! Files saved to: {OUTPUT_PATH}")
