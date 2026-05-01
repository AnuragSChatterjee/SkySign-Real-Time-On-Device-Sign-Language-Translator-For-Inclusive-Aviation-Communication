import numpy as np
import os

# -- CONFIG --
# We look in BOTH the symlinked old data and the fresh new processed folder
PROCESSED_DIRS = ["../data/processed", "../data/processed_link"]
OUTPUT_DIR     = "../data/train_ready"
TARGET_SAMPLES = 2000 # Increased because we have more data now!
SIGNS = ["ALLERGIC", "BATHROOM", "CALL", "EMERGENCY", "FOOD", "HELP",
         "LANDING", "PAIN", "REPEAT", "SEATBELT", "STOP", "THANK",
         "TAKE", "WATER", "YES"]

def normalize_and_augment(vec):
    try:
        arr = vec.reshape(21, 3).copy()
        arr -= arr[0] # Zeroing to wrist
        angle = np.radians(np.random.uniform(-15, 15))
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        arr = np.dot(arr, R)
        arr += np.random.normal(0, 0.005, arr.shape)
        max_val = np.max(np.abs(arr))
        if max_val > 1e-6: arr /= max_val
        return arr.flatten().astype(np.float32)
    except:
        return None

def balance_data(data_list, n_target):
    results = []
    if not data_list: return []
    for d in data_list:
        normed = normalize_and_augment(d)
        if normed is not None: results.append(normed)

    if len(results) == 0: return []
    while len(results) < n_target:
        idx = np.random.randint(len(data_list))
        aug = normalize_and_augment(data_list[idx])
        if aug is not None: results.append(aug)
    return results[:n_target]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, Y = [], []
    print("🚀 Starting Multi-User Dataset Prep...")

    for idx, sign in enumerate(SIGNS):
        class_raw_data = []
        
        for p_dir in PROCESSED_DIRS:
            if not os.path.exists(p_dir): continue
            
            files = [f for f in os.listdir(p_dir) if sign.upper() in f.upper() and f.endswith('.npy')]

            for f in files:
                file_path = os.path.join(p_dir, f)
                try:
                    if os.path.getsize(file_path) == 0: continue
                    data = np.load(file_path)
                    frames = data.reshape(-1, 63)
                    for frame in frames:
                        class_raw_data.append(frame)
                except:
                    continue

        if class_raw_data:
            print(f"✅ Found {len(class_raw_data)} for {sign}. Balancing to {TARGET_SAMPLES}...")
            balanced = balance_data(class_raw_data, TARGET_SAMPLES)
            for vec in balanced:
                X.append(vec)
                Y.append(idx)
        else:
            print(f"⚠️ Missing data for: {sign}")

    if len(X) > 0:
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), np.array(X))
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(Y))
        print(f"✨ SUCCESS: Prepared {len(X)} samples.")
    else:
        print("❌ ERROR: No data found.")
