# SkySign — Source Code Deep Dive

This document provides implementation-level technical detail for every script in `src/`. It is intended as a companion to the main repository README, which covers system overview, dataset statistics, and top-level results. Everything here goes one level deeper — explaining **why** each design decision was made, **how** the code works internally, and **what** the engineering trade-offs are.

---

## Contents

1. [Script Execution Order](#script-execution-order)
2. [extract_landmarks_new.py — Feature Extraction](#extract_landmarks_newpy)
3. [prepare_dataset_v3.py — Dataset Construction](#prepare_dataset_v3py)
4. [train_mlp_new.py — MLP Training](#train_mlp_newpy)
5. [train_cnn_new.py — CNN Training](#train_cnn_newpy)
6. [retrain_cnn_fixed_new.py — CNN Retraining](#retrain_cnn_fixed_newpy)
7. [quantize_new_videos.py — INT8 Quantization](#quantize_new_videospy)
8. [benchmark_new_videos.py — Model Comparison](#benchmark_new_videospy)
9. [robustness_test_cnn_videos.py — Robustness Testing](#robustness_test_cnn_videospy)
10. [test_video_fixed_all_NEW.py — Real-Video Validation](#test_video_fixed_all_newpy)
11. [inference_live — Deployment Script Internals](#inference-live-deployment-script-internals)
12. [Deep Dive: Model Trade-Off Engineering](#deep-dive-model-trade-off-engineering)
13. [Deep Dive: Latency Budget Breakdown](#deep-dive-latency-budget-breakdown)
14. [Deep Dive: Real-World Robustness Under Adverse Conditions](#deep-dive-real-world-robustness-under-adverse-conditions)

---

## Script Execution Order

```
extract_landmarks_new.py        (1) raw video → .npy landmark files
prepare_dataset_v3.py           (2) .npy files → balanced X.npy / y.npy
train_mlp_new.py                (3a) train MLP baseline
train_cnn_new.py                (3b) train CNN
retrain_cnn_fixed_new.py        (4)  retrain CNN → cnn_v3_multi_final.keras
quantize_new_videos.py          (5)  final.keras → INT8 .tflite
benchmark_new_videos.py         (6)  measure accuracy + latency
robustness_test_cnn_videos.py   (7)  stress test under degraded inputs
test_video_fixed_all_NEW.py     (8)  validate on raw held-out video
inference_live_...hardware.py   (9)  live deployment on Raspberry Pi 5
```

---

## `extract_landmarks_new.py`

### Internal mechanics

MediaPipe Hands runs a two-stage pipeline internally: a palm detector (BlazePalm) followed by a hand landmark model. Setting `static_image_mode=False` activates the tracking path — after the first frame, MediaPipe tracks the detected hand region rather than re-running the palm detector on every frame. This cuts per-frame processing time roughly in half and produces smoother landmark trajectories between frames, which is important for the temporal smoothing in the live inference script.

```python
hands = mp_hands.Hands(
    static_image_mode=False,    # tracking mode, not detection mode
    max_num_hands=2,            # extract first hand only, 2 for safety
    min_detection_confidence=0.5
)
```

### Why 0.5 detection confidence

This threshold was chosen through empirical testing across multiple lighting conditions:

- At 0.7+: frames with slightly dim lighting or a partially visible hand are dropped entirely, producing gaps in the landmark sequence and reducing training data
- At 0.3: occasional false positives where MediaPipe detects a non-existent hand in background texture, producing garbage landmark vectors
- At 0.5: stable across all tested conditions including window-backlit environments and desk-lamp-only lighting

### Filename aliasing for multi-subject consistency

Pramod used different naming conventions from Anurag for some signs. The aliasing map standardises these before saving:

```python
SIGN_MAP = {
    "THIRST":    "WATER",     # Pramod named it by meaning, not ASL name
    "LAND":      "LANDING",
    "TAKEOFF":   "TAKE",
    "THANK_YOU": "THANK",
    "THANK_YOU2":"THANK"
}
```

Without this map, WATER and THIRST would be treated as separate classes, creating a 16-class problem instead of 15 and splitting training data for the WATER class in half.

### Session timestamping

Output files are named `Anurag_EMERGENCY_20260424_1641.npy`. The `datetime.now().strftime("%Y%m%d_%H%M")` suffix means running the script twice in the same session creates new files rather than overwriting old ones. `prepare_dataset_v3.py` scans all files matching a sign name pattern, so all sessions are automatically pooled.

---

## `prepare_dataset_v3.py`

### The augmentation function in detail

```python
def normalize_and_augment(vec):
    arr = vec.reshape(21, 3).copy()
    arr -= arr[0]                                      # wrist centering
    angle = np.radians(np.random.uniform(-15, 15))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) # 3D rotation matrix
    arr = np.dot(arr, R)                               # XY rotation only
    arr += np.random.normal(0, 0.005, arr.shape)       # sensor noise
    max_val = np.max(np.abs(arr))
    if max_val > 1e-6: arr /= max_val                  # scale normalisation
    return arr.flatten().astype(np.float32)
```

**Why XY rotation only (Z axis fixed):** Hand signs are performed in a roughly vertical plane facing the camera. Rotating around the Z axis (the depth axis) simulates the hand being tilted left or right — a natural variation when a passenger's arm is at a different angle. Rotating around X or Y axes would simulate the hand being turned away from the camera, which does not correspond to realistic signing posture and would inject misleading training signal.

**Why σ=0.005 for noise:** The landmark coordinates are normalised to [-1, 1] after max-absolute scaling. MediaPipe's landmark detection uncertainty in clean video is approximately 0.002-0.005 in normalised units. Using σ=0.005 matches real detection noise — training on larger noise would teach the model to ignore legitimate signal differences between signs.

**Oversampling via repeated augmentation:** Classes with fewer raw samples (e.g. LANDING with only 1,017 raw frames) reach 2,000 by repeatedly calling `normalize_and_augment` on existing samples with new random seeds. Each call produces a geometrically distinct sample (different rotation, different noise draw), so oversampled classes do not simply duplicate data — they generate plausible geometric variations.

### Why 2,000 samples per class

At 15 classes × 2,000 = 30,000 total samples with an 80/20 split, each class has 1,600 training samples. Empirical testing during development showed:
- At 500/class: MLP accuracy ~87%, CNN ~85% — underfitting
- At 1,000/class: MLP accuracy ~95%, CNN ~96% — good
- At 2,000/class: MLP accuracy ~98.67%, CNN ~99.2% — target achieved
- At 5,000/class: No further improvement, longer training time

2,000 was the saturation point for this vocabulary size and model capacity.

---

## `train_mlp_new.py`

### Scikit-learn MLPClassifier internals

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
```

Scikit-learn's MLP uses a stochastic mini-batch gradient descent internally with Adam optimiser. Unlike TensorFlow/Keras, there is no explicit `batch_size` parameter — scikit-learn uses `min(200, n_samples)` by default. With 30,000 training samples this means batches of 200, giving 150 gradient updates per epoch.

### Why scikit-learn rather than Keras for the MLP

Three reasons:

1. **Inference independence:** The trained `.pkl` model runs with only `numpy` and `scikit-learn` — no TensorFlow import required. On a memory-constrained device, not loading TensorFlow at inference time saves ~300MB of RAM.

2. **Training speed:** 94 seconds on the Raspberry Pi 5 vs ~40 minutes for the CNN. This makes the MLP useful for rapid iteration and as a sanity check that the dataset is correctly formatted.

3. **Serialisation simplicity:** `pickle.dump(model, f)` produces a portable 693KB file. TFLite requires a separate conversion pipeline.

---

## `train_cnn_new.py`

### Why Conv1D treats landmarks as a sequence

The 21 MediaPipe landmarks follow strict anatomical ordering:

```
0: Wrist
1-4:   Thumb (CMC → MCP → IP → TIP)
5-8:   Index (MCP → PIP → DIP → TIP)
9-12:  Middle
13-16: Ring
17-20: Pinky
```

A Conv1D kernel of size 3 at position `i` sees landmarks `[i-1, i, i+1]` simultaneously. At position 5, it sees [wrist-proximal, index-MCP, index-PIP] — the base of the index finger and its first joint. This is exactly the kind of local joint relationship that distinguishes signs like CALL (pinky and thumb extended, middle fingers curled) from ALLERGIC (all fingers spread). The MLP sees all 63 values simultaneously without any notion of which values are adjacent joints.

### Dropout placement strategy

```python
Conv1D(64)  → BatchNorm → MaxPool → Dropout(0.2)  # after spatial reduction
Conv1D(128) → BatchNorm → GAP
Dense(128)  → Dropout(0.3)                          # before final classification
Dense(15)   → Softmax
```

Dropout after MaxPooling1D (not before) avoids dropping spatial information before it has been compressed. The higher dropout rate (0.3) before the final Dense layer reflects that this layer has the most parameters and is most prone to memorising training set quirks.

### Raspberry Pi stability constraints

```python
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
```

Unconstrained TensorFlow on the Raspberry Pi 5 uses all 4 Cortex-A76 cores simultaneously. At full load during CNN training, this draws enough current to trigger the Pi's thermal governor, reducing clock speed and causing SSH timeouts. Limiting to 2 threads keeps power draw stable and reduces core temperature by approximately 8-12°C during training, allowing the full 40-minute training run to complete without interruption.

---

## `retrain_cnn_fixed_new.py`

### Difference from `train_cnn_new.py`

| Parameter | `train_cnn_new.py` | `retrain_cnn_fixed_new.py` |
|---|---|---|
| Label format | One-hot (keras.utils.to_categorical) | Integer (sparse) |
| Loss function | categorical_crossentropy | sparse_categorical_crossentropy |
| Epochs | 80 | 40 |
| Batch size | 32 | 32 |
| Output file | cnn_v3_multi.keras | cnn_v3_multi_final.keras |

`retrain_cnn_fixed_new.py` produces `cnn_v3_multi_final.keras` which is the direct input to `quantize_new_videos.py`. Always run this script last before quantizing.

---

## `quantize_new_videos.py`

### How calibration determines quantization parameters

Post-training INT8 quantization maps each layer's floating-point weight tensor to an 8-bit integer range. The mapping requires two values per layer: a **scale** (float, maps integer range to float range) and a **zero point** (integer, maps float 0.0 to its integer representation).

For weights, these can be computed analytically from the weight tensor itself. For activations, they depend on the actual range of values that flow through each layer at runtime — which requires real data. The representative dataset provides this:

```python
def representative_data_gen():
    for i in range(500):
        yield [X[i:i+1].astype(np.float32)]
```

500 samples are passed through the model. At each layer, the minimum and maximum activation values are recorded. The scale and zero point are then:

```
scale     = (max_val - min_val) / 255
zero_point = round(-min_val / scale)
```

This is why using a representative dataset is critical — if calibration samples are not from the actual data distribution (e.g. if you accidentally use zeros or random noise), the quantization parameters will be wrong and accuracy will collapse.

### Why full INT8 (input and output both INT8)

Setting both `inference_input_type` and `inference_output_type` to `tf.int8` means the entire model — from the first operation to the last — runs in integer arithmetic. The alternative (mixed precision) keeps some layers in float32, requiring format conversion at layer boundaries and losing some of the latency benefit. Full INT8 enables the ARM XNNPACK delegate to process the entire graph as a single optimised integer kernel.

---

## `benchmark_new_videos.py`

### Latency measurement methodology

```python
start = time.perf_counter()
interpreter.set_tensor(in_det['index'], inp)
interpreter.invoke()
out = interpreter.get_tensor(out_det['index'])
lats.append((time.perf_counter() - start) * 1000)
```

`time.perf_counter()` uses the system's highest-resolution timer (nanosecond resolution on Linux). The measurement includes tensor copy time (`set_tensor`), actual inference (`invoke`), and output retrieval (`get_tensor`). This is the full inference cost as experienced by the calling code, not just the neural network computation in isolation.

### Why 1,000 samples for benchmarking

The benchmark draws 1,000 random samples from the 30,000-sample dataset. At 1,000 samples the standard error of the mean latency is small enough to be reliable, while keeping benchmark runtime under 60 seconds on the Pi. Running the full 30,000 samples would take ~10 minutes and is not necessary for a stable latency estimate.

---

## `robustness_test_cnn_videos.py`

### What jitter and shift represent physically

The two degradation parameters map directly to real cabin conditions:

**Gaussian jitter (σ parameter)** models two phenomena:
- **Camera shake from turbulence:** When the aircraft hits turbulence, a handheld phone camera vibrates. Each video frame captures a slightly different camera position, causing MediaPipe to detect landmark coordinates that fluctuate around their true values. Jitter with σ=0.03 corresponds approximately to the landmark displacement seen when the camera moves 2-3cm between frames.
- **MediaPipe detection uncertainty:** In compressed video streams (MJPEG at ~15 FPS), compression artefacts reduce edge sharpness around the hand. MediaPipe's landmark regression network produces slightly imprecise coordinates on blurry frames. This manifests as the same kind of zero-mean Gaussian noise.

**Spatial shift** models:
- **Hand repositioning between gestures:** A passenger may hold their hand at different positions in the frame for different signs. Wrist centering removes absolute position, but if the wrist landmark itself is detected slightly off-center (e.g. partially occluded by a sleeve), all downstream coordinates shift uniformly. The shift parameter simulates this.

### Why all 5 scenarios exceed 90%

The model is robust because the training augmentation (rotation ±15°, σ=0.005 noise) exposed it to a wider range of inputs than the test degradations. The test jitter (max σ=0.03) is 6× larger than training noise (σ=0.005), yet accuracy remains at 96.2%. This headroom exists because:

1. **Wrist centering removes the DC component of shift entirely** — a uniform translation of all landmarks by the shift value is completely cancelled when the wrist coordinate is subtracted
2. **Max-absolute scaling normalises jitter relative to hand size** — larger jitter on a large hand and smaller jitter on a small hand produce the same normalised perturbation
3. **The majority vote in the live inference script** absorbs single-frame misclassifications caused by extreme jitter frames

---

## `test_video_fixed_all_NEW.py`

### Why real-video accuracy differs from benchmark accuracy

The benchmark (`benchmark_new_videos.py`) evaluates on preprocessed, normalised landmark vectors from `data/train_ready/X.npy`. These were generated by the same pipeline that produced training data, so the test distribution matches training distribution.

The real-video validation evaluates on raw `.mp4` files that were not part of the training pipeline — MediaPipe runs on each frame fresh, normalization is applied at inference time, and the result is compared against the ground truth sign. This introduces several sources of difficulty not present in the benchmark:

- **Transition frames:** The first and last ~5 frames of each video show the hand moving into or out of the signing position. These frames contain partial signs that are genuinely ambiguous and are often misclassified.
- **Natural signing variation:** The same person signing EMERGENCY twice will use slightly different hand angles, speeds, and positions. The test videos were recorded independently of training data, capturing this natural variation.
- **10-frame window startup:** The majority vote window takes 10 frames to fill. During this period the output is dominated by whichever sign the model detects first, which may be a transition frame.

### Why some signs have lower real-video accuracy

Signs with accuracy below 60% (WATER 20%, SEATBELT 34%, ALLERGIC 43%) share visual similarity with higher-frequency signs in the dataset:

- **WATER:** Involves a W handshape that is geometrically similar to PAIN and REPEAT in normalized landmark space
- **SEATBELT:** A two-motion sign where the second motion (pulling the belt) is visually similar to STOP
- **ALLERGIC:** A scratching motion that in normalized form overlaps with REPEAT

This is a known limitation of static frame classification for multi-motion signs. A temporal model (LSTM, Transformer over frame sequences) would address this but is beyond the scope of this project.

---

## Inference Live — Deployment Script Internals

**Script:** `inference_live_vnc_videos_final_apicall_with_translation_optimized_with_accuracies_new_and_hardware.py`

### Thread architecture

The script runs two concurrent threads:

```
Main thread:  camera → MediaPipe → INT8 inference → GPIO → shared dict update → cv2.imshow
Flask thread: HTTP server reading shared dict → JSON responses to dashboard clients
```

The shared dictionary `current_data` is written by the main thread and read by Flask. In CPython, dictionary assignment is effectively atomic due to the Global Interpreter Lock (GIL), so no explicit locking is needed for this simple producer-consumer pattern.

### Normalization consistency guarantee

The inference-time normalization:
```python
coords -= coords[0]                         # wrist centering
mv = np.max(np.abs(coords))
coords = coords / mv if mv > 1e-6 else coords  # max-absolute scaling
```

must exactly match `prepare_dataset_v3.py`'s normalization. Any difference creates a distribution shift where the model receives inputs it was never trained on. Note that rotation augmentation from `prepare_dataset_v3.py` is deliberately **absent** at inference time — augmentation is a training technique to build robustness, not a preprocessing step applied to real inputs.

### The 1e-6 guard

```python
coords = coords / mv if mv > 1e-6 else coords
```

If MediaPipe detects a hand but all 21 landmarks are at exactly the same position (degenerate detection, can occur with a very small or partially occluded hand), `max_val` would be 0 and division would produce NaN. The guard skips normalisation in this case and passes the raw (near-zero) vector to the model, which will produce a low-confidence prediction that the majority vote window absorbs.

### 440Hz buzzer frequency rationale

The passive piezo buzzer on GPIO 18 is driven at 440Hz using `TonalBuzzer`. This frequency (musical note A4) was chosen because:
- It is the standard reference pitch in aviation audio warnings (used in cockpit altitude alerts and TCAS systems)
- It is clearly audible above typical aircraft cabin noise (~80dB) at the buzzer's output level
- It is distinct from the 1000Hz commonly used in hospital and retail alert systems, avoiding confusion

### Dashboard polling interval

```javascript
setInterval(update, 250);   // poll /status every 250ms
```

250ms was chosen as the polling interval because:
- Human visual perception cannot distinguish updates faster than ~100ms for text
- 250ms gives 4 updates per second, appearing smooth and responsive
- Lower intervals (e.g. 50ms) would increase Flask request load and are unnecessary
- The inference loop runs at ~15 FPS (~67ms per frame), so 250ms polling averages across approximately 3-4 inference cycles, further smoothing display transitions

---

## Deep Dive: Model Trade-Off Engineering

The main README presents the benchmark numbers. This section explains the engineering reasoning behind choosing TFLite INT8 as the deployment model.

### The embedded AI constraint hierarchy

For deployment on the Raspberry Pi 5 in an aviation context, constraints rank as follows:

1. **Reliability** — the model must work consistently across users and conditions
2. **Latency** — inference must complete within the frame budget
3. **Memory** — model + runtime must fit within available RAM
4. **Accuracy** — highest possible within the above constraints

### Why the MLP was not chosen for deployment despite competitive accuracy

The MLP achieves 98.67% accuracy with 0.307ms inference — very competitive. However:

- **No hardware acceleration path:** scikit-learn runs on pure Python/numpy with no SIMD optimisation on ARM. The TFLite runtime uses the XNNPACK delegate which generates optimised ARM Neon SIMD instructions, explaining the 17× latency difference (0.307ms vs 0.018ms) despite both models being small
- **Not the embedded AI approach:** Post-training quantization and TFLite deployment are standard embedded AI techniques. Using the MLP alone would demonstrate only a trained classifier, not an embedded AI system

### Why full Keras CNN was not chosen

The Keras CNN (99.2%, 0.25ms) is faster than the MLP because TensorFlow is already loaded. However:

- **191KB vs 23KB:** The full Keras model is 8.3× larger. On an embedded device where RAM is shared between the model, MediaPipe, Flask, OpenCV, and the OS, smaller is better
- **TF runtime overhead:** Loading the full TensorFlow/Keras runtime for inference adds ~200MB RAM usage that the TFLite runtime avoids
- **No quantization demonstration:** Using the full Keras model at deployment does not demonstrate the quantization trade-off that the course specifically asks for

### The chosen deployment model: TFLite INT8

TFLite INT8 (97.9%, 0.018ms, 23KB) wins on every embedded AI metric. The 1.3% accuracy reduction is absorbed by temporal smoothing. This is the correct engineering choice for an embedded AI project — sacrificing a small amount of accuracy for massive gains in efficiency.

---

## Deep Dive: Latency Budget Breakdown

The main README reports 0.018ms inference latency. This section shows where the rest of the frame budget goes.

### Per-component timing (measured on Raspberry Pi 5)

| Component | Typical latency | Variability |
|---|---|---|
| IP camera frame receive | 50–80ms | High — network dependent |
| OpenCV MJPEG decode | ~2ms | Low |
| cv2.resize to 480×360 | ~0.5ms | Low |
| cv2.flip | ~0.2ms | Low |
| MediaPipe Hands (model_complexity=0) | 30–50ms | Medium |
| Landmark array construction | ~0.1ms | Low |
| Wrist centering + scaling | ~0.1ms | Low |
| INT8 quantization of input | ~0.05ms | Low |
| **TFLite INT8 invoke** | **0.018ms** | **Very low** |
| INT8 dequantization of output | ~0.01ms | Low |
| deque append + Counter | ~0.01ms | Low |
| Shared dict update | ~0.005ms | Low |
| GPIO state check + update | ~1ms | Low |
| cv2.rectangle + putText | ~0.5ms | Low |
| cv2.imshow (VNC) | ~10ms | Medium |
| **Total frame pipeline** | **~95–145ms** | Network dominates |

### The actual bottleneck

The 0.018ms inference is not the bottleneck — it represents less than 0.02% of the total frame budget. The dominant cost is the IP camera stream latency (50–80ms) plus MediaPipe (30–50ms). These together consume 80–130ms of the ~100ms target budget.

This means further optimising the neural network would have no user-perceptible effect. The correct embedded AI approach is to optimise the bottleneck — which would mean using a lower-latency camera connection (USB camera directly connected to Pi would eliminate the 50-80ms network latency) or a lighter MediaPipe model.

### Why model_complexity=0 for MediaPipe

MediaPipe Hands offers three complexity levels (0, 1, 2). Higher complexity uses larger internal models for more precise landmark detection. For this application:
- complexity=0: ~30ms, sufficient precision for sign classification
- complexity=1: ~50ms, marginally better precision
- complexity=2: ~80ms, high precision, exceeds frame budget

The landmark-based classifier is robust to small landmark imprecision (as shown by the robustness tests), making the precision improvement from higher complexity not worth the latency cost.

---

## Deep Dive: Real-World Robustness Under Adverse Conditions

The main README presents robustness test numbers. This section explains the physical and engineering reasons why the system performs well in conditions that would defeat image-based classifiers.

### The architectural robustness guarantee

SkySign's core robustness comes from a single architectural decision: **classifying landmarks rather than pixels**. This decision propagates through every adverse condition:

```
Adverse condition          Image classifier        SkySign
────────────────────────────────────────────────────────────────
Dim lighting               Feature changes         Unchanged
Bright lighting            Feature changes         Unchanged
Different backgrounds      Feature changes         Unchanged
JPEG compression           Artifacts in pixels     Unchanged
Hand position in frame     Feature changes         Removed by wrist centering
Hand size / distance       Feature changes         Removed by max-abs scaling
Camera shake (small)       Feature changes         Absorbed by training noise
Camera shake (large)       Severe degradation      Absorbed by majority vote
```

### Tested real-world conditions

The following conditions were validated during live inference sessions. In all cases, the system maintained correct sign detection throughout:

**Lighting variations:**
- Bright overhead fluorescent lighting (typical office) ✅
- Single desk lamp only, other lights off ✅
- Mixed: bright window light on one side, shadow on other ✅
- Backlit conditions (bright window directly behind signer) ✅

MediaPipe uses a combination of color and depth cues. Backlit conditions that would silhouette a subject against a bright background — causing most image classifiers to fail — are handled because MediaPipe uses hand shape geometry, not color or texture, to locate and track landmarks.

**Simulated turbulence:**

During testing, the phone camera was physically shaken continuously while the signer performed each gesture. Despite the camera moving 5–10cm between frames, the system continued detecting the correct sign. Two mechanisms explain this:

1. **Wrist centering at inference time:** Each frame's landmarks are independently centered on the wrist before classification. A camera shift of 5cm corresponds to a shift of ~0.1 in normalised coordinates, which is completely removed by subtracting `coords[0]`. The classifier never sees the camera motion.

2. **5-frame majority vote:** Even if 1-2 frames during a turbulence spike produce incorrect predictions (due to motion blur causing MediaPipe imprecision), the majority vote over 5 frames suppresses these outliers as long as at least 3 frames are correct.

**Video stream quality:**
- Full quality MJPEG at 15 FPS ✅
- Compressed MJPEG with visible blocking artefacts ✅
- Brief stream dropouts (network hiccup) ✅ — deque holds last valid predictions

**Across subjects:**
The model was trained on Anurag and Pramod performing the signs. During live testing, both subjects achieved stable detection, confirming that the normalization pipeline successfully removes person-specific variation (hand size, finger length, natural hand angle) and the model generalises to the underlying gesture geometry.