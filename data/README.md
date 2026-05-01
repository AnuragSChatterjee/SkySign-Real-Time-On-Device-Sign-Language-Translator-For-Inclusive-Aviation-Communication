# Dataset

The full training dataset (30,000 samples, 15 classes) is not included
in this repository due to file size (>3GB of raw video + processed landmarks).

## Structure

- `raw_videos/` — MP4 recordings per subject and sign (not uploaded)
- `processed/` — Per-file MediaPipe landmark arrays as .npy (not uploaded)
- `train_ready/X.npy` — Final (30000, 63) feature matrix (not uploaded)
- `train_ready/y.npy` — Final (30000,) label array (not uploaded)
- `sample/` — 3 example .npy files showing the landmark format

## Sample File Format

Each .npy file contains shape (N_frames, 63):
- N_frames: number of video frames where MediaPipe detected a hand
- 63 values: 21 landmarks × (x, y, z) coordinates

## To Reproduce

1. Record MP4 videos of each sign
2. Run `src/extract_landmarks_new.py`
3. Run `src/prepare_dataset_v3.py`
4. Run `src/train_cnn_new.py`