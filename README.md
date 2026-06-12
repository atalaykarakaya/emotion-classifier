# Emotion Classifier — COE216 Final Project

## Installation

```bash
pip install librosa scikit-learn numpy pandas matplotlib seaborn joblib soundfile
```

---

## Project Files

| File | Description |
|---|---|
| `feature_extraction.py` | WAV -> feature vector (MFCC, ZCR, Pitch, etc.) |
| `train.py` | Model training, comparison and saving |
| `evaluate.py` | Full evaluation + error analysis |
| `predict.py` | Single file prediction (terminal) |
| `demo_gui.py` | Visual demo interface |

---

## Usage

### 1. Train the Model (PHASE 1)

```bash
python train.py --dataset /path/to/wav/folder
```

Outputs saved to `results/`:
- `confusion_matrix.png`
- `class_distribution.png`
- `feature_importance.png`
- `summary.json` — accuracy value is here

### 2. Full Evaluation (PHASE 2-3)

```bash
python evaluate.py --dataset /path/to/wav/folder
```

Saved to `results/evaluation/`.

### 3. Single File Prediction

```bash
python predict.py --file g01_d08_neutral.wav
python predict.py --file test.wav --verbose
```

### 4. Visual Demo (for Live Demo)

```bash
python demo_gui.py
```

---

## Feature Set

| Feature | Dims | Description |
|---|---|---|
| MFCC (mean+std) | 80 | Audio timbre fingerprint |
| Spectral Centroid | 2 | Frequency center of mass |
| Spectral Bandwidth | 2 | Frequency spread |
| Spectral Rolloff | 2 | High frequency cutoff |
| ZCR | 2 | Zero crossing rate |
| RMS Energy | 2 | Instantaneous energy |
| Chroma | 24 | Musical pitch class distribution |
| Mel Spectrogram | 256 | Mel-band energy distribution |
| Pitch (F0) | 2 | Fundamental frequency (pyin) |
| **TOTAL** | **~372** | |

---

## Filename Format

Supported format: `gXX_dXX_<emotion>.wav`

Example: `g01_d08_neutral.wav` -> **neutral**

Supported emotions: `happy`, `sad`, `angry`, `furious`, `neutral`, `surprised`

---

## Submit Your Score

https://bil216finalproje-woau7utnbhu7q6hbz8nuff.streamlit.app/

Get your accuracy value from `results/summary.json`.
