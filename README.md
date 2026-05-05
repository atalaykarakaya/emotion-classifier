# Emo Challenge 2026 — Group 11
## BIL216 Signals and Systems — Final Project

Automatic emotion classification from speech audio signals using audio feature extraction and machine learning.

## Phase 1

**Accuracy: 62.20%**  
**Model: SVM (RBF kernel)**  
**Feature Vector: 372 dimensions**

## Files

| File | Description |
|---|---|
| `feature_extraction.py` | Extracts audio features from WAV files (MFCC, ZCR, Pitch, etc.) |
| `train.py` | Model training, comparison and saving |
| `results/confusion_matrix.png` | Confusion matrix on test set |
| `results/class_distribution.png` | Class distribution of dataset |
| `results/summary.json` | Accuracy and classification report |

## Installation

```bash
pip install librosa scikit-learn numpy pandas matplotlib seaborn joblib soundfile
```

## Usage

```bash
python train.py --dataset /path/to/dataset
```

## Emotions

`Furious` `Happy` `Neutral` `Sad` `Shocked`

## Team

- Atalay Karakaya (240611025)
- Nazlı Baş (240611008)  
- Irmak Barni (230611049)
