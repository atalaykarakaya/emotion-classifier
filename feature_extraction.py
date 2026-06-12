"""
feature_extraction.py
---------------------
Extracts audio features from WAV files.
Supported features:
  - MFCC (40 coefficients, mean + std = 80 dims)
  - Spectral Centroid (mean + std)
  - Spectral Bandwidth (mean + std)
  - Spectral Rolloff (mean + std)
  - ZCR - Zero Crossing Rate (mean + std)
  - RMS Energy (mean + std)
  - Chroma Features (12 coefficients, mean + std = 24 dims)
  - Mel Spectrogram (128 bands, mean + std = 256 dims)
  - Pitch / F0 (mean + std)
  - Spectral Contrast (7 bands, mean + std = 14 dims)
  - Tonnetz (6 coefficients, mean + std = 12 dims)
  - MFCC Delta (40 coefficients, mean + std = 80 dims)
  - MFCC Delta-Delta (40 coefficients, mean + std = 80 dims)
Total: ~558-dimensional feature vector
"""

import os
import unicodedata
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


SAMPLE_RATE = 22050
N_MFCC      = 40
N_MELS      = 128
HOP_LENGTH  = 512



def extract_features(file_path: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        if len(y) < sr * 0.5:
            print(f"[WARNING] File too short, skipped: {file_path}")
            return None

        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.append(np.mean(sc))
        features.append(np.std(sc))

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.append(np.mean(sb))
        features.append(np.std(sb))

        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.append(np.mean(sr_feat))
        features.append(np.std(sr_feat))

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"), sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        features.append(float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0)
        features.append(float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))

        flatness = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)
        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"[ERROR] Problem processing {file_path}: {e}")
        return None

LABEL_MAP = {
    "ofkeli"   : "furious",
    "kizgin"   : "furious",
    "angry"    : "furious",
    "furious"  : "furious",
    "uzgun"    : "sad",
    "sad"      : "sad",
    "mutlu"    : "happy",
    "happy"    : "happy",
    "saskin"   : "shocked",
    "sasirma"  : "shocked",
    "sasirmis" : "shocked",
    "surprised": "shocked",
    "shocked"  : "shocked",
    "notr"     : "neutral",
    "tarafsiz" : "neutral",
    "neutral"  : "neutral",
    "mutsuz"   : "sad",
}

def normalize(text: str) -> str:
    """Türkçe karakterleri normalize et: ü→u, ö→o, ş→s, ı→i, ğ→g, ç→c"""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ü": "u", "ö": "o", "ş": "s", "ı": "i", "ğ": "g", "ç": "c",
        "Ü": "U", "Ö": "O", "Ş": "S", "İ": "I", "Ğ": "G", "Ç": "C"
    }
    for tr, en in replacements.items():
        text = text.replace(tr, en)
    return text

VALID_EMOTIONS = {"neutral", "happy", "furious", "sad", "shocked"}

def get_label_from_filename(filename: str) -> str | None:
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = [p for p in name.split("_") if p.strip()]
    if len(parts) >= 2:
        raw = parts[-2].lower().strip()
    elif len(parts) == 1:
        raw = parts[0].lower().strip()
    else:
        return None
    
    replacements = {"ü":"u","ö":"o","ş":"s","ı":"i","ğ":"g","ç":"c"}
    for tr, en in replacements.items():
        raw = raw.replace(tr, en)
    
    label = LABEL_MAP.get(raw, raw)
    
    if label not in VALID_EMOTIONS:
        return None
    
    return label

def load_dataset(dataset_dir: str):
    import glob

    wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(dataset_dir, "*.wav"))
    wav_files = list(set(wav_files))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in '{dataset_dir}'.")

    print(f"\nFound {len(wav_files)} WAV files. Starting feature extraction...\n")

    X, y, skipped = [], [], 0

    for i, fpath in enumerate(sorted(wav_files), 1):
        label = get_label_from_filename(fpath)
        if label is None:
            skipped += 1
            continue

        feats = extract_features(fpath)
        if feats is None:
            skipped += 1
            continue

        X.append(feats)
        y.append(label)

        if i % 20 == 0 or i == len(wav_files):
            print(f"  [{i}/{len(wav_files)}] processed — {len(X)} valid samples so far")

    print(f"\nFeature extraction complete: {len(X)} valid, {skipped} skipped.")
    return np.array(X, dtype=np.float32), y, skipped

