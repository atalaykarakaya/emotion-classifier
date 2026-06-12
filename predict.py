"""
predict.py
----------
Classifies a single WAV file using the trained model.

Usage:
    python predict.py --file /path/to/test.wav
    python predict.py --file /path/to/test.wav --verbose
"""

import os
import argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from feature_extraction import extract_features

MODEL_PATH   = "model.joblib"
SCALER_PATH  = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"

EMOTION_EMOJI = {
    "happy"    : "😊",
    "sad"      : "😢",
    "angry"    : "😠",
    "furious"  : "🤬",
    "neutral"  : "😐",
    "surprised": "😲",
}


def load_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        raise FileNotFoundError(
            "Model files not found. Run 'python train.py --dataset <folder>' first."
        )
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


def predict_file(file_path: str, verbose: bool = False):
    """
    Predicts the emotion for a single WAV file.
    Returns: (predicted_label, confidence_dict)
    """
    model, scaler, encoder = load_model()

    print(f"\nFile: {file_path}")
    feats = extract_features(file_path)
    if feats is None:
        print("[ERROR] Could not extract features.")
        return None, None

    feats_scaled = scaler.transform(feats.reshape(1, -1))

    # Prediction
    pred_idx   = model.predict(feats_scaled)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    # Probabilities (if model supports it)
    conf_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_scaled)[0]
        for cls, prob in zip(encoder.classes_, probs):
            conf_dict[cls] = round(float(prob), 4)

    # Output
    emoji = EMOTION_EMOJI.get(pred_label, "🎵")
    print(f"\n  Prediction : {emoji}  {pred_label.upper()}")

    if conf_dict:
        print("\n  Probability Distribution:")
        for cls in sorted(conf_dict, key=conf_dict.get, reverse=True):
            bar_len = int(conf_dict[cls] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " <--" if cls == pred_label else ""
            print(f"    {cls:<12} {bar}  {conf_dict[cls]:.2%}{marker}")

    if verbose:
        print(f"\n  Feature vector size: {feats.shape[0]}")

    return pred_label, conf_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction -- Single File")
    parser.add_argument("--file",    type=str, required=True, help="Path to WAV file")
    parser.add_argument("--verbose", action="store_true",     help="Verbose output")
    args = parser.parse_args()

    predict_file(args.file, verbose=args.verbose)
