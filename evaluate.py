"""
evaluate.py
-----------
Evaluates the trained model on the full dataset,
performs error analysis, and generates a detailed report.

Usage:
    python evaluate.py --dataset /path/to/wav/folder
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from feature_extraction import load_dataset

MODEL_PATH   = "model.joblib"
SCALER_PATH  = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"
EVAL_DIR     = "results/evaluation"


def plot_normalized_confusion_matrix(cm, classes, output_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix: {output_path}")


def plot_per_class_metrics(report_dict, classes, output_path):
    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report_dict[c][m] for c in classes] for m in metrics}

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, data[metric], width, label=metric.capitalize(), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Per-class metrics: {output_path}")


def plot_error_analysis(y_true_labels, y_pred_labels, classes, output_path):
    """Analyze which classes are being confused with each other."""
    errors = [(t, p) for t, p in zip(y_true_labels, y_pred_labels) if t != p]

    if not errors:
        print("  No errors — perfect classification!")
        return

    error_df = pd.DataFrame(errors, columns=["Actual", "Predicted"])
    error_counts = error_df.groupby(["Actual", "Predicted"]).size().reset_index(name="Count")
    error_counts = error_counts.sort_values("Count", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['Actual']} -> {r['Predicted']}" for _, r in error_counts.iterrows()]
    ax.barh(labels, error_counts["Count"], color="#C44E52")
    ax.set_xlabel("Error Count")
    ax.set_title("Most Frequent Misclassifications", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Error analysis: {output_path}")


def generate_text_report(accuracy, report_dict, classes, error_pairs, output_path):
    """Generate a plain-text detailed report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  EMOTION CLASSIFIER -- EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"\nOverall Accuracy: {accuracy:.4f}  ({accuracy:.2%})\n")

    lines.append("Per-Class Results:")
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 55)
    for cls in classes:
        r = report_dict[cls]
        lines.append(
            f"{cls:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1-score']:>10.4f} {int(r['support']):>10}"
        )

    lines.append("\n" + "-" * 55)
    macro = report_dict.get("macro avg", {})
    lines.append(
        f"{'Macro Avg':<15} {macro.get('precision', 0):>10.4f} "
        f"{macro.get('recall', 0):>10.4f} {macro.get('f1-score', 0):>10.4f}"
    )

    if error_pairs:
        lines.append("\nMost Frequent Errors (Actual -> Predicted):")
        for (t, p), count in sorted(error_pairs.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {t:<12} -> {p:<12}  ({count} times)")

    lines.append("\n" + "=" * 60)

    text = "\n".join(lines)
    print(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Text report: {output_path}")


def evaluate(dataset_dir: str):
    os.makedirs(EVAL_DIR, exist_ok=True)

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        raise FileNotFoundError("Model not found. Run train.py first.")

    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    classes = list(encoder.classes_)

    print("\nLoading dataset...")
    X, y_raw, _ = load_dataset(dataset_dir)

    X_scaled = scaler.transform(X)
    y_true   = encoder.transform(y_raw)
    y_pred   = model.predict(X_scaled)

    y_true_labels = encoder.inverse_transform(y_true)
    y_pred_labels = encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    from collections import Counter
    error_pairs = Counter(
        (t, p) for t, p in zip(y_true_labels, y_pred_labels) if t != p
    )

    plot_normalized_confusion_matrix(cm, classes, os.path.join(EVAL_DIR, "confusion_matrix_full.png"))
    plot_per_class_metrics(report, classes, os.path.join(EVAL_DIR, "per_class_metrics.png"))
    plot_error_analysis(y_true_labels, y_pred_labels, classes, os.path.join(EVAL_DIR, "error_analysis.png"))
    generate_text_report(accuracy, report, classes, dict(error_pairs), os.path.join(EVAL_DIR, "report.txt"))

    with open(os.path.join(EVAL_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(accuracy, 4),
            "classes": classes,
            "report": report,
            "top_errors": [
                {"actual": t, "predicted": p, "count": c}
                for (t, p), c in sorted(error_pairs.items(), key=lambda x: -x[1])[:10]
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Overall Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classifier -- Evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to folder containing WAV files")
    args = parser.parse_args()
    evaluate(args.dataset)
