"""
train.py
--------
Trains, evaluates, and saves the emotion classification model.

Usage:
    python train.py --dataset /path/to/wav/folder

Outputs:
    - model.joblib
    - scaler.joblib
    - label_encoder.joblib
    - results/  : plots and summary report
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.metrics           import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier






from feature_extraction import load_dataset

# -- Constants ---------------------------------------------------------------
RESULTS_DIR  = "results"
MODEL_PATH   = "model.joblib"
SCALER_PATH  = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"
TEST_SIZE    = 0.20
RANDOM_STATE = 42


# -- Helper functions --------------------------------------------------------

def save_confusion_matrix(cm, classes, accuracy, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.2%})", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


def save_class_distribution(y, output_path):
    from collections import Counter
    counts = Counter(y)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=plt.cm.Set2.colors[:len(labels)],
                  edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=10)
    ax.set_title("Class Distribution", fontsize=13)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Class distribution saved: {output_path}")


def save_feature_importance(model, output_path, top_n=20):
    """Show feature importances if model supports it (e.g. RandomForest)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices], color="steelblue")
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=13)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance Score")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(indices, rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved: {output_path}")


def compare_models(X_train, y_train, scaler):
    """Compare multiple models with 5-fold cross-validation."""
    print("\n-- Model Comparison (5-fold CV) --")

    X_scaled = scaler.transform(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Random Forest"    : RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)"        : SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE, probability=True),
        "KNN (k=7)"        : KNeighborsClassifier(n_neighbors=7),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE),
        "MLP"              : MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
                          random_state=RANDOM_STATE, eval_metric='mlogloss',
                          verbosity=0)
    }

    scores = {}
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, X_scaled, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        scores[name] = cv_scores
        print(f"  {name:<25} mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    best_name = max(scores, key=lambda k: scores[k].mean())
    print(f"\n  Best model: {best_name} ({scores[best_name].mean():.4f})")
    return best_name, candidates[best_name]

def train_voting_ensemble(X_train_s, y_train, encoder):
    """Top 3 modeli birleştiren Voting Classifier."""
    print("\n-- Voting Ensemble (Top 3 Models) --")
    
    gb  = GradientBoostingClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)
    svm = SVC(kernel="rbf", C=50, gamma="scale", probability=True, random_state=RANDOM_STATE)
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=800, 
                     early_stopping=True, random_state=RANDOM_STATE)
    
    voting = VotingClassifier(
        estimators=[("gb", gb), ("svm", svm), ("mlp", mlp)],
        voting="soft"
    )
    voting.fit(X_train_s, y_train)
    return voting


def optimize_model(best_name, best_clf, X_train, y_train, scaler):
    """GridSearchCV ile en iyi modelin parametrelerini optimize eder."""
    print(f"\n-- GridSearchCV Optimization for {best_name} --")
    
    X_scaled = scaler.transform(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grids = {
        "SVM (RBF)": {
            "C"    : [1, 5, 10, 50],
            "gamma": ["scale", "auto", 0.001, 0.01],
        },
        "Gradient Boosting": {
            "n_estimators" : [200, 300, 400],
            "max_depth"    : [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "Random Forest": {
            "n_estimators"     : [200, 300, 400],
            "max_depth"        : [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "MLP": {
            "hidden_layer_sizes": [(128, 64), (256, 128), (256, 128, 64)],
            "alpha"             : [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },
        "XGBoost": {
            "n_estimators" : [200, 300, 400],
            "max_depth"    : [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        }
    }

    if best_name not in param_grids:
        print(f"  No param grid defined for {best_name}, skipping optimization.")
        return best_clf

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(
        best_clf, param_grids[best_name],
        cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X_scaled, y_train)

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV acc : {grid.best_score_:.4f}")
    return grid.best_estimator_
    
    

    scores = {}
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, X_scaled, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        scores[name] = cv_scores
        print(f"  {name:<25} mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    best_name = max(scores, key=lambda k: scores[k].mean())
    print(f"\n  Best model: {best_name} ({scores[best_name].mean():.4f})")
    return best_name, candidates[best_name]


# -- Main training function --------------------------------------------------

def train(dataset_dir: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load dataset
    print("=" * 55)
    print("  EMOTION CLASSIFIER -- TRAINING")
    print("=" * 55)
    X, y_raw, skipped = load_dataset(dataset_dir)

    if len(X) == 0:
        print("[ERROR] No valid samples found!")
        return

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"\nClasses: {classes}")

    # 3. Class distribution plot
    save_class_distribution(y_raw, os.path.join(RESULTS_DIR, "class_distribution.png"))

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=300)
    X_train_s = selector.fit_transform(X_train_s, y_train)
    X_test_s  = selector.transform(X_test_s)
    joblib.dump(selector, "selector.joblib")
    print(f"  Feature selection: 560 → 200 dims")

    # 6. Compare models
    best_name, best_clf = compare_models(X_train, y_train, scaler)
    best_clf = optimize_model(best_name, best_clf, X_train, y_train, scaler)

    # 7. Train best model on full training set
    print(f"\n-- Training {best_name} on full training set...")
    best_clf.fit(X_train_s, y_train)

    # 7.5 Voting Ensemble dene
    print("\n-- Training Voting Ensemble (GB + SVM + MLP)...")
    from sklearn.ensemble import VotingClassifier
    gb  = GradientBoostingClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_STATE)
    voting = VotingClassifier(
        estimators=[("gb", gb), ("svm", svm), ("mlp", mlp), ("xgb", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric='mlogloss', verbosity=0))],
        voting="soft"
    )
    voting.fit(X_train_s, y_train)
    acc_v = accuracy_score(y_test, voting.predict(X_test_s))
    print(f"  Voting Ensemble Accuracy: {acc_v:.4f} ({acc_v:.2%})")
    acc_best = accuracy_score(y_test, best_clf.predict(X_test_s))
    if acc_v > acc_best:
        print("  → Voting Ensemble daha iyi, onu kullanıyoruz!")
        best_clf = voting
        best_name = "Voting Ensemble"

    # 8. Evaluate on test set
    y_pred     = best_clf.predict(X_test_s)
    accuracy   = accuracy_score(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)
    report     = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=classes)

    print(f"\n-- Test Results --")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy:.2%})")
    print(f"\n{report_str}")

    # 9. Save plots
    save_confusion_matrix(cm, classes, accuracy,
                          os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    save_feature_importance(best_clf,
                            os.path.join(RESULTS_DIR, "feature_importance.png"))

    # 10. Save summary JSON
    summary = {
        "model"     : best_name,
        "accuracy"  : round(accuracy, 4),
        "classes"   : classes,
        "train_size": len(X_train),
        "test_size" : len(X_test),
        "skipped"   : skipped,
        "report"    : report
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary JSON : {RESULTS_DIR}/summary.json")

    # 11. Save model artifacts
    joblib.dump(best_clf, MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(le,       ENCODER_PATH)
    print(f"  Model        : {MODEL_PATH}")
    print(f"  Scaler       : {SCALER_PATH}")
    print(f"  Encoder      : {ENCODER_PATH}")

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE -- Accuracy: {accuracy:.2%}")
    print("=" * 55)

    return accuracy, classes


# -- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classifier -- Model Training")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the folder containing WAV files"
    )
    args = parser.parse_args()
    train(args.dataset)
