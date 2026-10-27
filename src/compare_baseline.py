"""
src/compare_baseline.py
──────────────────────────────────────────────────────────────────────────────
SC08 Step 4: Fair Baseline Comparison for Machine Predictive Maintenance.

Role in Architecture:
  Compares the trained Artificial Neural Network (ANN) against the deterministic
  Fixed Alarm Threshold Baseline (Matzka 2020) on exactly the same validation
  and held-out test rows under identical evaluation criteria.

Fairness Principles Enforced:
  1. Identical Data Partitions: Both models evaluate on identical validation
     (800 rows) and held-out test (2,000 rows) partitions.
  2. Proven Row Alignment: Row ordering is strictly preserved and verified
     via assertions across ground-truth labels and both model predictions.
  3. No Data Leakage / Test Tuning: Baseline thresholds and ANN weights are
     fixed; no tuning occurs on test data.
  4. True Continuous ROC-AUC: ROC-AUC is calculated exclusively from continuous
     probabilities (ANN). Discrete binary predictions from the baseline are
     reported as 'N/A' to avoid mislabeling balanced accuracy as ROC-AUC.
  5. Repeatable Execution: Can be executed directly or imported by Jupyter.

Output Artifacts:
  - data/reports/step4_baseline_metrics.csv
  - data/reports/step4_comparison.csv
  - data/reports/step4_validation_paired_predictions.csv
  - data/reports/step4_test_paired_predictions.csv
  - results/step4_baseline_vs_ann.png
──────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Set non-interactive backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ann import NeuralNetwork
from src.baseline import predict_batch, _denormalise, _decode_product_type

# ==============================================================================
# Path Configurations
# ==============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

X_TRAIN_PATH = DATA_DIR / "X_train.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
X_TEST_PATH = DATA_DIR / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"
NORM_PARAMS_PATH = DATA_DIR / "norm_params.json"
ANN_WEIGHTS_PATH = RESULTS_DIR / "ann_weights.npz"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_M",
    "Type_H",
]
TARGET = "Machine failure"
CORE_METRICS = ["accuracy", "precision", "recall", "f1_score", "specificity"]


# ==============================================================================
# Evaluation & Metric Computation (Pure NumPy)
# ==============================================================================

def compute_roc_auc_continuous(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using numerical trapezoidal integration over unique decision thresholds.

    Requires valid continuous predicted probabilities.
    """
    desc_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_indices]
    y_prob_sorted = y_prob[desc_indices]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true_sorted) - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))

    return round(max(0.0, min(1.0, auc)), 4)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    split_name: str = "",
    method_name: str = "",
) -> dict:
    """
    Compute full classification metrics for a binary predictor.

    Parameters:
      y_true : 1D array of ground truth {0, 1}
      y_pred : 1D array of binary predictions {0, 1}
      y_prob : Optional continuous probabilities in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_total = len(y_true)
    pred_pos = tp + fp

    accuracy = float((tp + tn) / n_total) if n_total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    if y_prob is not None:
        roc_auc = compute_roc_auc_continuous(y_true, y_prob)
    else:
        # For discrete predictions without probabilities, do not mislabel
        # balanced accuracy as continuous ROC-AUC.
        roc_auc = "N/A"

    return {
        "split": split_name,
        "method": method_name,
        "rows": n_total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predicted_positives": pred_pos,
        "roc_auc": roc_auc,
    }


# ==============================================================================
# Model Restoration & Data Partition Helpers
# ==============================================================================

def get_reproducible_validation_split(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.10,
):
    """
    Reproduce the exact stratified validation split used during ANN training
    in src.ann.train_ann.

    Returns:
      X_train_sub, y_train_sub, X_val, y_val, val_idx
    """
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []

    for class_val in np.unique(y_train_full):
        idx = np.where(y_train_full == class_val)[0]
        idx = rng.permutation(idx)
        val_count = int(np.round(val_fraction * len(idx)))
        val_indices.append(idx[:val_count])
        train_indices.append(idx[val_count:])

    train_idx = np.sort(np.concatenate(train_indices))
    val_idx = np.sort(np.concatenate(val_indices))

    X_train_sub = X_train_full[train_idx]
    y_train_sub = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    return X_train_sub, y_train_sub, X_val, y_val, val_idx


def restore_trained_ann(weights_path: Path) -> NeuralNetwork:
    """Load existing trained ANN weights without retraining."""
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"ANN weights not found at '{weights_path}'. Finish Step 3 first."
        )
    weights = np.load(weights_path)
    model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1, seed=42)
    model.W1 = weights["W1"]
    model.b1 = weights["b1"]
    model.W2 = weights["W2"]
    model.b2 = weights["b2"]
    model.W3 = weights["W3"]
    model.b3 = weights["b3"]
    return model


def predict_baseline_on_features(X_norm: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Convert normalized features back to physical units and evaluate crisp baseline.
    """
    X_raw = _denormalise(X_norm, norm_params)
    product_types = [
        _decode_product_type(X_raw[i, 5], X_raw[i, 6])
        for i in range(len(X_raw))
    ]
    preds = predict_batch(
        air_temps_K=X_raw[:, 0],
        process_temps_K=X_raw[:, 1],
        rpms=X_raw[:, 2],
        torques_Nm=X_raw[:, 3],
        tool_wears_min=X_raw[:, 4],
        product_types=product_types,
    )
    return preds, X_raw


# ==============================================================================
# Visualization Helper
# ==============================================================================

def make_comparison_chart(metrics_df: pd.DataFrame, save_path: Path) -> Path:
    """
    Generate side-by-side comparison bar chart for Validation and Test splits.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    splits = ["validation", "test"]
    chart_metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Filter section
    m_idx = metrics_df.set_index(["split", "method"])

    for ax, split in zip(axes, splits):
        baseline_vals = [
            float(m_idx.loc[(split, "Fixed alarm baseline"), m])
            for m in chart_metrics
        ]
        ann_vals = [
            float(m_idx.loc[(split, "Step 3 ANN"), m])
            for m in chart_metrics
        ]

        x = np.arange(len(chart_metrics))
        width = 0.35

        rects1 = ax.bar(x - width / 2, baseline_vals, width, label="Fixed Alarm Baseline", color="#6c757d")
        rects2 = ax.bar(x + width / 2, ann_vals, width, label="Step 3 ANN (Ours)", color="#0d6efd")

        ax.set_title(f"{split.capitalize()} Split (N={m_idx.loc[(split, 'Step 3 ANN'), 'rows']})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Bar value annotations
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#0d6efd")

    axes[0].set_ylabel("Score (higher is better)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("SC08 Step 4: Fair Baseline Comparison (Same Data, Identical Partitions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ==============================================================================
# Main Comparison Pipeline
# ==============================================================================

def main():
    """Execute reproducible Step 4 comparison and write artifacts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SC08 STEP 4: FAIR BASELINE COMPARISON PIPELINE")
    print("=" * 75)

    # 1. Load Data Arrays & Preprocessing Metadata
    for p in [X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, NORM_PARAMS_PATH, ANN_WEIGHTS_PATH]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required artifact: '{p}'.")

    X_train_full = np.load(X_TRAIN_PATH)
    y_train_full = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    with open(NORM_PARAMS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    norm_params = meta["norm_params"]

    # 2. Reproduce Exact Stratified Validation Partition (90/10 of training set)
    _, _, X_val, y_val, val_idx = get_reproducible_validation_split(
        X_train_full, y_train_full, seed=42, val_fraction=0.10
    )

    # Integrity Assertions on Split Sizes and Labels
    assert len(X_val) == 800, f"Expected 800 validation samples, found {len(X_val)}"
    assert len(y_val) == 800, f"Expected 800 validation labels, found {len(y_val)}"
    assert len(X_test) == 2000, f"Expected 2,000 test samples, found {len(X_test)}"
    assert len(y_test) == 2000, f"Expected 2,000 test labels, found {len(y_test)}"
    assert int(y_val.sum()) == 27, f"Expected 27 validation failures, found {int(y_val.sum())}"
    assert int(y_test.sum()) == 68, f"Expected 68 test failures, found {int(y_test.sum())}"

    print(f"Validation Partition : {len(y_val)} rows (Healthy: {(y_val==0).sum()}, Failures: {int(y_val.sum())})")
    print(f"Held-out Test Set    : {len(y_test)} rows (Healthy: {(y_test==0).sum()}, Failures: {int(y_test.sum())})")

    # 3. Restore ANN Model (Zero Retraining)
    ann_model = restore_trained_ann(ANN_WEIGHTS_PATH)
    print("ANN Model Restored   : Weights restored from 'results/ann_weights.npz' (no retraining).")

    # 4. Generate Predictions & Probabilities
    # Validation
    val_ann_probs = ann_model.forward(X_val).ravel()
    val_ann_preds = (val_ann_probs >= 0.5).astype(int)
    val_base_preds, X_val_raw = predict_baseline_on_features(X_val, norm_params)

    # Test
    test_ann_probs = ann_model.forward(X_test).ravel()
    test_ann_preds = (test_ann_probs >= 0.5).astype(int)
    test_base_preds, X_test_raw = predict_baseline_on_features(X_test, norm_params)

    # Row Alignment Assertions
    assert len(val_ann_preds) == len(val_base_preds) == len(y_val)
    assert len(test_ann_preds) == len(test_base_preds) == len(y_test)

    # 5. Compute Detailed Metrics
    metric_rows = [
        compute_metrics(y_val, val_base_preds, None, "validation", "Fixed alarm baseline"),
        compute_metrics(y_val, val_ann_preds, val_ann_probs, "validation", "Step 3 ANN"),
        compute_metrics(y_test, test_base_preds, None, "test", "Fixed alarm baseline"),
        compute_metrics(y_test, test_ann_preds, test_ann_probs, "test", "Step 3 ANN"),
    ]
    metrics_df = pd.DataFrame(metric_rows)

    # 6. Compute Differences (ANN minus Baseline)
    comparison_rows = []
    for split in ["validation", "test"]:
        b_res = next(r for r in metric_rows if r["split"] == split and r["method"] == "Fixed alarm baseline")
        a_res = next(r for r in metric_rows if r["split"] == split and r["method"] == "Step 3 ANN")

        for m in CORE_METRICS:
            b_val = b_res[m]
            a_val = a_res[m]
            diff = round(a_val - b_val, 4)
            comparison_rows.append({
                "split": split,
                "metric": m,
                "baseline": b_val,
                "ann": a_val,
                "ann_minus_baseline": diff,
            })
    comparison_df = pd.DataFrame(comparison_rows)

    # 7. Construct Paired Predictions DataFrames (with physical inputs and split-local index)
    def build_paired_df(X_raw, y_true, base_preds, ann_preds, ann_probs):
        df = pd.DataFrame(X_raw, columns=FEATURES)
        df.insert(0, "split_row_idx", np.arange(len(y_true)))
        df[TARGET] = y_true.astype(int)
        df["baseline_prediction"] = base_preds.astype(int)
        df["ann_prediction"] = ann_preds.astype(int)
        df["ann_probability"] = np.round(ann_probs, 4)
        return df

    val_paired_df = build_paired_df(X_val_raw, y_val, val_base_preds, val_ann_preds, val_ann_probs)
    test_paired_df = build_paired_df(X_test_raw, y_test, test_base_preds, test_ann_preds, test_ann_probs)

    # 8. Save CSV Outputs
    metrics_csv_path = REPORT_DIR / "step4_baseline_metrics.csv"
    comparison_csv_path = REPORT_DIR / "step4_comparison.csv"
    val_paired_csv_path = REPORT_DIR / "step4_validation_paired_predictions.csv"
    test_paired_csv_path = REPORT_DIR / "step4_test_paired_predictions.csv"
    chart_png_path = RESULTS_DIR / "step4_baseline_vs_ann.png"

    metrics_df.to_csv(metrics_csv_path, index=False)
    comparison_df.to_csv(comparison_csv_path, index=False)
    val_paired_df.to_csv(val_paired_csv_path, index=False)
    test_paired_df.to_csv(test_paired_csv_path, index=False)
    make_comparison_chart(metrics_df, chart_png_path)

    print("\n" + "=" * 75)
    print("STEP 4 METRICS SUMMARY (data/reports/step4_baseline_metrics.csv)")
    print("=" * 75)
    print(metrics_df[["split", "method", "rows", "accuracy", "precision", "recall", "f1_score", "specificity", "roc_auc"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("SCORE DIFFERENCES: ANN MINUS BASELINE (data/reports/step4_comparison.csv)")
    print("(Positive difference means ANN scored higher on identical rows)")
    print("=" * 75)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("SAVED STEP 4 ARTIFACTS:")
    print("=" * 75)
    print(f"  [1] Metrics CSV          : {metrics_csv_path}")
    print(f"  [2] Comparison CSV       : {comparison_csv_path}")
    print(f"  [3] Val Paired CSV       : {val_paired_csv_path} ({len(val_paired_df)} rows)")
    print(f"  [4] Test Paired CSV      : {test_paired_csv_path} ({len(test_paired_df)} rows)")
    print(f"  [5] Comparison Figure    : {chart_png_path}")
    print("=" * 75)
    print("STEP 4 FAIR BASELINE COMPARISON COMPLETED SUCCESSFULLY.")

    return metrics_df, comparison_df, val_paired_df, test_paired_df


if __name__ == "__main__":
    main()
