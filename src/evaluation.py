"""
src/evaluation.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive Evaluation Module for Predictive Maintenance.

Role in the architecture
------------------------
Evaluates the complete predictive maintenance system on the held-out 2,000-sample test set:
  1. Artificial Neural Network (ANN) binary classifier
  2. Crisp Rule Reference Baseline (Matzka 2020)
  3. Hybrid ANN + Mamdani Fuzzy Severity & Priority System

Evaluation Philosophy
---------------------
- **Binary Classifiers (ANN & Crisp Baseline)**: Evaluated using standard classification metrics
  (Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC).
- **Hybrid Fuzzy Logic System**: Evaluated as an interpretable decision-support layer.
  CRITICAL: Pseudo-accuracy metrics are NOT invented for continuous fuzzy severity/priority.
  Instead, fuzzy output is evaluated using severity distribution statistics, priority breakdowns,
  rule activation frequencies, and correlation with ground-truth failures.

Notes
-----
- Test set size (2,000 samples) is verified via assertions and left completely unmutated.
- Pure NumPy metric computations are used — no scikit-learn, TensorFlow, PyTorch, or scikit-fuzzy.
- Saves results to results/metrics.json and visual figures to results/figures/.
"""

import json
import math
import os
import sys

# Ensure parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.ann import NeuralNetwork
from src.baseline import predict_batch, _denormalise, _decode_product_type
from src.fuzzy import FuzzyMaintenance


# ==============================================================================
# File Paths
# ==============================================================================

PROCESSED_DIR: str = os.path.join("data", "processed")
RESULTS_DIR: str = "results"
FIGURES_DIR: str = os.path.join(RESULTS_DIR, "figures")

X_TEST_PATH: str = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH: str = os.path.join(PROCESSED_DIR, "y_test.npy")
NORM_PARAMS_PATH: str = os.path.join(PROCESSED_DIR, "norm_params.json")
WEIGHTS_PATH: str = os.path.join(RESULTS_DIR, "ann_weights.npz")
METRICS_SAVE_PATH: str = os.path.join(RESULTS_DIR, "metrics.json")


# ==============================================================================
# Metric Computation Functions (Pure NumPy)
# ==============================================================================

def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using numerical trapezoidal integration over unique decision thresholds.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,) — ground truth binary labels {0, 1}
    y_prob : np.ndarray of shape (N,) — predicted failure probabilities in [0, 1]

    Returns
    -------
    float — ROC-AUC score in [0.0, 1.0]
    """
    # Sort samples by predicted probability in descending order
    desc_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_indices]
    y_prob_sorted = y_prob[desc_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined/trivial case

    # Identify unique probability threshold transitions
    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true_sorted) - 1]

    # Cumulative true positives and false positives at thresholds
    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]

    # Trapezoidal integration for area under curve
    # Support both np.trapezoid (NumPy 2.0+) and np.trapz (older NumPy)
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))

    return max(0.0, min(1.0, auc))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """
    Compute full classification metrics for binary predictions.

    Metrics computed:
      - Confusion Matrix: TP, TN, FP, FN
      - Accuracy
      - Precision
      - Recall (Sensitivity)
      - Specificity
      - F1-Score
      - ROC-AUC (if probabilities supplied)

    Returns
    -------
    dict of float / int metrics
    """
    y_true = y_true.astype(int).ravel()
    y_pred_bin = y_pred_bin.astype(int).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))

    n_total = len(y_true)
    accuracy = float((tp + tn) / n_total) if n_total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if precision + recall > 0:
        f1_score = float(2.0 * precision * recall / (precision + recall))
    else:
        f1_score = 0.0

    if y_prob is not None:
        roc_auc = compute_roc_auc(y_true, y_prob)
    else:
        # For discrete predictions without probabilities, ROC-AUC is equal to (TPR + TNR)/2
        roc_auc = float((recall + specificity) / 2.0)

    return {
        "tp":          tp,
        "tn":          tn,
        "fp":          fp,
        "fn":          fn,
        "accuracy":    round(accuracy, 4),
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score":    round(f1_score, 4),
        "roc_auc":     round(roc_auc, 4),
    }


# ==============================================================================
# Model Evaluation Functions
# ==============================================================================

def evaluate_ann_model(
    model: NeuralNetwork,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> tuple:
    """
    Evaluate trained ANN model on the 2,000-sample test set.

    Note on Classification Threshold:
      threshold = 0.5 is used as the standard default decision boundary for converting
      continuous ANN failure probabilities into binary predictions. It is NOT claimed
      to be the mathematically optimal threshold for a specific cost function.
    """
    y_prob = model.forward(X_test).ravel()
    y_pred_bin = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test, y_pred_bin, y_prob=y_prob)
    return metrics, y_prob, y_pred_bin


def evaluate_crisp_baseline_model(
    X_test_norm: np.ndarray,
    y_test: np.ndarray,
    norm_params: dict,
) -> tuple:
    """
    Evaluate Crisp Baseline rules on the 2,000-sample test set.

    Input Handling:
      - X_test_norm is denormalized back to original physical sensor units (X_test_raw).
      - Crisp rules process physical values (temperatures in K, RPM, torque in Nm, tool wear in min).
    """
    # Denormalise numeric features to physical units
    X_test_raw = _denormalise(X_test_norm, norm_params)

    # Recover product types
    product_types = [
        _decode_product_type(X_test_raw[i, 5], X_test_raw[i, 6])
        for i in range(len(X_test_raw))
    ]

    # Run crisp rules
    y_pred_bin = predict_batch(
        air_temps_K=X_test_raw[:, 0],
        process_temps_K=X_test_raw[:, 1],
        rpms=X_test_raw[:, 2],
        torques_Nm=X_test_raw[:, 3],
        tool_wears_min=X_test_raw[:, 4],
        product_types=product_types,
    )

    metrics = compute_classification_metrics(y_test, y_pred_bin, y_prob=None)
    return metrics, y_pred_bin, X_test_raw


def evaluate_hybrid_fuzzy_system(
    ann_probs: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Perform diagnostic evaluation of the Hybrid Fuzzy Logic System across the 2,000 test samples.

    CRITICAL INPUT HANDLING:
      - ANN probability (ann_probs) is the statistical failure likelihood in [0.0, 1.0].
      - X_test_raw contains unscaled physical sensor values (tool wear in min, torque in Nm).
      - Normalized X_test is NEVER passed directly into the fuzzy system as physical inputs.

    EVALUATION METHODOLOGY:
      - DOES NOT calculate synthetic classification accuracy for continuous fuzzy severity/priority.
      - Evaluates severity distribution statistics (min, max, mean, std, percentiles), priority
        level breakdowns (LOW/MEDIUM/HIGH/URGENT), priority vs ground-truth cross-tabulation,
        and rule activation frequencies across the dataset.
    """
    fuzzy = FuzzyMaintenance()

    severities = []
    priorities = []
    rule_counts = {r_idx: 0 for r_idx in range(1, 12)}
    rule_total_strengths = {r_idx: 0.0 for r_idx in range(1, 12)}

    n_samples = len(y_test)

    for i in range(n_samples):
        fp = float(ann_probs[i])
        tw = float(X_test_raw[i, 4])   # Tool wear [min] — raw physical value
        tq = float(X_test_raw[i, 3])   # Torque [Nm]     — raw physical value

        res = fuzzy.evaluate(fault_prob=fp, tool_wear_min=tw, torque_Nm=tq)

        severities.append(res["severity_index"])
        priorities.append(res["priority"])

        # Track fired rules
        for fr in res["fired_rules"]:
            r_num = fr["rule"]
            rule_counts[r_num] += 1
            rule_total_strengths[r_num] += fr["strength"]

    severities_arr = np.array(severities, dtype=np.float64)

    # 1. Severity Distribution Statistics
    sev_stats = {
        "min":    round(float(np.min(severities_arr)), 2),
        "max":    round(float(np.max(severities_arr)), 2),
        "mean":   round(float(np.mean(severities_arr)), 2),
        "std":    round(float(np.std(severities_arr)), 2),
        "p25":    round(float(np.percentile(severities_arr, 25)), 2),
        "median": round(float(np.median(severities_arr)), 2),
        "p75":    round(float(np.percentile(severities_arr, 75)), 2),
    }

    # 2. Priority Level Distribution
    prio_order = ["LOW", "MEDIUM", "HIGH", "URGENT"]
    prio_counts = {p: int(priorities.count(p)) for p in prio_order}
    prio_pcts = {p: round(100.0 * prio_counts[p] / n_samples, 2) for p in prio_order}

    # 3. Cross-Tabulation: Priority vs Ground-Truth Target
    # Shows how many actual failures (y=1) and healthy samples (y=0) map to each priority level
    prio_cross_tab = {p: {"y_0_healthy": 0, "y_1_failure": 0} for p in prio_order}
    for p, y_val in zip(priorities, y_test):
        if y_val == 1:
            prio_cross_tab[p]["y_1_failure"] += 1
        else:
            prio_cross_tab[p]["y_0_healthy"] += 1

    # 4. Rule Activation Statistics
    rule_stats = {}
    for r_idx in range(1, 12):
        count = rule_counts[r_idx]
        avg_str = float(rule_total_strengths[r_idx] / count) if count > 0 else 0.0
        rule_stats[f"R{r_idx}"] = {
            "times_fired": count,
            "fire_percentage": round(100.0 * count / n_samples, 2),
            "avg_strength_when_fired": round(avg_str, 4),
        }

    return {
        "severity_summary":       sev_stats,
        "priority_counts":        prio_counts,
        "priority_percentages":   prio_pcts,
        "priority_vs_ground_truth": prio_cross_tab,
        "rule_firing_statistics": rule_stats,
        "severities_array":       severities_arr,
        "priorities_list":        priorities,
    }


# ==============================================================================
# Visualization Functions (Matplotlib + Pure-Python SVG Fallback)
# ==============================================================================

def generate_evaluation_figures(
    ann_metrics: dict,
    baseline_metrics: dict,
    hybrid_analysis: dict,
    figures_dir: str = FIGURES_DIR,
):
    """
    Generate and save evaluation plots:
      1. ANN Confusion Matrix
      2. Baseline Confusion Matrix
      3. Metric Comparison Bar Chart (ANN vs Baseline)
      4. Hybrid Severity & Priority Distribution
    """
    os.makedirs(figures_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        # ── Figure 1 & 2: Confusion Matrices ──────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        def plot_cm_ax(ax, cm_dict, title, color_map):
            cm = np.array([
                [cm_dict["tn"], cm_dict["fp"]],
                [cm_dict["fn"], cm_dict["tp"]]
            ])
            im = ax.imshow(cm, cmap=color_map, interpolation='nearest')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['No Failure (0)', 'Failure (1)'])
            ax.set_yticklabels(['No Failure (0)', 'Failure (1)'])
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)

            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    color = "white" if val > cm.max() / 2 else "black"
                    ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontweight='bold', fontsize=12)

        plot_cm_ax(ax1, ann_metrics, "ANN Confusion Matrix (Test N=2000)", plt.cm.Blues)
        plot_cm_ax(ax2, baseline_metrics, "Crisp Baseline Confusion Matrix", plt.cm.Oranges)

        plt.tight_layout()
        cm_path = os.path.join(figures_dir, "confusion_matrices.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ── Figure 3: Metric Comparison Bar Chart ─────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        ann_vals = [ann_metrics[m] for m in metrics_to_plot]
        base_vals = [baseline_metrics[m] for m in metrics_to_plot]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, base_vals, width, label='Crisp Baseline (Rules)', color='#ff7f0e')
        rects2 = ax.bar(x + width/2, ann_vals, width, label='ANN Model (Data-Driven)', color='#1f77b4')

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Performance Comparison — ANN Model vs Crisp Baseline (Test N=2000)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(fontsize=10)

        # Value labels on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        comp_path = os.path.join(figures_dir, "ann_vs_baseline_metrics.png")
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ── Figure 4: Hybrid Priority & Severity Distribution ──────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Priority Counts Bar Chart
        prio_counts = hybrid_analysis["priority_counts"]
        prio_names = list(prio_counts.keys())
        prio_vals = list(prio_counts.values())
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

        ax1.bar(prio_names, prio_vals, color=colors, edgecolor='black', alpha=0.85)
        ax1.set_title('Fuzzy Maintenance Priority Breakdown (Test N=2000)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        for i, v in enumerate(prio_vals):
            ax1.text(i, v + 25, f"{v}\n({v/2000*100:.1f}%)", ha='center', fontsize=9, fontweight='bold')

        # Severity Histogram
        sevs = hybrid_analysis["severities_array"]
        ax2.hist(sevs, bins=30, color='#9467bd', edgecolor='black', alpha=0.75)
        ax2.set_title('Fuzzy Severity Index Distribution [0, 100]', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Severity Index', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        ax2.axvline(x=np.mean(sevs), color='red', linestyle='--', linewidth=2, label=f"Mean ({np.mean(sevs):.1f})")
        ax2.axvline(x=np.median(sevs), color='green', linestyle=':', linewidth=2, label=f"Median ({np.median(sevs):.1f})")
        ax2.legend(fontsize=9)

        plt.tight_layout()
        hybrid_path = os.path.join(figures_dir, "hybrid_priority_distribution.png")
        plt.savefig(hybrid_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[FIGURES SAVED] All Matplotlib evaluation figures saved to '{figures_dir}/'")

    except Exception as e:
        print(f"\n[NOTE] Matplotlib rendering skipped ({e}). Generating pure-Python SVG fallbacks...")
        _generate_svg_figures(ann_metrics, baseline_metrics, hybrid_analysis, figures_dir)


def _generate_svg_figures(ann_m: dict, base_m: dict, hybrid_a: dict, figures_dir: str):
    """Pure-Python SVG figure generator fallback."""
    comp_svg_path = os.path.join(figures_dir, "ann_vs_baseline_metrics.svg")
    width, height = 800, 400

    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    keys   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="400" y="30" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">Model Comparison — ANN vs Crisp Baseline (Test N=2000)</text>',
        '<rect x="80" y="50" width="680" height="280" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
    ]

    for i in range(6):
        val = i * 0.2
        py = 330 - val * 250
        svg.append(f'<line x1="80" y1="{py:.1f}" x2="760" y2="{py:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="70" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{val:.1f}</text>')

    for idx, key in enumerate(keys):
        cx = 130 + idx * 135
        bv = base_m[key]
        av = ann_m[key]

        bh = bv * 250
        by = 330 - bh
        svg.append(f'<rect x="{cx - 24}" y="{by:.1f}" width="22" height="{bh:.1f}" fill="#ff7f0e"/>')
        svg.append(f'<text x="{cx - 13}" y="{by - 5:.1f}" font-size="9" text-anchor="middle" fill="#333">{bv:.3f}</text>')

        ah = av * 250
        ay = 330 - ah
        svg.append(f'<rect x="{cx + 2}" y="{ay:.1f}" width="22" height="{ah:.1f}" fill="#1f77b4"/>')
        svg.append(f'<text x="{cx + 13}" y="{ay - 5:.1f}" font-size="9" font-weight="bold" text-anchor="middle" fill="#333">{av:.3f}</text>')

        svg.append(f'<text x="{cx}" y="350" font-size="11" text-anchor="middle" fill="#333">{labels[idx]}</text>')

    svg.append('<rect x="260" y="375" width="14" height="14" fill="#ff7f0e"/>')
    svg.append('<text x="280" y="387" font-size="11" fill="#333">Crisp Baseline (Rules)</text>')
    svg.append('<rect x="440" y="375" width="14" height="14" fill="#1f77b4"/>')
    svg.append('<text x="460" y="387" font-size="11" fill="#333">ANN Model (Data-Driven)</text>')
    svg.append('</svg>')

    with open(comp_svg_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"\n[FIGURE SAVED] SVG comparison chart saved to '{comp_svg_path}'")


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================

def main():
    """
    Execute full evaluation pipeline on the untouched 2,000 test set.
    """
    print("=" * 75)
    print("PREDICTIVE MAINTENANCE SYSTEM — COMPREHENSIVE EVALUATION")
    print("=" * 75)

    # ── 1. Guard & Assertion Checks ──────────────────────────────────────────
    required_files = [X_TEST_PATH, Y_TEST_PATH, NORM_PARAMS_PATH, WEIGHTS_PATH]
    for p in required_files:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[ERROR] Required file not found: '{p}'\n"
                "Please run preprocessing and ANN training first."
            )

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    # Assertions required by prompt
    assert len(X_test) == 2000, f"Expected 2000 test samples, got {len(X_test)}."
    assert len(y_test) == 2000, f"Expected 2000 test labels, got {len(y_test)}."
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test contains invalid labels."

    with open(NORM_PARAMS_PATH, "r", encoding="utf-8") as f:
        norm_params = json.load(f)["norm_params"]

    # Instantiate ANN and load pre-trained weights
    ann_model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1)
    weights_npz = np.load(WEIGHTS_PATH)
    ann_model.set_weights({k: weights_npz[k] for k in weights_npz.files})

    print(f"Test Set Loaded Successfully : {len(y_test)} samples (Positives/Failures: {y_test.sum()})")
    print(f"ANN Model Loaded Successfully: Weights restored from '{WEIGHTS_PATH}'")

    # ── 2. Evaluate ANN Model ─────────────────────────────────────────────────
    ann_metrics, ann_probs, ann_preds = evaluate_ann_model(ann_model, X_test, y_test, threshold=0.5)

    # ── 3. Evaluate Crisp Baseline Model ──────────────────────────────────────
    baseline_metrics, base_preds, X_test_raw = evaluate_crisp_baseline_model(X_test, y_test, norm_params)

    # ── 4. Evaluate Hybrid Fuzzy System ───────────────────────────────────────
    hybrid_analysis = evaluate_hybrid_fuzzy_system(ann_probs, X_test_raw, y_test)

    # ── 5. Print Comparison Table ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON (TEST SET N = 2,000)")
    print("=" * 75)
    print(f"{'Metric':<20s} | {'Crisp Baseline':<16s} | {'ANN Model (Ours)':<18s} | {'Difference':<12s}")
    print("-" * 75)

    metrics_keys = [
        ("Accuracy",    "accuracy"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("Specificity", "specificity"),
        ("F1-Score",    "f1_score"),
        ("ROC-AUC",     "roc_auc"),
    ]

    for label, key in metrics_keys:
        b_val = baseline_metrics[key]
        a_val = ann_metrics[key]
        diff = a_val - b_val
        sign = "+" if diff > 0 else ""
        print(f"{label:<20s} | {b_val:<16.4f} | {a_val:<18.4f} | {sign}{diff:<.4f}")

    print("-" * 75)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  Crisp Baseline : TP={baseline_metrics['tp']:3d} | TN={baseline_metrics['tn']:4d} | FP={baseline_metrics['fp']:3d} | FN={baseline_metrics['fn']:3d}")
    print(f"  ANN Model      : TP={ann_metrics['tp']:3d} | TN={ann_metrics['tn']:4d} | FP={ann_metrics['fp']:3d} | FN={ann_metrics['fn']:3d}")
    print("=" * 75)

    # ── 6. Print Hybrid Fuzzy Diagnostics Summary ────────────────────────────
    print("\n" + "=" * 75)
    print("HYBRID FUZZY SYSTEM — DIAGNOSTIC EVALUATION")
    print("(Interpretable Decision-Support Layer — Not evaluated as crisp classifier)")
    print("=" * 75)
    print("1. Severity Index Distribution Statistics [0, 100]:")
    ss = hybrid_analysis["severity_summary"]
    print(f"   Min: {ss['min']:.2f} | Max: {ss['max']:.2f} | Mean: {ss['mean']:.2f} | Std: {ss['std']:.2f}")
    print(f"   25th Percentile: {ss['p25']:.2f} | Median: {ss['median']:.2f} | 75th Percentile: {ss['p75']:.2f}")

    print("\n2. Maintenance Priority Breakdown (N = 2,000):")
    for prio in ["LOW", "MEDIUM", "HIGH", "URGENT"]:
        cnt = hybrid_analysis["priority_counts"][prio]
        pct = hybrid_analysis["priority_percentages"][prio]
        ct = hybrid_analysis["priority_vs_ground_truth"][prio]
        print(f"   {prio:<8s} : {cnt:4d} samples ({pct:5.2f}%)  [Healthy y=0: {ct['y_0_healthy']:4d} | Actual Failures y=1: {ct['y_1_failure']:3d}]")

    print("\n3. Top Fired Fuzzy Rules Across Test Set:")
    rule_stats = hybrid_analysis["rule_firing_statistics"]
    sorted_rules = sorted(rule_stats.items(), key=lambda x: x[1]["times_fired"], reverse=True)
    for r_name, r_info in sorted_rules[:5]:
        print(f"   {r_name} fired {r_info['times_fired']:4d} times ({r_info['fire_percentage']:5.2f}% of test set) with avg strength {r_info['avg_strength_when_fired']:.4f}")
    print("=" * 75)

    # ── 7. Save JSON Metrics File ─────────────────────────────────────────────
    save_payload = {
        "test_set_size": 2000,
        "true_failures_in_test_set": int(y_test.sum()),
        "ann_metrics": ann_metrics,
        "baseline_metrics": baseline_metrics,
        "hybrid_fuzzy_analysis": {
            "severity_summary": hybrid_analysis["severity_summary"],
            "priority_counts": hybrid_analysis["priority_counts"],
            "priority_percentages": hybrid_analysis["priority_percentages"],
            "priority_vs_ground_truth": hybrid_analysis["priority_vs_ground_truth"],
            "rule_firing_statistics": hybrid_analysis["rule_firing_statistics"],
        }
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)
    print(f"\n[METRICS SAVED] Complete numerical results saved to '{METRICS_SAVE_PATH}'")

    # ── 8. Generate Visual Figures ────────────────────────────────────────────
    generate_evaluation_figures(ann_metrics, baseline_metrics, hybrid_analysis, FIGURES_DIR)


if __name__ == "__main__":
    main()
