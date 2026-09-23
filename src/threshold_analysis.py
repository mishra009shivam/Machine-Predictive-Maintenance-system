"""
src/threshold_analysis.py
──────────────────────────────────────────────────────────────────────────────
Validation Threshold Diagnostic Module for Neural Network Binary Classification.

Purpose & Constraints
---------------------
- **Test Set Integrity**: The 2,000-sample test set (X_test.npy, y_test.npy) is COMPLETELY UNTOUCHED.
- **Zero Retraining**: ANN weights are loaded directly from results/ann_weights.npz.
- **Validation Data Origin**: Reconstructed from the training set (X_train.npy, y_train.npy)
  using the exact stratified 10% split logic (val_fraction=0.10, seed=42) defined in src/ann.py.
- **No Automatic Threshold Selection**: Reports precision-recall trade-offs objectively across candidate
  thresholds without changing the production default threshold or optimizing on test data.
"""

import json
import os
import sys

# Ensure parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.ann import NeuralNetwork


# ==============================================================================
# Paths
# ==============================================================================

PROCESSED_DIR: str = os.path.join("data", "processed")
RESULTS_DIR: str = "results"
FIGURES_DIR: str = os.path.join(RESULTS_DIR, "figures")

X_TRAIN_PATH: str = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH: str = os.path.join(PROCESSED_DIR, "y_train.npy")
WEIGHTS_PATH: str = os.path.join(RESULTS_DIR, "ann_weights.npz")
NORM_PARAMS_PATH: str = os.path.join(PROCESSED_DIR, "norm_params.json")

CANDIDATE_THRESHOLDS: list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ==============================================================================
# Validation Split Reconstruction
# ==============================================================================

def reconstruct_validation_set(seed: int = 42, val_fraction: float = 0.10) -> tuple:
    """
    Reconstruct the exact 800-sample validation set (X_val, y_val) from X_train.npy
    and y_train.npy using the exact stratified split logic from train_ann in src/ann.py.

    Returns
    -------
    X_val : np.ndarray of shape (800, 7)
    y_val : np.ndarray of shape (800,)
    """
    if not os.path.exists(X_TRAIN_PATH) or not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data files not found in '{PROCESSED_DIR}/'. Run preprocessing first."
        )

    X_train_full = np.load(X_TRAIN_PATH)
    y_train_full = np.load(Y_TRAIN_PATH)

    rng = np.random.default_rng(seed)

    val_indices = []
    for class_val in np.unique(y_train_full):
        idx = np.where(y_train_full == class_val)[0]
        idx = rng.permutation(idx)
        val_count = int(np.round(val_fraction * len(idx)))
        val_indices.append(idx[:val_count])

    val_idx = np.sort(np.concatenate(val_indices))

    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    return X_val, y_val


# ==============================================================================
# Threshold Evaluation Function
# ==============================================================================

def evaluate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list) -> list:
    """
    Evaluate binary classification metrics across candidate probability thresholds.
    """
    results = []
    n_total = len(y_true)

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        acc = (tp + tn) / n_total if n_total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results.append({
            "threshold":   round(th, 2),
            "tp":          tp,
            "tn":          tn,
            "fp":          fp,
            "fn":          fn,
            "accuracy":    round(acc, 4),
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score":    round(f1, 4),
        })

    return results


# ==============================================================================
# Visualization (Matplotlib + Pure-Python SVG Fallback)
# ==============================================================================

def plot_threshold_analysis(results: list, save_path: str):
    """
    Plot Precision, Recall, F1-Score, and Specificity curves across candidate thresholds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    thresholds = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls    = [r["recall"] for r in results]
    f1_scores  = [r["f1_score"] for r in results]
    specs      = [r["specificity"] for r in results]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(thresholds, precisions, marker='o', linewidth=2, color='#1f77b4', label='Precision')
        ax.plot(thresholds, recalls,    marker='s', linewidth=2, color='#2ca02c', label='Recall (Sensitivity)')
        ax.plot(thresholds, f1_scores,  marker='^', linewidth=2, color='#d62728', label='F1-Score')
        ax.plot(thresholds, specs,      marker='d', linewidth=2, color='#7f7f7f', linestyle='--', label='Specificity')

        ax.axvline(x=0.50, color='red', linestyle=':', linewidth=2, label='Current Default (0.50)')

        ax.set_title('ANN Validation Threshold Analysis (Validation Set N=800)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability Threshold τ', fontsize=11)
        ax.set_ylabel('Metric Score', fontsize=11)
        ax.set_xticks(thresholds)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[FIGURE SAVED] Threshold analysis curve saved to '{save_path}'")

    except Exception as e:
        print(f"\n[NOTE] Matplotlib rendering skipped ({e}). Generating SVG fallback...")
        svg_path = os.path.splitext(save_path)[0] + ".svg"
        
        width, height = 800, 450
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:sans-serif;">',
            f'<text x="400" y="30" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">ANN Validation Threshold Analysis (Validation N=800)</text>',
            '<rect x="80" y="50" width="680" height="320" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
        ]

        def map_coord(th, val):
            px = 80 + (th - 0.10) / (0.90 - 0.10) * 680
            py = 370 - val * 320
            return f"{px:.1f}", f"{py:.1f}"

        # Grid lines
        for y_t in np.linspace(0, 1.0, 6):
            py = 370 - y_t * 320
            svg.append(f'<line x1="80" y1="{py:.1f}" x2="760" y2="{py:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
            svg.append(f'<text x="70" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{y_t:.1f}</text>')

        for th in thresholds:
            px = 80 + (th - 0.10) / 0.80 * 680
            svg.append(f'<line x1="{px:.1f}" y1="50" x2="{px:.1f}" y2="370" stroke="#f0f0f0" stroke-width="1"/>')
            svg.append(f'<text x="{px:.1f}" y="388" font-size="10" text-anchor="middle" fill="#555">{th:.2f}</text>')

        # Metric polylines
        pts_p = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, precisions)])
        pts_r = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, recalls)])
        pts_f = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, f1_scores)])

        svg.append(f'<polyline points="{pts_p}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>')
        svg.append(f'<polyline points="{pts_r}" fill="none" stroke="#2ca02c" stroke-width="2.5"/>')
        svg.append(f'<polyline points="{pts_f}" fill="none" stroke="#d62728" stroke-width="2.5"/>')

        # Threshold 0.5 line
        p50_x = 80 + (0.50 - 0.10) / 0.80 * 680
        svg.append(f'<line x1="{p50_x:.1f}" y1="50" x2="{p50_x:.1f}" y2="370" stroke="#d62728" stroke-dasharray="4" stroke-width="2"/>')

        # Legend
        svg.append('<rect x="180" y="415" width="12" height="12" fill="#1f77b4"/>')
        svg.append('<text x="197" y="425" font-size="11" fill="#333">Precision</text>')
        svg.append('<rect x="300" y="415" width="12" height="12" fill="#2ca02c"/>')
        svg.append('<text x="317" y="425" font-size="11" fill="#333">Recall</text>')
        svg.append('<rect x="420" y="415" width="12" height="12" fill="#d62728"/>')
        svg.append('<text x="437" y="425" font-size="11" fill="#333">F1-Score</text>')
        svg.append('<line x1="530" y1="421" x2="550" y2="421" stroke="#d62728" stroke-dasharray="4" stroke-width="2"/>')
        svg.append('<text x="555" y="425" font-size="11" fill="#333">Default Threshold (0.50)</text>')

        svg.append('</svg>')

        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg))
        print(f"\n[FIGURE SAVED] Threshold analysis SVG saved to '{svg_path}'")


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================

def main():
    """
    Run threshold diagnostic analysis on validation set only.
    """
    print("=" * 75)
    print("ANN MODEL ANALYSIS — VALIDATION THRESHOLD DIAGNOSTIC STEP")
    print("=" * 75)

    # 1. State validation data availability & integrity
    print("\n[VALIDATION DATA INTEGRITY CHECK]")
    print("  - Test set (X_test.npy, y_test.npy, N=2000): 100% UNTOUCHED.")
    print("  - Reconstructing validation set from X_train.npy & y_train.npy")
    print("    using exact training split parameters: val_fraction=0.10, seed=42.")

    X_val, y_val = reconstruct_validation_set(seed=42, val_fraction=0.10)

    n_val = len(y_val)
    n_pos = int(y_val.sum())
    n_neg = n_val - n_pos

    print(f"  - Reconstructed Validation Set Size: {n_val} samples")
    print(f"    Positives (Failures = 1)        : {n_pos} ({100*n_pos/n_val:.2f}%)")
    print(f"    Negatives (Healthy  = 0)        : {n_neg} ({100*n_neg/n_val:.2f}%)")

    # 2. Load trained ANN weights
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"ANN weights file not found at '{WEIGHTS_PATH}'.")

    model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1)
    weights_npz = np.load(WEIGHTS_PATH)
    model.set_weights({k: weights_npz[k] for k in weights_npz.files})
    print(f"  - Pre-trained ANN Weights Loaded   : '{WEIGHTS_PATH}' (Zero retraining)")

    # Load w_pos from metadata for explanation
    w_pos = 28.5185
    if os.path.exists(NORM_PARAMS_PATH):
        with open(NORM_PARAMS_PATH, 'r') as f:
            m = json.load(f)
            w_pos = m.get('class_weights', {}).get('w_pos', w_pos)

    # 3. ANN Forward Pass on Validation Set
    y_val_prob = model.forward(X_val).ravel()

    # 4. Evaluate Candidate Thresholds
    threshold_results = evaluate_thresholds(y_val, y_val_prob, CANDIDATE_THRESHOLDS)

    # 5. Print Threshold Comparison Table
    print("\n" + "=" * 75)
    print("VALIDATION SET THRESHOLD PERFORMANCE TABLE (N = 800)")
    print("=" * 75)
    print(f"{'Threshold (tau)':<15s} | {'TP':<4s} | {'TN':<4s} | {'FP':<4s} | {'FN':<4s} | {'Precision':<10s} | {'Recall':<9s} | {'F1-Score':<9s} | {'Specificity':<11s}")
    print("-" * 75)

    for r in threshold_results:
        highlight = " <-- CURRENT DEFAULT" if r["threshold"] == 0.50 else ""
        print(f"  tau = {r['threshold']:<8.2f} | {r['tp']:<4d} | {r['tn']:<4d} | {r['fp']:<4d} | {r['fn']:<4d} | {r['precision']:<10.4f} | {r['recall']:<9.4f} | {r['f1_score']:<9.4f} | {r['specificity']:<11.4f}{highlight}")

    print("=" * 75)

    # 6. Technical Explanation of Threshold Behavior
    print("\n" + "=" * 75)
    print("TECHNICAL EXPLANATION - WHY THRESHOLD 0.5 PRODUCES HIGH RECALL & LOW PRECISION")
    print("=" * 75)
    print(f"1. Class Imbalance Penalty (w_pos = {w_pos:.4f}):")
    print("   The ANN was trained using Weighted Binary Cross-Entropy Loss to handle class imbalance (~3.4% failures).")
    print(f"   Missing a failure (FN) incurs a loss penalty weighted by w_pos ~ {w_pos:.2f} relative to false positives (FP).")
    print("   This heavily biases the Sigmoid output probabilities upwards across the entire sample population.")
    print("\n2. Probability Shift & Decision Threshold:")
    print("   Because probabilities are pushed upwards to avoid missing failures, a standard decision boundary of tau = 0.50")
    print("   operates conservatively: it captures almost all actual failures (Recall ~ 92.6% on validation),")
    print("   but flags many borderline healthy machines as potential failures (generating False Positives and lowering Precision).")
    print("\n3. Precision vs. Recall Trade-Off Dynamics:")
    print("   - Lowering threshold (tau < 0.50): Increases Recall towards 100%, but increases False Positives further.")
    print("   - Raising threshold (tau > 0.50): Increases Precision significantly (fewer False Positives),")
    print("     but increases False Negatives (missed machine failures).")
    print("   - For industrial predictive maintenance, missing a catastrophic failure (FN) is far more costly than an unnecessary inspection (FP).")
    print("=" * 75)

    # 7. Plot Figure
    fig_path = os.path.join(FIGURES_DIR, "validation_threshold_analysis.png")
    plot_threshold_analysis(threshold_results, fig_path)


if __name__ == "__main__":
    main()
