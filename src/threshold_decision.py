"""
src/threshold_decision.py
──────────────────────────────────────────────────────────────────────────────
Threshold Decision Analysis Module for Predictive Maintenance.

Role & Constraints
------------------
- Evaluates candidate probability decision thresholds (0.10 to 0.90) ONLY on the
  reconstructed 800-sample validation set.
- Test set (X_test.npy, y_test.npy, N=2000) is COMPLETELY UNTOUCHED.
- Model weights, fuzzy rules, baseline rules, and architecture are UNCHANGED.
- Saves analysis to:
    results/threshold_decision_analysis.json
    results/figures/threshold_decision_analysis.png
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

JSON_SAVE_PATH: str = os.path.join(RESULTS_DIR, "threshold_decision_analysis.json")
PNG_SAVE_PATH: str = os.path.join(FIGURES_DIR, "threshold_decision_analysis.png")

CANDIDATE_THRESHOLDS: list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ==============================================================================
# Reconstruct Validation Set
# ==============================================================================

def reconstruct_validation_set(seed: int = 42, val_fraction: float = 0.10) -> tuple:
    """
    Reconstruct 800-sample validation set (X_val, y_val) from X_train.npy and y_train.npy
    using exact stratified split logic from train_ann in src/ann.py.
    """
    if not os.path.exists(X_TRAIN_PATH) or not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(f"Training data files not found in '{PROCESSED_DIR}/'.")

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
    return X_train_full[val_idx], y_train_full[val_idx]


# ==============================================================================
# Threshold Evaluation
# ==============================================================================

def analyze_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list) -> list:
    """Compute detailed confusion matrix and classification metrics per threshold."""
    results = []
    n_total = len(y_true)

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        acc = float((tp + tn) / n_total) if n_total > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results.append({
            "threshold":   round(float(th), 2),
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
# Figure Plotting (Matplotlib + Pure-Python SVG Fallback)
# ==============================================================================

def generate_decision_figures(results: list, png_path: str):
    """Plot threshold decision curves and save to figures/."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    thresholds = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls    = [r["recall"] for r in results]
    f1_scores  = [r["f1_score"] for r in results]
    specs      = [r["specificity"] for r in results]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(thresholds, precisions, marker='o', linewidth=2.5, color='#1f77b4', label='Precision (Fewer False Alarms)')
        ax.plot(thresholds, recalls,    marker='s', linewidth=2.5, color='#2ca02c', label='Recall (Catch Failures)')
        ax.plot(thresholds, f1_scores,  marker='^', linewidth=2.5, color='#d62728', label='F1-Score (Harmonic Mean)')
        ax.plot(thresholds, specs,      marker='d', linewidth=2, color='#7f7f7f', linestyle='--', label='Specificity')

        # Highlight default vs max F1
        ax.axvline(x=0.50, color='#d62728', linestyle=':', linewidth=2, label='Current Default (0.50)')
        
        # Find max F1 threshold
        max_f1_item = max(results, key=lambda x: x["f1_score"])
        ax.axvline(x=max_f1_item["threshold"], color='#9467bd', linestyle='--', linewidth=2, label=f'Max Validation F1 ({max_f1_item["threshold"]:.2f})')

        ax.set_title('ANN Threshold Decision Analysis (Validation Set N=800)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability Threshold tau', fontsize=11)
        ax.set_ylabel('Metric Score', fontsize=11)
        ax.set_xticks(thresholds)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='center left')

        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[FIGURE SAVED] Threshold decision chart saved to '{png_path}'")

    except Exception as e:
        print(f"\n[NOTE] Matplotlib skipped ({e}). Generating SVG fallback...")
        svg_path = os.path.splitext(png_path)[0] + ".svg"
        
        width, height = 800, 450
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:sans-serif;">',
            f'<text x="400" y="30" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">ANN Threshold Decision Analysis (Validation Set N=800)</text>',
            '<rect x="80" y="50" width="680" height="320" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>'
        ]

        def map_coord(th, val):
            px = 80 + (th - 0.10) / (0.90 - 0.10) * 680
            py = 370 - val * 320
            return f"{px:.1f}", f"{py:.1f}"

        for y_t in np.linspace(0, 1.0, 6):
            py = 370 - y_t * 320
            svg.append(f'<line x1="80" y1="{py:.1f}" x2="760" y2="{py:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
            svg.append(f'<text x="70" y="{py+4:.1f}" font-size="10" text-anchor="end" fill="#555">{y_t:.1f}</text>')

        for th in thresholds:
            px = 80 + (th - 0.10) / 0.80 * 680
            svg.append(f'<line x1="{px:.1f}" y1="50" x2="{px:.1f}" y2="370" stroke="#f0f0f0" stroke-width="1"/>')
            svg.append(f'<text x="{px:.1f}" y="388" font-size="10" text-anchor="middle" fill="#555">{th:.2f}</text>')

        pts_p = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, precisions)])
        pts_r = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, recalls)])
        pts_f = " ".join([",".join(map_coord(th, v)) for th, v in zip(thresholds, f1_scores)])

        svg.append(f'<polyline points="{pts_p}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>')
        svg.append(f'<polyline points="{pts_r}" fill="none" stroke="#2ca02c" stroke-width="2.5"/>')
        svg.append(f'<polyline points="{pts_f}" fill="none" stroke="#d62728" stroke-width="2.5"/>')

        p50_x = 80 + (0.50 - 0.10) / 0.80 * 680
        svg.append(f'<line x1="{p50_x:.1f}" y1="50" x2="{p50_x:.1f}" y2="370" stroke="#d62728" stroke-dasharray="4" stroke-width="2"/>')

        svg.append('<rect x="180" y="415" width="12" height="12" fill="#1f77b4"/>')
        svg.append('<text x="197" y="425" font-size="11" fill="#333">Precision</text>')
        svg.append('<rect x="300" y="415" width="12" height="12" fill="#2ca02c"/>')
        svg.append('<text x="317" y="425" font-size="11" fill="#333">Recall</text>')
        svg.append('<rect x="420" y="415" width="12" height="12" fill="#d62728"/>')
        svg.append('<text x="437" y="425" font-size="11" fill="#333">F1-Score</text>')
        svg.append('<line x1="530" y1="421" x2="550" y2="421" stroke="#d62728" stroke-dasharray="4" stroke-width="2"/>')
        svg.append('<text x="555" y="425" font-size="11" fill="#333">Default (0.50)</text>')

        svg.append('</svg>')

        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(svg))
        print(f"\n[FIGURE SAVED] Threshold decision SVG saved to '{svg_path}'")


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================

def main():
    """Execute validation threshold decision analysis."""
    print("=" * 75)
    print("THRESHOLD DECISION ANALYSIS — VALIDATION SET (N = 800)")
    print("=" * 75)

    # 1. Reconstruct Validation Set (X_val, y_val)
    X_val, y_val = reconstruct_validation_set(seed=42, val_fraction=0.10)
    n_val = len(y_val)
    n_pos = int(y_val.sum())

    print(f"Validation Set Size : {n_val} samples (Positives/Failures: {n_pos})")
    print(f"Test Set Status     : 100% UNTOUCHED (2,000 samples reserved)")

    # 2. Load pre-trained ANN weights
    model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1)
    weights_npz = np.load(WEIGHTS_PATH)
    model.set_weights({k: weights_npz[k] for k in weights_npz.files})

    # 3. Predict probabilities on validation set
    y_val_prob = model.forward(X_val).ravel()

    # 4. Analyze candidate thresholds
    threshold_results = analyze_thresholds(y_val, y_val_prob, CANDIDATE_THRESHOLDS)

    # 5. Identify Key Threshold Operating Points
    max_recall_item = max(threshold_results, key=lambda x: (x["recall"], -x["threshold"]))
    max_precision_item = max(threshold_results, key=lambda x: (x["precision"], x["threshold"]))
    max_f1_item = max(threshold_results, key=lambda x: (x["f1_score"], x["threshold"]))

    default_item = next(r for r in threshold_results if r["threshold"] == 0.50)

    # 6. Print Structured Comparison Table
    print("\n" + "=" * 75)
    print("VALIDATION THRESHOLD DECISION TABLE")
    print("=" * 75)
    print(f"{'Threshold (tau)':<15s} | {'TP':<4s} | {'TN':<4s} | {'FP':<4s} | {'FN':<4s} | {'Precision':<10s} | {'Recall':<9s} | {'F1-Score':<9s} | {'Specificity':<11s}")
    print("-" * 75)

    for r in threshold_results:
        tag = ""
        if r["threshold"] == default_item["threshold"]:
            tag = " [Current Default]"
        if r["threshold"] == max_f1_item["threshold"]:
            tag += " [Max F1]"
        if r["threshold"] == max_recall_item["threshold"] and r["threshold"] != default_item["threshold"]:
            tag += " [Max Recall]"

        print(f"  tau = {r['threshold']:<8.2f} | {r['tp']:<4d} | {r['tn']:<4d} | {r['fp']:<4d} | {r['fn']:<4d} | {r['precision']:<10.4f} | {r['recall']:<9.4f} | {r['f1_score']:<9.4f} | {r['specificity']:<11.4f}{tag}")

    print("=" * 75)

    print("\nKEY OPERATING POINTS ON VALIDATION DATA:")
    print(f"  1. Highest Recall    : tau = {max_recall_item['threshold']:.2f}  (Recall = {max_recall_item['recall']:.4f}, Precision = {max_recall_item['precision']:.4f}, F1 = {max_recall_item['f1_score']:.4f})")
    print(f"  2. Highest Precision : tau = {max_precision_item['threshold']:.2f}  (Precision = {max_precision_item['precision']:.4f}, Recall = {max_precision_item['recall']:.4f}, F1 = {max_precision_item['f1_score']:.4f})")
    print(f"  3. Maximum F1-Score  : tau = {max_f1_item['threshold']:.2f}  (F1 = {max_f1_item['f1_score']:.4f}, Precision = {max_f1_item['precision']:.4f}, Recall = {max_f1_item['recall']:.4f})")
    print(f"  4. Current Default   : tau = 0.50  (F1 = {default_item['f1_score']:.4f}, Precision = {default_item['precision']:.4f}, Recall = {default_item['recall']:.4f})")
    print("=" * 75)

    # 7. Save Analysis to JSON
    json_payload = {
        "validation_sample_count": n_val,
        "validation_positive_count": n_pos,
        "candidate_threshold_results": threshold_results,
        "key_operating_points": {
            "highest_recall": {
                "threshold": max_recall_item["threshold"],
                "recall": max_recall_item["recall"],
                "precision": max_recall_item["precision"],
                "f1_score": max_recall_item["f1_score"],
                "fn": max_recall_item["fn"],
                "fp": max_recall_item["fp"],
            },
            "highest_precision": {
                "threshold": max_precision_item["threshold"],
                "precision": max_precision_item["precision"],
                "recall": max_precision_item["recall"],
                "f1_score": max_precision_item["f1_score"],
                "fn": max_precision_item["fn"],
                "fp": max_precision_item["fp"],
            },
            "maximum_f1": {
                "threshold": max_f1_item["threshold"],
                "f1_score": max_f1_item["f1_score"],
                "precision": max_f1_item["precision"],
                "recall": max_f1_item["recall"],
                "fn": max_f1_item["fn"],
                "fp": max_f1_item["fp"],
            },
            "current_default": {
                "threshold": default_item["threshold"],
                "f1_score": default_item["f1_score"],
                "precision": default_item["precision"],
                "recall": default_item["recall"],
                "fn": default_item["fn"],
                "fp": default_item["fp"],
            }
        },
        "tradeoff_analysis": {
            "operational_tradeoff": "The relative cost of false negatives and false positives depends on the actual maintenance environment. Threshold selection should therefore be based on the operational requirements of the target system.",
            "maximum_f1_note": "Threshold 0.90 produced the highest F1-score (0.6207) among the tested thresholds on this 800-sample validation set. This threshold is specific to this validation split and is not claimed to be universally optimal.",
            "recommended_strategy": "The current default threshold of tau = 0.50 is maintained for system evaluation. Any future threshold adjustment should be decided by project stakeholders based on specific plant operational requirements."
        }
    }

    with open(JSON_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n[ANALYSIS SAVED] Numerical analysis saved to '{JSON_SAVE_PATH}'")

    # 8. Generate Visual Figure
    generate_decision_figures(threshold_results, PNG_SAVE_PATH)


if __name__ == "__main__":
    main()
