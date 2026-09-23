"""
src/baseline.py
──────────────────────────────────────────────────────────────────────────────
Documented-rule crisp reference baseline for the AI4I 2020 dataset.

IMPORTANT — Academic honesty statement
---------------------------------------
This module encodes the failure-generation rules published by Matzka (2020),
which are the SAME rules used to generate the Machine failure labels in the
AI4I 2020 dataset.

This is NOT an independent industrial benchmark.
This is NOT a threshold system derived from real machine measurements.

Its purpose is:
  1. To establish what rigid, threshold-based classification looks like.
  2. To serve as a comparison point showing what the ANN must recover
     from data, without being told these rules explicitly.
  3. To demonstrate where crisp rules fail: noise, borderline inputs,
     probabilistic events (TWF randomness, RNF).

Failure modes encoded (from Matzka 2020)
------------------------------------------
  HDF  Heat Dissipation Failure:
       (process_temp_K - air_temp_K) < 8.6 K AND rotational_speed < 1380 rpm

  PWF  Power Failure:
       power = torque_Nm × (rpm × 2π / 60)  [Watts]
       power < 3500 W OR power > 9000 W

  TWF  Tool Wear Failure:
       Dataset: randomly triggered in [200, 240] min.
       Baseline: deterministic threshold at 220 min (midpoint of range).
       Acknowledged limitation: misses TWF outside [220, 240] min.

  OSF  Overstrain Failure:
       tool_wear_min × torque_Nm > threshold
       threshold: 11,000 (L) / 12,000 (M) / 13,000 (H) min·Nm

  RNF  Random Failure (0.1% chance):
       Cannot be encoded in any deterministic rule.
       NOT modelled. All RNF instances will be false negatives.
       This is documented as a known limitation.

Usage
-----
    python -m src.baseline     # Evaluate crisp baseline on test set
"""

import json
import math
import os
import sys

import numpy as np

# ==============================================================================
# File paths (relative to project root)
# ==============================================================================

_PROCESSED_DIR = os.path.join("data", "processed")
_X_TEST_PATH   = os.path.join(_PROCESSED_DIR, "X_test.npy")
_Y_TEST_PATH   = os.path.join(_PROCESSED_DIR, "y_test.npy")
_PARAMS_PATH   = os.path.join(_PROCESSED_DIR, "norm_params.json")

# Canonical feature order produced by preprocessing.py
# Index : Feature name
#   0   : Air temperature [K]
#   1   : Process temperature [K]
#   2   : Rotational speed [rpm]
#   3   : Torque [Nm]
#   4   : Tool wear [min]
#   5   : Type_M  (1 = product type M, else 0)
#   6   : Type_H  (1 = product type H, else 0)
_FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_M",
    "Type_H",
]

# OSF overstrain threshold per product type [min · Nm]
_OSF_THRESHOLD = {"L": 11_000, "M": 12_000, "H": 13_000}

# TWF deterministic threshold [min]
# The dataset triggers TWF randomly in [200, 240] min.
# The crisp baseline uses the midpoint 220 min as the best deterministic approximation.
# This is a documented limitation: TWF events below 220 min are missed,
# TWF events above 220 min that were NOT triggered in the dataset are false positives.
_TWF_THRESHOLD = 220


# ==============================================================================
# Individual failure-mode rule functions
# ==============================================================================

def _check_hdf(air_temp_K: float, process_temp_K: float, rpm: float) -> bool:
    """
    Heat Dissipation Failure (HDF).

    Rule (Matzka 2020):
        HDF = (process_temp_K − air_temp_K) < 8.6 K
              AND rotational_speed < 1380 rpm

    Physical interpretation:
        When the temperature differential between the process and the surrounding
        air drops below 8.6 K, heat is not dissipating fast enough. Combined with
        low rotational speed (insufficient cooling), this triggers thermal failure.
    """
    temp_diff = process_temp_K - air_temp_K
    return (temp_diff < 8.6) and (rpm < 1380)


def _check_pwf(rpm: float, torque_Nm: float) -> bool:
    """
    Power Failure (PWF).

    Rule (Matzka 2020):
        power [W] = torque_Nm × (rpm × 2π / 60)
        PWF = power < 3500 W OR power > 9000 W

    Physical interpretation:
        Power outside the operating range [3500, 9000] W indicates either an
        under-loaded (potential free-spin / loss-of-load) or over-loaded
        (mechanical overload) condition, both causing failure.
    """
    angular_velocity = rpm * 2.0 * math.pi / 60.0  # rad/s
    power_W = torque_Nm * angular_velocity          # Watts
    return (power_W < 3500.0) or (power_W > 9000.0)


def _check_twf(tool_wear_min: float) -> bool:
    """
    Tool Wear Failure (TWF).

    Rule (Matzka 2020):
        TWF is randomly triggered when tool wear is between 200 and 240 min.

    Crisp baseline approximation:
        TWF = tool_wear_min >= 220
        (midpoint of the [200, 240] range; see module docstring for limitation)

    Physical interpretation:
        Tools degrade over time. Beyond a wear threshold the cutting edge fails,
        causing sudden breakage or severe surface finish degradation.
    """
    return tool_wear_min >= _TWF_THRESHOLD


def _check_osf(tool_wear_min: float, torque_Nm: float, product_type: str) -> bool:
    """
    Overstrain Failure (OSF).

    Rule (Matzka 2020):
        OSF = (tool_wear_min × torque_Nm) > threshold
        threshold: L → 11,000 | M → 12,000 | H → 13,000  [min · Nm]

    Physical interpretation:
        The cumulative mechanical stress on the tool (wear × torque) exceeds the
        structural tolerance of the tool, leading to fracture or plastic deformation.
        Higher-quality (H) tools have higher tolerance than standard (L) tools.
    """
    product_type = product_type.strip().upper()
    if product_type not in _OSF_THRESHOLD:
        raise ValueError(
            f"Invalid product_type '{product_type}'. Must be 'L', 'M', or 'H'."
        )
    threshold = _OSF_THRESHOLD[product_type]
    return (tool_wear_min * torque_Nm) > threshold


# ==============================================================================
# Single-sample prediction
# ==============================================================================

def predict_single(
    air_temp_K: float,
    process_temp_K: float,
    rpm: float,
    torque_Nm: float,
    tool_wear_min: float,
    product_type: str,
) -> dict:
    """
    Apply the documented AI4I 2020 crisp failure rules to a single machine reading.

    Parameters
    ----------
    air_temp_K      : Air temperature [K]
    process_temp_K  : Process temperature [K]
    rpm             : Rotational speed [rpm]
    torque_Nm       : Torque [Nm]  — NOTE: NOT a vibration measurement.
    tool_wear_min   : Tool wear [min]
    product_type    : Product quality type: 'L', 'M', or 'H'

    Returns
    -------
    dict with keys:
        failure       : int   — 1 if any rule fires, 0 otherwise
        rules_fired   : list  — names of triggered failure modes (e.g. ['HDF', 'PWF'])
        explanation   : str   — human-readable summary of which rules fired and why
        hdf           : bool
        pwf           : bool
        twf           : bool
        osf           : bool
        rnf_modelled  : bool  — always False; RNF is not deterministically modelled
        power_W       : float — computed power [Watts] (useful for inspection)
        temp_diff_K   : float — process_temp − air_temp [K] (useful for inspection)
    """
    hdf = _check_hdf(air_temp_K, process_temp_K, rpm)
    pwf = _check_pwf(rpm, torque_Nm)
    twf = _check_twf(tool_wear_min)
    osf = _check_osf(tool_wear_min, torque_Nm, product_type)
    # RNF: 0.1% random failure — cannot be deterministically modelled.
    # All RNF instances will be false negatives in this baseline.
    rnf = False

    rules_fired = []
    explanations = []

    if hdf:
        rules_fired.append("HDF")
        temp_diff = process_temp_K - air_temp_K
        explanations.append(
            f"HDF: temp diff ({temp_diff:.2f} K) < 8.6 K AND rpm ({rpm:.0f}) < 1380"
        )
    if pwf:
        rules_fired.append("PWF")
        power = torque_Nm * rpm * 2.0 * math.pi / 60.0
        explanations.append(
            f"PWF: power ({power:.1f} W) outside [3500, 9000] W"
        )
    if twf:
        rules_fired.append("TWF")
        explanations.append(
            f"TWF: tool wear ({tool_wear_min:.0f} min) >= {_TWF_THRESHOLD} min (midpoint threshold)"
        )
    if osf:
        rules_fired.append("OSF")
        strain = tool_wear_min * torque_Nm
        threshold = _OSF_THRESHOLD[product_type.strip().upper()]
        explanations.append(
            f"OSF: wear × torque ({strain:.1f} min·Nm) > {threshold} min·Nm (type {product_type.upper()})"
        )

    failure = 1 if rules_fired else 0

    if failure:
        explanation = "FAILURE predicted. " + " | ".join(explanations)
    else:
        explanation = (
            "No failure. All crisp rules satisfied. "
            "(Note: RNF is not modelled — 0.1% random failures are always missed.)"
        )

    return {
        "failure":      failure,
        "rules_fired":  rules_fired,
        "explanation":  explanation,
        "hdf":          hdf,
        "pwf":          pwf,
        "twf":          twf,
        "osf":          osf,
        "rnf_modelled": rnf,
        "power_W":      torque_Nm * rpm * 2.0 * math.pi / 60.0,
        "temp_diff_K":  process_temp_K - air_temp_K,
    }


# ==============================================================================
# Batch prediction (operates on raw/unscaled sensor arrays)
# ==============================================================================

def predict_batch(
    air_temps_K: np.ndarray,
    process_temps_K: np.ndarray,
    rpms: np.ndarray,
    torques_Nm: np.ndarray,
    tool_wears_min: np.ndarray,
    product_types,          # list or array of 'L'/'M'/'H' strings
) -> np.ndarray:
    """
    Apply crisp rules to arrays of N unscaled sensor readings.

    Parameters
    ----------
    All arrays must have the same length N.
    product_types : list or array of str — each element is 'L', 'M', or 'H'.

    Returns
    -------
    y_pred : np.ndarray of shape (N,), dtype int
        1 = failure predicted, 0 = no failure predicted.
    """
    n = len(air_temps_K)
    y_pred = np.zeros(n, dtype=int)
    for i in range(n):
        result = predict_single(
            air_temp_K=float(air_temps_K[i]),
            process_temp_K=float(process_temps_K[i]),
            rpm=float(rpms[i]),
            torque_Nm=float(torques_Nm[i]),
            tool_wear_min=float(tool_wears_min[i]),
            product_type=str(product_types[i]),
        )
        y_pred[i] = result["failure"]
    return y_pred


# ==============================================================================
# Helpers: load and denormalise the processed test set
# ==============================================================================

def _load_norm_params(params_path: str = _PARAMS_PATH) -> dict:
    """Load normalisation parameters from norm_params.json."""
    with open(params_path, "r") as f:
        return json.load(f)["norm_params"]


def _denormalise(X_norm: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Reverse Min-Max normalisation for the 5 numeric features.

    X_norm columns (from preprocessing.py canonical order):
        0: Air temperature [K]   (normalised)
        1: Process temperature   (normalised)
        2: Rotational speed      (normalised)
        3: Torque [Nm]           (normalised)
        4: Tool wear [min]       (normalised)
        5: Type_M                (binary — no scaling applied)
        6: Type_H                (binary — no scaling applied)

    Returns a copy with columns 0–4 restored to original physical units.
    """
    X_raw = X_norm.copy()
    feature_order = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    for col_idx, feat_name in enumerate(feature_order):
        f_min = norm_params[feat_name]["min"]
        f_max = norm_params[feat_name]["max"]
        X_raw[:, col_idx] = X_norm[:, col_idx] * (f_max - f_min) + f_min
    return X_raw


def _decode_product_type(type_M: float, type_H: float) -> str:
    """
    Recover product type string from the two one-hot columns.
        Type_M=1, Type_H=0  →  'M'
        Type_M=0, Type_H=1  →  'H'
        Type_M=0, Type_H=0  →  'L'
    """
    if round(type_M) == 1:
        return "M"
    elif round(type_H) == 1:
        return "H"
    else:
        return "L"


# ==============================================================================
# Evaluate against the held-out test set
# ==============================================================================

def evaluate_on_test(
    x_test_path: str = _X_TEST_PATH,
    y_test_path: str = _Y_TEST_PATH,
    params_path: str = _PARAMS_PATH,
    verbose: bool = True,
) -> dict:
    """
    Load the pre-processed test set, denormalise it, apply crisp rules,
    and return predictions together with ground-truth labels.

    Does NOT compute metrics here — metrics are computed by evaluate.py
    so that all three approaches (baseline, ANN, hybrid) are compared in
    one place.  This function returns the raw prediction arrays needed for
    metric computation.

    Returns
    -------
    dict with keys:
        y_pred   : np.ndarray shape (N,) int  — baseline predictions
        y_true   : np.ndarray shape (N,) int  — ground-truth labels
        n_samples          : int
        n_predicted_failure: int
        n_true_failure     : int
    """
    # --- Guard: check files exist ---
    for path in (x_test_path, y_test_path, params_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: '{path}'\n"
                "Run 'python -m src.preprocessing' first."
            )

    # --- Load ---
    X_test_norm = np.load(x_test_path)   # shape (2000, 7), normalised
    y_true      = np.load(y_test_path)   # shape (2000,), int

    norm_params = _load_norm_params(params_path)

    # --- Denormalise numeric features (columns 0–4) ---
    X_test_raw = _denormalise(X_test_norm, norm_params)

    # --- Recover product type from one-hot columns 5 and 6 ---
    product_types = [
        _decode_product_type(X_test_raw[i, 5], X_test_raw[i, 6])
        for i in range(len(X_test_raw))
    ]

    # --- Extract unscaled sensor columns ---
    air_temps_K      = X_test_raw[:, 0]
    process_temps_K  = X_test_raw[:, 1]
    rpms             = X_test_raw[:, 2]
    torques_Nm       = X_test_raw[:, 3]
    tool_wears_min   = X_test_raw[:, 4]

    # --- Apply crisp rules ---
    y_pred = predict_batch(
        air_temps_K=air_temps_K,
        process_temps_K=process_temps_K,
        rpms=rpms,
        torques_Nm=torques_Nm,
        tool_wears_min=tool_wears_min,
        product_types=product_types,
    )

    n_predicted_failure = int(y_pred.sum())
    n_true_failure      = int(y_true.sum())
    n_samples           = len(y_true)

    if verbose:
        print("=" * 60)
        print("BASELINE — CRISP RULE EVALUATION ON TEST SET")
        print("=" * 60)
        print(f"  Test set size          : {n_samples} samples")
        print(f"  True failures (y=1)    : {n_true_failure} ({100*n_true_failure/n_samples:.2f}%)")
        print(f"  Predicted failures     : {n_predicted_failure} ({100*n_predicted_failure/n_samples:.2f}%)")
        print()
        print("  NOTE: RNF (Random Failure, ~5 events) is NOT modelled.")
        print("  TWF threshold is deterministic at 220 min (midpoint of [200,240]).")
        print("  Metric computation is performed in evaluate.py.")
        print("=" * 60)

    return {
        "y_pred":             y_pred,
        "y_true":             y_true,
        "n_samples":          n_samples,
        "n_predicted_failure": n_predicted_failure,
        "n_true_failure":     n_true_failure,
    }


# ==============================================================================
# Self-contained scenario tests (viva-friendly)
# ==============================================================================

def _print_result(label: str, result: dict):
    """Pretty-print a single predict_single() result."""
    status = "FAILURE" if result["failure"] else "OK"
    print(f"\n  [{status}] {label}")
    print(f"    Rules fired : {result['rules_fired'] if result['rules_fired'] else 'None'}")
    print(f"    Power       : {result['power_W']:.1f} W")
    print(f"    Temp diff   : {result['temp_diff_K']:.2f} K")
    print(f"    Explanation : {result['explanation']}")


def run_scenario_tests():
    """
    Run five hand-designed test scenarios to verify the crisp rules.

    Scenario values are chosen to clearly satisfy or violate each rule,
    making the expected outcome unambiguous for viva inspection.
    """
    print("=" * 60)
    print("BASELINE — SCENARIO TESTS")
    print("(Values chosen to clearly trigger or avoid each rule)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Normal healthy machine — no rules should fire
    # ------------------------------------------------------------------
    r1 = predict_single(
        air_temp_K=300.0,
        process_temp_K=310.0,  # diff = 10.0 K  (> 8.6 K  → HDF safe)
        rpm=1500,              # > 1380 rpm      → HDF safe
        torque_Nm=40.0,
        # Power = 40 × (1500 × 2π/60) = 40 × 157.08 = 6283 W  → in [3500,9000]
        tool_wear_min=100,     # < 220 min       → TWF safe
        product_type="M",
        # wear × torque = 100 × 40 = 4000 < 12000 → OSF safe
    )
    _print_result("Normal healthy machine (expect: OK)", r1)
    assert r1["failure"] == 0, "FAIL: Expected no failure for normal machine"
    assert r1["rules_fired"] == [], "FAIL: No rules should fire for normal machine"

    # ------------------------------------------------------------------
    # 2. HDF condition — low temp diff AND low rpm
    # ------------------------------------------------------------------
    r2 = predict_single(
        air_temp_K=303.0,
        process_temp_K=310.0,  # diff = 7.0 K < 8.6 K  → HDF condition 1 met
        rpm=1200,              # < 1380 rpm              → HDF condition 2 met
        torque_Nm=40.0,
        # Power = 40 × (1200 × 2π/60) = 40 × 125.66 = 5026 W  → in [3500,9000]
        tool_wear_min=50,      # < 220 min  → TWF safe
        product_type="M",
        # 50 × 40 = 2000 < 12000 → OSF safe
    )
    _print_result("HDF condition (temp diff=7.0 K, rpm=1200) — expect: HDF", r2)
    assert r2["failure"] == 1, "FAIL: Expected HDF failure"
    assert "HDF" in r2["rules_fired"], "FAIL: HDF should be in rules_fired"

    # ------------------------------------------------------------------
    # 3. PWF condition — power too low
    # ------------------------------------------------------------------
    # Power = torque × (rpm × 2π/60)
    # For power < 3500 W: need torque × rpm_rad < 3500
    # Example: torque=10, rpm=2000 → 10 × (2000×2π/60) = 10 × 209.4 = 2094 W < 3500
    r3 = predict_single(
        air_temp_K=298.0,
        process_temp_K=309.0,  # diff = 11.0 K > 8.6 K → HDF safe
        rpm=2000,              # > 1380 → HDF condition 2 not met
        torque_Nm=10.0,
        # Power = 10 × 209.4 = 2094 W < 3500 W → PWF
        tool_wear_min=50,      # < 220 → TWF safe
        product_type="L",
        # 50 × 10 = 500 < 11000 → OSF safe
    )
    _print_result("PWF condition (power ~ 2094 W < 3500 W) — expect: PWF", r3)
    assert r3["failure"] == 1, "FAIL: Expected PWF failure"
    assert "PWF" in r3["rules_fired"], "FAIL: PWF should be in rules_fired"

    # ------------------------------------------------------------------
    # 4. TWF condition — tool wear at/above 220 min
    # ------------------------------------------------------------------
    r4 = predict_single(
        air_temp_K=298.0,
        process_temp_K=309.0,  # diff = 11.0 K → HDF safe
        rpm=1500,              # > 1380 → HDF safe
        torque_Nm=30.0,
        # Power = 30 × 157.08 = 4712 W  → in [3500,9000]
        tool_wear_min=225,     # >= 220 → TWF triggered
        product_type="H",
        # 225 × 30 = 6750 < 13000 → OSF safe
    )
    _print_result("TWF condition (tool_wear=225 min >= 220) — expect: TWF", r4)
    assert r4["failure"] == 1, "FAIL: Expected TWF failure"
    assert "TWF" in r4["rules_fired"], "FAIL: TWF should be in rules_fired"

    # ------------------------------------------------------------------
    # 5. OSF condition — wear × torque exceeds type threshold
    # ------------------------------------------------------------------
    r5 = predict_single(
        air_temp_K=298.0,
        process_temp_K=309.0,  # diff = 11.0 K → HDF safe
        rpm=1500,              # > 1380 → HDF safe
        torque_Nm=60.0,
        # Power = 60 × 157.08 = 9425 W > 9000 W → this also triggers PWF!
        # Let's use rpm=1400 and torque=55: power = 55 × (1400×2π/60)
        # = 55 × 146.6 = 8063 W → in range
        tool_wear_min=220,     # 220 × 55 = 12100 > 12000 (type M) → OSF
        product_type="M",
    )
    # Re-do with safe power:
    r5 = predict_single(
        air_temp_K=298.0,
        process_temp_K=309.0,  # diff = 11.0 K → HDF safe
        rpm=1400,              # > 1380 → HDF safe; power = 55 × 146.6 = 8063 W
        torque_Nm=55.0,
        tool_wear_min=220,     # wear × torque = 220 × 55 = 12100 > 12000 → OSF (type M)
        product_type="M",
    )
    _print_result("OSF condition (220 min × 55 Nm = 12100 > 12000, type M) — expect: OSF", r5)
    assert r5["failure"] == 1, "FAIL: Expected OSF failure"
    assert "OSF" in r5["rules_fired"], "FAIL: OSF should be in rules_fired"

    print("\n" + "=" * 60)
    print("ALL 5 SCENARIO TESTS PASSED")
    print("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    # --- Scenario tests ---
    run_scenario_tests()

    # --- Evaluate on test set ---
    print()
    try:
        evaluate_on_test(verbose=True)
    except FileNotFoundError as e:
        print(f"\n[WARNING] Test-set evaluation skipped: {e}", file=sys.stderr)
        print("Run 'python -m src.preprocessing' first to generate processed arrays.",
              file=sys.stderr)
        sys.exit(1)

