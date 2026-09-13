"""
src/preprocessing.py
──────────────────────────────────────────────────────────────────────────────
Preprocessing pipeline for the AI4I 2020 Predictive Maintenance Dataset.

Dataset source:
    UCI Machine Learning Repository, ID 601
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Citation:
    Matzka, S. (2020). Explainable Artificial Intelligence for Predictive
    Maintenance Applications. In Proceedings of the Third International
    Conference on Artificial Intelligence for Industries (AI4I 2020), IEEE.

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
    https://creativecommons.org/licenses/by/4.0/

──────────────────────────────────────────────────────────────────────────────
IMPORTANT DATASET LIMITATION
──────────────────────────────────────────────────────────────────────────────
The AI4I 2020 dataset does NOT contain a vibration sensor.
Torque [Nm] is the documented torque feature — it is NOT a vibration proxy
and is NOT renamed or treated as vibration anywhere in this project.
──────────────────────────────────────────────────────────────────────────────

Pipeline (in order)
-------------------
1.  Load  data/ai4i2020.csv
2.  Verify shape (10 000 rows), expected columns, and no missing values
3.  Drop UID and Product ID (identifiers — they carry no predictive signal)
4.  One-hot encode Type (L/M/H) into Type_M and Type_H;
    Type_L is the reference category (both flags = 0)
5.  80/20 stratified train/test split (random_state=42)
6.  Min-Max normalise the five numeric features using TRAINING statistics only
    Formula: x_scaled = (x - train_min) / (train_max - train_min)
7.  Calculate class weights from the TRAINING labels only
    Formula: w_pos = N_negative / N_positive,  w_neg = 1.0
8.  Save processed arrays to data/processed/
9.  Print a full verification report

Final ANN input: 7 features in this order
    [Air temperature, Process temperature, Rotational speed,
     Torque, Tool wear, Type_M, Type_H]

ANN target: Machine failure  (0 = no failure, 1 = failure)

Notes
-----
- Torque [Nm] is used as Torque -- NOT as a vibration proxy.
- SMOTE is NOT used; class imbalance is handled via class-weighted loss in ann.py.
- The test set is NEVER used to fit any transformation (no data leakage).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
# sklearn is NOT imported here to avoid the scipy DLL issue on restricted systems.
# train_test_split is implemented directly using NumPy below (stratified_split).
# This is intentional: the project requires only pandas and numpy for preprocessing.

# ==============================================================================
# Constants
# ==============================================================================

RAW_CSV = os.path.join("data", "ai4i2020.csv")
PROCESSED_DIR = os.path.join("data", "processed")

# Columns that are pure row identifiers and carry no predictive signal.
# UDI  -- an arbitrary integer sequence 1-10 000 (UCI API name).
# UID  -- alternative header spelling used in some CSV releases.
# Product ID -- a letter + serial number string redundant with 'Type'.
# Retaining these would cause the model to memorise row identifiers rather
# than learning physical failure patterns.
#
# Note: When downloaded via the UCI API (ucimlrepo), these ID-role columns are
# already excluded from dataset.data.features and dataset.data.targets.
# The drop step below silently skips any columns not present in the CSV,
# so this script works with both the 12-column (API) and 14-column (zip) variants.
COLS_TO_DROP = ["UDI", "Product ID", "UID"]

# Five numeric sensor/process features to keep (real names from dataset).
# NOTE: Torque is Torque -- it is NOT a vibration measurement.
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Categorical feature to encode
TYPE_COL = "Type"

# Target column
TARGET_COL = "Machine failure"

# Sub-target columns (breakdown of the target; must NOT be used as ANN inputs)
SUBTARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Expected dataset dimensions (documented by Matzka 2020)
EXPECTED_N_ROWS = 10_000

# Final 7 ANN input feature names (canonical order)
FINAL_FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_M",
    "Type_H",
]


# ==============================================================================
# Pipeline functions
# ==============================================================================

def load_and_verify(path: str) -> pd.DataFrame:
    """
    Load the raw CSV and perform basic integrity checks.

    Checks:
    - File exists
    - Exactly 10 000 rows
    - No missing values
    - Target column 'Machine failure' is present
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Download from:\n"
            "  https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            "or run the download snippet in data/README.md"
        )

    df = pd.read_csv(path)

    print("=" * 70)
    print("VERIFICATION REPORT -- AI4I 2020 Predictive Maintenance Dataset")
    print("=" * 70)
    print(f"\n[1] Original dataset shape: {df.shape}")
    print(f"\n[2] Original column names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    # Row count check
    if df.shape[0] != EXPECTED_N_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_N_ROWS} rows, found {df.shape[0]}. "
            "Ensure the file is the complete AI4I 2020 dataset."
        )
    print(f"\n    [OK] Row count: {df.shape[0]} (expected {EXPECTED_N_ROWS})")

    # Missing value check
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        raise ValueError(
            f"Found {n_missing} missing value(s). "
            "The AI4I 2020 dataset should have no missing values."
        )
    print(f"    [OK] Missing values: {n_missing} (none expected)")

    # Target column check
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Columns present: {list(df.columns)}"
        )
    print(f"    [OK] Target column '{TARGET_COL}' present")

    return df


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove administrative identifier columns (UID / UDI / Product ID).

    Reason: These are row identifiers with no predictive value.
    - UDI/UID is an integer sequence 1-10 000 (perfectly correlated with row order).
    - Product ID is a concatenation of 'Type' letter + serial number; 'Type'
      is retained as a proper feature.
    Keeping identifiers would allow the model to memorise specific rows instead
    of learning physical sensor-to-failure relationships.
    """
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df_out = df.drop(columns=to_drop)
    print(f"\n[3] Dropped identifier columns: {to_drop}")
    print(f"    Shape after drop: {df_out.shape}")
    return df_out


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'Type' feature (L / M / H).

    Encoding scheme (Type_L is the reference / base category):
        L  -->  Type_M = 0,  Type_H = 0
        M  -->  Type_M = 1,  Type_H = 0
        H  -->  Type_M = 0,  Type_H = 1

    Reason for one-hot encoding:
        'Type' is a nominal categorical variable with no inherent ordinal
        relationship between L, M, and H. Encoding it as an integer (e.g.
        L=0, M=1, H=2) would incorrectly imply a linear ordering. One-hot
        encoding makes the encoding explicit, interpretable, and avoids
        imposing any false ordering.

    Reason for dropping Type_L (reference category):
        Using k-1 dummies instead of k avoids multicollinearity (the dummy
        variable trap). The ANN recovers L from Type_M=0 AND Type_H=0.
    """
    type_values = sorted(df[TYPE_COL].unique())
    print(f"\n[4] Encoding '{TYPE_COL}'. Unique values: {type_values}")

    df = df.copy()
    df["Type_M"] = (df[TYPE_COL] == "M").astype(int)
    df["Type_H"] = (df[TYPE_COL] == "H").astype(int)
    df = df.drop(columns=[TYPE_COL])

    # Value counts for verification
    for flag in ("Type_M", "Type_H"):
        counts = df[flag].value_counts().sort_index().to_dict()
        print(f"    {flag} counts: {counts}")
    return df


def split_features_target(df: pd.DataFrame):
    """
    Extract the 7-feature matrix X and binary target y.

    Sub-targets (TWF, HDF, PWF, OSF, RNF) are dropped here because they
    are components of the target label itself. Using them as ANN inputs
    would constitute direct data leakage (the ANN could trivially reconstruct
    Machine failure by OR-ing the sub-targets).
    """
    X_df = df[FINAL_FEATURE_NAMES].copy()
    y = df[TARGET_COL].values.astype(int)
    return X_df, y


def stratified_split(X_df: pd.DataFrame, y: np.ndarray):
    """
    80/20 stratified train/test split — implemented with NumPy only.

    Reason for stratification:
        The dataset has a ~3.4% positive (failure) rate. A simple random split
        could by chance assign very few positives to one partition, making the
        train or test class distribution unrepresentative. Stratification
        enforces that both partitions retain the original ~3.4% failure rate.

    Reason for random_state=42:
        Fixes the partition for reproducibility. Any researcher re-running
        this script on the same CSV obtains identical train/test sets.

    Implementation note:
        Implemented directly with NumPy to avoid importing sklearn/scipy.
        The algorithm: for each class separately, shuffle indices with the
        fixed random seed, take 80% for train (rounded to maintain exact
        8000/2000 total split) and 20% for test, then concatenate and sort.
    """
    rng = np.random.default_rng(seed=42)
    train_indices = []
    test_indices  = []

    for class_val in np.unique(y):
        class_idx = np.where(y == class_val)[0]        # indices for this class
        class_idx = rng.permutation(class_idx)          # shuffle within class
        split_at  = int(np.round(0.80 * len(class_idx)))
        train_indices.append(class_idx[:split_at])
        test_indices.append(class_idx[split_at:])

    train_idx = np.sort(np.concatenate(train_indices))  # sort to preserve order
    test_idx  = np.sort(np.concatenate(test_indices))

    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_test_df  = X_df.iloc[test_idx].reset_index(drop=True)
    y_train    = y[train_idx]
    y_test     = y[test_idx]

    return X_train_df, X_test_df, y_train, y_test


def fit_minmax(X_train_df: pd.DataFrame) -> dict:
    """
    Fit Min-Max scaling parameters on training data ONLY.

    Formula applied later: x_scaled = (x - train_min) / (train_max - train_min)

    Reason for fitting on training data only:
        Computing min/max from the full dataset (train + test) would leak
        information about test-set values into the normalisation parameters.
        This biases evaluation results optimistically. By fitting only on the
        training set and applying those same parameters to the test set, we
        replicate real deployment conditions where test/future data is unseen.

    Only the five numeric features are scaled. Type_M and Type_H are already
    binary {0, 1} and require no scaling.
    """
    norm_params = {}
    for col in NUMERIC_FEATURES:
        col_min = float(X_train_df[col].min())
        col_max = float(X_train_df[col].max())
        norm_params[col] = {"min": col_min, "max": col_max}
    return norm_params


def apply_minmax(X_df: pd.DataFrame, norm_params: dict) -> np.ndarray:
    """
    Apply pre-fitted Min-Max parameters to a feature DataFrame.

    The SAME training-derived min/max values are used for both the training
    and test sets. Test values outside the training range are allowed to fall
    outside [0, 1]; this is not corrected (it is informative about outliers).

    Returns a NumPy float64 array in the canonical 7-feature order.
    """
    X = X_df.copy()
    for col, params in norm_params.items():
        denom = params["max"] - params["min"]
        if denom == 0:
            # Constant feature edge case -- set to 0
            X[col] = 0.0
        else:
            X[col] = (X[col] - params["min"]) / denom
    return X[FINAL_FEATURE_NAMES].values.astype(np.float64)


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Calculate class weights from TRAINING labels only.

    Formula:
        w_pos (failure = 1)    = N_negative / N_positive
        w_neg (no failure = 0) = 1.0

    Reason for class weights:
        With ~3.4% positives, a model that always predicts 'no failure'
        achieves ~96.6% accuracy but zero recall on failures. Class weights
        penalise missed positive predictions more heavily, pushing the ANN
        towards useful sensitivity for the rare failure class.

    Reason for calculating from training data only:
        The test labels must remain completely invisible during any part of
        the training procedure, including the derivation of loss weights.
        Using test labels here would constitute data leakage.
    """
    n_total = len(y_train)
    n_positive = int(y_train.sum())
    n_negative = n_total - n_positive

    if n_positive == 0:
        raise ValueError("No positive (failure) samples found in training set. "
                         "Check stratified split.")

    w_pos = n_negative / n_positive   # heavier penalty for missing a failure
    w_neg = 1.0

    return {
        "w_pos": w_pos,
        "w_neg": w_neg,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_total": n_total,
    }


def save_outputs(X_train, X_test, y_train, y_test, norm_params, class_weights):
    """Save all processed arrays and parameters to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    output_json = {
        "norm_params": norm_params,
        "class_weights": class_weights,
        "feature_names": FINAL_FEATURE_NAMES,
        "target_name": TARGET_COL,
    }
    json_path = os.path.join(PROCESSED_DIR, "norm_params.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"\n[9] Saved files to '{PROCESSED_DIR}/':")
    for fname in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "norm_params.json"]:
        fpath = os.path.join(PROCESSED_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname:25s}  ({size_kb:.1f} KB)")


def print_summary(original_df, X_train, X_test, y_train, y_test,
                  norm_params, class_weights):
    """Print the complete verification/summary report."""
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE -- SUMMARY")
    print("=" * 70)

    print(f"\n[5] Final feature names ({len(FINAL_FEATURE_NAMES)}):")
    for i, name in enumerate(FINAL_FEATURE_NAMES, 1):
        print(f"    {i}. {name}")

    print(f"\n[6] Array shapes:")
    print(f"    X_train : {X_train.shape}")
    print(f"    X_test  : {X_test.shape}")
    print(f"    y_train : {y_train.shape}")
    print(f"    y_test  : {y_test.shape}")

    # Training class distribution
    train_fail  = int(y_train.sum())
    train_total = len(y_train)
    train_pct   = 100.0 * train_fail / train_total
    print(f"\n[7] Class distribution -- Training set (n={train_total}):")
    print(f"    Failures   (1) : {train_fail:5d}  ({train_pct:.2f}%)")
    print(f"    No-failure (0) : {train_total - train_fail:5d}  ({100 - train_pct:.2f}%)")

    # Test class distribution
    test_fail  = int(y_test.sum())
    test_total = len(y_test)
    test_pct   = 100.0 * test_fail / test_total
    print(f"\n[8] Class distribution -- Test set (n={test_total}):")
    print(f"    Failures   (1) : {test_fail:5d}  ({test_pct:.2f}%)")
    print(f"    No-failure (0) : {test_total - test_fail:5d}  ({100 - test_pct:.2f}%)")

    # Class weights
    print(f"\n[9] Training class weights (from training labels only):")
    print(f"    w_pos (failure=1)    = {class_weights['w_pos']:.4f}"
          f"  [{class_weights['n_negative']} neg / {class_weights['n_positive']} pos]")
    print(f"    w_neg (no-failure=0) = {class_weights['w_neg']:.4f}")

    # Normalisation parameters
    print(f"\n[10] Training Min-Max normalisation parameters (training set only):")
    print(f"    {'Feature':<35s}  {'Train Min':>12s}  {'Train Max':>12s}")
    print(f"    {'-'*35}  {'-'*12}  {'-'*12}")
    for feat, params in norm_params.items():
        print(f"    {feat:<35s}  {params['min']:>12.4f}  {params['max']:>12.4f}")

    print("\n" + "=" * 70)
    print("KEY REMINDERS")
    print("=" * 70)
    print("  * AI4I 2020 does NOT contain a vibration sensor.")
    print("  * Torque [Nm] is Torque -- NOT a vibration proxy.")
    print("  * Normalisation parameters fitted on TRAINING data only.")
    print("  * Class weights calculated from TRAINING labels only.")
    print("  * Test set was NEVER used to compute any preprocessing parameter.")
    print("=" * 70)


# ==============================================================================
# Main pipeline
# ==============================================================================

def run_preprocessing():
    """Execute the full preprocessing pipeline and save outputs."""

    # Step 1: Load and verify
    df = load_and_verify(RAW_CSV)
    original_df = df.copy()

    # Step 2: Drop identifier columns (UID / UDI / Product ID)
    df = drop_identifiers(df)

    # Step 3: One-hot encode Type (L/M/H -> Type_M, Type_H)
    df = encode_type(df)

    # Step 4: Extract 7-feature matrix X and binary target y
    X_df, y = split_features_target(df)
    print(f"\n[5] Feature matrix before split: {X_df.shape}")
    print(f"    Features : {list(X_df.columns)}")
    print(f"    Target   : '{TARGET_COL}' -- overall failure rate: "
          f"{y.sum()}/{len(y)} ({100*y.mean():.2f}%)")

    # Step 5: Stratified 80/20 train/test split
    X_train_df, X_test_df, y_train, y_test = stratified_split(X_df, y)
    print(f"\n[6] Stratified split (80/20, random_state=42):")
    print(f"    Train: {len(y_train)} rows  |  Test: {len(y_test)} rows")

    # Step 6: Fit Min-Max normalisation on training data only
    norm_params = fit_minmax(X_train_df)

    # Step 7: Apply same training norm_params to both train and test
    X_train = apply_minmax(X_train_df, norm_params)
    X_test  = apply_minmax(X_test_df,  norm_params)

    # Step 8: Calculate class weights from training labels only
    class_weights = compute_class_weights(y_train)

    # Step 9: Save all outputs
    save_outputs(X_train, X_test, y_train, y_test, norm_params, class_weights)

    # Step 10: Print full summary / verification report
    print_summary(original_df, X_train, X_test, y_train, y_test,
                  norm_params, class_weights)

    return X_train, X_test, y_train, y_test, norm_params, class_weights


if __name__ == "__main__":
    try:
        run_preprocessing()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[DATA ERROR] {e}", file=sys.stderr)
        sys.exit(1)
