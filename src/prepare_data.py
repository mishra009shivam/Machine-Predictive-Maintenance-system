"""
src/prepare_data.py
──────────────────────────────────────────────────────────────────────────────
Reproducible data-preparation pipeline entry point for SC08:
Machine Predictive Maintenance & Fault Priority System.

Dataset: UCI AI4I 2020 Predictive Maintenance Dataset (ID 601)
Raw location: data/raw/ai4i2020.csv
Output location: data/processed/

This script serves as the standardized entry point for Task 2 (Data Preparation).
It reuses the exact, verified preprocessing functions from src.preprocessing to
guarantee 100% pipeline consistency and prevent data leakage:
- Drops non-predictive row identifiers (UID / UDI / Product ID)
- One-hot encodes 'Type' (L/M/H -> Type_M, Type_H; Type_L reference category)
- Preserves the 7 canonical ANN input features in exact order
- Excludes individual sub-failure modes (TWF, HDF, PWF, OSF, RNF) from ANN inputs
- Performs an 80/20 stratified train/test split (random_state=42)
- Fits Min-Max normalization statistics on the training partition only
- Calculates training set class weights (w_pos ≈ 28.52)
- Saves processed NumPy arrays and normalization parameters to data/processed/
──────────────────────────────────────────────────────────────────────────────
"""

import sys
from src.preprocessing import run_preprocessing

def prepare_data():
    """Run the complete data preparation pipeline."""
    print("Executing SC08 Data Preparation Pipeline...")
    return run_preprocessing()

if __name__ == "__main__":
    try:
        prepare_data()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[DATA ERROR] {e}", file=sys.stderr)
        sys.exit(1)
