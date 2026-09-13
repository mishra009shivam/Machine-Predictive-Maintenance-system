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

# Implementation will be added in Step 5 (after Step 4 approval).
