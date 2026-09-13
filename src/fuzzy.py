"""
src/fuzzy.py
──────────────────────────────────────────────────────────────────────────────
Hand-coded Mamdani Fuzzy Logic system for severity and maintenance priority.
Implemented using NumPy only — no scikit-fuzzy or other fuzzy library.

Role in the hybrid pipeline
----------------------------
The Fuzzy system receives:
  (a) The ANN's fault probability output ∈ [0, 1]
  (b) Four original (unscaled) sensor readings from the dataset

And produces:
  - Severity Index ∈ [0, 100]   (continuous, defuzzified)
  - Maintenance Priority label  (URGENT / HIGH / MEDIUM / LOW)
  - Maintenance Recommendation  (text)

Input linguistic variables (5 inputs)
--------------------------------------
  1. fault_probability   — ANN output, range [0.0, 1.0]
  2. process_temp_K      — Process temperature [K], range [305, 315]
  3. rotational_speed    — Rotational speed [rpm], range [1000, 3000]
  4. torque_Nm           — Torque [Nm], range [0, 80]
                           NOTE: This is Torque — NOT a vibration proxy.
                           See data/README.md — Dataset Limitations.
  5. tool_wear_min       — Tool wear [min], range [0, 260]

Output linguistic variable (1 output)
--------------------------------------
  severity_index         — Severity Index [0, 100]

Membership function shapes: triangular and trapezoidal
Inference method         : Mamdani (min for AND, max for aggregation)
Defuzzification          : Centroid (centre of area) over 1000 discrete points

Unit tests
----------
Before finalising this module, unit tests must cover:
  - Each membership function at boundary and interior values
  - Rule activation: verify correct rules fire for known inputs
  - No-rule-fired case: system must return a safe default
  - Centroid defuzzification: verify against a hand-calculated example
  - Boundary values: inputs at universe min/max
  - Contradictory inputs: e.g., high fault_prob but all-normal sensors
  - Extreme values: all inputs at maximum simultaneously

Usage
-----
    from src.fuzzy import FuzzyMaintenance
    fm = FuzzyMaintenance()
    result = fm.evaluate(fault_prob=0.72, temp_K=312, rpm=1250,
                         torque=65, tool_wear=195)
    # result: {'severity': 81.4, 'priority': 'URGENT', 'recommendation': '...'}
"""

# Implementation will be added in Steps 6–7 (after Step 5 approval).
