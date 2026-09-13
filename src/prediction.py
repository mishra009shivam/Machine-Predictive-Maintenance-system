"""
src/prediction.py
──────────────────────────────────────────────────────────────────────────────
Full hybrid prediction pipeline.

Pipeline flow for a single new input
--------------------------------------
    Raw sensor values (unscaled)
          ↓
    Input validation (range checks, type checks)
          ↓
    Normalise using saved norm_params.json (training-set min/max)
          ↓
    ANN forward pass → fault_probability ∈ [0, 1]
          ↓
    Fuzzy Logic evaluation (fault_prob + unscaled sensors)
          ↓
    severity_index ∈ [0, 100]
          ↓
    Priority mapping → URGENT / HIGH / MEDIUM / LOW
          ↓
    Recommendation text
          ↓
    Return structured result dict

Usage
-----
    from src.prediction import predict
    result = predict(
        air_temp_K=300.5,
        process_temp_K=312.1,
        rpm=1250,
        torque_Nm=65.3,
        tool_wear_min=195,
        product_type='M'
    )
    print(result)
    # {
    #     'fault_probability': <float>,
    #     'severity_index'   : <float>,
    #     'priority'         : <str>,
    #     'recommendation'   : <str>,
    #     'rules_fired'      : <list>   ← for Fuzzy Inspector in dashboard
    # }
"""

# Implementation will be added in Step 8 (after Step 7 approval).
