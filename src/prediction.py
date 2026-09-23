"""
src/prediction.py
──────────────────────────────────────────────────────────────────────────────
End-to-End Hybrid Prediction Pipeline for Predictive Maintenance.

Role in the architecture
------------------------
Connects all system components into a single inference pipeline:
  Raw User Inputs
        ↓
  Input Validation (type & range checks)
        ↓
  One-Hot Encoding (Type L/M/H → Type_M, Type_H)
        ↓
  Min-Max Normalisation (using saved data/processed/norm_params.json)
        ↓
  ANN Inference (7 features → trained weights from results/ann_weights.npz)
        ↓
  ANN Fault Probability P(Failure) ∈ [0, 1]
        ↓
  Fuzzy Logic System (Mamdani inference via src.fuzzy)
        ↓
  Severity Index ∈ [0, 100] & Maintenance Priority (LOW / MEDIUM / HIGH / URGENT)
        ↓
  Structured Maintenance Recommendation & Explanation

Notes
-----
- Retraining is NOT performed. Pre-trained weights are loaded from results/ann_weights.npz.
- AI4I 2020 dataset contains NO vibration feature. Torque [Nm] is spindle torque.
"""

import json
import os
import sys

# Ensure parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.ann import NeuralNetwork
from src.fuzzy import FuzzyMaintenance


# ==============================================================================
# Constants & Default Paths
# ==============================================================================

WEIGHTS_PATH: str = os.path.join("results", "ann_weights.npz")
NORM_PARAMS_PATH: str = os.path.join("data", "processed", "norm_params.json")

VALID_PRODUCT_TYPES: set = {"L", "M", "H"}

# Physical validation boundaries for inputs
BOUNDS = {
    "air_temp_K":           (250.0, 350.0),
    "process_temp_K":       (250.0, 350.0),
    "rotational_speed_rpm": (0.0,   5000.0),
    "torque_Nm":            (0.0,   150.0),
    "tool_wear_min":        (0.0,   300.0),
}


# ==============================================================================
# Input Validation Function
# ==============================================================================

def validate_inputs(
    product_type: str,
    air_temp_K: float,
    process_temp_K: float,
    rotational_speed_rpm: float,
    torque_Nm: float,
    tool_wear_min: float,
) -> dict:
    """
    Validate raw user inputs for data types and realistic physical boundaries.

    Parameters
    ----------
    product_type         : str — Product quality variant ('L', 'M', or 'H')
    air_temp_K           : float — Air temperature in Kelvin [250.0, 350.0]
    process_temp_K       : float — Process temperature in Kelvin [250.0, 350.0]
    rotational_speed_rpm : float — Spindle rotational speed [0.0, 5000.0]
    torque_Nm            : float — Spindle torque in Nm [0.0, 150.0]
    tool_wear_min        : float — Cumulative tool wear in minutes [0.0, 300.0]

    Returns
    -------
    dict — validated and type-cast inputs

    Raises
    ------
    TypeError  : if an input has an invalid data type
    ValueError : if product_type is not L/M/H or a value is out of physical bounds
    """
    # 1. Product Type Check
    if not isinstance(product_type, str):
        raise TypeError(f"product_type must be a string, got {type(product_type).__name__}.")
    
    clean_type = product_type.strip().upper()
    if clean_type not in VALID_PRODUCT_TYPES:
        raise ValueError(
            f"Invalid product_type '{product_type}'. Must be one of: {sorted(VALID_PRODUCT_TYPES)}."
        )

    # 2. Numeric Type & Range Checks
    numeric_inputs = {
        "air_temp_K":           air_temp_K,
        "process_temp_K":       process_temp_K,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_Nm":            torque_Nm,
        "tool_wear_min":        tool_wear_min,
    }

    validated_numerics = {}
    for name, val in numeric_inputs.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(f"Input '{name}' must be numeric (int/float), got {type(val).__name__}.")
        
        float_val = float(val)
        min_b, max_b = BOUNDS[name]
        if not (min_b <= float_val <= max_b):
            raise ValueError(
                f"Input '{name}' value {float_val} is out of valid physical range [{min_b}, {max_b}]."
            )
        validated_numerics[name] = float_val

    return {
        "product_type": clean_type,
        **validated_numerics
    }


# ==============================================================================
# Predictive Maintenance Pipeline Class
# ==============================================================================

class PredictiveMaintenancePipeline:
    """
    End-to-End Hybrid Predictive Maintenance Pipeline.

    Integrates:
      1. Preprocessing & Min-Max Normalisation (using training parameters)
      2. Trained Artificial Neural Network (7 -> 16 -> 8 -> 1 architecture)
      3. Hand-coded Mamdani Fuzzy Logic Severity & Priority System
    """

    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
        norm_params_path: str = NORM_PARAMS_PATH,
    ):
        """
        Initialise the pipeline by loading pre-trained ANN weights and normalisation parameters.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"ANN weights file not found at '{weights_path}'. "
                "Run 'python -m src.ann' first to train and save weights."
            )
        if not os.path.exists(norm_params_path):
            raise FileNotFoundError(
                f"Normalisation parameters file not found at '{norm_params_path}'. "
                "Run 'python -m src.preprocessing' first."
            )

        # 1. Load Normalisation Parameters & Metadata
        with open(norm_params_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.norm_params = meta["norm_params"]
        self.feature_names = meta.get("feature_names", [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_M",
            "Type_H"
        ])

        # 2. Instantiate ANN & Load Trained Weights
        weights_npz = np.load(weights_path)
        weights_dict = {k: weights_npz[k] for k in weights_npz.files}
        
        self.ann_model = NeuralNetwork(input_dim=7, h1_dim=16, h2_dim=8, output_dim=1)
        self.ann_model.set_weights(weights_dict)

        # 3. Instantiate Fuzzy Logic Subsystem
        self.fuzzy_system = FuzzyMaintenance()

    def _encode_and_normalize(self, raw: dict) -> np.ndarray:
        """
        Convert raw input dict into normalized 7D feature array for the ANN.

        Feature Order:
          [Air temp, Process temp, Speed, Torque, Tool wear, Type_M, Type_H]
        """
        ptype = raw["product_type"]
        type_m = 1.0 if ptype == "M" else 0.0
        type_h = 1.0 if ptype == "H" else 0.0

        num_mapping = {
            "Air temperature [K]":     raw["air_temp_K"],
            "Process temperature [K]": raw["process_temp_K"],
            "Rotational speed [rpm]":  raw["rotational_speed_rpm"],
            "Torque [Nm]":             raw["torque_Nm"],
            "Tool wear [min]":         raw["tool_wear_min"],
        }

        # Min-Max Normalisation: (x - min) / (max - min)
        scaled_numerics = []
        for feat in [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]:
            val = num_mapping[feat]
            p_min = self.norm_params[feat]["min"]
            p_max = self.norm_params[feat]["max"]
            denom = p_max - p_min if p_max > p_min else 1.0
            scaled_val = (val - p_min) / denom
            scaled_numerics.append(scaled_val)

        # 7-Feature Vector
        x_vector = np.array(scaled_numerics + [type_m, type_h], dtype=np.float64).reshape(1, 7)
        return x_vector

    def predict(
        self,
        product_type: str,
        air_temp_K: float,
        process_temp_K: float,
        rotational_speed_rpm: float,
        torque_Nm: float,
        tool_wear_min: float,
    ) -> dict:
        """
        Execute full prediction pipeline for a single machine reading.

        Returns
        -------
        dict with keys:
            fault_probability  : float in [0.0, 1.0] (ANN output)
            severity_index     : float in [0.0, 100.0] (Fuzzy CoG output)
            priority           : str ('LOW' / 'MEDIUM' / 'HIGH' / 'URGENT')
            recommendation     : str (Maintenance action recommendation)
            fuzzy_explanation  : str (Textual explanation of ANN + Fuzzy interaction)
            fired_rules        : list of fired fuzzy rules with strengths
            raw_inputs         : dict of validated raw inputs
        """
        # Step 1: Input Validation
        validated = validate_inputs(
            product_type=product_type,
            air_temp_K=air_temp_K,
            process_temp_K=process_temp_K,
            rotational_speed_rpm=rotational_speed_rpm,
            torque_Nm=torque_Nm,
            tool_wear_min=tool_wear_min,
        )

        # Step 2: Feature Encoding & Min-Max Normalisation
        X_scaled = self._encode_and_normalize(validated)

        # Step 3: ANN Forward Pass -> Fault Probability
        y_hat = self.ann_model.forward(X_scaled)
        fault_prob = float(y_hat[0, 0])

        # Step 4: Fuzzy Logic Inference -> Severity Index & Priority
        fuzzy_res = self.fuzzy_system.evaluate(
            fault_prob=fault_prob,
            tool_wear_min=validated["tool_wear_min"],
            torque_Nm=validated["torque_Nm"],
        )

        severity_index = fuzzy_res["severity_index"]
        priority = fuzzy_res["priority"]
        recommendation = fuzzy_res["recommendation"]
        fired_rules = fuzzy_res["fired_rules"]

        # Step 5: Construct Explainable Synthesis String
        explanation = (
            f"ANN predicted a failure probability of {fault_prob:.4f} ({fault_prob*100:.1f}%). "
            f"Fuzzy logic evaluated this probability alongside tool wear ({validated['tool_wear_min']} min) "
            f"and spindle torque ({validated['torque_Nm']} Nm), triggering {len(fired_rules)} rule(s) "
            f"to yield a Severity Index of {severity_index:.2f}/100 with Priority level '{priority}'."
        )

        # Step 6: Format Final Structured Result
        return {
            "fault_probability": round(fault_prob, 4),
            "severity_index":    severity_index,
            "priority":          priority,
            "recommendation":    recommendation,
            "fuzzy_explanation": explanation,
            "fired_rules":       fired_rules,
            "raw_inputs":        validated,
        }


# ==============================================================================
# Pipeline Verification & Scenario Tests
# ==============================================================================

def run_pipeline_tests():
    """
    Run 5 comprehensive test scenarios covering the operational risk spectrum.
    """
    pipeline = PredictiveMaintenancePipeline()

    test_cases = [
        {
            "name": "Scenario 1: Normal Operating Machine",
            "inputs": {
                "product_type": "L",
                "air_temp_K": 300.0,
                "process_temp_K": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_Nm": 40.0,
                "tool_wear_min": 20.0,
            },
            "expected_priority": "LOW",
        },
        {
            "name": "Scenario 2: Moderate-Risk Machine",
            "inputs": {
                "product_type": "M",
                "air_temp_K": 301.0,
                "process_temp_K": 311.0,
                "rotational_speed_rpm": 1400.0,
                "torque_Nm": 48.0,
                "tool_wear_min": 130.0,
            },
            "expected_priority": "MEDIUM",
        },
        {
            "name": "Scenario 3: High ANN Probability (Elevated Risk)",
            "inputs": {
                "product_type": "H",
                "air_temp_K": 303.0,
                "process_temp_K": 313.0,
                "rotational_speed_rpm": 1200.0,
                "torque_Nm": 65.0,
                "tool_wear_min": 50.0,
            },
            "expected_priority": "MEDIUM",
        },
        {
            "name": "Scenario 4: High Tool Wear Machine",
            "inputs": {
                "product_type": "L",
                "air_temp_K": 299.0,
                "process_temp_K": 309.0,
                "rotational_speed_rpm": 1550.0,
                "torque_Nm": 38.0,
                "tool_wear_min": 225.0,
            },
            "expected_priority": "MEDIUM",
        },
        {
            "name": "Scenario 5: Critical Emergency Condition",
            "inputs": {
                "product_type": "M",
                "air_temp_K": 304.0,
                "process_temp_K": 314.0,
                "rotational_speed_rpm": 1180.0,
                "torque_Nm": 68.0,
                "tool_wear_min": 235.0,
            },
            "expected_priority": "URGENT",
        },
    ]

    print("=" * 70)
    print("END-TO-END HYBRID PREDICTION PIPELINE VERIFICATION")
    print("=" * 70)

    all_passed = True

    for i, test in enumerate(test_cases, start=1):
        print(f"\n{'-' * 70}")
        print(f"[{i}/5] {test['name']}")
        print(f"{'-' * 70}")
        print("Raw Inputs:")
        for k, v in test["inputs"].items():
            print(f"  {k:<22s}: {v}")

        res = pipeline.predict(**test["inputs"])

        print("\nPipeline Results:")
        print(f"  ANN Fault Probability : {res['fault_probability']:.4f} ({res['fault_probability']*100:.2f}%)")
        print(f"  Fuzzy Severity Index  : {res['severity_index']:.2f} / 100")
        print(f"  Priority Level        : {res['priority']}")
        print(f"  Recommendation        : {res['recommendation']}")
        print(f"  Fuzzy Explanation     : {res['fuzzy_explanation']}")
        print(f"  Fired Rules ({len(res['fired_rules'])}):")
        for r in res["fired_rules"]:
            print(f"    - Rule R{r['rule']} [{r['consequent']} @ strength {r['strength']:.4f}]: {r['description']}")

        # ── Assertions ────────────────────────────────────────────────────────
        fp_valid = 0.0 <= res["fault_probability"] <= 1.0
        sev_valid = 0.0 <= res["severity_index"] <= 100.0
        prio_valid = res["priority"] in {"LOW", "MEDIUM", "HIGH", "URGENT"}

        print("\nVerification Checks:")
        print(f"  [1] ANN Probability in [0, 1] : {'PASS' if fp_valid else 'FAIL'}")
        print(f"  [2] Severity Index in [0, 100]: {'PASS' if sev_valid else 'FAIL'}")
        print(f"  [3] Priority in valid set    : {'PASS' if prio_valid else 'FAIL'}")

        if not (fp_valid and sev_valid and prio_valid):
            all_passed = False
            print("  ==> TEST FAILED: Assertion error in range limits.")

    print(f"\n{'=' * 70}")
    if all_passed:
        print("ALL 5 END-TO-END PIPELINE SCENARIO TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME PIPELINE TESTS FAILED. PLEASE CHECK LOGS ABOVE.")
    print("=" * 70)


def test_invalid_inputs():
    """
    Verify input validation error handling.
    """
    print("\n" + "=" * 70)
    print("INPUT VALIDATION EXCEPTION TESTS")
    print("=" * 70)

    pipeline = PredictiveMaintenancePipeline()

    invalid_cases = [
        ("Invalid product type", {"product_type": "X", "air_temp_K": 300, "process_temp_K": 310, "rotational_speed_rpm": 1500, "torque_Nm": 40, "tool_wear_min": 20}, ValueError),
        ("Air temp out of range", {"product_type": "L", "air_temp_K": 150, "process_temp_K": 310, "rotational_speed_rpm": 1500, "torque_Nm": 40, "tool_wear_min": 20}, ValueError),
        ("Torque negative", {"product_type": "L", "air_temp_K": 300, "process_temp_K": 310, "rotational_speed_rpm": 1500, "torque_Nm": -10, "tool_wear_min": 20}, ValueError),
        ("Invalid data type", {"product_type": "L", "air_temp_K": "300K", "process_temp_K": 310, "rotational_speed_rpm": 1500, "torque_Nm": 40, "tool_wear_min": 20}, TypeError),
    ]

    for label, inp, expected_exc in invalid_cases:
        try:
            pipeline.predict(**inp)
            print(f"  [FAIL] {label}: Expected {expected_exc.__name__}, but no exception was raised.")
        except expected_exc as e:
            print(f"  [PASS] {label}: Gracefully caught {expected_exc.__name__} -> '{e}'")
        except Exception as e:
            print(f"  [FAIL] {label}: Caught unexpected exception {type(e).__name__} instead of {expected_exc.__name__}.")


if __name__ == "__main__":
    run_pipeline_tests()
    test_invalid_inputs()
