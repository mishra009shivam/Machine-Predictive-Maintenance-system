"""
src/step1_validator.py
──────────────────────────────────────────────────────────────────────────────
Isolated Step 1 Input Contract Validator and Starter Fixture Runner.

Reuses the canonical validation bounds (`BOUNDS`) and product-type set
(`VALID_PRODUCT_TYPES`) from `src.prediction` without modifying any Steps 2-6
production modules.
"""

import csv
import math
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure repository root is in sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.prediction import BOUNDS, VALID_PRODUCT_TYPES

FUZZY_OPERATING_LIMITS: Dict[str, Tuple[float, float]] = {
    "torque_Nm": (0.0, 80.0),
    "tool_wear_min": (0.0, 260.0),
}

NUMERIC_FIELDS: Tuple[str, ...] = (
    "air_temp_K",
    "process_temp_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
)

DEFAULT_FIXTURE_PATH = os.path.join("data", "fixtures", "step1_starter_sensors.csv")


def validate_step1_inputs(
    product_type: Any,
    air_temp_K: Any,
    process_temp_K: Any,
    rotational_speed_rpm: Any,
    torque_Nm: Any,
    tool_wear_min: Any,
) -> Dict[str, Any]:
    """
    Validate raw machine operating inputs against the Step 1 contract.

    Checks:
      1. Missing values (`None` or blank strings)
      2. Incorrect data types (`bool` or non-numeric values for sensor fields)
      3. Invalid product types (must be 'L', 'M', or 'H')
      4. Non-finite values (`NaN`, `+Inf`, `-Inf`)
      5. Physical range boundaries (`BOUNDS` from `src.prediction`)
    """
    if product_type is None:
        raise ValueError("Missing required field 'product_type' (got None).")
    if not isinstance(product_type, str):
        raise TypeError(
            f"product_type must be a string ('L', 'M', or 'H'), got {type(product_type).__name__}."
        )

    clean_type = product_type.strip().upper()
    if not clean_type:
        raise ValueError("Missing required field 'product_type' (empty string).")
    if clean_type not in VALID_PRODUCT_TYPES:
        raise ValueError(
            f"Invalid product_type '{product_type}'. Must be one of {sorted(VALID_PRODUCT_TYPES)}."
        )

    raw_numerics = {
        "air_temp_K": air_temp_K,
        "process_temp_K": process_temp_K,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_Nm": torque_Nm,
        "tool_wear_min": tool_wear_min,
    }

    validated: Dict[str, Any] = {"product_type": clean_type}
    for field, val in raw_numerics.items():
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(f"Missing required numeric field '{field}'.")
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(
                f"Input '{field}' must be numeric (int/float), got {type(val).__name__}."
            )

        float_val = float(val)
        if not math.isfinite(float_val):
            raise ValueError(
                f"Input '{field}' must be finite (no NaN or Inf), got {float_val}."
            )

        min_b, max_b = BOUNDS[field]
        if not (min_b <= float_val <= max_b):
            raise ValueError(
                f"Input '{field}' value {float_val} is outside allowed range [{min_b}, {max_b}]."
            )
        validated[field] = float_val

    return validated


def check_fuzzy_universe_notes(validated: Dict[str, Any]) -> List[str]:
    """
    Return non-fatal notes when valid pipeline inputs exceed the fuzzy universe
    ceilings (torque_Nm > 80.0 or tool_wear_min > 260.0).
    """
    notes: List[str] = []
    for field, (f_min, f_max) in FUZZY_OPERATING_LIMITS.items():
        val = validated[field]
        if not (f_min <= val <= f_max):
            notes.append(
                f"{field}={val} exceeds fuzzy universe [{f_min}, {f_max}] "
                "(valid for pipeline BOUNDS, capped in Streamlit UI)."
            )
    return notes


def _parse_csv_numeric(field: str, raw_value: str) -> float:
    """Parse a CSV string cell into a float, raising explicit errors on missing/non-numeric text."""
    if raw_value is None or not raw_value.strip():
        raise ValueError(f"Missing required numeric field '{field}'.")
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise TypeError(
            f"Input '{field}' must be numeric (int/float), got string '{raw_value}'."
        ) from exc


def validate_fixture_csv(csv_path: str = DEFAULT_FIXTURE_PATH) -> Dict[str, Any]:
    """
    Validate all rows in a starter sensor CSV fixture and report valid cases
    and expected-invalid rejections separately.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fixture CSV not found at '{csv_path}'.")

    valid_results: List[Dict[str, Any]] = []
    expected_rejections: List[Dict[str, Any]] = []
    unexpected_failures: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]
            expected = row["expected_status"].strip().lower()
            try:
                parsed = {
                    "product_type": row["product_type"],
                    "air_temp_K": _parse_csv_numeric("air_temp_K", row["air_temp_K"]),
                    "process_temp_K": _parse_csv_numeric("process_temp_K", row["process_temp_K"]),
                    "rotational_speed_rpm": _parse_csv_numeric(
                        "rotational_speed_rpm", row["rotational_speed_rpm"]
                    ),
                    "torque_Nm": _parse_csv_numeric("torque_Nm", row["torque_Nm"]),
                    "tool_wear_min": _parse_csv_numeric("tool_wear_min", row["tool_wear_min"]),
                }
                validated = validate_step1_inputs(**parsed)
                fuzzy_notes = check_fuzzy_universe_notes(validated)

                if expected == "valid":
                    valid_results.append({
                        "case_id": case_id,
                        "validated": validated,
                        "fuzzy_notes": fuzzy_notes,
                    })
                else:
                    unexpected_failures.append({
                        "case_id": case_id,
                        "reason": "Expected rejection, but input unexpectedly passed validation.",
                    })
            except (TypeError, ValueError) as exc:
                if expected == "expected_invalid":
                    expected_rejections.append({
                        "case_id": case_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    })
                else:
                    unexpected_failures.append({
                        "case_id": case_id,
                        "reason": f"Expected valid, but failed with {type(exc).__name__}: {exc}",
                    })

    return {
        "valid_results": valid_results,
        "expected_rejections": expected_rejections,
        "unexpected_failures": unexpected_failures,
    }


def main() -> int:
    """CLI entry point for running Step 1 starter fixture validation."""
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIXTURE_PATH
    summary = validate_fixture_csv(csv_path)

    print("=== STEP 1 FIXTURE VALIDATION REPORT ===")
    print(f"Fixture: {csv_path}\n")

    print(f"[1] VALID INPUT CASES PASSED ({len(summary['valid_results'])})")
    for item in summary["valid_results"]:
        v = item["validated"]
        print(
            f"  PASS  {item['case_id']}: Type={v['product_type']}, "
            f"T_air={v['air_temp_K']} K, T_proc={v['process_temp_K']} K, "
            f"Speed={v['rotational_speed_rpm']} rpm, Torque={v['torque_Nm']} Nm, "
            f"Wear={v['tool_wear_min']} min"
        )
        for note in item["fuzzy_notes"]:
            print(f"        NOTE: {note}")

    print(f"\n[2] EXPECTED INVALID INPUTS REJECTED ({len(summary['expected_rejections'])})")
    for item in summary["expected_rejections"]:
        print(f"  REJECTED (EXPECTED)  {item['case_id']}: [{item['error_type']}] {item['message']}")

    print(f"\n[3] UNEXPECTED FAILURES ({len(summary['unexpected_failures'])})")
    for item in summary["unexpected_failures"]:
        print(f"  FAIL  {item['case_id']}: {item['reason']}")

    print(
        f"\nSummary: {len(summary['valid_results'])} valid passed, "
        f"{len(summary['expected_rejections'])} expected invalid rejected, "
        f"{len(summary['unexpected_failures'])} unexpected failures."
    )
    return 1 if summary["unexpected_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())

