"""
tests/test_step1_contract.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the SC08 Step 1 machine input contract, isolated validator,
and starter sensor CSV fixture.

Note on invalid-input tests:
  Invalid-input tests assert that `validate_step1_inputs` raises the expected
  `TypeError` or `ValueError`. A test PASSES when the validator correctly
  rejects invalid input.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import BOUNDS, VALID_PRODUCT_TYPES
from src.step1_validator import (
    DEFAULT_FIXTURE_PATH,
    check_fuzzy_universe_notes,
    validate_fixture_csv,
    validate_step1_inputs,
)


class TestStep1Contract(unittest.TestCase):
    """Test cases verifying Step 1 input contract and fixture validation."""

    def test_s1_01_valid_normal_risky_high_wear_cases(self):
        """S1-01: Valid normal, risky, and high-wear inputs pass and normalize types."""
        cases = [
            ("l", 300, 310, 1500, 40, 20),
            (" m ", 303.0, 313.0, 1200.0, 65.0, 180.0),
            ("H", 299.0, 309.0, 1550.0, 38.0, 225.0),
        ]
        for ptype, t_air, t_proc, rpm, tq, tw in cases:
            with self.subTest(product_type=ptype, torque=tq, tool_wear=tw):
                res = validate_step1_inputs(ptype, t_air, t_proc, rpm, tq, tw)
                self.assertIn(res["product_type"], VALID_PRODUCT_TYPES)
                self.assertIsInstance(res["air_temp_K"], float)
                self.assertEqual(check_fuzzy_universe_notes(res), [])

    def test_s1_02_boundary_inputs_and_fuzzy_gap_note(self):
        """S1-02: Lower, fuzzy-upper, and pipeline-BOUNDS upper boundaries pass."""
        lower = validate_step1_inputs("L", 250.0, 250.0, 0.0, 0.0, 0.0)
        self.assertEqual(lower["torque_Nm"], 0.0)

        fuzzy_upper = validate_step1_inputs("M", 304.0, 314.0, 1200.0, 80.0, 260.0)
        self.assertEqual(check_fuzzy_universe_notes(fuzzy_upper), [])

        pipeline_upper = validate_step1_inputs("H", 350.0, 350.0, 5000.0, 150.0, 300.0)
        notes = check_fuzzy_universe_notes(pipeline_upper)
        self.assertEqual(len(notes), 2)

    def test_s1_03_missing_values_rejected(self):
        """S1-03 (Expected Rejection): Missing/None/empty inputs raise ValueError."""
        missing_cases = [
            (None, 300.0, 310.0, 1500.0, 40.0, 20.0),
            ("", 300.0, 310.0, 1500.0, 40.0, 20.0),
            ("L", None, 310.0, 1500.0, 40.0, 20.0),
            ("L", 300.0, 310.0, 1500.0, "", 20.0),
        ]
        for args in missing_cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    validate_step1_inputs(*args)

    def test_s1_04_invalid_types_rejected(self):
        """S1-04 (Expected Rejection): Wrong types (text, bool, int product_type) raise TypeError."""
        bad_type_cases = [
            (123, 300.0, 310.0, 1500.0, 40.0, 20.0),
            ("L", "300K", 310.0, 1500.0, 40.0, 20.0),
            ("L", 300.0, 310.0, 1500.0, True, 20.0),
            ("L", 300.0, 310.0, 1500.0, 40.0, "worn"),
        ]
        for args in bad_type_cases:
            with self.subTest(args=args):
                with self.assertRaises(TypeError):
                    validate_step1_inputs(*args)

    def test_s1_05_invalid_product_type_rejected(self):
        """S1-05 (Expected Rejection): Product types outside L/M/H raise ValueError."""
        for bad_type in ("X", "LOW", "LM"):
            with self.subTest(product_type=bad_type):
                with self.assertRaises(ValueError):
                    validate_step1_inputs(bad_type, 300.0, 310.0, 1500.0, 40.0, 20.0)

    def test_s1_06_out_of_range_inputs_rejected(self):
        """S1-06 (Expected Rejection): Values outside BOUNDS raise ValueError."""
        out_of_range = [
            ("air_temp_low", ("L", BOUNDS["air_temp_K"][0] - 0.1, 310.0, 1500.0, 40.0, 20.0)),
            ("air_temp_high", ("L", BOUNDS["air_temp_K"][1] + 0.1, 310.0, 1500.0, 40.0, 20.0)),
            ("speed_high", ("L", 300.0, 310.0, BOUNDS["rotational_speed_rpm"][1] + 1.0, 40.0, 20.0)),
            ("torque_high", ("L", 300.0, 310.0, 1500.0, BOUNDS["torque_Nm"][1] + 0.1, 20.0)),
            ("wear_neg", ("L", 300.0, 310.0, 1500.0, 40.0, -0.1)),
            ("wear_high", ("L", 300.0, 310.0, 1500.0, 40.0, BOUNDS["tool_wear_min"][1] + 1.0)),
        ]
        for label, args in out_of_range:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    validate_step1_inputs(*args)

    def test_s1_07_nan_and_inf_rejected(self):
        """S1-07 (Expected Rejection): NaN and +/-Inf raise ValueError."""
        non_finite = [
            ("nan_air", ("L", float("nan"), 310.0, 1500.0, 40.0, 20.0)),
            ("inf_torque", ("L", 300.0, 310.0, 1500.0, float("inf"), 20.0)),
            ("ninf_wear", ("L", 300.0, 310.0, 1500.0, 40.0, float("-inf"))),
        ]
        for label, args in non_finite:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    validate_step1_inputs(*args)

    def test_s1_08_starter_fixture_csv_validation(self):
        """S1-08: Starter CSV fixture produces 5 valid passes, 6 expected rejections, 0 unexpected failures."""
        summary = validate_fixture_csv(DEFAULT_FIXTURE_PATH)
        self.assertEqual(len(summary["valid_results"]), 5)
        self.assertEqual(len(summary["expected_rejections"]), 6)
        self.assertEqual(len(summary["unexpected_failures"]), 0)


if __name__ == "__main__":
    unittest.main()

