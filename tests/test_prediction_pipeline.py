"""
tests/test_prediction_pipeline.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Automated unit tests for the SC08 Predictive Maintenance Pipeline.

Covers:
  - T01: Normal prediction
  - T02: Lower-boundary inputs
  - T03: Upper-boundary inputs
  - T04: Out-of-range inputs
  - T05: Invalid text in numeric fields
  - T06: Missing or None inputs
  - T09: Direct pipeline vs application invocation consistency
"""

import math
import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import PredictiveMaintenancePipeline


class TestPredictionPipeline(unittest.TestCase):
    """Unit test cases for PredictiveMaintenancePipeline."""

    @classmethod
    def setUpClass(cls):
        """Instantiate pipeline once for testing."""
        cls.pipeline = PredictiveMaintenancePipeline()

    def assert_prediction_structure(self, res: dict):
        """Helper to assert that returned dictionary matches expected schema."""
        expected_keys = {
            "fault_probability",
            "severity_index",
            "priority",
            "recommendation",
            "fuzzy_explanation",
            "fired_rules",
            "raw_inputs",
        }
        self.assertIsInstance(res, dict, "Result should be a dictionary.")
        self.assertTrue(expected_keys.issubset(res.keys()), f"Missing keys in result: {expected_keys - res.keys()}")

        # Value domain assertions
        self.assertIsInstance(res["fault_probability"], (float, int))
        self.assertGreaterEqual(res["fault_probability"], 0.0)
        self.assertLessEqual(res["fault_probability"], 1.0)

        self.assertIsInstance(res["severity_index"], (float, int))
        self.assertGreaterEqual(res["severity_index"], 0.0)
        self.assertLessEqual(res["severity_index"], 100.0)

        self.assertIn(res["priority"], {"LOW", "MEDIUM", "HIGH", "URGENT"})
        self.assertIsInstance(res["recommendation"], str)
        self.assertGreater(len(res["recommendation"]), 0)

        self.assertIsInstance(res["fuzzy_explanation"], str)
        self.assertGreater(len(res["fuzzy_explanation"]), 0)

        self.assertIsInstance(res["fired_rules"], list)
        self.assertIsInstance(res["raw_inputs"], dict)

    # â”€â”€ T01: Normal Prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t01_normal_prediction(self):
        """T01: Verify prediction under nominal machine operating conditions."""
        inputs = {
            "product_type": "L",
            "air_temp_K": 300.0,
            "process_temp_K": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_Nm": 40.0,
            "tool_wear_min": 20.0,
        }
        res = self.pipeline.predict(**inputs)
        self.assert_prediction_structure(res)
        self.assertIn(res["priority"], {"LOW", "MEDIUM"})
        self.assertGreater(len(res["fired_rules"]), 0)

    # â”€â”€ T02: Lower-Boundary Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t02_lower_boundary_inputs(self):
        """T02: Verify system stability at absolute physical lower boundary limits."""
        inputs = {
            "product_type": "L",
            "air_temp_K": 250.0,
            "process_temp_K": 250.0,
            "rotational_speed_rpm": 0.0,
            "torque_Nm": 0.0,
            "tool_wear_min": 0.0,
        }
        res = self.pipeline.predict(**inputs)
        self.assert_prediction_structure(res)
        self.assertFalse(math.isnan(res["fault_probability"]))
        self.assertFalse(math.isnan(res["severity_index"]))

    # â”€â”€ T03: Upper-Boundary Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t03_upper_boundary_inputs(self):
        """T03: Verify system behavior at supported upper boundary limits."""
        # Test 1: Supported upper boundary preset (Type M, 80 Nm, 260 min)
        inputs_upper_preset = {
            "product_type": "M",
            "air_temp_K": 304.0,
            "process_temp_K": 314.0,
            "rotational_speed_rpm": 1200.0,
            "torque_Nm": 80.0,
            "tool_wear_min": 260.0,
        }
        res1 = self.pipeline.predict(**inputs_upper_preset)
        self.assert_prediction_structure(res1)
        self.assertIn(res1["priority"], {"HIGH", "URGENT"})
        self.assertGreater(len(res1["fired_rules"]), 0)

        # Test 2: Theoretical maximum supported boundary (Type H, 350 K, 5000 rpm, 80 Nm, 260 min)
        inputs_theoretical_max = {
            "product_type": "H",
            "air_temp_K": 350.0,
            "process_temp_K": 350.0,
            "rotational_speed_rpm": 5000.0,
            "torque_Nm": 80.0,
            "tool_wear_min": 260.0,
        }
        res2 = self.pipeline.predict(**inputs_theoretical_max)
        self.assert_prediction_structure(res2)
        self.assertIn(res2["priority"], {"HIGH", "URGENT"})
        self.assertGreater(len(res2["fired_rules"]), 0)

        # Subtest: Out-of-fuzzy-universe behavior (> 80 Nm or > 260 min)
        # Documents that values beyond the fuzzy universe trigger the safe fallback severity (50.0 -> MEDIUM)
        inputs_out_of_universe = {
            "product_type": "H",
            "air_temp_K": 350.0,
            "process_temp_K": 350.0,
            "rotational_speed_rpm": 5000.0,
            "torque_Nm": 150.0,
            "tool_wear_min": 300.0,
        }
        res_fallback = self.pipeline.predict(**inputs_out_of_universe)
        self.assert_prediction_structure(res_fallback)
        self.assertEqual(res_fallback["severity_index"], 50.0)
        self.assertEqual(res_fallback["priority"], "MEDIUM")
        self.assertEqual(len(res_fallback["fired_rules"]), 0)

    # â”€â”€ T04: Out-of-Range Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t04_out_of_range_inputs(self):
        """T04: Verify defensive ValueError exception when inputs exceed physical bounds."""
        base_valid = {
            "product_type": "L",
            "air_temp_K": 300.0,
            "process_temp_K": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_Nm": 40.0,
            "tool_wear_min": 20.0,
        }

        out_of_range_cases = [
            ("air_temp_K", 240.0),       # Below 250.0
            ("air_temp_K", 360.0),       # Above 350.0
            ("process_temp_K", 240.0),   # Below 250.0
            ("process_temp_K", 360.0),   # Above 350.0
            ("rotational_speed_rpm", -10.0),   # Below 0.0
            ("rotational_speed_rpm", 5500.0),  # Above 5000.0
            ("torque_Nm", -5.0),         # Below 0.0
            ("torque_Nm", 160.0),        # Above 150.0
            ("tool_wear_min", -1.0),     # Below 0.0
            ("tool_wear_min", 310.0),    # Above 300.0
        ]

        for field, invalid_val in out_of_range_cases:
            with self.subTest(field=field, value=invalid_val):
                test_inputs = base_valid.copy()
                test_inputs[field] = invalid_val
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {field}={invalid_val}"):
                    self.pipeline.predict(**test_inputs)

        # Invalid product type string
        with self.subTest(field="product_type", value="X"):
            test_inputs = base_valid.copy()
            test_inputs["product_type"] = "X"
            with self.assertRaises(ValueError):
                self.pipeline.predict(**test_inputs)

    # â”€â”€ T05: Invalid Text in Numeric Fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t05_invalid_text_in_numeric_fields(self):
        """T05: Verify defensive TypeError when non-numeric types are supplied."""
        base_valid = {
            "product_type": "L",
            "air_temp_K": 300.0,
            "process_temp_K": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_Nm": 40.0,
            "tool_wear_min": 20.0,
        }

        invalid_type_cases = [
            ("air_temp_K", "300K"),
            ("process_temp_K", "high"),
            ("rotational_speed_rpm", "1500rpm"),
            ("torque_Nm", "40Nm"),
            ("tool_wear_min", "worn"),
            ("torque_Nm", True),        # Booleans must not pass as numbers
            ("air_temp_K", [300.0]),
        ]

        for field, invalid_val in invalid_type_cases:
            with self.subTest(field=field, value=invalid_val):
                test_inputs = base_valid.copy()
                test_inputs[field] = invalid_val
                with self.assertRaises(TypeError, msg=f"Expected TypeError for {field}={invalid_val}"):
                    self.pipeline.predict(**test_inputs)

    # â”€â”€ T06: Missing or None Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t06_missing_or_none_inputs(self):
        """T06: Verify handling when required inputs are None or empty strings."""
        base_valid = {
            "product_type": "L",
            "air_temp_K": 300.0,
            "process_temp_K": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_Nm": 40.0,
            "tool_wear_min": 20.0,
        }

        # None inputs
        with self.subTest(case="product_type is None"):
            test_inputs = base_valid.copy()
            test_inputs["product_type"] = None
            with self.assertRaises(TypeError):
                self.pipeline.predict(**test_inputs)

        with self.subTest(case="numeric is None"):
            test_inputs = base_valid.copy()
            test_inputs["air_temp_K"] = None
            with self.assertRaises(TypeError):
                self.pipeline.predict(**test_inputs)

        # Empty string inputs
        with self.subTest(case="product_type is empty string"):
            test_inputs = base_valid.copy()
            test_inputs["product_type"] = ""
            with self.assertRaises(ValueError):
                self.pipeline.predict(**test_inputs)

        with self.subTest(case="numeric is empty string"):
            test_inputs = base_valid.copy()
            test_inputs["torque_Nm"] = ""
            with self.assertRaises(TypeError):
                self.pipeline.predict(**test_inputs)

        # NaN / Infinity inputs
        with self.subTest(case="numeric is NaN"):
            test_inputs = base_valid.copy()
            test_inputs["air_temp_K"] = float("nan")
            with self.assertRaises(ValueError):
                self.pipeline.predict(**test_inputs)

        with self.subTest(case="numeric is Inf"):
            test_inputs = base_valid.copy()
            test_inputs["air_temp_K"] = float("inf")
            with self.assertRaises(ValueError):
                self.pipeline.predict(**test_inputs)

    # â”€â”€ T09: Direct Pipeline Consistency Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def test_t09_pipeline_consistency(self):
        """
        T09: Verify repeatability and consistency across identical invocations.
        Compares multiple calls with identical inputs to verify deterministic results.
        """
        test_inputs = {
            "product_type": "M",
            "air_temp_K": 301.0,
            "process_temp_K": 311.0,
            "rotational_speed_rpm": 1400.0,
            "torque_Nm": 48.0,
            "tool_wear_min": 130.0,
        }

        res_a = self.pipeline.predict(**test_inputs)
        res_b = self.pipeline.predict(**test_inputs)

        # Compare numeric outputs within tight floating-point tolerance
        self.assertAlmostEqual(res_a["fault_probability"], res_b["fault_probability"], places=6)
        self.assertAlmostEqual(res_a["severity_index"], res_b["severity_index"], places=4)

        # Exact string/categorical matches
        self.assertEqual(res_a["priority"], res_b["priority"])
        self.assertEqual(res_a["recommendation"], res_b["recommendation"])
        self.assertEqual(res_a["fuzzy_explanation"], res_b["fuzzy_explanation"])
        self.assertEqual(len(res_a["fired_rules"]), len(res_b["fired_rules"]))


if __name__ == "__main__":
    unittest.main()
