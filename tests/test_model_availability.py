"""
tests/test_model_availability.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Tests for model dependency availability and recovery handling.

Covers:
  - T07: Missing ANN model file (isolated non-destructive check)
  - T08: Recovery after missing-model test (re-instantiation with valid assets)
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import PredictiveMaintenancePipeline


class TestModelAvailability(unittest.TestCase):
    """Verify behavior when model weights are missing or restored."""

    def test_t07_missing_ann_model_file(self):
        """
        T07: Verify FileNotFoundError when ANN weights file is nonexistent.
        Uses an isolated dummy path without moving, renaming, or deleting real files.
        """
        isolated_dummy_path = os.path.join("results", "nonexistent_isolated_test_weights.npz")
        self.assertFalse(os.path.exists(isolated_dummy_path), "Isolated dummy path must not exist.")

        with self.assertRaises(FileNotFoundError) as context:
            PredictiveMaintenancePipeline(weights_path=isolated_dummy_path)

        # Verify informative error message
        err_msg = str(context.exception)
        self.assertIn("ANN weights file not found", err_msg)
        self.assertIn(isolated_dummy_path, err_msg)

    def test_t07_missing_normalization_file(self):
        """
        T07 Subtest: Verify FileNotFoundError when norm_params.json is nonexistent.
        Uses an isolated dummy path without modifying real files.
        """
        isolated_dummy_path = os.path.join("data", "processed", "nonexistent_isolated_norm_params.json")
        self.assertFalse(os.path.exists(isolated_dummy_path), "Isolated dummy path must not exist.")

        with self.assertRaises(FileNotFoundError) as context:
            PredictiveMaintenancePipeline(norm_params_path=isolated_dummy_path)

        err_msg = str(context.exception)
        self.assertIn("Normalisation parameters file not found", err_msg)
        self.assertIn(isolated_dummy_path, err_msg)

    def test_t08_recovery_after_missing_model_test(self):
        """
        T08: Verify system recovery when pipeline is initialized with real model assets.
        Confirms that the missing-model check caused no persistent corruption.
        """
        # Re-instantiate pipeline using real default paths
        pipeline = PredictiveMaintenancePipeline()
        self.assertIsNotNone(pipeline.ann_model)
        self.assertIsNotNone(pipeline.fuzzy_system)

        # Execute normal prediction to confirm operational status
        res = pipeline.predict(
            product_type="L",
            air_temp_K=300.0,
            process_temp_K=310.0,
            rotational_speed_rpm=1500.0,
            torque_Nm=40.0,
            tool_wear_min=20.0,
        )
        self.assertIsInstance(res, dict)
        self.assertIn(res["priority"], {"LOW", "MEDIUM", "HIGH", "URGENT"})
        self.assertGreaterEqual(res["fault_probability"], 0.0)
        self.assertLessEqual(res["fault_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
