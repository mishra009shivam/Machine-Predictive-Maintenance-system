# Step 1 — Problem Framing, Input/Output Contracts, Baseline & Validation

> **Course / Project Code:** SC08 — Soft Computing
>
> **Project Title:** Machine Predictive Maintenance & Fault Priority System
>
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)
>
> **Revision Branch:** `step-1-contract-revision`

---

## 1. Machine Input Contract

The system accepts **6 raw operational parameters** from the machine operator or sensor feed. During preprocessing ([`src/preprocessing.py`](../src/preprocessing.py)), `product_type` is one-hot encoded so the ANN receives **7 features** in canonical order.

### 1.1 Raw Inputs, Units, Types, and Allowed Ranges

| # | Parameter Key | Dataset / Feature Name | Data Type | Unit | Pipeline `BOUNDS` ([`src/prediction.py`](../src/prediction.py)) | Fuzzy / UI Operating Range ([`src/fuzzy.py`](../src/fuzzy.py), [`app.py`](../app.py)) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `product_type` | `Type` | `str` | — | `{'L', 'M', 'H'}` | `{'L', 'M', 'H'}` |
| 2 | `air_temp_K` | `Air temperature [K]` | `float` | K | `[250.0, 350.0]` | `[250.0, 350.0]` |
| 3 | `process_temp_K` | `Process temperature [K]` | `float` | K | `[250.0, 350.0]` | `[250.0, 350.0]` |
| 4 | `rotational_speed_rpm` | `Rotational speed [rpm]` | `float` | rpm | `[0.0, 5000.0]` | `[0.0, 5000.0]` |
| 5 | `torque_Nm` | `Torque [Nm]` | `float` | Nm | `[0.0, 150.0]` | `[0.0, 80.0]` |
| 6 | `tool_wear_min` | `Tool wear [min]` | `float` | min | `[0.0, 300.0]` | `[0.0, 260.0]` |

> **Important Dataset Note:** The AI4I 2020 dataset does **not** contain a vibration sensor. `Torque [Nm]` represents mechanical spindle torque and is never treated as a vibration proxy.

### 1.2 Pipeline Input Bounds vs. Fuzzy-System Operating Ranges

- **`src/prediction.py` (`BOUNDS`):** Allows `torque_Nm` up to `150.0 Nm` and `tool_wear_min` up to `300.0 min` for defensive physical validation.
- **`src/fuzzy.py` & `app.py`:** Fuzzy membership supports end at `80.0 Nm` and `260.0 min` (covering the training distribution `3.8–76.6 Nm` and `0–253 min`). Inputs in the gap (`80–150 Nm` or `260–300 min`) pass `BOUNDS` validation, but fall outside fuzzy MF support (triggering the safe `50.0` fallback severity if called directly); `app.py` caps UI inputs at `80.0 Nm` and `260.0 min`.

### 1.3 Product Type Encoding & 7-Feature ANN Input Order

`Type` is normalized via `.strip().upper()` (accepting lowercase and whitespace-padded `'L'`, `'M'`, `'H'`, matching [`src/prediction.py`](../src/prediction.py)) and one-hot encoded with `L` as the reference category:
- `L` $\to$ `Type_M = 0.0`, `Type_H = 0.0`
- `M` $\to$ `Type_M = 1.0`, `Type_H = 0.0`
- `H` $\to$ `Type_M = 0.0`, `Type_H = 1.0`

Exact 7-feature ANN input vector order (Min-Max normalized using training statistics in [`data/processed/norm_params.json`](../data/processed/norm_params.json)):
1. `Air temperature [K]` (`[295.3, 304.5]`)
2. `Process temperature [K]` (`[305.7, 313.8]`)
3. `Rotational speed [rpm]` (`[1168.0, 2886.0]`)
4. `Torque [Nm]` (`[3.8, 76.6]`)
5. `Tool wear [min]` (`[0.0, 253.0]`)
6. `Type_M`
7. `Type_H`

### 1.4 Valid and Invalid Input Examples

- **Valid Normal Example:** `product_type='L'`, `air_temp_K=300.0`, `process_temp_K=310.0`, `rotational_speed_rpm=1500.0`, `torque_Nm=40.0`, `tool_wear_min=20.0`
- **Valid Boundary Examples:** Lower (`'L', 250.0, 250.0, 0.0, 0.0, 0.0`); Fuzzy Upper (`'M', 304.0, 314.0, 1200.0, 80.0, 260.0`)
- **Invalid Examples (Expected Rejections):** Missing (`torque_Nm=''` or `None`), bad type (`air_temp_K='300K'`, `torque_Nm=True`), bad product type (`'X'`), out-of-range (`rotational_speed_rpm=5500.0`), non-finite (`nan`, `inf`).

---

## 2. Fault-Risk Output Contract

The hybrid pipeline ([`src/prediction.py`](../src/prediction.py)) produces structured outputs from two complementary stages:

| Output Field | Type / Range | Source Stage | Meaning |
| :--- | :---: | :--- | :--- |
| `fault_probability` | `float` in `[0.0, 1.0]` | ANN (`7 -> 16 -> 8 -> 1`, Sigmoid) | Statistical likelihood $P(\text{Machine failure} = 1 \mid X)$ learned from training data. Binary decision threshold in UI is `0.50`. |
| `severity_index` | `float` in `[0.0, 100.0]` | Mamdani Fuzzy System (CoG) | Continuous operational risk score combining `fault_probability`, `tool_wear_min`, and `torque_Nm` across 11 IF–THEN rules. |
| `priority` | `'LOW' \| 'MEDIUM' \| 'HIGH' \| 'URGENT'` | Crisp Priority Classifier | Actionable tier mapped from `severity_index`: `< 33` (`LOW`), `[33, 60)` (`MEDIUM`), `[60, 78)` (`HIGH`), `>= 78` (`URGENT`). |
| `recommendation` | `str` | Priority Classifier | Concrete maintenance instruction corresponding to the assigned priority tier. |
| `fuzzy_explanation` / `fired_rules` | `str` / `list[dict]` | Fuzzy Engine + Synthesis | Human-readable trace of which rules fired and at what activation strength. |

**How the Outputs Differ:**
- **ANN Failure Probability (`[0, 1]`):** Measures statistical pattern match to historical failures across all 7 normalized features.
- **Fuzzy Severity Score (`0–100`):** Measures operational consequence by weighing the ANN probability against physical tool degradation and mechanical load.
- **Priority (`LOW`–`URGENT`):** Translates the continuous `0–100` severity index into discrete maintenance scheduling actions.

---

## 3. Threshold Baseline ([`src/baseline.py`](../src/baseline.py))

The project defines a deterministic crisp rule-based baseline encoding the 5 AI4I 2020 failure-mode rules published by Matzka (2020):

| Mode | Crisp Rule Definition | Physical Meaning |
| :---: | :--- | :--- |
| **HDF** | `(process_temp_K - air_temp_K) < 8.6` AND `rotational_speed_rpm < 1380` | Insufficient cooling / heat dissipation failure. |
| **PWF** | `power < 3500 W` OR `power > 9000 W`, where `power = torque_Nm * (rpm * 2π / 60)` | Under-loaded or over-loaded spindle power failure. |
| **TWF** | `tool_wear_min >= 220` | Deterministic midpoint approximation of the `[200, 240] min` random wear zone. |
| **OSF** | `(tool_wear_min * torque_Nm) > threshold` (`11,000` L / `12,000` M / `13,000` H min·Nm) | Mechanical overstrain failure. |
| **RNF** | Not modeled (0.1% random chance) | Unpredictable random failure; all RNF cases become false negatives. |

**Purpose & Limitations:**
- **Purpose:** Establishes a rigid, transparent threshold benchmark showing what the ANN must learn from data without being explicitly programmed with these rules.
- **Limitations:** Cannot model stochastic failures (`RNF`), approximates the random `[200, 240] min` `TWF` band with a hard cutoff at `220 min` (missing `< 220` failures and over-flagging `>= 220` non-failures), and reproduces synthetic AI4I 2020 generation logic rather than independent factory thresholds.

---

## 4. Starter Sensor Fixture & Isolated Validator

- **Starter Fixture CSV:** [`data/fixtures/step1_starter_sensors.csv`](../data/fixtures/step1_starter_sensors.csv) — Separate from the training dataset (`data/raw/ai4i2020.csv`). Contains **11 cases**: 5 valid cases (normal, risky, high-wear, lower boundary, fuzzy upper boundary) and 6 intentional invalid-input cases (missing value, non-numeric text, invalid product type, out-of-range speed, `NaN`, `Inf`).
- **Isolated Validator:** [`src/step1_validator.py`](../src/step1_validator.py) — Reuses `BOUNDS` and `VALID_PRODUCT_TYPES` from [`src/prediction.py`](../src/prediction.py) without modifying production code. Validates missing values, data types (`bool`/text rejection), product types (`.strip().upper()` normalization matching production), `math.isfinite()` (`NaN`/`Inf`), and physical range bounds, and reports valid passes separately from expected invalid-input rejections.

---

## 5. Tests, Expected Rejections & Reproduction Commands

- **Unit Test Suite:** [`tests/test_step1_contract.py`](../tests/test_step1_contract.py) (8 test methods covering valid, boundary, missing, wrong-type, invalid product type, out-of-range, `NaN`/`Inf`, and CSV fixture validation).
- **Expected-Failure Semantics:** Invalid-input unit tests use `assertRaises(TypeError | ValueError)` and **PASS** only when the validator accurately rejects invalid data. In the fixture runner, valid cases and expected rejections are tallied in separate sections.
- **Recorded Evidence:** [`docs/evidence/step1/step1-test-output.txt`](evidence/step1/step1-test-output.txt)

### Reproduction Commands

```bash
python -m unittest tests.test_step1_contract -v
python -m src.step1_validator
python -m unittest discover -s tests -v
```

---

## 6. Product Visuals

Genuine dashboard screenshots from the application (committed in Stage 1 and retained in `screenshots/`) illustrate normal, elevated-risk, and critical/urgent operational states:

- **Normal / Low Priority:** [`screenshots/dashboard-normal-low.png`](../screenshots/dashboard-normal-low.png)
- **Elevated / High Risk:** [`screenshots/dashboard-high-risk.png`](../screenshots/dashboard-high-risk.png)
- **Critical / Urgent Priority:** [`screenshots/dashboard-critical-urgent.png`](../screenshots/dashboard-critical-urgent.png)

