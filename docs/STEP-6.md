# Step 6 â€” Test and Validate the Application

> **Academic Milestone:** Step 6 â€” Test and Validate the Application
> **Course / Project Code:** SC08 â€” Soft Computing
> **Project Title:** Machine Predictive Maintenance & Fault Priority System
> **Current Branch:** `step-6-testing`
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)

---

## 1. Project and Environment Information

- **Project Title:** SC08 â€” Machine Predictive Maintenance & Fault Priority System
- **Repository Remote:** `https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system.git`
- **Current Branch:** `step-6-testing`
- **Python Version:** 3.12.9 (64-bit, Windows)
- **Active Dependencies ([`requirements.txt`](../requirements.txt)):**
  - `numpy>=1.24,<2.0`
  - `pandas>=2.0,<3.0`
  - `scikit-learn>=1.3,<2.0`
  - `matplotlib>=3.7,<4.0`
  - `streamlit>=1.35,<2.0`
  - `joblib>=1.3,<2.0`
- **Testing Framework:** Python standard library `unittest` (requires zero external package dependencies).

---

## 2. Test Execution Overview & Scope

The testing framework evaluates the pipeline, defensive validation, exception handling, and asset recovery using Python's built-in `unittest` runner.

### Automated Unit Test Command
```bash
python -m unittest discover -s tests -v
```

### Verified Terminal Execution Output
```
test_t07_missing_ann_model_file (test_model_availability.TestModelAvailability.test_t07_missing_ann_model_file)
T07: Verify FileNotFoundError when ANN weights file is nonexistent. ... ok
test_t07_missing_normalization_file (test_model_availability.TestModelAvailability.test_t07_missing_normalization_file)
T07 Subtest: Verify FileNotFoundError when norm_params.json is nonexistent. ... ok
test_t08_recovery_after_missing_model_test (test_model_availability.TestModelAvailability.test_t08_recovery_after_missing_model_test)
T08: Verify system recovery when pipeline is initialized with real model assets. ... ok
test_t01_normal_prediction (test_prediction_pipeline.TestPredictionPipeline.test_t01_normal_prediction)
T01: Verify prediction under nominal machine operating conditions. ... ok
test_t02_lower_boundary_inputs (test_prediction_pipeline.TestPredictionPipeline.test_t02_lower_boundary_inputs)
T02: Verify system stability at absolute physical lower boundary limits. ... ok
test_t03_upper_boundary_inputs (test_prediction_pipeline.TestPredictionPipeline.test_t03_upper_boundary_inputs)
T03: Verify system behavior at supported upper boundary limits. ... ok
test_t04_out_of_range_inputs (test_prediction_pipeline.TestPredictionPipeline.test_t04_out_of_range_inputs)
T04: Verify defensive ValueError exception when inputs exceed physical bounds. ... ok
test_t05_invalid_text_in_numeric_fields (test_prediction_pipeline.TestPredictionPipeline.test_t05_invalid_text_in_numeric_fields)
T05: Verify defensive TypeError when non-numeric types are supplied. ... ok
test_t06_missing_or_none_inputs (test_prediction_pipeline.TestPredictionPipeline.test_t06_missing_or_none_inputs)
T06: Verify handling when required inputs are None or empty strings. ... ok
test_t09_pipeline_consistency (test_prediction_pipeline.TestPredictionPipeline.test_t09_pipeline_consistency)
T09: Verify repeatability and consistency across identical invocations. ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.038s

OK
```

*Summary:* **10 tests run, 0 failures, 0 errors, status OK.**
*Evidence Log File:* [`docs/evidence/step6/unittest-output.txt`](evidence/step6/unittest-output.txt)

> **Scope Clarification on UI vs. Pipeline Testing:**
> The automated test suite strictly evaluates the Python computational modules (`src/prediction.py`, `src/ann.py`, `src/fuzzy.py`, and model asset loaders). Interactive Streamlit browser rendering (HTML/CSS layout, button clicks, widget state resets) constitutes a separate UI verification layer and is not claimed as passed by the `unittest` execution.

---

## 3. Step 6 Test Matrix

### 3.1 Automated Pipeline Unit Test Matrix (T01 â€“ T10)

| Test ID | Test Name | Objective | Inputs | Expected Result | Actual Result | Evidence Path |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **T01** | **Normal Prediction** | Verify pipeline prediction under nominal operational conditions. | Type: `L`<br>$T_{\text{air}} = 300.0\,\text{K}, T_{\text{proc}} = 310.0\,\text{K}$<br>$\omega = 1500.0\,\text{rpm}, \tau = 40.0\,\text{Nm}, t_{\text{wear}} = 20.0\,\text{min}$ | Valid prediction dictionary returned; $P(\text{Failure}) \in [0, 1]$, Severity $\in [0, 100]$, Priority is `LOW` or `MEDIUM`, Rule R1 activated. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T02** | **Lower-Boundary Inputs** | Verify stability at physical lower limits without zero-division. | Type: `L`<br>$T_{\text{air}} = 250.0\,\text{K}, T_{\text{proc}} = 250.0\,\text{K}$<br>$\omega = 0.0\,\text{rpm}, \tau = 0.0\,\text{Nm}, t_{\text{wear}} = 0.0\,\text{min}$ | Valid prediction dictionary returned; probability and severity are non-NaN numbers within limits. Priority is valid. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T03** | **Upper-Boundary Inputs** | Verify behavior at supported upper boundary limits ($\le 80\,\text{Nm}, \le 260\,\text{min}$). | Type: `M`<br>$T_{\text{air}} = 304.0\,\text{K}, T_{\text{proc}} = 314.0\,\text{K}$<br>$\omega = 1200.0\,\text{rpm}, \tau = 80.0\,\text{Nm}, t_{\text{wear}} = 260.0\,\text{min}$ | Valid prediction dictionary returned; $P(\text{Failure}) \ge 0.50$, Priority is `HIGH` or `URGENT`, Rules R9 and R10 activated. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T04** | **Out-of-Range Inputs** | Verify defensive `ValueError` when inputs exceed physical bounds. | 1. $T_{\text{air}} < 250\,\text{K}$ or $> 350\,\text{K}$<br>2. $\omega < 0\,\text{rpm}$ or $> 5000\,\text{rpm}$<br>3. $\tau < 0\,\text{Nm}$ or $> 150\,\text{Nm}$<br>4. $t_{\text{wear}} < 0\,\text{min}$ or $> 300\,\text{min}$<br>5. Product type = `"X"` | Pipeline raises explicit `ValueError` identifying invalid field and physical boundary bounds. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T05** | **Invalid Text in Numeric Fields** | Verify defensive `TypeError` when non-numeric types are supplied. | $T_{\text{air}} = \text{"300K"}$, $\tau = \text{"40Nm"}$, $\tau = \text{True}$, $t_{\text{wear}} = \text{"worn"}$ | Pipeline raises explicit `TypeError` stating input must be numeric (`int` or `float`). | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T06** | **Missing or None Inputs** | Verify defensive handling for `None`, empty string, and `NaN`/`Inf`. | 1. `product_type = None` or `""`<br>2. `air_temp_K = None` or `""`<br>3. `air_temp_K = float("nan")` or `float("inf")` | Pipeline raises `TypeError` for `None`/empty numeric, and `ValueError` for empty type string or `nan`/`inf`. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T07** | **Missing ANN Model File** | Verify error handling when `results/ann_weights.npz` is unavailable. | Pass isolated nonexistent path: `weights_path="results/nonexistent_test.npz"` | Pipeline raises explicit `FileNotFoundError` with informative message. App displays error banner and halts. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T07 (Subtest)** | **Missing Normalization File** | Verify error handling when `norm_params.json` is unavailable. | Pass isolated nonexistent path: `norm_params_path="data/processed/nonexistent_test.json"` | Pipeline raises explicit `FileNotFoundError` with informative message. App displays error banner and halts. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T08** | **Restore Model and Recovery** | Verify system recovery when pipeline is re-instantiated with valid assets. | Standard initialization with real weights and normalization assets. | Pipeline initializes cleanly; prediction executes normally with valid return dictionary. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T09** | **Pipeline Consistency Test** | Verify repeatability between direct pipeline calls and application invocation. | Multiple identical invocations across standard test presets. | Unrounded probabilities match within floating-point tolerance ($10^{-6}$), and categorical outputs match exactly. | **PASS** | `docs/evidence/step6/unittest-output.txt` |
| **T10** | **Automated Test Discovery & Execution** | Automated batch test discovery across the entire test suite. | Command: `python -m unittest discover -s tests -v` | All 10 unit test methods across the test suite execute and report `OK` with 0 failures and 0 errors. | **PASS** | `docs/evidence/step6/unittest-output.txt` |

---

### Structure of Test Execution: T07 Subtests & T10 Runner

1. **Dual Verification in T07 (Missing Asset Integrity):**
   Test T07 contains two distinct, isolated missing-file checks within [`tests/test_model_availability.py`](../tests/test_model_availability.py):
   - `test_t07_missing_ann_model_file`: Confirms `FileNotFoundError` when `ann_weights.npz` is unavailable.
   - `test_t07_missing_normalization_file`: Confirms `FileNotFoundError` when `norm_params.json` is unavailable.
   Both tests use non-existent dummy paths, ensuring that production weights and normalization assets are never deleted, moved, or renamed.

2. **Nature of T10:**
   T10 does not represent an isolated twelfth test case; rather, it represents the automated test suite execution command (`python -m unittest discover -s tests -v`). It discovers and executes all 10 unit test methods (T01 through T09, including the T07 subtest) and validates that the complete batch passes with zero failures.

---

### 3.2 Manual Streamlit UI Verification Matrix

Interactive browser and UI rendering tests evaluate frontend presentation, form responsiveness, and user interaction components in Streamlit. Manual browser verification was performed against the official root entry point (`app.py`), and genuine application screenshot evidence has been archived in [`docs/evidence/step6/`](evidence/step6/).

| Test ID | Test Case | Actual Result | Test Status | Evidence Reference & Status |
| :---: | :--- | :--- | :---: | :--- |
| **UI-01** | **Input form visibility** | All six input fields and Run Prediction Pipeline button were visible | **PASS** | [`docs/evidence/step6/UI-01-input-form.png`](evidence/step6/UI-01-input-form.png)<br>*(Available â€” verified application capture)* |
| **UI-02** | **Normal prediction** | Probability 2.16%, raw probability 0.0216, fuzzy severity 15.53/100, priority LOW, MACHINE HEALTHY status | **PASS** | [`docs/evidence/step6/UI-02-normal-prediction.png`](evidence/step6/UI-02-normal-prediction.png)<br>*(Available â€” verified application capture)* |
| **UI-03** | **Urgent prediction** | Probability 99.99%, raw probability 0.9999, fuzzy severity 84.47/100, priority URGENT | **PASS** | [`docs/evidence/step6/UI-03-urgent-prediction.png`](evidence/step6/UI-03-urgent-prediction.png)<br>*(Available â€” verified application capture)* |
| **UI-04** | **Rule explainability** | Urgent maintenance recommendation was displayed correctly. The live Streamlit rule explanation container confirmed that fuzzy rules R9 and R10 fired. | **PASS** | [`docs/evidence/step6/UI-04-rule-explainability.png`](evidence/step6/UI-04-rule-explainability.png)<br>*(Available â€” verified recommendation capture)* |
| **UI-05** | **Dark mode** | Dark mode was manually verified; input labels, prediction results, priority status, rule explanation and maintenance recommendation were readable | **PASS** | [`docs/evidence/step6/UI-05-dark-mode.png`](evidence/step6/UI-05-dark-mode.png)<br>*(Available â€” verified application capture)* |

#### UI-04 Verified Maintenance Recommendation & Confirmed Fired Rules

- **Observed Maintenance Recommendation (Exact Output):**
  > "STOP MACHINE IMMEDIATELY. Conduct full mechanical inspection and replace worn components before resuming operation. Do not continue under current load conditions."

- **Confirmed Fired Fuzzy Rules:**
  The live Streamlit rule explanation container confirmed that fuzzy rules **R9** and **R10** fired:
  - **Rule R9:** $\text{FP} = \text{HIGH}$ and $\text{TW} = \text{HIGH}$. This rule represents high ANN failure probability combined with high tool wear, confirming that physical degradation corroborates the machine learning anomaly flag.
  - **Rule R10:** $\text{FP} = \text{HIGH}$ and $\text{TQ} = \text{HIGH}$. This rule represents high ANN failure probability combined with high torque, confirming that heavy mechanical spindle loading significantly amplifies failure risk.

- **Linked Evidence:**
  - Actionable recommendation banner detail: [`docs/evidence/step6/UI-04-rule-explainability.png`](evidence/step6/UI-04-rule-explainability.png)
  - Full urgent dashboard and fired rule activations (light mode): [`docs/evidence/step6/UI-03-urgent-prediction.png`](evidence/step6/UI-03-urgent-prediction.png)
  - Full urgent dashboard and fired rule activations (dark mode): [`docs/evidence/step6/UI-05-dark-mode.png`](evidence/step6/UI-05-dark-mode.png)

#### Evidence Screenshot Inventory

All five genuine screenshot files have been saved to [`docs/evidence/step6/`](evidence/step6/):
- [`docs/evidence/step6/UI-01-input-form.png`](evidence/step6/UI-01-input-form.png) â€” **Available** (Machine operating parameters sidebar with all six input controls and Run Prediction Pipeline button).
- [`docs/evidence/step6/UI-02-normal-prediction.png`](evidence/step6/UI-02-normal-prediction.png) â€” **Available** (Normal prediction under nominal operational conditions, green healthy banner, 2.16% probability, LOW priority).
- [`docs/evidence/step6/UI-03-urgent-prediction.png`](evidence/step6/UI-03-urgent-prediction.png) â€” **Available** (Critical failure regime, red failure banner, 99.99% probability, URGENT priority, 84.47 severity).
- [`docs/evidence/step6/UI-04-rule-explainability.png`](evidence/step6/UI-04-rule-explainability.png) â€” **Available** (Verified high-resolution crop of the Actionable Maintenance Recommendation banner).
- [`docs/evidence/step6/UI-05-dark-mode.png`](evidence/step6/UI-05-dark-mode.png) â€” **Available** (Complete dashboard rendered in Streamlit dark mode verifying readability and color contrast).

---

### 3.3 Manual Streamlit Model Availability & Recovery Verification (T07 & T08)

To complement the automated unit tests, manual browser verification of missing model error handling and system recovery was conducted on the Streamlit web interface using an isolated temporary project copy:

| Test ID | Test Scenario | Execution Environment | Actual Result | Status | Evidence Type |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **T07** | **Missing Model File UI Handling** | Isolated temporary project copy (Port 8502) | Missing ANN weights caused a clear Streamlit error message (`Failed to load predictive maintenance pipeline: ANN weights file not found at 'results\ann_weights.npz' ...`), and the sidebar and prediction form were completely hidden. | **PASS** | Manually verified *(No screenshot invented)* |
| **T08** | **Model Restoration & UI Recovery** | Isolated temporary project copy (Port 8502) | After restoring the original `results/` and `data/` folders in the isolated copy and restarting Streamlit, the interface loaded successfully. Running a normal prediction returned 2.16% failure probability, 15.53 fuzzy severity, and LOW priority (`MACHINE HEALTHY`). | **PASS** | Manually verified |

- **Port Isolation & Asset Protection:** Testing was performed strictly on isolated port **8502** using a temporary project copy. Original project model files (`results/ann_weights.npz` and `data/processed/norm_params.json`) in the primary repository were not modified, renamed, or deleted.

---

## 4. Known Limitations & Technical Findings

1. **Boundary Divergence Between Pipeline and Fuzzy Universe:**
   - In [`src/prediction.py`](../src/prediction.py), `BOUNDS` permits `torque_Nm` up to $150.0\,\text{Nm}$ and `tool_wear_min` up to $300.0\,\text{min}$.
   - In [`src/fuzzy.py`](../src/fuzzy.py), membership function support ends at $80.0\,\text{Nm}$ and $260.0\,\text{min}$. Inputs beyond these limits produce $0.0$ membership across all rules, resulting in an unactivated distribution where the safe fallback severity ($50.0 \to \text{MEDIUM}$) is returned.
   - The Streamlit application ([`app.py`](../app.py)) enforces UI caps at $80.0\,\text{Nm}$ and $260.0\,\text{min}$ with Python validation guards. Callers invoking `PredictiveMaintenancePipeline.predict()` directly must be aware that values in the gap ($80\text{--}150\,\text{Nm}$, $260\text{--}300\,\text{min}$) trigger the fallback severity rather than an exception.
2. **Streamlit Number Input Clearing Behavior:**
   - In the web dashboard, deleting all text in a number input does not pass `None` or empty string to the backend; Streamlit automatically restores the initialized `value` parameter upon losing focus. Direct pipeline tests (T06) independently verify that if `None` or `""` reaches the backend, `TypeError`/`ValueError` is properly raised.
3. **Absence of Vibration Sensor:**
   - The AI4I 2020 dataset contains no accelerometer or vibration transducer. Mechanical resistance is monitored via `Torque [Nm]`.

---

## 5. Fixes and Retests

During the initial execution of the automated unit test suite, one boundary test specification was refined:

- **Identified Discrepancy in T03:** The initial draft of `test_t03_upper_boundary_inputs` attempted to assert `URGENT` priority on inputs of $150.0\,\text{Nm}$ and $300.0\,\text{min}$. Because these values exceed the fuzzy membership universe ceilings ($80\,\text{Nm}$, $260\,\text{min}$), no rules fired, triggering the safe fallback severity ($50.0 \to \text{MEDIUM}$).
- **Corrective Action:** The test assertion was corrected to evaluate the true supported upper boundary limits ($80.0\,\text{Nm}$ and $260.0\,\text{min}$), which activate Rules R9 and R10 to yield $84.47$ (`URGENT`). An explicit subtest was added to document and verify the out-of-universe fallback behavior ($50.0 \to \text{MEDIUM}$).
- **Retest Result:** All 10 unit tests pass cleanly with zero failures.

---

## 6. Team Responsibilities â€” Step 6

| Team Member | Responsibility |
|---|---|
| Shivam Mishra | Automated testing, prediction pipeline validation and integration of test results |
| Anuj | Manual UI testing and verification of normal and urgent prediction scenarios |
| Ayush Kumar | Evidence screenshot organization and verification |
| Siddharth Anand | Test report preparation, documentation review and test case validation |

---

## 7. Step 6 Milestone Status & Remaining Incomplete Requirements

The overall Step 6 milestone status remains **IN PROGRESS** until team contribution attribution is resolved. Component statuses are as follows:

- [x] **Automated Testing:** COMPLETE (10/10 tests passing in 0.038s; verified in [`docs/evidence/step6/unittest-output.txt`](evidence/step6/unittest-output.txt)).
- [x] **Manual UI Testing:** COMPLETE (All five manual UI test cases UI-01 through UI-05 verified as PASS).
- [x] **Screenshot Evidence Archival:** COMPLETE (All 5 screenshots saved in [`docs/evidence/step6/`](evidence/step6/): `UI-01-input-form.png`, `UI-02-normal-prediction.png`, `UI-03-urgent-prediction.png`, `UI-04-rule-explainability.png`, and `UI-05-dark-mode.png`).
- [x] **Rule Activation Verification:** COMPLETE (Fuzzy rules R9 and R10 confirmed in live UI rule container).
- [ ] **Team Contribution Attribution:** PENDING (Section 6 awaiting individual member names and verified contribution breakdowns upon team sign-off).
