# SC08 — Presentation Content Reference (8 Slides)

> **Project:** SC08 — Machine Predictive Maintenance & Fault Priority System  
> **University:** Galgotias University  
> **Branch:** `step-7-final-documentation` (based on `0f5b55a`)  
> **Note:** Verified technical facts, metrics, workflow, and suggested existing screenshots for the 8-slide presentation.

---

## Slide 1 — Project and Team
- **Title:** SC08 — Machine Predictive Maintenance & Fault Priority System
- **Subtitle:** Hybrid NumPy Artificial Neural Network + Mamdani Fuzzy Logic Decision Support
- **Team:** Shivam Mishra, Anuj, Ayush, Siddharth (confirm exact per-member roles before finalizing)
- **Repository:** `https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system`

---

## Slide 2 — Problem and User
- **Problem:** Unplanned milling machine breakdowns cause costly downtime, while overly sensitive alarms waste inspection effort.
- **Target User:** Maintenance operators and reliability engineers needing both failure probability and actionable priority.
- **Decision Supported:** `LOW` (routine monitoring), `MEDIUM` (2–3 days), `HIGH` (within 24h), `URGENT` (stop machine immediately).

---

## Slide 3 — Data and Preparation
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (ID 601, Matzka 2020, CC BY 4.0) — `data/raw/ai4i2020.csv`.
- **Size & Imbalance:** `10,000` records, `14` columns, `0` missing values, `339` failures (`~3.39%`).
- **Splits:** `7,200` train / `800` stratified validation (`27` failures) / `2,000` held-out test (`68` failures).
- **Preprocessing (`src/preprocessing.py`, `data/processed/norm_params.json`):** Drop `UID`/`Product ID`, one-hot encode `Type` (`Type_M`, `Type_H`), Min-Max scale 5 numeric features using training-set min/max only.
- **Important Note:** Synthetic dataset with **no vibration sensor**; `Torque [Nm]` is mechanical spindle torque.

---

## Slide 4 — Method and Workflow
- **Workflow:** `6 Raw Inputs` → `Validation & Min-Max Scaling` → `NumPy ANN (7 → 16 → 8 → 1)` → `P(Failure) ∈ [0, 1]` → `Mamdani Fuzzy System` → `Severity [0–100]` → `Priority`.
- **ANN (`src/ann.py`):** Pure NumPy, ReLU hidden layers, Sigmoid output, `273` trainable parameters, Weighted BCE (`w_pos = 28.5203`), mini-batch SGD (`lr = 0.01`, `batch_size = 64`), early stopping (best epoch `71`).
- **Fuzzy System (`src/fuzzy.py`):** Pure NumPy Mamdani FIS, 3 inputs (`fault_probability`, `tool_wear_min` `[0, 260]`, `torque_Nm` `[0, 80]`), `11` rules, `min`-AND, `max`-aggregation, Centroid (CoG) defuzzification.
- **Suggested Visuals:** `results/figures/ann_learning_curve.svg`

---

## Slide 5 — Baseline and Comparison
- **Baseline (`src/baseline.py`):** Deterministic Matzka (2020) rules (`HDF OR PWF OR TWF OR OSF`, `TWF >= 220` min, `RNF` unmodeled).
- **Fair Comparison (`src/compare_baseline.py`, `data/reports/step4_comparison.csv`):** Evaluated on identical `800` validation and `2,000` test records.
- **Test-Set Comparison (`2,000` records, `68` failures):**
  - **Baseline:** Accuracy `0.9845`, Precision `0.7284`, Recall `0.8676`, F1 `0.7919`, Specificity `0.9886`
  - **ANN:** Accuracy `0.8815`, Precision `0.2116`, Recall `0.9118`, F1 `0.3435`, Specificity `0.8804`, ROC-AUC `0.9581`
- **Suggested Visual:** `results/step7_test_comparison.png` (or `results/step4_baseline_vs_ann.png`)

---

## Slide 6 — Results and Errors
- **Test Confusion Counts (`results/metrics.json`, `docs/STEP-4.md`):**
  - **Baseline:** `TP=59`, `TN=1,910`, `FP=22`, `FN=9`
  - **ANN:** `TP=62`, `TN=1,701`, `FP=231`, `FN=6`
- **Paired Test Agreement (`68` failures):** `58` detected by both, `4` ANN-only, `1` baseline-only, `5` missed by both.
- **Trade-off:** ANN achieves higher recall (`0.9118` vs. `0.8676`) due to weighted BCE, but produces substantially more false positives (`231` vs. `22`). Neither method is universally better; choice depends on $C_{\text{FN}}$ vs. $C_{\text{FP}}$.
- **Hybrid Fuzzy Output (`results/metrics.json`):** `LOW` `68.05%`, `MEDIUM` `18.60%`, `HIGH` `6.25%`, `URGENT` `7.10%`.

---

## Slide 7 — App and Test Evidence
- **Streamlit Product V1 (`app.py`):** Launch with `streamlit run app.py`. Displays separate ANN threshold banner (`0.50`), Severity Index (`0–100`), Priority badge, recommendation, and fired rules.
- **Fresh-Copy Unit Tests (`data/reports/step7_fresh_check.txt`):** `python -m unittest discover -s tests -v` — **18 tests passed** (`OK`). *(Note: `pip` blocked by Windows AppLocker in fresh-copy check.)*
- **Suggested Screenshots:**
  - Normal (`2.16%`, `15.53/100`, `LOW`): `docs/evidence/step5/02-normal-prediction.png` or `docs/evidence/step6/UI-02-normal-prediction.png`
  - Urgent (`99.82%` / `99.99%`, `84.47/100`, `URGENT`): `docs/evidence/step5/03-urgent-prediction.png` or `docs/evidence/step6/UI-03-urgent-prediction.png`
  - Rule Explainability: `docs/evidence/step5/04-explanation-and-rules.png`

---

## Slide 8 — Conclusion and Contributions
- **Conclusion:** Pure NumPy hybrid ANN + Mamdani Fuzzy system provides high failure recall (`91.18%`) with transparent rule-based severity/priority.
- **Limitations:** Synthetic AI4I 2020 data, no vibration sensor, educational prototype (not IEC 61508 / SIL certified), high ANN false-positive count (`231` FP).
- **Team Contributions:** Shivam Mishra (technical implementation, experimental analysis, report preparation per `docs/STEP-4.md`); Anuj, Ayush, Siddharth (confirm exact roles before finalizing).

