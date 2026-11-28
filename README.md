# SC08 — Machine Predictive Maintenance & Fault Priority System

> **Course:** Soft Computing (SC08) — B.Tech Project  
> **University:** Galgotias University  
> **Repository:** [https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system](https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system)  
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI ID: 601)

---

## Overview

**SC08** is a hybrid Soft Computing decision-support system for industrial milling machine predictive maintenance. It combines:

1. A **hand-coded NumPy Artificial Neural Network (ANN)** (`7 -> 16 -> 8 -> 1`) trained with weighted binary cross-entropy to estimate machine failure probability $P(\text{Failure} \mid X) \in [0, 1]$.
2. A **hand-coded NumPy Mamdani Fuzzy Inference System** (3 inputs, 11 IF–THEN rules, Centroid defuzzification) that combines ANN probability with physical tool wear and spindle torque to compute a continuous **Severity Index (`0–100`)** and discrete **Maintenance Priority** (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).
3. A **Streamlit web application** (`app.py`) providing live predictions, rule-level explainability, and actionable maintenance recommendations.
4. A **deterministic fixed-alarm baseline** (`src/baseline.py`) encoding Matzka (2020) failure-mode rules for fair comparison.

---

## Problem & Features

- **Problem:** Unplanned machine failures cause costly downtime, while overly sensitive alarms cause unnecessary inspection overhead.
- **Features:**
  - Pure NumPy ANN and Mamdani Fuzzy Logic (no TensorFlow, PyTorch, Keras, or `scikit-fuzzy`).
  - Separate ANN binary threshold banner ($\tau = 0.50$) and fuzzy severity/priority assessment.
  - Transparent rule-firing breakdown (R1–R11) with activation strengths.
  - Defensive input validation and missing-model error handling.

---

## Technology Stack

- **Python:** Verified on `Python 3.14.7` (`requirements.txt` specifies `Python >= 3.9`)
- **Core Libraries (`requirements.txt`):**
  - `numpy>=1.24,<2.0`
  - `pandas>=2.0,<3.0`
  - `scikit-learn>=1.3,<2.0` (used only for `train_test_split` and evaluation metrics)
  - `matplotlib>=3.7,<4.0`
  - `streamlit>=1.35,<2.0`
  - `joblib>=1.3,<2.0`

---

## Architecture & Workflow

```
6 Raw Inputs (Type, Air Temp, Process Temp, Speed, Torque, Tool Wear)
   │
   ├──► Min-Max Scaling (norm_params.json) + One-Hot (Type_M, Type_H)
   │       └──► NumPy ANN (7 → 16 ReLU → 8 ReLU → 1 Sigmoid) ──► P(Failure) ∈ [0, 1]
   │
   └──► Mamdani Fuzzy System (P(Failure), Tool Wear [0,260], Torque [0,80])
           └──► 11 Rules (min-AND, max-agg, CoG) ──► Severity [0–100] ──► Priority (LOW/MEDIUM/HIGH/URGENT)
```

---

## Dataset Source

- **Name:** AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)
- **Source:** UCI Machine Learning Repository (Dataset ID: 601), CC BY 4.0
- **Size:** 10,000 records, 14 columns, 339 failures (~3.39%), 0 missing values
- **Splits:** 7,200 train / 800 validation (27 failures) / 2,000 test (68 failures)
- **Important Note:** The dataset contains **no vibration sensor**; `Torque [Nm]` represents mechanical spindle torque.

---

## Repository Structure

```text
├── app.py                     # Canonical Streamlit web application entry point
├── app/app.py                 # Synchronized subfolder alias
├── requirements.txt           # Pinned Python dependencies
├── src/
│   ├── preprocessing.py       # Split, one-hot encoding, Min-Max scaling
│   ├── ann.py                 # Pure NumPy 7->16->8->1 ANN & training
│   ├── fuzzy.py               # Pure NumPy Mamdani Fuzzy Inference System
│   ├── prediction.py          # Unified PredictiveMaintenancePipeline
│   ├── baseline.py            # Deterministic Matzka (2020) alarm baseline
│   ├── compare_baseline.py    # Paired validation & test comparison
│   └── step1_validator.py     # Step 1 input contract validator
├── data/
│   ├── raw/ai4i2020.csv       # Raw UCI AI4I 2020 dataset
│   ├── processed/             # X_train.npy, X_test.npy, y_train.npy, y_test.npy, norm_params.json
│   ├── fixtures/              # step1_starter_sensors.csv
│   └── reports/               # Step 3 & Step 4 CSV/MD reports
├── results/                   # ann_weights.npz, metrics.json, figures, plots
├── tests/                     # Unit tests (unittest)
├── docs/                      # STEP-1.md, STEP-4.md, STEP-5.md, STEP-6.md, STEP-7.md, evidence/
└── report/                    # final_report.md
```

---

## Setup & Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Example Inputs & Expected Outputs

| Scenario | Inputs (`Type`, `T_air`, `T_proc`, `RPM`, `Torque`, `Wear`) | Expected ANN Prob | Expected Severity | Expected Priority |
| :--- | :--- | :---: | :---: | :---: |
| **Normal** | `L`, `300.0 K`, `310.0 K`, `1500.0 rpm`, `40.0 Nm`, `20.0 min` | `0.0216` (`2.16%`) | `15.53 / 100` | `LOW` |
| **Urgent (UI Max)** | `M`, `304.0 K`, `314.0 K`, `1200.0 rpm`, `80.0 Nm`, `260.0 min` | `0.9999` (`99.99%`) | `84.47 / 100` | `URGENT` |

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

---

## Reports & Evidence Paths

- **Final Report:** `report/final_report.md`
- **Step 7 Feedback Record:** `docs/STEP-7.md`
- **Milestone Docs:** `docs/STEP-1.md`, `docs/STEP-4.md`, `docs/STEP-5.md`, `docs/STEP-6.md`
- **Saved Metrics & Comparisons:** `results/metrics.json`, `data/reports/step4_comparison.csv`, `data/reports/step4_test_paired_predictions.csv`
- **UI Screenshots:** `docs/evidence/step5/`, `docs/evidence/step6/`

---

## Limitations & Scope

- **Educational Prototype:** Developed for B.Tech Soft Computing coursework (SC08). Not certified under IEC 61508 / SIL standards and must not control real industrial machinery.
- **Synthetic Data:** Trained and evaluated on synthetic AI4I 2020 data; real-world factory performance is not validated.

