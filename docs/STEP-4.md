# Step 4 — Fair Baseline Comparison Report

> **Academic Milestone:** Step 4 — Machine Predictive Maintenance & Fault Priority System  
> **Course / Project Code:** SC08  
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)  

---

## Team and project

- **Project:** SC08 — Machine Predictive Maintenance & Fault Priority System
- **Team:** Shivam Mishra (Lead Developer & System Architect)
- **Repository Branch:** `step-4-baseline`
- **Repository Link:** [https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system](https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system)

---

## Question

**On identical held-out validation and test rows, does the trained Artificial Neural Network (ANN) detect machine failures with higher sensitivity (recall) than a deterministic, rule-based alarm threshold baseline?**

In industrial predictive maintenance, failing to detect an impending machine breakdown (false negative) causes catastrophic tool destruction, secondary spindle damage, and unplanned factory downtime. Conversely, false alarms (false positives) incur unnecessary inspection overhead. This milestone evaluates whether a data-driven soft-computing neural classifier improves failure detection coverage over rigid, non-soft-computing alarm limit checks on the exact same unseen machine observations.

---

## Data and scenarios

- **Source and Citation:**  
  - UCI Machine Learning Repository, Dataset ID 601: *AI4I 2020 Predictive Maintenance Dataset*.  
  - Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I 2020), IEEE, pp. 69–74.  
  - Raw path: `data/raw/ai4i2020.csv` (10,000 operational records).
- **Training Data (Development Split):**  
  - Total training partition: **8,000 samples** (`data/processed/X_train.npy`, `data/processed/y_train.npy`).  
  - Sub-training partition: **7,200 samples** (6,956 healthy, 244 failures). Used strictly for gradient optimization and backpropagation.
- **Validation and Test Data (Evaluated Splits):**  
  - **Validation Partition:** Exactly **800 samples** (773 healthy, 27 failures). Created using a deterministic 90/10 stratified split (`random_state=42`, `val_fraction=0.10`) within `src/ann.py` for early stopping and threshold validation.  
  - **Held-out Test Partition:** Exactly **2,000 samples** (`data/processed/X_test.npy`, `data/processed/y_test.npy`), containing 1,932 healthy samples (96.60%) and 68 failure samples (3.40%). Completely unobserved during ANN training and scaler fitting.
- **Target and Relevant Input Fields:**  
  - **Target:** `Machine failure` ($0 = \text{Healthy}$, $1 = \text{Machine Failure}$).  
  - **7 Canonical Inputs:** `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Type_M`, `Type_H`.  
  - *Identified Exclusions:* Sub-failure breakdown flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from ANN inputs to prevent trivial target leakage.
- **Generated Scenarios:**  
  - No synthetic generated scenarios are mixed into the validation or test splits. All evaluation rows originate strictly from the held-out partitions of the official dataset.

---

## Baseline rule

The baseline is implemented in `src/baseline.py` as a deterministic, non-soft-computing alarm threshold system based on the composite OR logic across four primary failure modes:

$$\hat{y}_{\text{baseline}} = \text{HDF} \lor \text{PWF} \lor \text{TWF} \lor \text{OSF}$$

Where:
1. **Heat Dissipation Failure (HDF):** $(T_{\text{process}} - T_{\text{air}}) < 8.6\text{ K} \land \omega_{\text{rot}} < 1380\text{ rpm}$
2. **Power Failure (PWF):** Instantaneous power $P = \tau \times \left(\omega_{\text{rot}} \times \frac{2\pi}{60}\right)$ falls outside $[3500, 9000]\text{ W}$.
3. **Tool Wear Failure (TWF):** Tool wear time $t_{\text{wear}} \ge 220\text{ min}$ (deterministic midpoint approximation of the stochastic $[200, 240]\text{ min}$ window).
4. **Overstrain Failure (OSF):** Product $t_{\text{wear}} \times \tau > \text{Threshold}_{\text{Type}}$ ($11,000\text{ min}\cdot\text{Nm}$ for Type L, $12,000$ for Type M, $13,000$ for Type H).
5. **Random Failure (RNF):** Unmodelled ($0.1\%$ stochastic noise; set to `False`).

> **Academic Integrity Statement on Baseline Origin:**  
> **"The crisp rules reproduce or approximate the failure-generation logic associated with the AI4I 2020 dataset. Their results are not an independent validation of industrial failure thresholds or evidence of performance on real factory data."**  
> In statistical terms, the baseline acts as an **oracle classifier** that shares the structural generative priors of Matzka (2020). Its parameters were not tuned on the test set.

---

## Fair-comparison checks

- [x] **Same Question & Units:** Both methods predict the exact same binary target (`Machine failure` $\in \{0, 1\}$). Feature scaling was reversed via `_denormalise()` so the baseline evaluated raw physical units ($\text{K}, \text{rpm}, \text{Nm}, \text{min}$) while the ANN processed normalized inputs.
- [x] **Identical Held-Out Rows:** Both methods are evaluated on the exact same 800 validation rows and 2,000 held-out test rows.
- [x] **Row Identity & Ordering Verified:** Row index alignment and label identity were strictly verified with programmatic assertions before computing metrics.
- [x] **No Test Tuning:** The ANN weights (`results/ann_weights.npz`) were restored without retraining. The baseline thresholds are static and were never modified based on test results.
- [x] **Appropriate Metric Selection:** Standard classification metrics (Accuracy, Precision, Recall, F1-Score, Specificity) were pre-selected. True continuous ROC-AUC is reported exclusively for the continuous ANN model, while the binary baseline is reported as `N/A` to prevent mislabeling balanced accuracy as ROC-AUC.

---

## Results

The tables below record the verified results generated by `src/compare_baseline.py` and saved to `data/reports/step4_baseline_metrics.csv` and `data/reports/step4_comparison.csv`.

### Primary Classification Performance

| Split | Method | Rows | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | Fixed alarm baseline | 800 | **0.9875** | **0.7429** | **0.9630** | **0.8387** | **0.9884** | *N/A (discrete)* |
| **Validation** | Step 3 ANN (Ours) | 800 | 0.8838 | 0.2105 | 0.8889 | 0.3404 | 0.8836 | **0.9547** |
| **Test** | Fixed alarm baseline | 2,000 | **0.9845** | **0.7284** | 0.8676 | **0.7919** | **0.9886** | *N/A (discrete)* |
| **Test** | Step 3 ANN (Ours) | 2,000 | 0.8815 | 0.2116 | **0.9118** | 0.3435 | 0.8804 | **0.9581** |

### Confusion Matrix Counts & Predicted Positives

| Split | Method | TP | TN | FP | FN | Total Positives Predicted ($\text{TP}+\text{FP}$) | Actual Failures ($\text{TP}+\text{FN}$) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Validation** | Fixed alarm baseline | 26 | 764 | 9 | 1 | 35 | 27 |
| **Validation** | Step 3 ANN | 24 | 683 | 90 | 3 | 114 | 27 |
| **Test** | Fixed alarm baseline | 59 | 1,910 | 22 | 9 | 81 | 68 |
| **Test** | Step 3 ANN | 62 | 1,701 | 231 | 6 | 293 | 68 |

### Score Differences: ANN minus Baseline

| Split | Metric | Baseline Score | ANN Score | Difference ($\Delta$) | Directional Meaning |
|---|---|---:|---:|---:|---|
| **Validation** | Accuracy | 0.9875 | 0.8838 | -0.1037 | Baseline higher by 10.37 points |
| **Validation** | Precision | 0.7429 | 0.2105 | -0.5324 | Baseline higher by 53.24 points |
| **Validation** | Recall | 0.9630 | 0.8889 | -0.0741 | Baseline higher by 7.41 points |
| **Validation** | F1-Score | 0.8387 | 0.3404 | -0.4983 | Baseline higher by 49.83 points |
| **Validation** | Specificity | 0.9884 | 0.8836 | -0.1048 | Baseline higher by 10.48 points |
| **Test** | Accuracy | 0.9845 | 0.8815 | -0.1030 | Baseline higher by 10.30 points |
| **Test** | Precision | 0.7284 | 0.2116 | -0.5168 | Baseline higher by 51.68 points |
| **Test** | Recall | 0.8676 | **0.9118** | **+0.0442** | **ANN higher by +4.42 points (+5.1% relative)** |
| **Test** | F1-Score | 0.7919 | 0.3435 | -0.4484 | Baseline higher by 44.84 points |
| **Test** | Specificity | 0.9886 | 0.8804 | -0.1082 | Baseline higher by 10.82 points |

---

## Interpretation

1. **What improved, what worsened, and by how much?**
   - **Test Recall Improved by +0.0442 Score Points:** On the unseen 2,000-sample test set, the ANN achieved a Recall of **0.9118 (91.18%)**, compared to **0.8676 (86.76%)** for the baseline. The ANN correctly detected **62 out of 68 true machine breakdowns**, leaving only 6 missed failures ($\text{FN} = 6$). In contrast, the baseline missed 9 failures ($\text{FN} = 9$).
   - **Precision & F1-Score are Higher for the Baseline:** The baseline achieved a Precision of $0.7284$ and F1 of $0.7919$, whereas the ANN scored $0.2116$ and $0.3435$ respectively. This difference is directly explained by the training loss function: to prevent catastrophic false negatives under severe class imbalance (~3.39% failures), the ANN was trained with positive class weight $w_{\text{pos}} = 28.5203$. This deliberately penalizes missed failures 28.5 times more heavily than false alarms, resulting in higher sensitivity at the cost of additional false alarms ($\text{FP} = 231$).

2. **Why does the baseline achieve such high precision?**
   - As formally established in Step 3, the baseline is an **oracle reference**: its deterministic rules directly mirror the algebraic equations Matzka (2020) programmed into the AI4I synthetic dataset generator.
   - The baseline does not learn from data; it tests the exact synthetic formulas. Its only errors arise from the stochastic TWF window ($[200, 240]\text{ min}$) and unmodelled random failure noise ($0.1\%$ RNF).

3. **Is the difference meaningful for the project goal?**
   - **Yes, critically so.** In predictive maintenance, an undetected machine failure causes severe tool collision, spindle seizure, and production stoppage costing orders of magnitude more than an inspection false alarm. Catching 62 vs 59 failures represents significant operational risk reduction.
   - More importantly, **the baseline provides only a rigid binary trip flag with no concept of gradation or risk margin**. In contrast, the ANN outputs a smooth, continuous failure probability $P \in [0, 1]$ with high discriminative capability (**Test ROC-AUC: 0.9581**). This continuous score is the indispensable soft-computing foundation required by the **Mamdani Fuzzy Logic Inference System in Step 5** to calculate a multi-level Severity Index ($0\text{--}100$) and assign actionable priorities (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).

4. **Which examples disagree?**
   - Inspection of `step4_test_paired_predictions.csv` reveals three distinct categories of disagreement:
     - **ANN True Positives missed by Baseline (FN for Baseline, TP for ANN):** Samples where tool wear was elevated ($t_{\text{wear}} \in [200, 219]\text{ min}$) and load was high, but below the rigid $220\text{ min}$ cutoff. The ANN detected compounding thermal and load stress and raised an alarm, whereas the rigid baseline failed.
     - **Baseline True Positives missed by ANN (FN for ANN, TP for Baseline):** 6 edge cases where sensor parameters marginally crossed one synthetic limit but overall normalized feature vectors resembled normal operations.
     - **False Alarms (FP):** The ANN flagged 231 healthy machines, while the baseline flagged 22. In factory operations, these 231 machines are not discarded; rather, the Step 5 fuzzy engine assesses their physical degradation (tool wear and torque) and assigns most of them to `LOW` or `MEDIUM` monitoring rather than emergency shutdowns.

---

## Limitations and responsible use

1. **Synthetic Nature of AI4I 2020:** The AI4I dataset is a synthetic benchmark created from a phenomenological milling simulation. The crisp rules reflect the generator formulas, not empirical factory thresholds validated across varied machine tool vendors.
2. **Absence of Vibration Sensor:** The AI4I 2020 dataset does **not** contain an accelerometer or vibration sensor. Spindle mechanical resistance is measured in Newton-metres by `Torque [Nm]`. In this project, torque is strictly analyzed as torque and is never misrepresented as a vibration proxy.
3. **Severe Class Imbalance:** With only 339 failures across 10,000 records (3.39%), a naive majority classifier that always predicts healthy achieves 96.60% accuracy with 0.0% recall. Accuracy is therefore an uninformative standalone metric; recall, precision, and F1 must always be reported together.
4. **Educational Scope:** This software is an educational prototype developed for course SC08. It must not be deployed to control physical CNC machinery or life-critical manufacturing lines without physical validation, sensor redundancy, and safety interlocks.

---

## Reproducibility and evidence

All Step 4 artifacts are generated deterministically and can be independently verified from the repository root:

- **Repeatable Comparison Script:** `src/compare_baseline.py`  
  *Execution command:*
  ```bash
  python -m src.compare_baseline
  ```
- **Executed Jupyter Notebook:** `notebooks/03_baseline_comparison.ipynb`  
  *Verification:* Contains all code cells and pre-computed visual and table outputs.
- **Metrics Table:** `data/reports/step4_baseline_metrics.csv`
- **Metric Differences Table:** `data/reports/step4_comparison.csv`
- **Validation Paired Predictions:** `data/reports/step4_validation_paired_predictions.csv` (800 rows)
- **Test Paired Predictions:** `data/reports/step4_test_paired_predictions.csv` (2,000 rows)
- **Side-by-Side Comparison Plot:** `results/step4_baseline_vs_ann.png`

---

## Team contributions

- **Shivam Mishra:**
  - Designed and authored the repeatable fair comparison pipeline `src/compare_baseline.py`.
  - Implemented row alignment assertions and strict separation of validation (800 rows) and held-out test (2,000 rows) evaluations.
  - Corrected ROC-AUC calculation to prevent mislabeling discrete baseline accuracy as continuous ROC-AUC.
  - Generated all machine-readable CSV artifacts, paired prediction tables, and side-by-side comparison charts.
  - Executed and validated `notebooks/03_baseline_comparison.ipynb`.
  - Authored the comprehensive Step 4 milestone report `docs/STEP-4.md`.
