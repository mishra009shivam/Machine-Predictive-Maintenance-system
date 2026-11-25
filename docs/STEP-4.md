# Step 4 — Fair Baseline Comparison Report

> **Academic Milestone:** Step 4 — Machine Predictive Maintenance & Fault Priority System  
> **Course / Project Code:** SC08  
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)  

---

## Team and project

- **Project:** SC08 — Machine Predictive Maintenance & Fault Priority System
- **Team:** Shivam Mishra, Anuj, Ayush, Siddharth
- **Repository Branch:** `step-4-baseline-revision` (revised from `step-4-baseline`)
- **Repository Link:** [https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system](https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system)

---

## Question

**On identical held-out validation and test rows, does the trained Artificial Neural Network (ANN) detect machine failures with higher sensitivity (recall) than a deterministic, rule-based alarm threshold baseline?**

In industrial predictive maintenance, failing to detect an impending machine breakdown (false negative) carries equipment-damage and unplanned-downtime risk, whereas false alarms (false positives) incur inspection overhead. This milestone evaluates how a data-driven soft-computing neural classifier compares against rigid, non-soft-computing alarm limit checks on the exact same unseen machine observations, including both sensitivity gains and false-alarm trade-offs.

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

### Model Agreement & Disagreement Breakdown (Paired Row Analysis)

Verified from `data/reports/step4_validation_paired_predictions.csv` and `data/reports/step4_test_paired_predictions.csv`, the table below separates **total false negatives** ($\text{FN}$) from **single-model detections** (cases caught by only one method):

| Split | Records | Actual Failures | Both Detect | ANN-Only Detected | Baseline-Only Detected | Both Miss | ANN Total TP / FN | Baseline Total TP / FN | ANN FP (Shared / ANN-Only) | Baseline FP (Shared / Base-Only) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | 800 | 27 | 24 | **0** | **2** | 1 | 24 / 3 | 26 / 1 | 90 (7 / 83) | 9 (7 / 2) |
| **Test** | 2,000 | 68 | 58 | **4** | **1** | 5 | 62 / 6 | 59 / 9 | 231 (14 / 217) | 22 (14 / 8) |

> **Key Distinction:** On the 2,000-row test set, the ANN has `FN = 6` total missed failures (`5` missed by both models + `1` missed only by ANN), **not** 6 baseline-only detections. Similarly, the ANN's 62 TPs vs. the baseline's 59 TPs reflect `58` shared detections, `4` ANN-only detections, and `1` baseline-only detection (a net difference of `+3` failures caught).

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
   - **Test Recall Improved by +0.0442 Score Points:** On the unseen 2,000-sample test set, the ANN achieved a Recall of **0.9118 (91.18%)**, compared to **0.8676 (86.76%)** for the baseline. The ANN correctly detected **62 out of 68 true machine breakdowns** (`FN = 6`), whereas the baseline detected **59 out of 68** (`FN = 9`), yielding a net gain of 3 detected failures (`4` ANN-only detections minus `1` baseline-only detection; `58` shared detections; `5` missed by both). On the 800-row validation split, the baseline detected `26/27` (`FN = 1`) vs. `24/27` (`FN = 3`) for the ANN (`24` shared, `0` ANN-only, `2` baseline-only, `1` missed by both).
   - **Precision & F1-Score are Higher for the Baseline:** The baseline achieved a test Precision of $0.7284$ and F1 of $0.7919$, whereas the ANN scored $0.2116$ and $0.3435$ respectively. Neither method is universally superior: the ANN prioritizes recall at the expense of substantially more false alarms, whereas the baseline prioritizes precision and rule fidelity.

2. **Why does the ANN generate 231 false alarms while the baseline generates only 22?**
   - **Why the ANN generates 231 false alarms (`14` shared + `217` ANN-only):** Under severe class imbalance (~3.39% failures), the ANN was trained with positive class weight $w_{\text{pos}} = 28.5203$ (`w_neg = 1.0`), penalizing missed failures 28.5 times more heavily than false alarms. Combined with evaluating at the default $0.50$ probability cutoff without post-training threshold calibration, this shifts the decision boundary across borderline high-load and elevated-wear healthy samples.
   - **Why the baseline generates 22 false alarms (`14` shared + `8` baseline-only):** As established in Step 3, the baseline is an **oracle reference** encoding the exact algebraic Matzka (2020) rules used to generate the synthetic AI4I 2020 labels. Consequently, its false alarms occur solely in the stochastic `TWF` $[200, 240]\text{ min}$ band where `tool_wear >= 220` triggers on non-failing samples.
   - **Limitations of fixed threshold rules:** Fixed rules cannot model stochastic failure noise (`RNF`), miss `TWF` failures in `[200, 219]` min due to the hard `220` min cutoff, and emit only a binary trip flag without continuous risk margins.
   - **Operational trade-off:** The choice between the two models depends on the relative cost of missed failures versus unnecessary inspections. While the Step 5 fuzzy engine scores ANN outputs using physical tool wear and torque to separate `LOW`/`MEDIUM` alerts from `HIGH`/`URGENT` actions, raw binary thresholding at `0.50` produces 209 more test false alarms than the baseline.

3. **Why is a continuous probability output still valuable for the project?**
   - **Continuous risk margin for downstream fuzzy logic:** Unlike the baseline's rigid binary trip flag, the ANN outputs a continuous failure probability $P \in [0, 1]$ (**Test ROC-AUC: 0.9581**). This continuous score provides the soft-computing input required by the **Mamdani Fuzzy Logic Inference System in Step 5** to compute a $0\text{--}100$ Severity Index and assign tiered priorities (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).

4. **Which examples disagree? (Test Set: 230 Total Disagreements)**
   - Inspection of `data/reports/step4_test_paired_predictions.csv` identifies the exact disagreement counts:
     - **ANN-Only True Positives (`4` cases; rows 614, 909, 959, 1947):** Failures detected by the ANN (`ann_probability` from `0.5156` to `0.8426`) that the baseline missed (`FN` for baseline, `TP` for ANN).
     - **Baseline-Only True Positives (`1` case; row 1016):** Exactly **1** failure (`ann_probability = 0.2840`) detected by the baseline and missed by the ANN (`FN` for ANN, `TP` for baseline). *(Note: the remaining 5 of the ANN's 6 total `FN`s — rows 19, 979, 1044, 1327, 1913 — were missed by both models.)*
     - **False-Positive Disagreements (`225` cases):** `217` healthy records flagged only by the ANN and `8` healthy records flagged only by the baseline (`14` flagged by both, totaling `231` ANN FPs vs. `22` baseline FPs).

---

## Illustrative Cost & Break-Even Analysis (Test Set)

To evaluate the operational trade-off between **3 net fewer missed failures** ($6$ vs. $9$) and **209 net additional false alarms** ($231$ vs. $22$) on the 2,000-record test set, let $C_{\text{FN}}$ denote the cost of one missed failure and $C_{\text{FP}}$ denote the cost of one false alarm inspection:

- **ANN Expected Cost:** $6 \times C_{\text{FN}} + 231 \times C_{\text{FP}}$
- **Baseline Expected Cost:** $9 \times C_{\text{FN}} + 22 \times C_{\text{FP}}$

The ANN yields a lower total expected cost than the baseline only when:

$$6\,C_{\text{FN}} + 231\,C_{\text{FP}} < 9\,C_{\text{FN}} + 22\,C_{\text{FP}} \implies 209\,C_{\text{FP}} < 3\,C_{\text{FN}} \implies \frac{C_{\text{FN}}}{C_{\text{FP}}} > \frac{209}{3} \approx 69.7$$

### Hypothetical Cost-Ratio Sensitivity Table (Normalized with $C_{\text{FP}} = 1$)

| Hypothetical Cost Ratio ($C_{\text{FN}} : C_{\text{FP}}$) | ANN Cost ($6\,C_{\text{FN}} + 231$) | Baseline Cost ($9\,C_{\text{FN}} + 22$) | Lower-Cost Model (Under Assumption) |
| :---: | :---: | :---: | :--- |
| **10 : 1** | $6(10) + 231 = \mathbf{291}$ | $9(10) + 22 = \mathbf{112}$ | Baseline (by 179 units) |
| **20 : 1** | $6(20) + 231 = \mathbf{351}$ | $9(20) + 22 = \mathbf{202}$ | Baseline (by 149 units) |
| **50 : 1** | $6(50) + 231 = \mathbf{531}$ | $9(50) + 22 = \mathbf{472}$ | Baseline (by 59 units) |
| **70 : 1** | $6(70) + 231 = \mathbf{651}$ | $9(70) + 22 = \mathbf{652}$ | ANN (by 1 unit; just above break-even $69.7:1$) |
| **100 : 1** | $6(100) + 231 = \mathbf{831}$ | $9(100) + 22 = \mathbf{922}$ | ANN (by 91 units) |

> **Important Disclaimer:** These ratios (`10:1` through `100:1`) are **illustrative hypothetical assumptions, not verified factory costs**. No empirical monetary figures or actual plant savings are claimed without real industrial cost data. At lower failure-to-inspection cost ratios ($< 69.7:1$), the fixed baseline has lower expected cost at the uncalibrated `0.50` ANN threshold.

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

### Team Members and Recognition

The SC08 project was undertaken by the following team members:

1. Shivam Mishra
2. Anuj
3. Ayush
4. Siddharth

All four members are recognized as members of the SC08 project team. The technical implementation, experimental analysis, and report preparation were led by Shivam Mishra.
