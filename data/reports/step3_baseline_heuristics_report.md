# Step 3 Milestone Report: Non-Soft-Computing Baseline Rule Heuristics

**Project:** SC08 — Machine Predictive Maintenance & Fault Priority System  
**Stage:** Step 3 (Non-Soft-Computing Baseline Evaluation & Threshold Traceability)  
**Target Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)  
**Evaluation Set:** Held-out Stratified Test Partition ($N = 2,000$ samples, $68$ failures, `data/processed/X_test.npy`, `data/processed/y_test.npy`)  

---

## 1. Project Objective and the Role of a Non-Soft-Computing Baseline

### 1.1 Project Objective
The primary objective of Project SC08 is to design, implement, and evaluate an interpretable, data-driven machine predictive maintenance framework. Operating on physical milling machine parameters, the architecture couples:
1. A hand-coded **Artificial Neural Network (ANN)** to learn non-linear sensor relationships and estimate a continuous failure probability $P(\text{Machine failure} = 1 \mid X) \in [0, 1]$.
2. A hand-coded **Mamdani Fuzzy Logic Inference Engine** to fuse failure probability with physical degradation indicators (tool wear and spindle torque), producing an interpretable continuous Severity Index ($0\text{--}100$) and assigning operational Maintenance Priority levels (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).

### 1.2 Purpose of a Non-Soft-Computing Baseline in Step 3
Before deploying soft-computing methods (ANNs and fuzzy inference), rigorous engineering methodology requires establishing a non-soft-computing benchmark. In industrial automation, traditional supervisory control and data acquisition (SCADA) and programmable logic controllers (PLCs) rely heavily on rigid, deterministic threshold limits and Boolean logic.

The non-soft-computing baseline serves three critical functions:
- **Benchmark of Rigid Threshold Logic:** Establishes the performance and boundary behaviors of rigid, step-function limit checks in contrast to adaptive and fuzzy decision boundaries.
- **Reference of Target Signal Representation:** Illustrates the degree to which deterministic rules can capture the underlying data patterns and highlights failure modes that crisp rules inherently miss.
- **Academic and Experimental Control:** Provides a clear standard against which the complexity, generalization, and computational trade-offs of subsequent neural and fuzzy modules must be judged.

---

## 2. Description of the Crisp Rule-Based Classifier

The baseline classifier is implemented in `src/baseline.py`. It operates as a purely deterministic, non-soft-computing decision system.

### 2.1 OR-Based Composite Decision Logic
In industrial machine health monitoring, if any primary subsystem or failure mode violates its critical operating limits, an alarm or trip condition must be declared. The baseline classifier applies an **inclusive OR logic** across all individual modeled failure modes:

$$\hat{y}_{\text{baseline}} = \text{HDF} \lor \text{PWF} \lor \text{TWF} \lor \text{OSF}$$

Where:
- $\hat{y}_{\text{baseline}} \in \{0, 1\}$ ($0 = \text{Healthy / No Failure}$, $1 = \text{Machine Failure predicted}$).
- If one or more failure conditions evaluate to `True`, $\hat{y}_{\text{baseline}} = 1$.
- If all four failure conditions evaluate to `False`, $\hat{y}_{\text{baseline}} = 0$.

### 2.2 Operational Characteristic
The classifier is completely non-soft-computing:
- It uses no artificial neural network layers, matrix weights, bias offsets, or gradient optimization.
- It uses no fuzzy membership functions, linguistic hedges, or defuzzification calculations.
- It relies entirely on standard mathematical arithmetic, physical conversions, and Boolean logic operations (`<`, `>`, `AND`, `OR`).

---

## 3. Threshold Traceability Matrix

The baseline failure-detection rules encode the physical threshold mechanisms described by Matzka (2020) for the AI4I 2020 dataset.

| Failure Mode | Mathematical Condition | Input Features & Physical Units | Threshold Limit(s) | Source & Justification | Known Limitations & Assumptions |
|---|---|---|---|---|---|
| **Heat Dissipation Failure (HDF)** | $(T_{\text{process}} - T_{\text{air}}) < 8.6\text{ K}$ $\land$ $\omega_{\text{rot}} < 1380\text{ rpm}$ | - $T_{\text{air}}$: Air temperature $[\text{K}]$<br>- $T_{\text{process}}$: Process temperature $[\text{K}]$<br>- $\omega_{\text{rot}}$: Spindle rotational speed $[\text{rpm}]$ | $\Delta T < 8.6\text{ K}$<br>$\text{Speed} < 1380\text{ rpm}$ | Matzka (2020). Models insufficient convective cooling when temperature difference is minimal and fan/spindle air circulation is low. | Assumes sharp step-function cutoff. Does not account for thermal inertia, transient load changes, or measurement noise around $8.6\text{ K}$. |
| **Power Failure (PWF)** | $P < 3500\text{ W} \lor P > 9000\text{ W}$<br>where $P = \tau \times \left(\omega_{\text{rot}} \times \frac{2\pi}{60}\right)$ | - $\tau$: Spindle torque $[\text{Nm}]$<br>- $\omega_{\text{rot}}$: Spindle rotational speed $[\text{rpm}]$ | $P_{\text{min}} = 3500.0\text{ W}$<br>$P_{\text{max}} = 9000.0\text{ W}$ | Matzka (2020). Power required for machining. $P < 3500\text{ W}$ indicates drive slipping or no-load fault; $P > 9000\text{ W}$ indicates severe mechanical overload. | Calculates instantaneous electric/mechanical power. Ignores motor efficiency curves, acceleration power spikes, and power factor variation. |
| **Tool Wear Failure (TWF)** | $t_{\text{wear}} \ge 220\text{ min}$ | - $t_{\text{wear}}$: Cumulative tool wear time $[\text{min}]$ | $t_{\text{threshold}} = 220\text{ min}$ | Matzka (2020). Represents cutting tool flank/crater wear reaching maximum allowable operational endurance. | Deterministic approximation of a stochastic phenomenon. See Section 4 for detailed analysis. |
| **Overstrain Failure (OSF)** | $(t_{\text{wear}} \times \tau) > \text{Threshold}_{\text{Type}}$ | - $t_{\text{wear}}$: Tool wear $[\text{min}]$<br>- $\tau$: Spindle torque $[\text{Nm}]$<br>- Product Quality Type ($\text{L}, \text{M}, \text{H}$) | Type L: $11,000\text{ min}\cdot\text{Nm}$<br>Type M: $12,000\text{ min}\cdot\text{Nm}$<br>Type H: $13,000\text{ min}\cdot\text{Nm}$ | Matzka (2020). Models mechanical overstrain as product of tool wear and applied torque. Higher-quality variants tolerate higher stress. | Product type variants assumed to have perfectly uniform material limits with zero fatigue variability across batches. |

---

## 4. Analysis of Tool Wear Failure (TWF) Approximation

In the generative model of the AI4I 2020 dataset (Matzka, 2020), Tool Wear Failure is **not** a deterministic threshold. Instead, it is triggered stochastically within an interval:

$$\text{TWF}_{\text{generative}} \sim \text{Random Event Triggered when } t_{\text{wear}} \in [200, 240]\text{ minutes}$$

Because a deterministic, non-soft-computing rule cannot model non-deterministic triggers without past state or random generators, `src/baseline.py` adopts the **interval midpoint**:

$$t_{\text{TWF\_cutoff}} = 220\text{ minutes}$$

### Practical Consequences on Baseline Performance:
1. **False Negatives ($t_{\text{wear}} \in [200, 220)\text{ min}$):** Tools that suffered premature structural breakdown or micro-chipping in the dataset between $200$ and $219$ minutes are classified as healthy ($\hat{y}=0$) by the deterministic baseline.
2. **False Positives ($t_{\text{wear}} \in [220, 240]\text{ min}$):** Tools that maintained cutting integrity past $220$ minutes without failing before finishing the cycle are incorrectly flagged as failures ($\hat{y}=1$) by the deterministic baseline.

This approximation represents a known, unavoidable limitation of static threshold heuristics when addressing stochastic wear dynamics.

---

## 5. Stochastic Nature of Random Failures (RNF)

In the AI4I 2020 dataset, **Random Failure (RNF)** is explicitly simulated with a fixed probability of approximately $0.1\%$ per operational cycle, regardless of process parameters. Across the full 10,000-record dataset, exactly 19 RNF instances occur.

### Baseline Treatment and Limitation:
- **Unmodelled Status:** In `src/baseline.py`, RNF is set to `False` unconditionally.
- **Physical Rationale:** Purely random failures (e.g., sudden electrical surge, external debris, unmeasured material defect) possess zero mathematical correlation with the six recorded operational features.
- **Impact on Evaluation:** Any true machine failure caused solely by RNF will inevitably result in a **False Negative ($\text{FN}$)** in the crisp baseline. This design choice is mathematically sound: attempting to fit a deterministic rule to zero-signal noise would induce severe overfitting without physical validity.

---

## 6. Critical Feature Clarification: Absence of Vibration

A critical technical constraint of the AI4I 2020 dataset must be documented for academic integrity:

> **The AI4I 2020 dataset does NOT contain a direct vibration sensor feature.**

- **Torque is Not Vibration:** Feature 4 in the canonical input order is `Torque [Nm]`, which measures spindle mechanical resistance and motor loading. It is measured in Newton-metres ($\text{Nm}$).
- **Vibration:** Vibration sensors (such as piezoelectric accelerometers) measure dynamic oscillatory acceleration in $\text{m/s}^2$ or velocity in $\text{mm/s}$ over continuous frequency spectra.
- **Methodological Rule:** In this project, `Torque [Nm]` is utilized strictly as spindle torque. It is never renamed, substituted, or claimed to serve as a vibration proxy.

---

## 7. Test-Set Performance Evaluation

The crisp baseline was evaluated on the held-out, unmutated test partition (`data/processed/X_test.npy`, `data/processed/y_test.npy`). The results documented below reflect the exact values stored in `results/metrics.json`.

### 7.1 Test Partition Summary
- **Total Test Samples ($N$):** 2,000
- **Negative Class (Healthy, $y=0$):** 1,932 ($96.60\%$)
- **Positive Class (Failure, $y=1$):** 68 ($3.40\%$)
- **Preprocessing Protocol:** Test samples are denormalized using parameters fitted strictly on the training set (`data/processed/norm_params.json`). The test partition was never observed during normalisation parameter estimation.

### 7.2 Confusion Matrix (Crisp Baseline)

| Actual \ Predicted | Predicted Healthy ($\hat{y} = 0$) | Predicted Failure ($\hat{y} = 1$) | Total |
|---|---|---|---|
| **Actual Healthy ($y = 0$)** | $\text{TN} = 1,910$ | $\text{FP} = 22$ | 1,932 |
| **Actual Failure ($y = 1$)** | $\text{FN} = 9$ | $\text{TP} = 59$ | 68 |
| **Total** | 1,919 | 81 | 2,000 |

### 7.3 Reported Numerical Metrics

| Metric | Formula / Definition | Baseline Value | Reported in `results/metrics.json` |
|---|---|---|---|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ | **0.9845** | `0.9845` ($98.45\%$) |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | **0.7284** | `0.7284` ($72.84\%$) |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **0.8676** | `0.8676` ($86.76\%$) |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | **0.9886** | `0.9886` ($98.86\%$) |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | **0.7919** | `0.7919` |
| **ROC-AUC** | $\frac{\text{Recall} + \text{Specificity}}{2}$ *(discrete binary)* | **0.9281** | `0.9281` |

*(Note: For comparison, the trained ANN metrics recorded in `results/metrics.json` are: Accuracy: 0.8815, Precision: 0.2116, Recall: 0.9118, Specificity: 0.8804, F1-Score: 0.3435, ROC-AUC: 0.9581; $\text{TP}=62, \text{TN}=1701, \text{FP}=231, \text{FN}=6$).*

---

## 8. Impact of Class Imbalance (~3.4% Failure Rate)

### 8.1 The Accuracy Paradox
The AI4I 2020 dataset exhibits severe class imbalance: only $339$ out of $10,000$ operational records ($3.39\%$) represent machine failures. In the 2,000-sample test set, exactly $68$ failures ($3.40\%$) exist against $1,932$ healthy records.

Under this distribution, a naive or degenerate classifier that uniformly predicts "Healthy" ($\hat{y} = 0$) for every sample achieves:

$$\text{Accuracy}_{\text{trivial}} = \frac{1932}{2000} = 96.60\%$$

Despite a seemingly impressive accuracy of $96.60\%$, such a classifier has a **Recall of 0.00%**, failing to detect a single failing machine and offering zero practical utility in a manufacturing environment.

### 8.2 Why Precision, Recall, and F1-Score are Essential
- **Recall ($86.76\%$):** Measures the proportion of actual machine failures correctly intercepted. In predictive maintenance, missed failures ($\text{FN} = 9$) can lead to catastrophic tool breakage and unbudgeted assembly line stoppage.
- **Precision ($72.84\%$):** Measures the proportion of flagged alarms that represent genuine failures. A lower precision incurs unnecessary inspection overhead and maintenance team fatigue due to false alarms ($\text{FP} = 22$).
- **F1-Score ($0.7919$):** The harmonic mean of precision and recall balances detection sensitivity against false alarm rates, making it the primary objective metric for model comparison.

---

## 9. Academic Integrity: Data-Generation Equations vs. Industrial Heuristics

An essential distinction must be formally recognized regarding the nature of `src/baseline.py`:

### 9.1 The Baseline Reflects Data-Generating Equations
The crisp baseline achieves a high F1-score ($0.7919$) and accuracy ($98.45\%$) primarily because **it directly implements the mathematical rules Matzka (2020) used to create the failure labels in the dataset**.
- In statistical terms, the baseline acts as an **oracle classifier** that shares the exact structural priors of the synthetic generator.
- Its imperfect score ($F_1 = 0.7919$ rather than $1.0$) is driven entirely by the two failure mechanisms that crisp static rules cannot capture: the stochastic TWF window ($[200, 240]\text{ min}$) and unmodelled stochastic RNF ($0.1\%$).

### 9.2 It is Not an Independently Validated Industrial Heuristic
- This baseline **must not** be presented to faculty or reviewers as an independent industrial thresholding system validated across physical milling equipment.
- In real factory environments, exact physical threshold boundaries (such as exactly $8.6\text{ K}$ or exactly $11,000\text{ min}\cdot\text{Nm}$) are unknown, machine-specific, and contaminated by multi-source sensor degradation and ambient temperature swings.
- The baseline's true academic purpose is to serve as a **theoretical upper-bound reference of crisp rule representation** against which learning algorithms (which must infer boundaries without being provided the formulas) can be evaluated.

---

## 10. Motivation for Soft-Computing Methods (ANN & Fuzzy Logic)

The limitations observed in the crisp baseline provide the theoretical and practical justification for introducing soft-computing techniques:

### 10.1 Role of the Artificial Neural Network (ANN)
- **Data-Driven Boundary Learning:** In real-world predictive maintenance, the explicit algebraic formulas governing failure are unavailable. The ANN demonstrates how a machine learning model can approximate these complex, non-linear failure surfaces purely from observational data without explicit hard-coding.
- **Handling Interdependent Features:** While crisp rules require strict step thresholds across individual feature pairs (e.g., speed and temperature difference), an ANN naturally learns continuous, multi-dimensional decision boundaries across all 7 input features simultaneously.

### 10.2 Role of Mamdani Fuzzy Logic
- **Gradual Degradation vs. Step-Function Cliffs:** In physical machining, tool degradation does not transition abruptly from perfectly safe at $219.9\text{ min}$ to critically failed at $220.0\text{ min}$. Fuzzy membership functions model gradual wear and transitional risk zones (e.g., `NORMAL`, `WARNING`, `CRITICAL`).
- **Operational Prioritization:** The crisp baseline produces only a binary flag ($0$ or $1$) with no measure of urgency. The Mamdani fuzzy system combines the ANN failure probability with tool wear and torque to compute a continuous **Severity Index ($0\text{--}100$)** and assign operational **Maintenance Priorities (`LOW`, `MEDIUM`, `HIGH`, `URGENT`)**, directly addressing plant maintenance planning.

---

## 11. Reproducibility Instructions

All data structures, preprocessing routines, and baseline evaluation scripts can be executed and verified directly using the commands documented below from the repository root:

### 11.1 Re-run Preprocessing and Test Array Generation
```bash
python -m src.prepare_data
```
*Expected Outputs:* Generates `data/processed/X_train.npy`, `data/processed/X_test.npy`, `data/processed/y_train.npy`, `data/processed/y_test.npy`, and `data/processed/norm_params.json`.

### 11.2 Execute Crisp Baseline Evaluation
```bash
python -m src.baseline
```
*Expected Output:* Runs the 5 scenario verification unit tests, evaluates the 2,000-sample test set, and prints the summary breakdown.

### 11.3 Execute Complete Evaluation Suite
```bash
python -m src.evaluation
```
*Expected Output:* Computes classification metrics for both the Crisp Baseline and ANN, executes the Hybrid Fuzzy diagnostic analysis across all 2,000 test samples, updates `results/metrics.json`, and renders visual plots to `results/figures/`.

---

## 12. References

1. **Matzka, S. (2020).** "Explainable Artificial Intelligence for Predictive Maintenance Applications." In *Proceedings of the Third International Conference on Artificial Intelligence for Industries (AI4I 2020)*, IEEE, pp. 69–74. DOI: 10.1109/AI4I49448.2020.00023.
2. **UCI Machine Learning Repository.** "AI4I 2020 Predictive Maintenance Dataset." Dataset ID: 601. URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
3. **SC08 Project Documentation.** `data/reports/data_quality_report.md` (Step 2 Verification Report) and `README.md`.
