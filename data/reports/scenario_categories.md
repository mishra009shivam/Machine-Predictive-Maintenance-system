# Data & Scenario Validation Framework — SC08

## 1. Overview & Conceptual Distinction

In the SC08 **Machine Predictive Maintenance & Fault Priority System**, data inputs are evaluated through a hybrid architecture combining an **Artificial Neural Network (ANN)** and a **Mamdani Fuzzy Inference System**.

To ensure scientific rigor, validation transparency, and proper domain context, this document formalizes the distinction between **real empirical dataset records** and **synthetic evaluation scenarios**.

---

## 2. Category Distinction

### Category A: Empirical Dataset Records (AI4I 2020)
- **Source:** UCI AI4I 2020 Predictive Maintenance Dataset (`data/raw/ai4i2020.csv`).
- **Nature:** Statistical distributions representing synthetic milling machine operation with 10,000 continuous/categorical observations.
- **Purpose:** Used strictly for ANN model training, validation loss monitoring, threshold tuning, and objective statistical evaluation ($F_1$-score, Precision, Recall, ROC-AUC).
- **Scope:** Defines the empirical domain boundaries under which model weights and normalization parameters were learned.

### Category B: Generated Testing & Benchmark Scenarios
- **Source:** Defined synthetic test cases (e.g. preset UI scenarios in Streamlit dashboard and pipeline test suites in `src/prediction.py`).
- **Nature:** Purpose-built edge cases designed to test hybrid ANN + Fuzzy decision logic, rule firing combinations, and emergency priority escalation.
- **Purpose:** Demonstrates interpretable fuzzy reasoning, boundary handling, and safety-critical maintenance escalation (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).
- **Explicit Note:** **These generated benchmark scenarios are synthetic test inputs, NOT real physical sensor observations.** They serve as structured diagnostic probes to evaluate system safety and fuzzy rule activation.

---

## 3. Scenario Taxonomy & Categories

For systematic pipeline testing and verification, test cases are grouped into five distinct operational categories:

| Scenario Category | Description | Primary Evaluation Objective | Expected System Behavior |
|---|---|---|---|
| **1. Normal** | Standard operating parameters (ambient temp ~300 K, low tool wear < 50 min, moderate torque ~35 Nm). | Verify baseline stability and absence of false alarms. | Low ANN failure probability ($P < 0.10$), Low Severity Index (< 33), `LOW` maintenance priority. |
| **2. Boundary** | Operating conditions near operational decision thresholds (e.g. tool wear ~180 min, torque ~55 Nm). | Test fuzzy membership transition regions and intermediate risk scoring. | Gradual severity scaling, transition between `MEDIUM` and `HIGH` priority levels. |
| **3. Stress / Extreme** | Compounding physical degradation (tool wear > 220 min, high process temp > 311 K, torque > 60 Nm). | Validate critical failure detection and rapid escalation. | High ANN probability ($P > 0.90$), Severity Index > 78, `URGENT` maintenance priority with immediate stop recommendation. |
| **4. Contradictory** | Divergence between ANN probability and physical indicators (e.g. high ANN probability with low tool wear, or low ANN probability with critical tool wear). | Demonstrate fuzzy rule-base risk dampening or safety overrides. | Fuzzy rules act as domain-aware safeguards (e.g. Rule R7 dampening severity when tool wear is low, or Rule R9 escalating severity when physical tool wear is critical). |
| **5. Invalid / Out-of-Range** | Out-of-bounds input values (e.g. negative temperatures, zero rotational speed with non-zero torque, tool wear > 300 min). | Verify input validation and robust error handling. | Input validation flags errors or clips inputs gracefully before pipeline execution. |

---

## 4. Verification & Usage Guidelines

1. **Reporting Integrity:** Benchmark scenario evaluations must be explicitly labeled as "Synthetic Benchmark Probe" in all documentation, reports, and dashboard interfaces.
2. **No Data Contamination:** Synthetic benchmark scenarios are never mixed with empirical training/test partitions (`X_train`, `X_test`).
3. **Fuzzy Explainability:** Every test scenario output must log activated fuzzy rules, rule activation strengths ($\alpha_i$), and the defuzzified Centre of Gravity (CoG) output for auditability.
