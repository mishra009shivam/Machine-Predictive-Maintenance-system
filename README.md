# SC08 — Machine Predictive Maintenance & Fault Priority System

A university Soft Computing capstone project that combines a hand-coded **Artificial Neural Network (ANN)** and a hand-coded **Mamdani Fuzzy Logic system** to evaluate machine failure probability, reason about fault severity, assign maintenance priorities, and provide actionable maintenance recommendations.

---

## Problem Statement

Modern industrial manufacturing relies heavily on continuous equipment operation. Unscheduled equipment breakdowns lead to costly unplanned downtime, expensive secondary machine damage, lost production capacity, and severe operational hazards. Traditional schedule-based maintenance often leads to unnecessary over-servicing of healthy machinery or fails to catch sudden, unexpected failures.

However, developing effective automated maintenance systems presents key challenges:
- **Limitations of Raw Sensor Signals & Binary Flags:** Standard binary classification models output a raw failure probability (e.g., 0.72), but plant operators cannot translate a single probability score directly into prioritized operational actions. Furthermore, sensor data is noisy, failure events are rare (severe class imbalance), and binary labels cannot express failure severity or distinguish between minor component wear and imminent catastrophic breakdown.
- **Why ANN + Fuzzy Logic Hybrid:** Artificial Neural Networks excel at learning complex, non-linear relationships from multi-sensor data to predict failure probability. However, ANNs act as black-box estimators and lack transparent decision reasoning. Fuzzy Logic bridges this gap by incorporating domain knowledge and expert reasoning using continuous linguistic rules (e.g., *IF Failure Probability is High AND Tool Wear is Extreme, THEN Severity is Critical*).
- **End-to-End Decision Support:** The final hybrid system processes raw machine sensor readings, computes failure probability via the hand-coded ANN, evaluates operational severity via the Mamdani Fuzzy Inference System, maps severity to discrete Maintenance Priorities (*URGENT*, *HIGH*, *MEDIUM*, *LOW*), and generates specific, actionable maintenance recommendations.

---

## Objectives

- **Predict Machine Failure Probability:** Develop and train a hand-coded Artificial Neural Network (NumPy only) to estimate machine failure probability from operational sensor readings.
- **Handle Class Imbalance:** Implement class-weighted Binary Cross-Entropy (BCE) loss to handle rare machine failure cases (~3.4% positive rate) without synthetic data generation.
- **Implement Multi-Factor Severity Reasoning:** Design a hand-coded Mamdani Fuzzy Inference System (NumPy only) to evaluate fault severity based on failure probability and operational stress inputs.
- **Generate Maintenance Priorities & Recommendations:** Map continuous fuzzy severity indices to operational maintenance priority levels and clear maintenance actions.
- **Compare Against Baseline:** Compare model behavior against a documented-rule crisp reference baseline.
- **Rigorous Test Evaluation:** Evaluate model generalization and prediction performance on unseen held-out test data.
- **Interactive User Interface:** Provide a clean, user-friendly Streamlit dashboard interface for real-time risk assessment and decision support.

---

## Project Overview

| Item | Detail |
|---|---|
| **Course** | Soft Computing |
| **Project Code** | SC08 |
| **Dataset** | AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository) |
| **Dataset License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Python Version** | ≥ 3.9 |

### System Pipeline

```
Machine sensor readings
        ↓
Input validation
        ↓
Data preprocessing / normalization
        ↓
ANN (NumPy)
        ↓
Fault probability [0,1]
        ↓
Mamdani Fuzzy Logic (NumPy)
        ↓
Severity Index [0,100]
        ↓
Maintenance Priority
        ↓
Maintenance Recommendation
        ↓
Streamlit Dashboard
```

### Approaches Compared

| Approach | Description |
|---|---|
| **Documented-rule crisp reference baseline** | Encodes the published AI4I failure-generation rules (Matzka, 2020). Serves as a deterministic, crisp reference rule set — **not** an independent industrial benchmark. |
| **ANN only** | Hand-coded NumPy neural network trained on normalized sensor features to output raw failure probability. |
| **ANN + Fuzzy hybrid** | Hybrid architecture combining ANN failure probability with Mamdani Fuzzy Logic inferencing to compute severity indices, maintenance priorities, and maintenance recommendations. |

---

## Methodology

### A. Data Preprocessing
- **Validation & Cleaning:** Verification of dataset integrity, feature data types, and absence of missing values. Identifier attributes (`UDI`, `Product ID`) are removed.
- **Categorical Encoding:** One-hot encoding of machine product type (`Type` L/M/H) into `Type_M` and `Type_H`, using `Type_L` as the implicit reference category to prevent multicollinearity.
- **Stratified Train/Test Split:** 80/20 stratified split implemented in pure NumPy to preserve the ~3.4% failure class distribution across training and test sets.
- **Min-Max Normalization:** Numeric features are scaled to $[0, 1]$ using minimum and maximum bounds computed strictly from training data to avoid data leakage.
- **Class Weighting:** Positive class weight $w_{\text{pos}} = N_{\text{neg}} / N_{\text{pos}}$ calculated from training labels for weighted loss computation.

### B. Artificial Neural Network (ANN)
- **Implementation:** Fully connected multi-layer perceptron built entirely from scratch using **NumPy**.
- **Loss Function:** Class-weighted Binary Cross-Entropy (BCE) loss to penalize false negatives on rare failure instances.
- **Optimization & Activation:** Forward pass with ReLU activation for hidden layers and Sigmoid activation for the output layer, optimized via backpropagation and gradient descent.

### C. Fuzzy Logic System
- **Implementation:** Hand-coded Mamdani Fuzzy Inference System built entirely with **NumPy**.
- **Fuzzification:** Triangular and trapezoidal membership functions convert crisp ANN failure probability and operational stress inputs into linguistic variables.
- **Inference Engine:** Rule base applying min-max composition (Mamdani implication) to evaluate rule antecedents and aggregate fuzzy output sets.
- **Defuzzification:** Center of Gravity (CoG) method converts aggregated fuzzy output distributions into a continuous Severity Index $S \in [0, 100]$.

### D. Hybrid Prediction Pipeline
- Integrates the trained ANN probability output into the Mamdani Fuzzy Inference System to derive a combined risk score, assign operational priority categories (*LOW*, *MEDIUM*, *HIGH*, *URGENT*), and generate specific maintenance recommendations.

### E. Evaluation Framework
- Methodological protocol designed to compare the baseline, ANN-only, and hybrid models on held-out test data across standard metrics (F1 score, Precision, Recall, ROC-AUC, confusion matrices) and edge-case operational scenarios.

---

## Dataset

The project utilizes the **AI4I 2020 Predictive Maintenance Dataset**, sourced from the UCI Machine Learning Repository.

- **Source:** UCI Machine Learning Repository (Dataset ID 601)
- **Volume:** 10,000 total records
- **Class Distribution:** 339 machine failure events (~3.4% failure rate)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Matzka, S. (2020)

> **Important Dataset Note:**
> The AI4I 2020 dataset does not contain a vibration sensor. The project therefore uses the available documented variables without renaming Torque as vibration.

---

## Current Status

- [x] Project scaffold
- [x] Dataset acquisition/documentation
- [x] Data preprocessing
- [x] Train/validation/test split
- [x] Class-weighted loss implementation
- [x] Hand-coded NumPy ANN
- [x] ANN training and saved weights
- [x] Hand-coded NumPy Fuzzy Logic
- [x] Baseline implementation
- [x] Hybrid prediction pipeline
- [ ] Final evaluation
- [ ] Edge-case experiments
- [ ] Streamlit dashboard
- [ ] Public deployment
- [ ] Final report

---

## Repository Structure

```
SC08-Predictive-Maintenance/
│
├── app/                        ← Streamlit dashboard directory (dashboard pending implementation)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← Data loading, encoding, normalization, and NumPy stratified split
│   ├── ann.py                  ← Hand-coded ANN implementation (NumPy only)
│   ├── fuzzy.py                ← Hand-coded Mamdani Fuzzy Logic system (NumPy only)
│   ├── baseline.py             ← Documented-rule crisp reference baseline
│   └── prediction.py           ← Full hybrid prediction pipeline
│
├── data/
│   ├── ai4i2020.csv            ← Raw dataset (10,000 records)
│   ├── README.md               ← Full dataset citation, license, and limitations
│   └── processed/              ← Pre-processed NumPy arrays (generated)
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── norm_params.json
│
├── results/
│   ├── ann_weights.npz         ← Saved trained ANN weights
│   ├── training_history.json   ← Training loss and history
│   └── figures/                ← Saved plots
│       └── ann_learning_curve.svg
│
├── report/                     ← Project report directory (pending final report)
├── requirements.txt            ← Project dependencies
├── README.md                   ← Project documentation
└── .gitignore                  ← Git ignore configuration
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system.git
cd SC08-Predictive-Maintenance
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset verification

The raw dataset is stored in `data/ai4i2020.csv`. If re-downloading is required:
1. Visit: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Download and place `ai4i2020.csv` in `data/`.

### 5. Run preprocessing

```bash
python -m src.preprocessing
```

This generates normalized train/test arrays and `norm_params.json` in `data/processed/`.

### 6. Train the ANN

```bash
python -m src.ann --train
```

Saves `results/ann_weights.npz` and `results/training_history.json`.

---

## Implementation Notes

- **NumPy-Only ANN & Fuzzy Mathematics:** Both the Artificial Neural Network and Mamdani Fuzzy Inference System are built entirely using **NumPy** without deep learning frameworks (TensorFlow, PyTorch, Keras) or fuzzy libraries (scikit-fuzzy).
- **Pure NumPy Preprocessing:** Preprocessing functions, including Min-Max normalization, one-hot encoding, and 80/20 stratified train/test split, are implemented directly using **NumPy** and **pandas**.
- **Role of Dependencies:** `scikit-learn` is included in `requirements.txt` for standardized evaluation metric computation (F1 score, ROC-AUC, confusion matrix), but is not used in core model math or preprocessing routines.
- **Handling Imbalance:** Class imbalance (~3.4% failure rate) is handled strictly via **class-weighted binary cross-entropy loss**, avoiding synthetic sampling techniques like SMOTE.
- **Sensor Domain Accuracy:** The dataset contains no vibration measurements; Torque [Nm] is processed as documented and is not converted or renamed.

---

## Limitations

- **Lack of Vibration Data:** The AI4I 2020 dataset does not contain vibration sensor readings, which are typically critical in industrial mechanical fault diagnostics.
- **Synthetic Data Source:** The dataset is synthetically constructed based on physical models, which may not capture all random, unmodeled environmental dynamics of real manufacturing facilities.
- **Crisp Baseline Nature:** The documented-rule crisp reference baseline encodes the exact mathematical formulas used to generate failures in the dataset; it functions as a reference check rather than an external industrial benchmark.
- **Fuzzy Membership & Threshold Assumptions:** Fuzzy membership bounds, rule weights, and severity priority thresholds represent academic design choices rather than validated industrial standards.

---

## Future Work

- **Integration of Physical Vibration Signals:** Incorporating real multi-axis vibration data streams to enhance fault diagnostic capabilities.
- **Validation on Real Industrial Datasets:** Testing and evaluating the hybrid framework on operational data from real-world manufacturing plants.
- **Advanced Model Calibration:** Exploring probability calibration techniques to improve risk estimation under extreme class imbalance.
- **Automated Parameter Optimization:** Implementing Neuro-Fuzzy (ANFIS) or Genetic Algorithms to optimize membership function shapes and rule weights automatically.
- **Real-Time Sensor Integration & UI Deployment:** Developing real-time streaming interfaces and deploying the Streamlit application for live operational decision support.

---

## Citation

```bibtex
@inproceedings{matzka2020explainable,
  title={Explainable Artificial Intelligence for Predictive Maintenance Applications},
  author={Matzka, Stephan},
  booktitle={Third International Conference on Artificial Intelligence for Industries (AI4I)},
  year={2020},
  publisher={IEEE}
}
```

- **Dataset:** UCI Machine Learning Repository, ID 601.  
- **URL:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset  
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Reproducibility

- **Random Generator Seeding:** Stochastic operations (stratified data splitting in `src/preprocessing.py`, ANN weight initialization, validation splitting, and mini-batch shuffling in `src/ann.py`) instantiate explicit NumPy generator instances using `np.random.default_rng(seed=42)`.
- **Explicit Hyperparameters:** Neural network architecture dimensions (7-16-8-1), learning rates, loss weights, and fuzzy membership sets are explicitly defined in `src/ann.py` and `src/fuzzy.py`.
- **Execution Consistency:** Re-running preprocessing and training routines within the same Python/NumPy environment reproduces generated dataset splits, trained model weights, and history loss curves.
