# SC08 — Machine Predictive Maintenance & Fault Priority System

A university Soft Computing capstone project that combines a hand-implemented
**Artificial Neural Network (ANN)** and a hand-implemented **Fuzzy Logic system**
to build a decision-support tool for predictive maintenance.

---

## Project Overview

| Item | Detail |
|---|---|
| **Course** | Soft Computing |
| **Project code** | SC08 |
| **Dataset** | AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository) |
| **Dataset license** | CC BY 4.0 |
| **Python version** | ≥ 3.9 |

### System Pipeline

```
Machine sensor readings
        ↓
  Input validation
        ↓
 Data preprocessing / normalisation
        ↓
    ANN (NumPy)
        ↓
  Fault probability ∈ [0, 1]
        ↓
  Fuzzy Logic system (NumPy)
        ↓
  Severity index [0–100]
        ↓
  Maintenance priority (URGENT / HIGH / MEDIUM / LOW)
        ↓
  Maintenance recommendation
        ↓
  Visual dashboard (Streamlit)
```

### Approaches Compared

| Approach | Description |
|---|---|
| **Documented-rule crisp reference baseline** | Encodes the published AI4I failure-generation rules (Matzka 2020). Serves as a crisp, rule-based reference — **not** an independent industrial benchmark. |
| **ANN only** | Hand-coded NumPy neural network trained on the dataset. Predicts fault probability. |
| **ANN + Fuzzy hybrid** | ANN fault probability fed into a Mamdani fuzzy system that outputs severity and maintenance priority. |

---

## Repository Structure

```
SC08-Predictive-Maintenance/
│
├── app/
│   └── app.py                  ← Streamlit dashboard
│
├── src/
│   ├── preprocessing.py        ← Load, encode, normalise, split
│   ├── ann.py                  ← Hand-coded ANN (NumPy only)
│   ├── fuzzy.py                ← Hand-coded Fuzzy Logic (NumPy only)
│   ├── baseline.py             ← Documented-rule crisp reference baseline
│   ├── prediction.py           ← Full hybrid pipeline
│   └── evaluation.py           ← Metrics, comparison, edge-case tests
│
├── data/
│   ├── ai4i2020.csv            ← Raw dataset (download instructions below)
│   ├── processed/              ← Pre-processed NumPy arrays (generated)
│   └── README.md               ← Full dataset citation, license, and limitations
│
├── results/
│   ├── ann_weights.npz         ← Saved trained ANN weights
│   ├── training_history.json   ← Loss and F1 per epoch
│   ├── metrics.json            ← Final evaluation metrics
│   └── figures/                ← Saved plots
│
├── report/                     ← Project report
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd SC08-Predictive-Maintenance
```

### 2. Create a virtual environment

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

### 4. Download the dataset

The AI4I 2020 dataset must be downloaded manually (CC BY 4.0 licence):

1. Go to: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Click **Download**
3. Extract and place `ai4i2020.csv` in the `data/` folder

See `data/README.md` for full citation, licence text, and dataset limitations.

### 5. Run preprocessing

```bash
python -m src.preprocessing
```

This generates the normalised train/test arrays and `norm_params.json` in `data/processed/`.

### 6. Train the ANN

```bash
python -m src.ann --train
```

Saves `results/ann_weights.npz` and `results/training_history.json`.

### 7. Run evaluation

```bash
python -m src.evaluation
```

Prints the comparison table (Baseline / ANN / Hybrid) and saves figures to `results/figures/`.

### 8. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## Implementation Notes

- The ANN is implemented entirely with **NumPy** — no TensorFlow, PyTorch, or Keras.
- The Fuzzy Logic system is implemented entirely with **NumPy** — no scikit-fuzzy.
- `scikit-learn` is used **only** for `train_test_split` and evaluation metrics (F1, ROC-AUC, confusion matrix).
- Class imbalance (~3.4% positive rate) is handled via **class-weighted binary cross-entropy loss**, not oversampling.
- The dataset does **not** contain a vibration sensor. See `data/README.md` for details.

---

## Citation

```
Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance
Applications. In Proceedings of the Third International Conference on Artificial
Intelligence for Industries (AI4I 2020).

Dataset: UCI Machine Learning Repository, ID 601.
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
License: CC BY 4.0
```

---

## Reproducibility

- `random_state=42` used for all stochastic operations.
- All hyperparameters are documented in `src/ann.py` and `src/fuzzy.py`.
- All evaluation metrics are computed on the held-out test set only.
- Training can be re-run from scratch to reproduce results.
