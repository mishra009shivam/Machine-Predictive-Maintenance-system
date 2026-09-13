# Data Directory — AI4I 2020 Predictive Maintenance Dataset

## Dataset Identity

| Field | Value |
|---|---|
| **Name** | AI4I 2020 Predictive Maintenance Dataset |
| **Repository** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| **Dataset ID** | 601 |
| **Donated** | 2020-08-29 |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **License URL** | https://creativecommons.org/licenses/by/4.0/ |

## Citation

```
Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance
Applications. In Proceedings of the Third International Conference on Artificial
Intelligence for Industries (AI4I 2020), IEEE.

Bibtex:
@inproceedings{matzka2020explainable,
  title={Explainable Artificial Intelligence for Predictive Maintenance Applications},
  author={Matzka, Stephan},
  booktitle={2020 Third International Conference on Artificial Intelligence for Industries (AI4I)},
  year={2020},
  organization={IEEE}
}
```

## Dataset Description

The AI4I 2020 dataset is a **synthetic** dataset created to reflect real industrial
predictive-maintenance scenarios. It is not derived from a real machine installation,
but its generative process was designed to mirror realistic failure patterns found
in industry.

### Size

- **10,000 rows** (data points)
- **14 columns** (including identifiers and targets)
- **No missing values**

### Variables

| Variable | Role | Type | Unit | Notes |
|---|---|---|---|---|
| UID | Identifier | Integer | — | Unique ID 1–10,000; dropped in preprocessing |
| Product ID | Identifier | Categorical | — | Letter + serial number; dropped in preprocessing |
| Type | Feature | Categorical | — | Product quality: L (50%), M (30%), H (20%) |
| Air temperature | Feature | Continuous | K | Random walk, σ=2 K around 300 K |
| Process temperature | Feature | Continuous | K | Air temp + 10 K + noise, σ=1 K |
| Rotational speed | Feature | Integer | rpm | Derived from 2860 W power + noise |
| Torque | Feature | Continuous | Nm | Normal(μ=40, σ=10), no negatives |
| Tool wear | Feature | Integer | min | Increments by 5/3/2 min for H/M/L type |
| Machine failure | **Target** | Binary | — | 1 if any failure mode is active |
| TWF | Sub-target | Binary | — | Tool Wear Failure |
| HDF | Sub-target | Binary | — | Heat Dissipation Failure |
| PWF | Sub-target | Binary | — | Power Failure |
| OSF | Sub-target | Binary | — | Overstrain Failure |
| RNF | Sub-target | Binary | — | Random Failure |

### Failure Modes (label-generation rules, published by Matzka 2020)

The `Machine failure` label is set to 1 if **any** of the following conditions is true:

| Mode | Rule | Approx. count |
|---|---|---|
| **TWF** | Tool wear is randomly selected to fail between 200–240 min | ~120 |
| **HDF** | (Process_temp − Air_temp) < 8.6 K **AND** Rotational_speed < 1380 rpm | ~115 |
| **PWF** | Power = Torque × (rpm × 2π/60) < 3500 W **OR** > 9000 W | ~95 |
| **OSF** | Tool_wear × Torque > 11,000 (L) / 12,000 (M) / 13,000 (H) min·Nm | ~98 |
| **RNF** | Each process has a 0.1% random failure chance | ~5 |

**Overall positive rate: ~3.4% (339 / 10,000)**

This imbalance is addressed via class-weighted loss during ANN training.
The test set is evaluated on the original class distribution.

---

## Preprocessing Applied in This Project

1. Drop `UID` and `Product ID` (identifiers, not predictors)
2. One-hot encode `Type` → `Type_M` (0/1) and `Type_H` (0/1); `Type_L` is the reference category
3. Split 80% train / 20% test, stratified on `Machine failure`, `random_state=42`
4. Min-Max normalise all five numeric features using **training-set statistics only**:
   `x_norm = (x − x_min_train) / (x_max_train − x_min_train)`
5. Normalisation parameters saved to `processed/norm_params.json`
6. Arrays saved as `processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

The five numeric features after preprocessing (in order):

1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]

Plus two binary encoded variables: `Type_M`, `Type_H` — giving **7 total ANN inputs**.

---

## Dataset Limitations Relevant to This Project

### 1. No vibration sensor

The AI4I 2020 dataset does **not** include a vibration measurement. The project
specification mentions "Vibration" as a candidate sensor input. This feature is
**absent** from the chosen dataset and has **not** been simulated or substituted.

- `Torque [Nm]` is used as Torque — not as a vibration proxy or surrogate.
  It is a real, documented feature in the dataset that captures mechanical load.
  It is **not** equivalent to vibration and is **not** presented as such anywhere
  in this project.

If a direct vibration measurement is required, a different dataset (e.g., CWRU
Bearing Dataset, NASA FEMTO, or IMS Bearing datasets) would be needed. Those
datasets are condition-monitoring datasets for specific bearing faults and do not
provide the same multi-feature, binary machine-failure classification task as AI4I 2020.

### 2. Synthetic origin

The dataset is synthetically generated. Its failure patterns reflect the documented
generative rules above. Real machine data may exhibit more complex, correlated,
and environment-dependent failure patterns not captured here.

### 3. Baseline relationship with dataset labels

The "Documented-rule crisp reference baseline" (`src/baseline.py`) encodes the
same failure rules used to generate the dataset labels. It is **not** an independent
industrial benchmark. Its purpose is to establish what rigid, rule-based classification
looks like, and to compare it against a data-driven ANN that must recover these
patterns from training examples without being told the rules.

### 4. TWF randomness

Tool Wear Failure is generated probabilistically in the dataset (randomly triggered
between 200–240 min). The crisp baseline uses a deterministic threshold (midpoint
220 min) as there is no exact equivalent. This is acknowledged as a limitation of
the baseline.

### 5. RNF not modelled in baseline

Random Failure (0.1% chance) cannot be encoded in any deterministic rule. The
crisp baseline will miss all RNF instances. This is documented and expected.

---

## Download Instructions

The dataset file `ai4i2020.csv` must be downloaded separately:

1. Visit: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Click the **Download** button (509.8 KB zip file)
3. Extract the zip and move `ai4i2020.csv` to this `data/` directory
4. Run `python -m src.preprocessing` to generate the processed arrays

Alternatively, using the UCI Python package:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=601)
df = dataset.data.features.join(dataset.data.targets)
df.to_csv("data/ai4i2020.csv", index=False)
```
