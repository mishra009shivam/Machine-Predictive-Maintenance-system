# Data Quality & Preprocessing Report — SC08

## 1. Dataset Source
- **Repository:** UCI Machine Learning Repository (Dataset ID: 601)
- **Dataset Title:** AI4I 2020 Predictive Maintenance Dataset
- **URL:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Citation:** Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. In *Proceedings of the Third International Conference on Artificial Intelligence for Industries (AI4I 2020)*, IEEE.
- **Local File Path:** `data/raw/ai4i2020.csv`

## 2. Dataset License
- **License Type:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License URL:** https://creativecommons.org/licenses/by/4.0/

## 3. Number of Rows
- **Total Records:** 10,000 operational observations.

## 4. Number of Columns
- **Total Raw Columns:** 12 columns in UCI API release (14 columns in zip archive containing `UDI` / `Product ID`).
- **Feature Schema:**
  - 1 Categorical feature: `Type` (`L`, `M`, `H`)
  - 5 Continuous numeric features: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
  - 1 Primary Target: `Machine failure` (binary 0/1)
  - 5 Sub-failure modes: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

## 5. Missing-Value Count by Column

| Column Name | Data Type | Missing Count | Missing Percentage |
|---|---|---|---|
| `Type` | object | 0 | 0.00% |
| `Air temperature [K]` | float64 | 0 | 0.00% |
| `Process temperature [K]` | float64 | 0 | 0.00% |
| `Rotational speed [rpm]` | int64 | 0 | 0.00% |
| `Torque [Nm]` | float64 | 0 | 0.00% |
| `Tool wear [min]` | int64 | 0 | 0.00% |
| `Machine failure` | int64 (Target) | 0 | 0.00% |
| `TWF` | int64 | 0 | 0.00% |
| `HDF` | int64 | 0 | 0.00% |
| `PWF` | int64 | 0 | 0.00% |
| `OSF` | int64 | 0 | 0.00% |
| `RNF` | int64 | 0 | 0.00% |

**Total Missing Values Across Dataset:** 0

## 6. Duplicate-Row Count
- **Duplicate Rows:** 0 (all 10,000 rows represent unique feature combinations).

## 7. Numeric Feature Ranges (Full Dataset)

| Feature | Min | Max | Mean | Std |
|---|---|---|---|---|
| `Air temperature [K]` | 295.30 | 304.50 | 300.0049 | 2.0003 |
| `Process temperature [K]` | 305.70 | 313.80 | 310.0056 | 1.4837 |
| `Rotational speed [rpm]` | 1168.00 | 2886.00 | 1538.7761 | 179.2841 |
| `Torque [Nm]` | 3.80 | 76.60 | 39.9869 | 9.9689 |
| `Tool wear [min]` | 0.00 | 253.00 | 107.9510 | 63.6541 |

## 8. Categorical Value Counts

| Category (`Type`) | Count | Percentage |
|---|---|---|
| `L` (Low quality variant) | 6,000 | 60.00% |
| `M` (Medium quality variant) | 2,997 | 29.97% |
| `H` (High quality variant) | 1,003 | 10.03% |
| **Total** | **10,000** | **100.00%** |

## 9. Machine-Failure Class Counts and Percentages

| Label (`Machine failure`) | Description | Count | Percentage |
|---|---|---|---|
| `0` | Healthy / Operational | 9,661 | 96.61% |
| `1` | Machine Failure | 339 | 3.39% |
| **Total** | | **10,000** | **100.00%** |

### Breakdown of Sub-failure Modes
- **TWF (Tool Wear Failure):** 46 instances
- **HDF (Heat Dissipation Failure):** 115 instances
- **PWF (Power Failure):** 95 instances
- **OSF (Overstrain Failure):** 98 instances
- **RNF (Random Failure):** 19 instances

*(Note: Sum exceeds 339 because several failures trigger multiple failure modes simultaneously).*

## 10. Train/Test Split Counts
- **Split Ratio:** 80% Training / 20% Test (Stratified by `Machine failure`)
- **Training Set (`X_train.npy`, `y_train.npy`):** 8,000 rows
  - Healthy (`0`): 7,729 (96.61%)
  - Failure (`1`): 271 (3.39%)
- **Test Set (`X_test.npy`, `y_test.npy`):** 2,000 rows
  - Healthy (`0`): 1,932 (96.60%)
  - Failure (`1`): 68 (3.40%)

## 11. Train/Validation Split Information
- Within the ANN training pipeline (`src/ann.py`), the 8,000-sample training set is split 90/10 into:
  - **Training Partition:** 7,200 samples
  - **Validation Partition:** 800 samples
- Used for validation loss tracking, early stopping (patience = 15 epochs), and validation threshold tuning ($\tau = 0.90$).
- The 2,000-sample test set remains untouched during all training and parameter fitting steps.

## 12. Selected ANN Features (7 Canonical Inputs)
1. `Air temperature [K]` (Numeric)
2. `Process temperature [K]` (Numeric)
3. `Rotational speed [rpm]` (Numeric)
4. `Torque [Nm]` (Numeric)
5. `Tool wear [min]` (Numeric)
6. `Type_M` (Binary dummy: 1 if Type is M, 0 otherwise)
7. `Type_H` (Binary dummy: 1 if Type is H, 0 otherwise)

*(Note: Product `Type_L` serves as reference category when `Type_M = 0` and `Type_H = 0` to prevent dummy variable trap).*

## 13. Excluded Identifier Fields
- `UDI` / `UID`: Integer sequence (1–10,000) representing row index.
- `Product ID`: String concatenation of product type and serial number (e.g. `L47180`).
- **Reason for Exclusion:** Pure administrative identifiers that carry no physical predictive signal. Retaining them would allow the neural network to memorize specific rows instead of learning physical failure relationships.

## 14. Excluded Failure-Mode Fields
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- **Reason for Exclusion:** These 5 sub-failure columns represent component breakdowns that constitute the target label itself (`Machine failure = TWF | HDF | PWF | OSF | RNF`). Passing sub-targets as inputs to the ANN would cause direct **data leakage** (allowing the ANN to trivially predict failure with 100% accuracy without learning from physical sensor parameters).

## 15. Normalization Method
- **Method:** Min-Max Normalization to range $[0, 1]$.
- **Formula:**
  $$x_{\text{scaled}} = \frac{x - x_{\text{train\_min}}}{x_{\text{train\_max}} - x_{\text{train\_min}}}$$
- **Data Leakage Prevention:** Min-Max parameters ($x_{\text{train\_min}}, x_{\text{train\_max}}$) are computed **only** from the 8,000 training samples and stored in `data/processed/norm_params.json`. The same parameters are subsequently applied to the 2,000 test set samples.

### Saved Training Normalization Parameters
- `Air temperature [K]`: Min = 295.3 K, Max = 304.5 K
- `Process temperature [K]`: Min = 305.7 K, Max = 313.8 K
- `Rotational speed [rpm]`: Min = 1168.0 rpm, Max = 2886.0 rpm
- `Torque [Nm]`: Min = 3.8 Nm, Max = 76.6 Nm
- `Tool wear [min]`: Min = 0.0 min, Max = 253.0 min

## 16. Random Seed
- `random_state = 42` is strictly enforced across all NumPy stratified splitting routines to guarantee exact reproducibility of train/test partitions.

## 17. Reproducibility Instructions
To regenerate all processed arrays and verify dataset integrity:
```bash
python -m src.prepare_data
```
Outputs generated in `data/processed/`:
- `X_train.npy` (8000, 7)
- `X_test.npy` (2000, 7)
- `y_train.npy` (8000,)
- `y_test.npy` (2000,)
- `norm_params.json` (scaling dictionary + class weights $w_{\text{pos}} = 28.5203$)

## 18. Important Assumptions & Limitations
1. **No Vibration Sensor:** The AI4I 2020 dataset does not contain a vibration sensor. `Torque [Nm]` is used as spindle torque (mechanical load indicator) and is **not** a vibration proxy.
2. **Synthetic Data Origin:** The AI4I 2020 dataset is synthetically generated according to mathematical failure rules specified by Matzka (2020).
3. **Class Imbalance:** Extreme positive class imbalance (~3.39% failures) is handled via class-weighted loss ($w_{\text{pos}} \approx 28.52$) during ANN training, preserving natural class distributions in evaluation sets.
4. **TWF Probabilistic Noise:** Tool Wear Failure contains intentional probabilistic variance in raw data generation (failing randomly between 200–240 min).
