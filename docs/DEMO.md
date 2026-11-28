# SC08 — Application Demo & UI Evidence

## 1. Project Overview

The Machine Predictive Maintenance & Fault Priority System is a Streamlit-based application that combines a Pure NumPy Artificial Neural Network (ANN) and a Mamdani Fuzzy Inference System (FIS).

It predicts machine failure probability and assigns a maintenance priority: LOW, MEDIUM, HIGH or URGENT.

## 2. Application Screenshots

The following screenshots show the application's interface and prediction results.

### Input Form
![Application Input Form](evidence/step6/01-input-form.png)

### Normal Prediction
![Normal Prediction](evidence/step6/02-normal-prediction.png)

### Urgent Prediction
![Urgent Prediction](evidence/step6/03-urgent-prediction.png)

### Fuzzy Rule Explanation
![Fuzzy Rule Explanation](evidence/step6/04-rule-explanation.png)

### Dark Mode
![Dark Mode](evidence/step6/05-dark-mode.png)

## 3. Example Predictions

| Scenario | Failure Probability | Severity | Priority |
|---|---:|---:|---|
| Normal | 2.16% | 15.53 | LOW |
| Urgent | 99.82% | 84.47 | URGENT |

## 4. How to Run the Application

### Requirements
- Python 3.9 or higher
- Project dependencies installed

### Installation

Clone the repository:

```bash
git clone https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system.git
cd Machine-Predictive-Maintenance-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 5. Application Status

This is a locally executed Streamlit application. It is not currently deployed as a public hosted service.

The screenshots above provide evidence of the tested application interface and prediction scenarios.

## 6. Project Repository

[SC08 — Machine Predictive Maintenance & Fault Priority System](https://github.com/mishra009shivam/Machine-Predictive-Maintenance-system)
