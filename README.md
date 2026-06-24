# 🏠 Home Credit Default Risk Prediction

## 📌 Project Overview

This project was developed as a complete end-to-end Machine Learning pipeline for the Home Credit Default Risk competition.

The objective is to predict whether a loan applicant will experience payment difficulties based on application information, credit history, previous loans, installment payments, POS cash behavior, and credit card activity.

The project covers the entire machine learning workflow, including:

* Business Understanding
* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Baseline Modeling
* Feature Selection
* Hyperparameter Optimization (Optuna)
* Cross Validation
* Model Explainability (SHAP)
* Threshold Optimization
* Ensemble Learning
* Model Persistence

---

## 🎯 Business Problem

Home Credit aims to expand financial inclusion by providing loans to individuals with limited or non-traditional credit histories.

The challenge is to accurately predict the probability of loan default while minimizing risk and maintaining access to credit.

This project builds a predictive model that estimates customer default risk using historical financial behavior.

---

## 📂 Dataset

This project uses the Home Credit Default Risk dataset from Kaggle.

Dataset Link:

https://www.kaggle.com/competitions/home-credit-default-risk

### Main Files

* application_train.csv
* application_test.csv
* bureau.csv
* bureau_balance.csv
* previous_application.csv
* POS_CASH_balance.csv
* installments_payments.csv
* credit_card_balance.csv

### Note

The dataset is not included in this repository because of its large size and Kaggle licensing restrictions.

Please download the dataset directly from Kaggle and place the files inside:

```text
data/
```

before running the notebook.

---

## 🏗 Project Structure

```text
Home-Credit-Default-Risk/

├── models/
│
├── notebooks/
│   ├── archive/
│   │   └── home_credit_default_risk_v1.ipynb
│   └── home_credit_default_risk.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Feature Engineering

Features were engineered from multiple auxiliary datasets:

### Bureau Features

Customer historical credit behavior.

### Previous Application Features

Information about previously submitted loan applications.

### Installments Payments Features

Repayment patterns and installment delays.

### POS Cash Features

Point-of-sale loan activity and delinquency behavior.

### Credit Card Features

Credit utilization, payment ratios, and delinquency indicators.

---

## 🤖 Models Evaluated

### LightGBM Baseline

Initial benchmark model.

### Feature Selection LightGBM

Reduced feature space using feature importance.

### Optuna Tuned LightGBM

Automated hyperparameter optimization.

### XGBoost

Gradient boosting benchmark.

### CatBoost

Categorical boosting benchmark.

### Voting Ensemble

Combination of multiple models for improved performance.

---

## 📊 Model Performance

| Model                      | ROC-AUC  |
| -------------------------- | -------- |
| Baseline LightGBM          | 0.775580 |
| Feature Selection LightGBM | 0.775606 |
| Optuna Tuned LightGBM      | 0.777266 |
| XGBoost                    | 0.777300 |
| CatBoost                   | 0.776691 |
| Voting Ensemble            | 0.779883 |

### Best Model

Voting Ensemble achieved the highest ROC-AUC score:

```text
ROC-AUC = 0.779883
```

---

## 🔍 Model Explainability

SHAP (SHapley Additive Explanations) was used to interpret model predictions.

Top influential features included:

* EXT_SOURCE_1
* EXT_SOURCE_2
* EXT_SOURCE_3
* POS_FUTURE_INSTALLMENTS_MEAN
* INST_DELAY_MAX
* INST_COUNT
* AMT_CREDIT
* DAYS_BIRTH

This helped explain how customer characteristics influenced default risk.

---

## 🎯 Threshold Optimization

The default classification threshold was optimized using validation data.

Best Threshold:

```text
0.65
```

Optimized Metrics:

```text
Precision : 0.2634
Recall    : 0.4302
F1 Score  : 0.3267
```

---

## 🔄 Cross Validation

5-Fold Stratified Cross Validation was performed.

Results:

```text
Mean ROC-AUC : 0.772707
Std ROC-AUC  : 0.002695
```

The low standard deviation indicates stable model performance across folds.

---

## 💾 Model Saving

The final model can be saved using:

```python
import joblib

joblib.dump(model, "models/model.pkl")
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/morsycoo/Home-Credit-Default-Risk
cd Home-Credit-Default-Risk
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

Open:

```text
notebooks/home_credit_default_risk.ipynb
```

and execute the notebook sequentially.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* XGBoost
* CatBoost
* Optuna
* SHAP
* Matplotlib
* Seaborn

---

## 👨‍💻 Author

Mahmoud Morsy

AI Engineer | Machine Learning | Deep Learning | Data Science
