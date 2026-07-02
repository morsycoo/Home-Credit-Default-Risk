# 🏠 Home Credit Default Risk Prediction

<div align="center">

# Credit Risk Intelligence System

**An End-to-End Machine Learning & MLOps-ready Credit Risk Prediction System built with LightGBM, FastAPI, Docker, and Pytest.**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green?logo=fastapi)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-success)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Pytest](https://img.shields.io/badge/Pytest-Tested-success?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

</div>

---

# 📖 Table of Contents

- Project Overview
- Business Problem
- Project Highlights
- System Architecture
- Project Structure
- Dataset
- Machine Learning Pipeline
- Feature Engineering
- Models
- Model Performance
- Threshold Optimization
- Explainability (SHAP)
- REST API
- Docker
- Installation
- Running the Project
- API Examples
- Testing
- Logging
- Technologies Used
- Future Improvements
- License
- Author

---

# 📌 Project Overview

Home Credit provides loans to customers who often have little or no traditional credit history.

The goal of this project is to predict whether a customer will experience payment difficulties using historical financial information, previous loans, bureau reports, installment behavior, POS cash balances, and credit card activity.

Instead of building only a machine learning notebook, this repository contains a complete production-style machine learning application.

The project includes:

- Complete data preprocessing pipeline
- Feature engineering
- Feature selection
- Hyperparameter optimization
- Threshold optimization
- Model persistence
- Production inference pipeline
- FastAPI REST API
- Docker containerization
- Logging system
- Automated API tests
- Modular project architecture

This repository is designed to resemble the workflow used in real-world machine learning projects.

---

# 🎯 Business Problem

Financial institutions lose millions of dollars every year because of loan defaults.

Rejecting every risky applicant reduces profit, while approving everyone increases financial losses.

The objective is therefore to estimate the probability that a customer will default on a loan.

The model allows lenders to:

- Reduce financial risk
- Improve approval decisions
- Increase profitability
- Expand financial inclusion
- Support automated decision making

---

# ⭐ Project Highlights

✔ End-to-End Machine Learning Pipeline

✔ Production-ready FastAPI Application

✔ Docker Container

✔ Modular Python Codebase

✔ Feature Engineering Pipeline

✔ Feature Selection

✔ Optuna Hyperparameter Optimization

✔ Threshold Optimization

✔ SHAP Explainability

✔ Logging System

✔ REST API

✔ Swagger Documentation

✔ Automated Testing with Pytest

✔ Production Model Loading

✔ Ready for Cloud Deployment

---

# 🏗 System Architecture

```text
                     Client
                        │
                        ▼
                 FastAPI REST API
                        │
                        ▼
              Request Validation
                 (Pydantic Models)
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
             Feature Engineering
                        │
                        ▼
          Selected Features Loader
                        │
                        ▼
            Trained LightGBM Model
                        │
                        ▼
          Probability Prediction
                        │
                        ▼
        Threshold Optimization
                        │
                        ▼
               JSON Response
```

---

# 📂 Project Structure

```text
Home-Credit-Default-Risk/

│
├── api/
│   └── app.py
│
├── artifacts/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│   └── app.log
│
├── models/
│   ├── home_credit_optuna_lgbm.pkl
│   ├── selected_features.pkl
│   ├── best_threshold.pkl
│   └── final_model_results.csv
│
├── notebooks/
│   └── home_credit_default_risk.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── artifacts.py
│   ├── inference.py
│   ├── model.py
│   ├── logger.py
│   ├── schema.py
│   ├── config.py
│   └── utils.py
│
├── tests/
│   ├── conftest.py
│   └── test_api.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Why This Project?

Most machine learning repositories stop after training a model inside a notebook.

This project goes much further by transforming the trained model into a production-ready application.

It demonstrates practical machine learning engineering skills including:

- Software engineering
- API development
- Docker
- Model deployment
- Testing
- Logging
- Modular architecture
- Production inference

This makes the project suitable for showcasing Machine Learning Engineering skills rather than only Data Science experimentation.

---

# 🎯 Main Features

### Machine Learning

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Feature Selection
- LightGBM
- XGBoost
- CatBoost
- Optuna
- SHAP
- Cross Validation
- Threshold Optimization

---

### Backend

- FastAPI
- Pydantic Validation
- Swagger UI
- REST API
- JSON Responses

---

### Production

- Docker
- Logging
- Joblib Model Loading
- Automated Testing
- Modular Source Code

---

# 📈 Project Workflow

```text
Raw CSV Files
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Threshold Optimization
        │
        ▼
Model Saving
        │
        ▼
Inference Pipeline
        │
        ▼
FastAPI
        │
        ▼
Docker
        │
        ▼
REST API
```

---

# 🎯 Repository Goals

This project aims to demonstrate practical knowledge of:

- Machine Learning
- Model Deployment
- API Development
- Software Engineering
- MLOps Foundations
- Production Machine Learning

rather than only model training.

---

# 📊 Dataset

This project uses the **Home Credit Default Risk** dataset provided by Kaggle.

The competition focuses on predicting whether a client will experience payment difficulties.

## Dataset Source

https://www.kaggle.com/competitions/home-credit-default-risk

---

## Main Files

| File | Description |
|------|-------------|
| application_train.csv | Training dataset |
| application_test.csv | Test dataset |
| bureau.csv | Previous credits from other institutions |
| bureau_balance.csv | Monthly bureau balances |
| previous_application.csv | Previous loan applications |
| POS_CASH_balance.csv | Point-of-sale loan history |
| installments_payments.csv | Installment payment history |
| credit_card_balance.csv | Credit card monthly balance |

---

## Dataset Size

| Property | Value |
|----------|------:|
| Training Samples | 307,511 |
| Test Samples | 48,744 |
| Original Features | 122 |
| Engineered Features | 700+ |

---

## Dataset Note

The dataset is **not included** in this repository because:

- Kaggle licensing restrictions
- Large file size

Download it manually and place the CSV files inside:

```text
data/raw/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Home-Credit-Default-Risk.git
```

Move into the project

```bash
cd Home-Credit-Default-Risk
```

Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Main libraries

- Python 3.13
- NumPy
- Pandas
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- SHAP
- FastAPI
- Uvicorn
- Joblib
- Pytest

---

# ▶ Running the Notebook

Open

```text
notebooks/home_credit_default_risk.ipynb
```

Execute all cells sequentially.

This notebook performs:

- Data Loading
- Cleaning
- EDA
- Feature Engineering
- Model Training
- Hyperparameter Optimization
- Model Evaluation
- Threshold Optimization
- Model Saving

---

# 🚀 Running the API

Start the FastAPI server

```bash
python -m uvicorn api.app:app --reload
```

The API will be available at

```text
http://127.0.0.1:8000
```

Swagger UI

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON

```text
http://127.0.0.1:8000/openapi.json
```

---

# 🐳 Docker

Build the Docker image

```bash
docker build -t credit-risk-api .
```

Run the container

```bash
docker run -p 8000:8000 credit-risk-api
```

Verify the running container

```bash
docker ps
```

View logs

```bash
docker logs <container_id>
```

Stop container

```bash
docker stop <container_id>
```

Remove container

```bash
docker rm <container_id>
```

---

# 🌐 REST API

## Base URL

```text
http://127.0.0.1:8000
```

---

## Health Check

Endpoint

```http
GET /
```

Example Response

```json
{
    "message": "Credit Risk Intelligence API is running."
}
```

---

## Predict Credit Risk

Endpoint

```http
POST /predict
```

Content-Type

```text
application/json
```

---

# Example Request

```json
{
    "features": {
        "EXT_SOURCE_1": 0.083,
        "EXT_SOURCE_2": 0.622,
        "EXT_SOURCE_3": 0.746,
        "PAYMENT_RATE": 0.045,
        "CREDIT_TO_ANNUITY_RATIO": 22.0,
        "BUREAU_DAYS_CREDIT_MAX": -500,
        "POS_FUTURE_INSTALLMENTS_MEAN": 2.3,
        "ANNUITY_INCOME_RATIO": 0.18,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000
    }
}
```

---

# Example Response

```json
{
    "prediction": 1,
    "risk_score": 0.6801,
    "threshold": 0.65
}
```

---

## Response Fields

| Field | Description |
|------|-------------|
| prediction | Final prediction (0 or 1) |
| risk_score | Probability of default |
| threshold | Decision threshold |

---

## Status Codes

| Code | Description |
|-----:|-------------|
| 200 | Successful prediction |
| 400 | Invalid request |
| 422 | Validation error |
| 500 | Internal server error |

---

# 📖 Interactive Documentation

FastAPI automatically generates interactive API documentation.

Swagger UI

```text
http://127.0.0.1:8000/docs
```

ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 📷 API Preview

> Add screenshots inside an `assets/` directory.

Example:

```text
assets/
│
├── swagger_home.png
├── swagger_predict.png
├── prediction_response.png
└── docker_running.png
```

Then display them:

```md
![Swagger](assets/swagger_home.png)

![Prediction](assets/prediction_response.png)

![Docker](assets/docker_running.png)

---

# 🧠 Machine Learning Pipeline

The project follows a complete end-to-end machine learning workflow inspired by production ML systems.

```text
Business Understanding
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Baseline Model
        │
        ▼
Model Comparison
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Threshold Optimization
        │
        ▼
Model Explainability
        │
        ▼
Model Serialization
        │
        ▼
Inference Pipeline
        │
        ▼
FastAPI Deployment
        │
        ▼
Docker Container
```

---

# 🧹 Data Preprocessing

Several preprocessing techniques were applied before training the models.

### Missing Values

- Median Imputation
- Zero Filling
- Domain-Specific Imputation

---

### Data Cleaning

- Duplicate removal
- Invalid value handling
- Infinite value replacement
- Data consistency checks

---

### Feature Encoding

Categorical variables were encoded using:

- Label Encoding
- Frequency Encoding (where applicable)

---

### Feature Scaling

Tree-based algorithms (LightGBM, XGBoost, CatBoost) do not require feature scaling, therefore no standardization was applied during training.

---

# ⚙ Feature Engineering

A large number of predictive features were engineered from multiple auxiliary datasets.

The goal was to transform raw financial records into meaningful customer-level statistics.

---

## Bureau Features

Historical credit information was aggregated into customer-level metrics.

Examples include:

- Number of previous loans
- Average credit duration
- Maximum overdue days
- Credit debt ratios
- Credit end dates

---

## Previous Applications

Features extracted include:

- Approval rate
- Refusal rate
- Previous loan amount
- Previous credit amount
- Previous annuity
- Previous application timing

---

## Installment Payments

Payment behavior features such as:

- Payment delays
- Maximum delay
- Average delay
- Number of installments
- Late payment ratio

---

## POS Cash Balance

Generated features include:

- Remaining installments
- Future installments
- Average delinquency
- Contract status statistics

---

## Credit Card Balance

Extracted statistics include:

- Credit utilization
- Balance ratios
- Payment ratios
- Cash withdrawal ratios

---

## Application Features

Derived financial indicators including:

- Credit to Income Ratio
- Credit to Annuity Ratio
- Annuity to Income Ratio
- Employment Length
- Age
- Income per Family Member

---

# 📉 Feature Selection

Feature importance obtained from LightGBM was used to remove less informative variables.

Benefits:

- Reduced overfitting
- Faster inference
- Lower memory usage
- Better generalization

The final production model uses only the selected features stored in:

```text
models/selected_features.pkl
```

---

# 🤖 Models Evaluated

Several gradient boosting algorithms were evaluated.

| Model | Purpose |
|--------|---------|
| LightGBM | Baseline |
| Feature Selection LightGBM | Reduced feature space |
| Optuna Tuned LightGBM | Hyperparameter optimization |
| XGBoost | Benchmark |
| CatBoost | Benchmark |
| Voting Ensemble | Final comparison |

---

# 📊 Model Performance

| Model | ROC-AUC |
|--------|--------:|
| Baseline LightGBM | 0.775580 |
| Feature Selection LightGBM | 0.775606 |
| Optuna LightGBM | 0.777266 |
| XGBoost | 0.777300 |
| CatBoost | 0.776691 |
| Voting Ensemble | **0.779883** |

---

## Best Individual Model

The production API uses the **Optuna-tuned LightGBM** model because it provides an excellent balance between:

- Accuracy
- Inference Speed
- Memory Usage
- Deployment Simplicity

The serialized model is stored as:

```text
models/home_credit_optuna_lgbm.pkl
```

---

# ⚡ Hyperparameter Optimization

Hyperparameter tuning was performed using **Optuna**.

Optimized parameters included:

- Number of Trees
- Learning Rate
- Maximum Depth
- Feature Fraction
- Bagging Fraction
- Minimum Child Samples
- Regularization Parameters

This significantly improved model performance compared to the baseline model.

---

# 🎯 Threshold Optimization

Instead of using the default threshold (0.50), the optimal classification threshold was determined using validation data.

Default Threshold

```text
0.50
```

Optimized Threshold

```text
0.65
```

Benefits:

- Higher Precision
- Better Recall Balance
- Improved F1 Score
- Lower False Positives

The threshold is saved inside:

```text
models/best_threshold.pkl
```

---

# 📈 Cross Validation

A **5-Fold Stratified Cross Validation** strategy was used.

Results:

```text
Mean ROC-AUC : 0.772707

Standard Deviation : 0.002695
```

The low variance demonstrates that the model generalizes consistently across different validation folds.

---

# 🔍 Model Explainability

Model interpretability was performed using **SHAP (SHapley Additive Explanations)**.

SHAP was used to:

- Rank feature importance
- Explain individual predictions
- Interpret model behavior
- Increase model transparency

Top Important Features

- EXT_SOURCE_1
- EXT_SOURCE_2
- EXT_SOURCE_3
- PAYMENT_RATE
- CREDIT_TO_ANNUITY_RATIO
- INST_DELAY_MAX
- POS_FUTURE_INSTALLMENTS_MEAN
- AMT_CREDIT
- DAYS_BIRTH

These variables have the strongest influence on the predicted probability of default.

---

# 💾 Model Persistence

Production artifacts are serialized using **Joblib**.

Stored artifacts include:

```text
models/

├── home_credit_optuna_lgbm.pkl
├── selected_features.pkl
├── best_threshold.pkl
└── final_model_results.csv
```

During inference, these artifacts are loaded automatically by the prediction pipeline.

---

# 🔄 Inference Pipeline

```text
Incoming JSON
        │
        ▼
Pydantic Validation
        │
        ▼
Preprocessing
        │
        ▼
Feature Alignment
        │
        ▼
Selected Features
        │
        ▼
LightGBM Prediction
        │
        ▼
Threshold Optimization
        │
        ▼
JSON Response
```

<div align="center">

⭐ **If you found this repository useful, consider giving it a star!**

</div>
