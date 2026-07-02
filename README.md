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

<div align="center">

⭐ **If you found this repository useful, consider giving it a star!**

</div>
