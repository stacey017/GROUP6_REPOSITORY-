# GROUP6_REPOSITORY-
Machine Learning Lab Practical Group Assignment 
# G6 | MediCare Diagnostics
## Heart Disease Risk Screening using Machine Learning

> **BCDA 5P126 - Machine Learning Lab Mini-Project**

---

## 1. Project Overview

**MediCare Diagnostics** is an educational machine learning project focused on
risk screening for coronary heart disease.

The objective is to develop a machine learning classification system that
analyzes patient-related clinical attributes and predicts whether a patient
is at risk of coronary heart disease.

This project follows an end-to-end Enterprise Machine Learning workflow,
covering data acquisition, preprocessing, exploratory data analysis,
feature engineering, model development, evaluation, interpretation, and
business-oriented visualization.

> **Important:** This project is strictly for educational risk screening
> purposes and is NOT a clinical diagnostic system.

---

## 2. Client Persona

### Client
**MediCare Diagnostics**

### Domain
**Clinical Health / Healthcare Analytics**

### Business Problem

Healthcare organizations need reliable analytical tools to identify
patients who may require further medical evaluation.

The objective of this project is to build a machine learning-based
educational screening system that predicts coronary heart disease risk
from patient health attributes.

The system is designed to support early risk identification while
highlighting the importance of minimizing false-negative predictions.

---

## 3. Problem Statement

Build a binary classification model using the UCI Heart Disease dataset
to predict the presence or absence of coronary heart disease.

The machine learning pipeline should:

- Acquire and validate the dataset
- Clean missing and invalid values
- Perform exploratory data analysis
- Engineer and select relevant features
- Apply leakage-free preprocessing
- Train baseline and machine learning models
- Evaluate model performance
- Analyze classification errors
- Integrate predictions into business-oriented analytics
- Present key findings through an interactive dashboard

---

## 4. Dataset

### Dataset Name
**UCI Heart Disease Dataset**

### Dataset Type
Clinical / Medical Classification Dataset

### Target
Heart disease presence / absence

### Source
UCI Machine Learning Repository

The dataset is used as specified in the Machine Learning Lab project
guide for Group 6.

### Dataset-Specific Considerations

- The dataset is relatively small.
- Missing values and data quality must be handled carefully.
- Feature preprocessing must prevent data leakage.
- Model limitations must be clearly documented.
- Results should not be interpreted as medical diagnosis.

---

## 5. Primary Evaluation Metrics

The primary metrics specified for Group 6 are:

### Recall (Sensitivity)

Recall is particularly important because false-negative predictions,
where a patient at risk is classified as low-risk, represent an important
screening concern.

### Log-Loss

Log-Loss evaluates the quality of predicted probabilities and penalizes
overconfident incorrect predictions.

### Additional Metrics

Where applicable, the project also evaluates:

- Accuracy
- Precision
- F1-Score
- Confusion Matrix
- ROC-AUC

The project does not rely on accuracy alone when interpreting model
performance.

---

## 6. Machine Learning Workflow

The project follows an end-to-end machine learning lifecycle:

```text
Problem Definition
        ↓
Dataset Acquisition
        ↓
Data Validation & Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Leakage-Free Preprocessing
        ↓
Baseline Model
        ↓
Machine Learning Models
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Error Analysis
        ↓
Prediction / Inference
        ↓
Dashboard
        ↓
Business Insights & Recommendations
