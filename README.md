# Loan-Approval-Prediction

An end-to-end machine learning pipeline that predicts whether a loan application should be approved or rejected — automatically, accurately, and without human bias.

---

## Problem Definition

Banks receive thousands of loan applications daily. Manual review is slow, costly, and inconsistent. This project builds a classification model to automate and improve the loan approval process.

**Goal:** Predict `LoanApproved` (1 = Approved, 0 = Rejected) based on the applicant's financial and personal profile.

**Problem type:** Supervised Learning — Binary Classification

---

## Dataset

Source: Kaggle — Loan Risk Prediction Dataset

| Feature | Type | Description |
|---|---|---|
| Age | Integer | Applicant's age |
| Income | Float | Annual income |
| LoanAmount | Float | Requested loan amount |
| CreditScore | Float | Credit history score (300–850) |
| YearsExperience | Integer | Years of professional experience |
| Gender | Categorical | Male / Female |
| Education | Categorical | Bachelors / High School / Masters / PhD |
| City | Categorical | Chicago / Houston / New York / San Francisco |
| EmploymentType | Categorical | Salaried / Self-Employed / Unemployed |
| LoanApproved | Binary (Target) | 1 = Approved, 0 = Rejected |

**Size:** 5,000 rows — Class split: 77% Rejected / 23% Approved

---

## Project Pipeline

### 1. Data Cleaning
- Converted negative values in `Income` and `LoanAmount` to positive using `abs()`
- Filled missing values in `Income` and `CreditScore` with column mean
- Filled missing values in `Education` with mode
- Confirmed no duplicate rows

### 2. Exploratory Data Analysis (EDA)

**Key findings:**
- `CreditScore` is the strongest predictor of loan approval (r = 0.465)
- `Income` shows a moderate positive correlation (r = 0.19)
- `Age`, `LoanAmount`, and `YearsExperience` show near-zero correlation with the target
- Dataset is imbalanced: 3,849 Rejected vs 1,151 Approved (77% / 23%)

### 3. Data Preprocessing
- Label Encoding for categorical features (Gender, Education, City, EmploymentType)
- StandardScaler applied to normalize all numeric features
- 80/20 Train/Test Split

### 4. Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 86.5% | 75.3% | 59.6% | 66.5% |
| **Random Forest** | **96.5%** | **95.2%** | **88.9%** | **91.9%** |

**Best model: Random Forest** (100 trees, F1 = 0.9195)

**Cross-validation score (5-fold):** Mean = 97.05%, Std = 0.53%

### 5. Deployment

Interactive Streamlit web app — enter applicant details and get an instant loan approval prediction.

---

## How to Run

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Run the app:**
```
streamlit run app.py
```

**3. Open in browser:**
```
localhost:8501
```

---

## Project Structure

```
├── app.py
├── loan_model.pkl
├── loan_risk_prediction_dataset.csv
├── Loan_Approval_Prediction_System.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

---

## Results

The Random Forest model achieved **96.5% accuracy** on the test set:
- 765 out of 775 rejected loans correctly identified
- 200 out of 225 approved loans correctly identified
- Only 10 false positives and 25 false negatives

**Top predictors:** CreditScore → Income → LoanAmount

---

## Future Work

- Add debt-to-income ratio as a feature
- Deploy as a cloud API
- Integrate real banking data
- Explore deep learning models

---

## Team Members
1. **Alaa Samy Mahmoud**
2. **Basmala Ahmed Younis**
3. **Haneen Ahmed Omar**
4. **Noura Ashraf Abuelyazed**
5. **Ziad Ashraf Mohammed**
