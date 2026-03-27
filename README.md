# 📊 Regression & Customer Churn Prediction — ML Project

A machine learning project exploring **nonlinear regression techniques** and **classification models** to understand complex data patterns and predict customer churn.

---

## 🚀 Overview
This project is divided into two main parts:
- 📈 **Nonlinear Regression Analysis**
- 📉 **Customer Churn Prediction (Classification)**

We study how model complexity and regularization affect performance and generalization.

---

## 🎯 Objectives
- Analyze **bias vs variance trade-off**
- Compare **different regression models**
- Build a **robust churn prediction system**
- Evaluate models using proper metrics (ROC, AUC, etc.)

---

## 📊 Part 1 — Nonlinear Regression

### 🔹 Polynomial Ridge Regression
- Polynomial features up to degree 9  
- Ridge regularization applied  
- Dataset: 25 synthetic data points  

#### 🔧 Hyperparameter Tuning
- λ values tested: `0, 0.001, 0.01, 0.1, 1`

#### 📈 Key Results
- λ = 0 → ❌ Overfitting  
- λ = 0.01 → ✅ Best generalization  
- λ ≥ 0.1 → ❌ Underfitting  

👉 Insight: Proper regularization balances model complexity.

---

### 🔹 RBF (Radial Basis Function) Regression
- Gaussian basis functions  
- Centers tested: `1, 5, 10, 50`

#### 📈 Results
| RBF Count | Behavior |
|----------|--------|
| 1        | Underfitting |
| 5        | Good fit |
| 10       | Best generalization |
| 50       | Overfitting |

#### 📊 R² Scores
- 1 → 0.0189  
- 5 → 0.3121  
- 10 → **0.9699**  
- 50 → 1.0000 (overfit)

👉 Insight: Increasing model capacity improves fit but risks overfitting.

---

## 📉 Part 2 — Customer Churn Prediction

### 🧠 Model: Logistic Regression
- Linear + Polynomial features  
- Degrees tested: `2, 5, 9`

---

### 🔧 Data Preprocessing
- Missing values → Median imputation  
- Outliers → Capped (1%–98%)  
- Scaling → Min-Max normalization  

---

### 📊 Data Split
- Train: 2500  
- Validation: 500  
- Test: 500  

---

### 📈 Results

| Model | Performance |
|------|-----------|
| Degree 2 | Underfitting |
| Degree 5 | ✅ Best |
| Degree 9 | Overfitting |

👉 **Best Model:** Polynomial Logistic Regression (Degree 5)

---

### 📊 ROC Curve
- **AUC = 0.9853**  

👉 Excellent separation between churn and non-churn customers.

---

## 🔍 Key Insights
- Regularization is critical for controlling overfitting  
- Model complexity must match dataset size  
- Validation set is essential for model selection  
- Polynomial features significantly improve logistic regression  

---

## ⚙️ Tech Stack
- Python  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---
🔮 Future Work
Try more advanced models (XGBoost, Neural Networks)
Feature engineering improvements
Cross-validation
Hyperparameter optimization
