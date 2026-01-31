# Customer Attrition Modeling (R)

**Author:** J. Casey Brookshier  
**Tools:** R, caret, pROC, randomForest, gbm

---

## 📌 Project Overview

This project analyzes customer attrition using a real-world banking dataset.
Multiple machine learning models are developed and evaluated to identify
drivers of customer churn and predict attrition risk.

The goal is to compare traditional statistical models with modern
tree-based machine learning approaches using accuracy and AUC,
with special consideration for imbalanced data.

---

## 🧠 Models Implemented

- Logistic Regression (initial & refined)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Null (baseline) model
- Random Forest
- Gradient Boosting Machine (GBM)

---

## 📊 Key Results (Test Set)

| Model | Accuracy | AUC |
|------|---------|-----|
| Logistic Regression | ~0.90 | ~0.92 |
| Refined Logistic Regression | ~0.90 | ~0.91 |
| KNN | ~0.85 | ~0.54 |
| Naive Bayes | ~0.88 | ~0.86 |
| Random Forest | ~0.96 | ~0.99 |
| Gradient Boosting | **~0.97** | **~0.99** |

Tree-based models significantly outperform linear models,
highlighting nonlinear customer behavior patterns.

---

## 🔑 Business Insights

- Declining transaction activity is the strongest indicator of attrition
- Inactive accounts exhibit sharply higher churn probability
- Ensemble models capture complex interactions missed by linear approaches

---

## 📁 Repository Structure

```text
├── run_analysis.r     # Main analysis script (one-click run)
├── data/              # Input data (CSV)
├── figures/           # Saved plots and figures
├── metrics/           # Model metrics & comparison tables
└── output/            # Additional generated artifacts


## How to Run
```bash
git clone git@github.com:CrassSax7/customer-attrition-modeling.git
cd customer-attrition-modeling
Rscript run_analysis.r


