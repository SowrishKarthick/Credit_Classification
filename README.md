# Credit Risk Classification – German Credit Dataset

## Overview

This project develops a supervised machine learning framework to predict borrower default risk using the Statlog (German Credit) dataset from the UCI Machine Learning Repository.

The objective is to evaluate multiple classification algorithms under class imbalance conditions and identify a model that balances predictive performance with interpretability — a key requirement in real-world credit scoring systems.

---

## Dataset

- **Source:** UCI Statlog (German Credit Data)
- **Observations:** 1,000 loan applicants
- **Features:** 20 predictive variables  
  - 13 categorical attributes  
  - 7 numerical attributes  
- **Target:**  
  - 0 = Good Credit  
  - 1 = Bad Credit  

Class distribution:
- 700 Good Credit
- 300 Bad Credit

The dataset represents a moderately imbalanced binary classification problem.

---

## Methodology

### Data Preprocessing

- One-hot encoding for categorical variables
- Target remapped to binary format (0/1)
- Stratified 70–30 train-test split
- Feature scaling using `StandardScaler`
- No missing values present

Pipeline:

---

## Class Imbalance Handling

To mitigate bias toward the majority class, the following resampling strategies were applied **only to the training set**:

- **Original (Baseline)**
- **RandomOverSampler**
- **SMOTE (Synthetic Minority Oversampling Technique)**

This ensured fair evaluation without data leakage.

---

## Models Evaluated

The following supervised classification models were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Support Vector Machine (RBF Kernel)
- K-Nearest Neighbors
- Naive Bayes
- Explainable Boosting Machine (EBM)

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC–AUC

---

## Key Results

### Best Performing Model

**Logistic Regression + RandomOverSampler**

- ROC–AUC: **0.808**
- Strong recall for minority (default) class
- Balanced precision–recall tradeoff
- High interpretability

This result demonstrates that interpretable linear models, when supported by appropriate resampling, can achieve competitive performance while maintaining transparency — a crucial property in financial credit scoring systems.

---

## Error Analysis

The confusion matrix analysis highlights:

- **False Positives (Type I Error):**  
  Risky applicants incorrectly classified as creditworthy  
  → Financially costly due to potential default

- **False Negatives (Type II Error):**  
  Creditworthy applicants incorrectly rejected  
  → Opportunity cost and lost business

The selected model maintains a strong balance between risk containment (high recall for bad credit) and lending efficiency.

---

## Key Insights

- Class imbalance significantly affects recall for default cases.
- Resampling improves minority-class detection without severe overfitting.
- Logistic Regression provides competitive performance with superior interpretability compared to complex ensemble models.
- In financial applications, model transparency can be as important as raw predictive power.

---

## Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib / Seaborn

---


