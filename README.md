# Loan Approval Classification

## Project Overview
This project predicts whether a loan application is approved or rejected using applicant, loan and credit-history features.

**Dataset:** Loan Approval Classification Dataset (Kaggle)  
**Rows:** 45,000  
**Columns:** 14  
**Target:** `loan_status` (`1 = approved`, `0 = rejected`)

## Workflow
1. Load and inspect the dataset
2. Check missing values and duplicates
3. Perform exploratory data analysis
4. Encode categorical features with One-Hot Encoding
5. Standardize numerical features
6. Split data into 80% training and 20% testing using stratification
7. Train:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
8. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
   - ROC-AUC
9. Compare model performance
10. Analyze Random Forest feature importance

## Results from the supplied dataset
Using `random_state=42` and a stratified 80/20 split:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8993 | 0.7888 | 0.7470 | 0.7673 | 0.9562 |
| Random Forest | **0.9298** | **0.8968** | 0.7730 | **0.8303** | **0.9747** |
| Gradient Boosting | 0.9279 | 0.8880 | 0.7730 | 0.8265 | 0.9744 |

Random Forest performed best overall on this split, particularly on accuracy, precision, F1 and ROC-AUC. Logistic Regression is useful as a simple baseline, while Gradient Boosting performs very similarly to Random Forest.

## Dataset Notes
The supplied CSV contains no missing values and no duplicate rows. The target is imbalanced, with approximately 77.8% rejected and 22.2% approved applications, so metrics beyond accuracy are important.

## Files
- `loan_data.csv` — dataset
- `loan_approval_classification.ipynb` — complete analysis and model notebook
- `README.md` — project documentation
- `requirements.txt` — Python dependencies

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook loan_approval_classification.ipynb
```

## Dataset Source
Kaggle: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

## Model Comparison
This project compares Logistic Regression, Random Forest, and Gradient Boosting models for loan approval classification

