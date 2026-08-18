1. Project Overview

I developed a Customer Churn Prediction system using machine learning to identify customers who are likely to discontinue a company's service. The problem was formulated as a binary classification task where Churn = 1 represents a customer who is likely to leave and Churn = 0 represents a customer who is likely to stay.

The dataset contains 1,000 customer records and features such as age, gender, tenure, monthly charges, contract type, internet service, total charges, and technical support.

I first performed exploratory data analysis to understand customer behavior and identify relationships between the features and churn. I handled missing values using a preprocessing pipeline and encoded categorical variables using one-hot encoding. I removed CustomerID because it is an identifier and has no predictive meaning.

I used a stratified train-test split because the target variable was highly imbalanced, with approximately 88.3% churn and 11.7% non-churn customers.

I initially trained Logistic Regression as a baseline model. It achieved approximately 93.5% accuracy, 100% precision, 92.66% recall, 96.19% F1-score, and 99.26% ROC-AUC.

I then trained a Random Forest classifier, performed 5-fold stratified cross-validation and hyperparameter tuning using GridSearchCV. The Random Forest achieved approximately 100% ROC-AUC during cross-validation and 100% performance on the held-out test set.

I also saved the trained model using Joblib and developed a prediction pipeline that takes customer information and returns churn probability and a risk category such as Low, Medium, or High.

Finally, I worked toward integrating SHAP explainability and Streamlit so that the application can not only predict churn but also explain the major factors influencing a prediction.

2. Project Architecture

Know this diagram extremely well:

                    Customer Dataset
                          │
                          ▼
                 Data Understanding
                          │
                          ▼
                       EDA
                          │
                          ▼
                  Data Preprocessing
                          │
             ┌────────────┴────────────┐
             │                         │
      Numerical Features       Categorical Features
             │                         │
       Imputation + Scaling      Imputation + Encoding
             │                         │
             └────────────┬────────────┘
                          ▼
                  Train/Test Split
                          │
                          ▼
                Logistic Regression
                          │
                          ▼
                     Baseline
                          │
                          ▼
                  Random Forest
                          │
                          ▼
                 Cross Validation
                          │
                          ▼
                Hyperparameter Tuning
                          │
                          ▼
                  Final Model
                          │
                          ▼
                 Churn Probability
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Risk Level               SHAP Explanation
             │                         │
             └────────────┬────────────┘
                          ▼
                    Streamlit App
