# Telco Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline built to identify high-risk churn profiles in telecom customer data. This project uses scikit-learn pipelines and XGBoost, optimizing for Recall to minimize false negatives (missed churners) for the customer retention team.

## Project Structure
- `01_eda.ipynb`: Exploratory analysis and feature target distribution checking.
- `02_preprocessing.ipynb`: Data cleaning, train-test splitting, and Feature Pipeline definition.
- `03_modeling.ipynb`: Model training, hyperparameter tuning, and threshold evaluation.
- `data/`: Local directory for raw and processed datasets (excluded from git via .gitignore).

## Environment Setup
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter pyarrow

Download the dataset from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it to data/WA_Fn-UseC_-Telco-Customer-Churn.csv.

Run the notebooks in order:

01_eda.ipynb — exploration and visualization
02_preprocessing.ipynb — cleaning and feature engineering
03_modeling.ipynb — training and evaluation


## Pipeline Architecture & Design Choices
To ensure strict prevention of data leakage, all feature transformations (Standard Scaling for numeric variables, One-Hot Encoding for categories) are contained within a scikit-learn `ColumnTransformer`. This transformer is fit **exclusively on the training split** and is combined directly into individual classifier `Pipeline` objects. This keeps our data transformations and execution routines tightly coupled for reproducible validation and clean inference.

## Evaluation & Strategic Focus
- **Baseline Churn State:** ~26.5% overall churn rate, signaling a mild class imbalance.
- **Top Metrics:** Logistic Regression and XGBoost converged at a tie with a **validation ROC-AUC of ~0.846**.
- **Business Alignment:** For customer retention campaigns, optimizing for **Recall** is prioritized over raw accuracy. Missing an actual churner (False Negative) is significantly more expensive than reaching out to a customer who wasn't planning to leave (False Positive).

## Engineering Trade-offs & Roadblocks
- **The `TotalCharges` Type Quirks:** The raw column drops in as a string/object data type. This occurs because brand-new customers with a tenure of 0 months have blank whitespace spaces instead of numbers. Coercing this introduces NaN values, which I mitigated by tracking and applying the training median.
- **Handling Imbalance Safely:** Instead of relying on oversampling methods like SMOTE (which can synthetically overcomplicate a dataset of this size and distort precision boundaries), I enforced stratified train-test splitting and hyperparameter constraints (`max_depth=4` and low learning rates in XGBoost) to curb overfitting.