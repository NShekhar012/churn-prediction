# Churn Prediction

Predicts customer churn on telecom data using scikit-learn and XGBoost.

## Requirements

- Python 3.x
- pip

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter pyarrow
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it to `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Running

Run the notebooks in order:

1. `01_eda.ipynb` — exploration and visualization
2. `02_preprocessing.ipynb` — cleaning and feature engineering
3. `03_modeling.ipynb` — training and evaluation

## Architecture

Three-notebook pipeline: EDA → preprocessing → modeling. Preprocessing uses a scikit-learn `ColumnTransformer` fit only on training data to avoid leakage. Models are wrapped in `Pipeline` objects so preprocessing and inference stay coupled.

## Performance

Best ROC-AUC: 0.846 (logistic regression and XGBoost), tested on 20% holdout. A naive baseline that predicts no churn for every customer scores 0.5 ROC-AUC and misses 100% of churners; the model catches the majority of actual churners while maintaining reasonable precision.

## Notes

TotalCharges type issue: the column ships as a string in the raw CSV despite containing numeric values. Blank entries (new customers with no billing history) cause the coercion to fail silently, producing nulls that have to be filled before training.

Estimator tuning: initially ran both Random Forest and XGBoost at 200 estimators, which caused XGBoost to hang for over 30 minutes. Reduced to 100 and 50 respectively to diagnose, then restored to 200 once the bottleneck was confirmed as runtime rather than a bug. At 200 trees XGBoost has four times as many sequential correction steps compared to 50, which meaningfully improves predictive power but introduces overfitting risk, mitigated here by setting a low learning rate (0.05) and capping tree depth at 4.