# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using Python and scikit-learn. The goal is to identify customers who may be at risk of leaving and provide insights that can support customer-retention strategies.

## Project Overview

- **Problem type:** Binary classification
- **Dataset:** Telco Customer Churn
- **Total records:** 7,043 customers
- **Target:** `Churn` (`Yes` / `No`)
- **Churn rate:** 26.54%
- **Best evaluated model:** Logistic Regression
- **Selected classification threshold:** 0.35

## Results

The Logistic Regression model was evaluated on a stratified 20% hold-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 76.51% |
| Precision | 54.43% |
| Recall | 70.59% |
| F1-score | 61.47% |
| ROC-AUC | 84.19% |
| Average Precision | 63.34% |

The classification threshold was selected using cross-validation predictions on the training set. Compared with the default 0.50 threshold, 0.35 identifies more churners, but it also produces more false-positive predictions. The chosen threshold prioritizes finding at-risk customers; the best threshold may differ depending on retention-campaign costs and business goals.

## Key Findings

Permutation importance for the Logistic Regression pipeline identified these leading predictive features:

1. Tenure
2. Internet service type
3. Contract type
4. Monthly charges
5. Total charges

Exploratory analysis also found higher churn rates among month-to-month customers and customers with shorter tenure. These are predictive associations, not evidence that a feature causes churn.

## Workflow

1. Load and inspect the dataset.
2. Clean data and convert `TotalCharges` to numeric.
3. Explore churn patterns using visualizations and summary statistics.
4. Perform statistical analysis of customer characteristics and churn.
5. Split the data into stratified training and test sets.
6. Build preprocessing pipelines for numeric and categorical variables.
7. Compare a Dummy Classifier, Logistic Regression, and Random Forest.
8. Evaluate classification metrics and confusion matrix.
9. Tune the classification threshold using training-only cross-validation predictions.
10. Plot ROC and Precision–Recall curves.
11. Analyze permutation feature importance.
12. Save and test the trained pipeline.

## Project Structure

Adapt the paths below to match your actual notebook locations.

```text
Customer Churn/
├── data/
│   └── customer-churn-data.csv
├── models/
│   └── customer_churn_logistic_regression.pkl
├── 01_data_cleaning.ipynb
├── 02_eda.ipynb
├── 03_statistical_analysis.ipynb
├── 04_predictive_modeling.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset at:

```text
data/customer-churn-data.csv
```

Update notebook file paths if your local directory differs. Do not commit the dataset unless its license permits redistribution.

### 5. Run the notebooks

Open Jupyter Notebook or JupyterLab and run the notebooks in order, from data preparation and EDA through statistical analysis and predictive modeling.

## Making a Prediction

The saved file in `models/` contains the preprocessing pipeline and Logistic Regression estimator. Load it only if you trust its source. The project uses a classification threshold of `0.35`; this threshold is separate from the saved pipeline and must be applied when converting predicted probabilities into class labels.

```python
import joblib
import pandas as pd

model = joblib.load(
    "models/customer_churn_logistic_regression.pkl"
)

# customer_data must contain the same input columns used during training,
# excluding the target column `Churn` and identifier `customerID`.
probabilities = model.predict_proba(customer_data)[:, 1]
predictions = (probabilities >= 0.35).astype(int)
```

Here, `1` means predicted churn and `0` means predicted non-churn.

## Limitations

- Model performance may change when applied to new customers or future periods.
- False positives and false negatives are both possible.
- The selected threshold reflects a precision–recall trade-off and should be validated against business costs.
- Feature importance describes predictive contribution, not causality.
- The model should support, not replace, informed retention decisions.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib
- scikit-learn
- Statsmodels
- SciPy
- Jupyter Notebook
- Joblib

## Future Improvements

- Compare additional models and tune hyperparameters with cross-validation.
- Evaluate calibration and business-specific threshold/cost trade-offs.
- Use explainability methods to support individual predictions.
- Monitor performance and data drift on new customer data.
- Validate model performance on a later time period if timestamps are available.

## Disclaimer

This project is for educational and portfolio purposes. Predictions indicate estimated risk, not certainty, and should not be treated as causal explanations.
