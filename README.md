# Real-Time Fraud Detection

A machine-learning service that scores card transactions and flags the ones that look fraudulent. Trained on the well-known ULB credit-card dataset (284,807 real transactions, 492 of them fraud) and wrapped in a small Gradio app so you can try it in the browser.

## What it does

You give it a transaction (amount, hour of day, and a handful of the anonymised PCA signals) and it returns a fraud probability plus a plain verdict. The decision threshold is set at 0.20 instead of the usual 0.50 because in fraud you would rather catch a few extra suspicious ones than miss a real one.

## Data

The dataset comes from OpenML (fetch_openml("CreditCardFraudDetection")) - the transactions Kaggle/ULB released for research. It is heavily imbalanced: only about 0.17% of the rows are fraud, so accuracy on its own is meaningless here. Most of the columns (V1..V28) are already PCA components; the only untouched fields are Time and Amount.

## Features

On top of the raw columns I added a few that made the model noticeably better:

- hour - hour of day derived from Time
- log_amount - log of the amount (the amounts are very skewed)
- amount_z - how far the amount sits from the average
- gap_since_prev - seconds since the previous transaction (velocity)
- roll_count_50 - rolling count over a short window (velocity / burst signal)

## Model

A RandomForestClassifier with 200 trees and class_weight="balanced" to deal with the imbalance. Split was stratified 75/25.

## Results (held-out test set)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9449 |
| PR-AUC | 0.8429 |
| Precision (fraud) | 0.9468 |
| Recall (fraud) | 0.7236 |

Confusion matrix at the 0.20 threshold:

```
              predicted legit   predicted fraud
actual legit       71074               5
actual fraud          34              89
```

The most important features were V14, V10 and V12 - the same signals that show up in most public write-ups of this dataset, which is a nice sanity check.

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

The app expects fraud_model.joblib, fraud_scaler.joblib and feature_cols.json next to it. It opens on http://127.0.0.1:7860.

## Deploy on Hugging Face Spaces

1. Create a new Space and pick the Gradio SDK.
2. Upload app.py, requirements.txt and the three artifact files.
3. The Space builds itself and the demo goes live.

## Notes

This is a demo / learning project, not a production fraud system. Real fraud pipelines retrain constantly and use far more signals than a public dataset can offer.
