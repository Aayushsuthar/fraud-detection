"""Train the fraud model from scratch and save the artifacts app.py needs.

Run this once before launching the app:

    python train.py

It pulls the public ULB credit-card dataset from OpenML, builds the same
features the app uses, trains a balanced random forest and writes out
fraud_model.joblib, fraud_scaler.joblib and feature_cols.json.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, confusion_matrix
)


def load_data():
    print("downloading dataset from OpenML ...")
    ds = fetch_openml("CreditCardFraudDetection", version=1, as_frame=True)
    df = ds.frame.copy()
    df["Class"] = df["Class"].astype(int)
    return df.sort_values("Time").reset_index(drop=True)


def add_features(df):
    df["hour"] = (df["Time"] / 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_z"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["gap_since_prev"] = df["Time"].diff().fillna(0)
    df["roll_count_50"] = df["Time"].rolling(50).count().fillna(1)
    return df


def main():
    df = add_features(load_data())
    feature_cols = [c for c in df.columns if c not in ("Time", "Class")]

    X = df[feature_cols].values
    y = df["Class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)

    proba = rf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.20).astype(int)
    print(classification_report(y_te, pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_te, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_te, proba), 4))
    print("confusion matrix:", confusion_matrix(y_te, pred).tolist())

    joblib.dump(rf, "fraud_model.joblib")
    joblib.dump(scaler, "fraud_scaler.joblib")
    json.dump(feature_cols, open("feature_cols.json", "w"))
    print("saved fraud_model.joblib, fraud_scaler.joblib, feature_cols.json")


if __name__ == "__main__":
    main()
