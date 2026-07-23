import gradio as gr
import numpy as np
import pandas as pd
import joblib, json

# trained artifacts from the notebook
model = joblib.load("fraud_model.joblib")
scaler = joblib.load("fraud_scaler.joblib")
FEATURES = json.load(open("feature_cols.json"))

# a couple of real rows pulled from the test set so people can try it fast
DECISION_T = 0.20  # tuned in the notebook to catch more of the fraud

def build_row(amount, hour, v14, v10, v12, v4, v17):
    row = {c: 0.0 for c in FEATURES}
    row["Amount"] = amount
    row["hour"] = hour
    row["V14"] = v14
    row["V10"] = v10
    row["V12"] = v12
    row["V4"] = v4
    row["V17"] = v17
    row["log_amount"] = np.log1p(max(amount, 0))
    row["amount_z"] = (amount - 88.0) / 250.0
    row["gap_since_prev"] = 0.0
    row["roll_count_50"] = 1.0
    return pd.DataFrame([row])[FEATURES]

def score(amount, hour, v14, v10, v12, v4, v17):
    x = build_row(amount, hour, v14, v10, v12, v4, v17)
    p = float(model.predict_proba(scaler.transform(x))[0, 1])
    verdict = "FRAUD - review this transaction" if p >= DECISION_T else "looks legitimate"
    return {"fraud": p, "legit": 1 - p}, verdict

with gr.Blocks(title="Real-Time Fraud Detection") as demo:
    gr.Markdown(
        "# Real-Time Fraud Detection\n"
        "Enter a transaction and the model scores how likely it is to be fraud.\n"
        "The V-features are the anonymised PCA signals that ship with the card dataset."
    )
    with gr.Row():
        with gr.Column():
            amount = gr.Number(label="Amount", value=12.99)
            hour = gr.Slider(0, 23, value=11, step=1, label="Hour of day")
            v14 = gr.Number(label="V14", value=-0.31)
            v10 = gr.Number(label="V10", value=0.09)
            v12 = gr.Number(label="V12", value=0.53)
            v4 = gr.Number(label="V4", value=-0.42)
            v17 = gr.Number(label="V17", value=0.21)
            go = gr.Button("Check transaction", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Probability")
            out_text = gr.Textbox(label="Verdict")
    go.click(score, [amount, hour, v14, v10, v12, v4, v17], [out_label, out_text])
    gr.Examples(
        [[12.99, 11, -0.31, 0.09, 0.53, -0.42, 0.21],
         [0.0, 2, -9.8, -6.4, -8.1, 6.2, -11.3]],
        [amount, hour, v14, v10, v12, v4, v17],
    )

if __name__ == "__main__":
    demo.launch()
import gradio as gr
import numpy as np
import pandas as pd
import joblib, json

# trained artifacts from the notebook
model = joblib.load("fraud_model.joblib")
scaler = joblib.load("fraud_scaler.joblib")
FEATURES = json.load(open("feature_cols.json"))

# a couple of real rows pulled from the test set so people can try it fast
LEGIT = {"Amount": 12.99, "hour": 11, "V14": -0.31, "V10": 0.09, "V12": 0.53, "V4": -0.42, "V17": 0.21}
FRAUD = {"Amount": 0.0, "hour": 2, "V14": -9.8, "V10": -6.4, "V12": -8.1, "V4": 6.2, "V17": -11.3}
DECISION_T = 0.20  # tuned in the notebook to catch more fraud


def build_row(amount, hour, v14, v10, v12, v4, v17):
    row = {c: 0.0 for c in FEATURES}
    row["Amount"] = amount
    row["hour"] = hour
    row["V14"] = v14
    row["V10"] = v10
    row["V12"] = v12
    row["V4"] = v4
    row["V17"] = v17
    row["log_amount"] = np.log1p(max(amount, 0))
    row["amount_z"] = (amount - 88.0) / 250.0
    row["gap_since_prev"] = 0.0
    row["roll_count_50"] = 1.0
    return pd.DataFrame([row])[FEATURES]


def score(amount, hour, v14, v10, v12, v4, v17):
    x = build_row(amount, hour, v14, v10, v12, v4, v17)
    p = float(model.predict_proba(scaler.transform(x.values))[0, 1])
    verdict = "FRAUD - review this transaction" if p >= DECISION_T else "looks legitimate"
    label = {"fraud": p, "legit": 1 - p}
    return label, verdict


with gr.Blocks(title="Real-Time Fraud Detection") as demo:
    gr.Markdown("# Real-Time Fraud Detection\nEnter a transaction and the model scores how likely it is to be fraud.\nThe V-features are the anonymised PCA signals from the card-transaction dataset.")
    with gr.Row():
        with gr.Column():
            amount = gr.Number(label="Amount", value=12.99)
            hour = gr.Slider(0, 23, value=11, step=1, label="Hour of day")
            v14 = gr.Number(label="V14", value=-0.31)
            v10 = gr.Number(label="V10", value=0.09)
            v12 = gr.Number(label="V12", value=0.53)
            v4 = gr.Number(label="V4", value=-0.42)
            v17 = gr.Number(label="V17", value=0.21)
            go = gr.Button("Check transaction", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Probability")
            out_text = gr.Textbox(label="Verdict")
    go.click(score, [amount, hour, v14, v10, v12, v4, v17], [out_label, out_text])
    gr.Examples([[12.99, 11, -0.31, 0.09, 0.53, -0.42, 0.21], [0.0, 2, -9.8, -6.4, -8.1, 6.2, -11.3]], [amount, hour, v14, v10, v12, v4, v17])

if __name__ == "__main__":
    demo.launch()
