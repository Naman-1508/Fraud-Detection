import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "model/fraud_model.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error":"no input"}), 400

    if "rows" in data:
        df = pd.DataFrame(data["rows"])
    else:
        df = pd.DataFrame([data])

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    df["Amount"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)

    expected_cols = [
        "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
    ]

    df = df[expected_cols]

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = [{"pred": int(p), "probability": float(prob)} for p, prob in zip(preds, probs)]

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
