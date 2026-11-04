from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "logistic_regression.joblib"
TEST_DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="WDBC Classifier Evaluation", layout="wide")
st.title("Breast Cancer Classification")
st.caption("UCI Breast Cancer Wisconsin Diagnostic dataset | Logistic Regression")

if not MODEL_PATH.exists() or not TEST_DATA_PATH.exists():
    st.warning("Model artifacts are not available yet. Run `python source/train_models.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)
X_test = test_data.drop(columns="target")
y_test = test_data["target"]
y_pred = model.predict(X_test)
positive_class_index = list(model.classes_).index(1)
y_score = model.predict_proba(X_test)[:, positive_class_index]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_score),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader("Evaluation metrics")
metric_items = list(metrics.items())
for row_start in range(0, len(metric_items), 3):
    metric_columns = st.columns(3)
    for column, (metric_name, metric_value) in zip(
        metric_columns,
        metric_items[row_start : row_start + 3],
    ):
        column.metric(metric_name, f"{metric_value:.4f}")
st.caption("Positive class: benign (label 1). Metrics use the fixed held-out test split.")

st.subheader("Model implementation status")
st.markdown(
    """
    | Model | Status |
    | --- | --- |
    | Logistic Regression | Implemented |
    | Decision Tree | Not implemented yet |
    | K-Nearest Neighbors | Not implemented yet |
    | Naive Bayes | Not implemented yet |
    | Support Vector Machine | Not implemented yet |
    """
)

with st.expander("Dataset and test split"):
    st.write(f"Held-out instances: {len(test_data)}")
    st.write(f"Input features: {X_test.shape[1]}")
    st.write("The saved pipeline applies feature scaling before Logistic Regression.")