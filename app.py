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

if not MODEL_PATH.exists():
    st.warning("The trained model is not available yet. Run `python source/train_models.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
expected_features = list(model.feature_names_in_)

st.subheader("Test data")
uploaded_file = st.file_uploader(
    "Upload test data as CSV",
    type=["csv"],
    help="Include all 30 model features and a target column (0 = malignant, 1 = benign).",
)

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    data_source = uploaded_file.name
else:
    if not TEST_DATA_PATH.exists():
        st.error("No CSV was uploaded and the default test_data.csv file is missing.")
        st.stop()
    test_data = pd.read_csv(TEST_DATA_PATH)
    data_source = "Bundled test_data.csv"

if "target" not in test_data.columns:
    st.error("The CSV must include a `target` column with labels 0 or 1.")
    st.stop()

missing_features = [feature for feature in expected_features if feature not in test_data.columns]
if missing_features:
    st.error(f"The CSV is missing required model features: {', '.join(missing_features)}")
    st.stop()

X_test = test_data.loc[:, expected_features].apply(pd.to_numeric, errors="coerce")
y_test = pd.to_numeric(test_data["target"], errors="coerce")
if X_test.empty:
    st.error("The CSV must contain at least one test row.")
    st.stop()
if X_test.isna().any().any() or y_test.isna().any():
    st.error("Feature and target values must be present and numeric.")
    st.stop()
if not set(y_test.unique()).issubset(set(model.classes_)):
    st.error(f"Target values must be 0 or 1; received {sorted(y_test.unique())}.")
    st.stop()
if y_test.nunique() < 2:
    st.error("The test CSV must contain both target classes to calculate AUC.")
    st.stop()

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
st.caption(f"Evaluated data: {data_source} ({len(test_data)} rows)")
metrics_table = "| " + " | ".join(metrics) + " |\n"
metrics_table += "|" + " ---: |" * len(metrics) + "\n"
metrics_table += "| " + " | ".join(f"{value:.4f}" for value in metrics.values()) + " |"
st.markdown(metrics_table)
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
    st.write(f"Data source: {data_source}")
    st.write("The saved pipeline applies feature scaling before Logistic Regression.")