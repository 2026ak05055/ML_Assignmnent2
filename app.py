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
MODEL_PATHS = {
    "Logistic Regression": ROOT / "model" / "logistic_regression.joblib",
    "Decision Tree": ROOT / "model" / "decision_tree.joblib",
    "KNN": ROOT / "model" / "knn.joblib",
    "Naive Bayes": ROOT / "model" / "naive_bayes.joblib",
    "Support Vector Machine": ROOT / "model" / "support_vector_machine.joblib",
}
TEST_DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="Machine Learning - Classifier Evaluation", layout="wide")
st.title("Machine Learning Assignment 2: Classifier Evaluator")
st.caption("UCI Breast Cancer Wisconsin Diagnostic dataset")

missing_models = [name for name, path in MODEL_PATHS.items() if not path.exists()]
if missing_models:
    st.warning("Train all classifiers first by running `python source/train_models.py`.")
    st.stop()

models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
expected_features = list(next(iter(models.values())).feature_names_in_)

st.subheader("Test data")
uploaded_file = st.file_uploader(
    "Upload test data as CSV",
    type=["csv"],
    help="Include all model features and a target column (0 = malignant, 1 = benign).",
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
        st.error("Upload a test CSV or add the default test_data.csv file.")
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
if not set(y_test.unique()).issubset({0, 1}):
    st.error(f"Target values must be 0 or 1; received {sorted(y_test.unique())}.")
    st.stop()
if y_test.nunique() < 2:
    st.error("The test CSV must contain both target classes to calculate AUC.")
    st.stop()

metric_names = [
    "Accuracy",
    "AUC Score",
    "Precision",
    "Recall",
    "F1 Score",
    "Matthews Correlation Coefficient (MCC Score)",
]
results = []
for model_name, classifier in models.items():
    y_pred = classifier.predict(X_test)
    if hasattr(classifier, "predict_proba"):
        positive_class_index = list(classifier.classes_).index(1)
        y_score = classifier.predict_proba(X_test)[:, positive_class_index]
    else:
        y_score = classifier.decision_function(X_test)
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_score),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "Matthews Correlation Coefficient (MCC Score)": matthews_corrcoef(
                y_test,
                y_pred,
            ),
        }
    )

st.subheader("Evaluation metrics")
st.caption(f"Evaluated data: {data_source} ({len(test_data)} rows)")
metrics_table = "| Model | " + " | ".join(metric_names) + " |\n"
metrics_table += "| --- |" + " ---: |" * len(metric_names) + "\n"
for result in results:
    metrics_table += (
        f"| {result['Model']} | "
        + " | ".join(f"{result[name]:.4f}" for name in metric_names)
        + " |\n"
    )
st.markdown(metrics_table)
st.caption("Positive class: benign (label 1). All classifiers use the same test data.")

st.subheader("Model implementation status")
st.markdown(
    "| Model | Status |\n| --- | --- |\n"
    + "\n".join(f"| {name} | Trained and evaluated |" for name in models)
)

with st.expander("Dataset and test split"):
    st.write(f"Held-out instances: {len(test_data)}")
    st.write(f"Input features: {X_test.shape[1]}")
    st.write(f"Data source: {data_source}")
    st.write("Each classifier uses its saved preprocessing and fitted estimator.")