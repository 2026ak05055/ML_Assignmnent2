from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "wdbc.csv"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.csv"
RANDOM_STATE = 42


# Build all five classifiers with shared preprocessing where needed.
def build_models() -> dict[str, object]:
    # Scaling keeps feature magnitudes comparable for linear, distance-based, and margin-based models.
    # Decision Trees split by thresholds and do not require standardized inputs.
    return {
        # Logistic Regression models class log-odds as a linear function of the features.
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        # A Decision Tree learns feature thresholds recursively, without distance-based scaling.
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # KNN classifies by the majority label among the five nearest scaled samples.
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        # Gaussian Naive Bayes applies Bayes' rule with a Gaussian likelihood per feature.
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        # SVM finds a maximum-margin boundary in standardized feature space.
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(random_state=RANDOM_STATE)),
            ]
        ),
    }


# Calculate the six requested metrics for one fitted classifier.
def calculate_metrics(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        # AUC is measured for the positive class: benign (label 1).
        positive_class_index = list(model.classes_).index(1)
        positive_scores = model.predict_proba(X_test)[:, positive_class_index]
    else:
        # SVM supplies continuous decision scores even when probability calibration is disabled.
        positive_scores = model.decision_function(X_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC Score": roc_auc_score(y_test, positive_scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "Matthews Correlation Coefficient (MCC Score)": matthews_corrcoef(
            y_test,
            predictions,
        ),
    }


# Load the dataset, train each model, and save the artifacts and metrics.
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    if "target" not in dataset.columns:
        raise ValueError("The dataset CSV must contain a 'target' column.")

    features = dataset.drop(columns="target")
    target = pd.to_numeric(dataset["target"], errors="raise").rename("target")

    if features.shape[1] < 12 or features.shape[0] < 500:
        raise ValueError("The selected dataset does not meet the assignment minimums.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("The target must use 0 for malignant and 1 for benign.")

    # Preserve class proportions and use one fixed holdout for fair model comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save the same test rows for app evaluation and reproducibility.
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        TEST_DATA_PATH,
        index=False,
    )

    results = []
    for model_name, model in build_models().items():
        # Fit only on training data; the untouched test set is used for metrics.
        model.fit(X_train, y_train)
        results.append({"Model": model_name, **calculate_metrics(model, X_test, y_test)})
        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")

    results_frame = pd.DataFrame(results).set_index("Model")
    results_frame.to_csv(METRICS_PATH)
    print("\nHeld-out evaluation metrics:")
    print(results_frame.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Dataset: {DATA_PATH}")
    print(f"Instances: {len(features)}; features: {features.shape[1]}")
    print(f"Training instances: {len(X_train)}; test instances: {len(X_test)}")
    print(f"Saved test data: {TEST_DATA_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()