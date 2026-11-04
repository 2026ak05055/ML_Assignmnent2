from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
TEST_DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target.rename("target")

    if features.shape[1] < 12 or features.shape[0] < 500:
        raise ValueError("The selected dataset does not meet the assignment minimums.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        TEST_DATA_PATH,
        index=False,
    )

    print(f"Dataset: {dataset.DESCR.splitlines()[0]}")
    print(f"Instances: {len(features)}; features: {features.shape[1]}")
    print(f"Training instances: {len(X_train)}; test instances: {len(X_test)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved test data: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()