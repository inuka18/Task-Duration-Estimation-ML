
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "task_duration_dataset.csv"
MODEL_PATH = ROOT / "models" / "duration_model.joblib"
DURATION_CLASSES = ["short (<=60 min)", "medium (61-180 min)", "long (>180 min)"]


def duration_class(minutes):
    """Map durations to readable bands for classification metrics."""
    return np.select(
        [minutes <= 60, minutes <= 180],
        DURATION_CLASSES[:2],
        default=DURATION_CLASSES[2],
    )


def duration_to_minutes(value):
    text = str(value).lower().replace(",", "").strip()

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b",
        text
    )

    if match:
        number = float(match.group(1))
        unit = match.group(2)

        if unit.startswith(("hour", "hr")):
            return number * 60

        return number

    try:
        # Numeric values are assumed to be minutes.
        return float(text)
    except ValueError:
        return np.nan


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "task": "task_name",
        "resource role": "resource_role",
        "duration": "duration_minutes"
    })

    required = {"task_name", "resource_role", "duration_minutes"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns: {required}. "
            f"Found: {list(df.columns)}"
        )

    df["task_name"] = df["task_name"].astype(str).str.strip()
    df["resource_role"] = df["resource_role"].astype(str).str.strip()
    df["duration_minutes"] = df["duration_minutes"].apply(
        duration_to_minutes
    )

    df = df.dropna(subset=list(required))
    df = df[
        (df["task_name"] != "") &
        (df["resource_role"] != "") &
        (df["duration_minutes"] > 0)
    ]

    X = df[["task_name", "resource_role"]]
    y = df["duration_minutes"]

    # Must match train_model.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    predictions = np.maximum(model.predict(X_test), 0)

    # Baseline: predict the median training duration for every task.
    baseline_predictions = np.full(
        len(y_test), y_train.median()
    )

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    model_r2 = r2_score(y_test, predictions)
    absolute_percentage_errors = (
        np.abs(y_test.to_numpy() - predictions) /
        np.maximum(y_test.to_numpy(), 1)
    )
    mape = absolute_percentage_errors.mean() * 100
    within_30_minutes = (
        np.abs(y_test.to_numpy() - predictions) <= 30
    ).mean() * 100

    actual_classes = duration_class(y_test.to_numpy())
    predicted_classes = duration_class(predictions)
    class_accuracy = accuracy_score(actual_classes, predicted_classes)
    class_precision = precision_score(
        actual_classes, predicted_classes,
        labels=DURATION_CLASSES, average="macro", zero_division=0
    )
    class_recall = recall_score(
        actual_classes, predicted_classes,
        labels=DURATION_CLASSES, average="macro", zero_division=0
    )

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    print("\n=== Task Duration Model Evaluation ===")
    print(f"Test rows: {len(y_test)}")

    print("\nTF-IDF Ridge:")
    print(f"MAE:  {model_mae:.2f} minutes")
    print(f"RMSE: {model_rmse:.2f} minutes")
    print(f"R2:   {model_r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Within +/-30 minutes: {within_30_minutes:.2f}%")
    print("\nDuration-band classification (short <=60, medium 61-180, long >180 min):")
    print(f"Accuracy:  {class_accuracy * 100:.2f}%")
    print(f"Precision: {class_precision * 100:.2f}% (macro average)")
    print(f"Recall:    {class_recall * 100:.2f}% (macro average)")

    print("\nMedian Baseline:")
    print(f"MAE:  {baseline_mae:.2f} minutes")
    print(f"RMSE: {baseline_rmse:.2f} minutes")

    print("\nComparison:")
    if model_mae < baseline_mae:
        print("TF-IDF Ridge has lower MAE than the baseline.")
    else:
        print("TF-IDF Ridge does not beat the baseline on MAE.")

    if model_rmse < baseline_rmse:
        print("TF-IDF Ridge has lower RMSE than the baseline.")
    else:
        print("TF-IDF Ridge does not beat the baseline on RMSE.")

    # Save predictions for further inspection.
    results = X_test.copy()
    results["actual_minutes"] = y_test
    results["predicted_minutes"] = predictions
    results["absolute_error_minutes"] = np.abs(
        y_test.to_numpy() - predictions
    )
    results["actual_duration_band"] = actual_classes
    results["predicted_duration_band"] = predicted_classes

    output_path = ROOT / "evaluation_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nDetailed predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
