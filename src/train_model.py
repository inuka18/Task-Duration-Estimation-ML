from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "task_duration_dataset.csv"
MODEL_PATH = ROOT / "models" / "duration_model.joblib"


def duration_to_minutes(value):
    """Convert durations such as '1.5 hours' or '90 minutes'."""
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
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
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
            f"Required columns: {required}. "
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

    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required.")

    X = df[["task_name", "resource_role"]]
    y = df["duration_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocess = ColumnTransformer([
        (
            "task_text",
            TfidfVectorizer(ngram_range=(1, 2)),
            "task_name"
        ),
        (
            "role",
            OneHotEncoder(handle_unknown="ignore"),
            ["resource_role"]
        )
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Random Forest Evaluation ---")
    print(f"Valid rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
