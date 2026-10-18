from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
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

    # Task names are mostly unique, so a tree model cannot reliably memorize
    # them. Word and character n-grams generalize across similar task phrases.
    preprocess = ColumnTransformer([
        (
            "task_words",
            TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                max_features=40000
            ),
            "task_name"
        ),
        (
            "task_characters",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=2,
                max_features=50000
            ),
            "task_name"
        ),
        (
            "role",
            OneHotEncoder(handle_unknown="ignore"),
            ["resource_role"]
        )
    ])

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=30))
    ])

    # Choose regularization using only the training portion. This reduces
    # overfitting while keeping the final test set independent for evaluation.
    search = GridSearchCV(
        pipeline,
        param_grid={"regressor__alpha": [1, 3, 10, 30, 100, 300]},
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    predictions = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Tuned Word/Character TF-IDF Ridge Evaluation ---")
    print(f"Valid rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"Selected Ridge alpha: {search.best_params_['regressor__alpha']}")
    print(
        "Training CV MAE: "
        f"{-search.best_score_:.2f} minutes"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
