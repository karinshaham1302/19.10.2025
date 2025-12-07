from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import json
import math  # for RMSE calculation (sqrt of MSE)
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Paths and metadata handling
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = MODELS_DIR / "models_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metadata file if it does not exist
if not METADATA_PATH.exists():
    METADATA_PATH.write_text("[]", encoding="utf-8")


# -----------------------------
# Features / target definition
# -----------------------------
FEATURE_COLUMNS: List[str] = [
    "subject",
    "student_level",
    "lesson_minutes",
    "teacher_experience_years",
    "is_online",
    "city",
]

TARGET_COLUMN: str = "lesson_price"


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataset contains all required columns.
    Raises ValueError if columns are missing.
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Work on a copy to avoid modifying the original DataFrame
    return df.copy()


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessing pipeline:
    - OneHotEncoding for categorical columns
    - passthrough for numerical columns
    Returns:
        preprocessor, categorical_cols, numerical_cols
    """
    feature_df = df[FEATURE_COLUMNS]

    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    return preprocessor, categorical_cols, numerical_cols


def create_model(
    model_name: str,
    model_params: Optional[Dict[str, Any]],
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Create a scikit-learn Pipeline with:
    - preprocessing (ColumnTransformer)
    - chosen regressor (linear / decision_tree / random_forest)
    """
    if model_params is None:
        model_params = {}

    name = model_name.lower().strip()

    if name == "linear":
        estimator = LinearRegression(**model_params)
    elif name == "decision_tree":
        estimator = DecisionTreeRegressor(**model_params)
    elif name == "random_forest":
        estimator = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("regressor", estimator),
        ]
    )

    return pipeline


def train_model(
    df: pd.DataFrame,
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a regression model on the given DataFrame.

    Steps:
    1. Validate the dataset (required columns).
    2. Split into train/test.
    3. Build preprocessing (OneHotEncoder + numeric passthrough).
    4. Create the requested model (linear / decision_tree / random_forest).
    5. Fit on training data.
    6. Evaluate on test data:
       - R²
       - MAE
       - MSE
       - RMSE (sqrt of MSE)
    7. Save model and metadata.

    Returns a dict containing:
    - "model_info": metadata about the trained model
    - "metrics": evaluation metrics (r2, mae, mse, rmse)
    - "model_path": path to the saved .pkl file
    """
    # Ensure all required columns are present
    df = validate_dataset(df)

    # Features (X) and target (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build preprocessing and get column lists
    preprocessor, categorical_cols, numerical_cols = build_preprocessor(df)

    # Create the chosen model as a Pipeline (preprocessing + regressor)
    pipeline = create_model(
        model_name=model_name,
        model_params=model_params,
        preprocessor=preprocessor,
    )

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = pipeline.predict(X_test)

    # Raw metrics (before rounding)
    r2_raw = r2_score(y_test, y_pred)
    mae_raw = mean_absolute_error(y_test, y_pred)
    mse_raw = mean_squared_error(y_test, y_pred)
    rmse_raw = math.sqrt(mse_raw)

    # Round everything to 2 decimal places for cleaner API output
    metrics = {
        "r2": round(float(r2_raw), 2),
        "mae": round(float(mae_raw), 2),
        "mse": round(float(mse_raw), 2),
        "rmse": round(float(rmse_raw), 2),
    }

    # Save model and metadata
    model_path, model_record = save_model_with_metadata(
        pipeline=pipeline,
        model_name=model_name,
        metrics=metrics,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        n_samples=len(df),
    )

    result = {
        "model_info": model_record,
        "metrics": metrics,
        "model_path": model_path,
    }

    return result


def _load_metadata() -> List[Dict[str, Any]]:
    """
    Load the list of model metadata records from JSON file.
    If anything goes wrong, return an empty list.
    """
    try:
        content = METADATA_PATH.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_metadata(metadata: List[Dict[str, Any]]) -> None:
    """
    Save the list of model metadata records back to JSON file.
    """
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_model_with_metadata(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, Any],
    categorical_cols: List[str],
    numerical_cols: List[str],
    n_samples: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Save trained pipeline as .pkl and record its metadata.

    Returns:
        (model_path, model_record)
    """
    # Unique filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    model_path = MODELS_DIR / filename

    # Save the pipeline to disk
    joblib.dump(pipeline, model_path)

    # Load existing metadata
    metadata = _load_metadata()
    if metadata:
        new_id = max(m.get("model_id", 0) for m in metadata) + 1
    else:
        new_id = 1

    # Build a record for the new model
    model_record = {
        "model_id": new_id,
        "model_name": model_name,
        "model_type": pipeline.named_steps["regressor"].__class__.__name__,
        "trained_at": datetime.utcnow().isoformat(),
        "features_used": FEATURE_COLUMNS,
        "label_column": TARGET_COLUMN,
        "n_samples": n_samples,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "metrics": metrics,
        "model_path": str(model_path),
    }

    # Append and save back to JSON
    metadata.append(model_record)
    _save_metadata(metadata)

    return str(model_path), model_record


def get_all_models() -> List[Dict[str, Any]]:
    """
    Return all trained models metadata as a list of dicts.
    """
    return _load_metadata()


def get_latest_model_record(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest trained model record for the given model name.
    If no model exists for that name, return None.
    """
    metadata = _load_metadata()
    candidates = [m for m in metadata if m.get("model_name") == model_name]

    if not candidates:
        return None

    # Sort by trained_at in descending order (newest first)
    candidates.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return candidates[0]


def load_model_from_record(record: Dict[str, Any]) -> Pipeline:
    """
    Load a pipeline from a metadata record.

    The record must contain a valid "model_path".
    """
    model_path = Path(record["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    return pipeline
