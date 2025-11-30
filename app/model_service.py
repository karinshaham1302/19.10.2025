from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import json
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


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = MODELS_DIR / "models_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not METADATA_PATH.exists():
    METADATA_PATH.write_text("[]", encoding="utf-8")


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
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df.copy()


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessing pipeline:
    - OneHotEncoding for categorical columns
    - passthrough for numerical columns
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
    Create a scikit-learn Pipeline with preprocessing + chosen regressor.
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
    Returns:
    - model_info: metadata about the trained model
    - metrics: evaluation metrics (r2, mae, mse)
    - model_path: where the .pkl file was saved
    """
    df = validate_dataset(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor, categorical_cols, numerical_cols = build_preprocessor(df)

    pipeline = create_model(model_name=model_name, model_params=model_params, preprocessor=preprocessor)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
    }

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
    try:
        content = METADATA_PATH.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_metadata(metadata: List[Dict[str, Any]]) -> None:
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
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    model_path = MODELS_DIR / filename

    joblib.dump(pipeline, model_path)

    metadata = _load_metadata()
    if metadata:
        new_id = max(m.get("model_id", 0) for m in metadata) + 1
    else:
        new_id = 1

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

    metadata.append(model_record)
    _save_metadata(metadata)

    return str(model_path), model_record


def get_all_models() -> List[Dict[str, Any]]:
    """
    Return all trained models metadata.
    """
    return _load_metadata()


def get_latest_model_record(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest trained model record for the given model name.
    """
    metadata = _load_metadata()
    candidates = [m for m in metadata if m.get("model_name") == model_name]

    if not candidates:
        return None

    candidates.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return candidates[0]


def load_model_from_record(record: Dict[str, Any]) -> Pipeline:
    """
    Load a pipeline from a metadata record.
    """
    model_path = Path(record["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    return pipeline
