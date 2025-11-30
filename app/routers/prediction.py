from __future__ import annotations

import logging
from typing import List, Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from app.model_service import get_all_models, get_latest_model_record, load_model_from_record
from app.schemas import PredictRequest, PredictResponse, ModelsListResponse, ModelSummary
from app.auth_service import get_current_user, require_tokens
from app.config import PREDICT_TOKENS_COST

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger("prediction")


@router.get("/", response_model=ModelsListResponse)
def list_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Return a list of all trained models with basic metrics.
    Does not consume tokens.
    """
    metadata = get_all_models()

    summaries: List[ModelSummary] = []

    for m in metadata:
        metrics = m.get("metrics", {})
        summaries.append(
            ModelSummary(
                model_id=m["model_id"],
                model_name=m["model_name"],
                model_type=m["model_type"],
                trained_at=m["trained_at"],
                r2=float(metrics.get("r2", 0.0)),
                mae=float(metrics.get("mae", 0.0)),
                mse=float(metrics.get("mse", 0.0)),
            )
        )

    logger.info(f"User {current_user['username']} listed models (count={len(summaries)})")

    return ModelsListResponse(models=summaries)


@router.post("/predict/{model_name}", response_model=PredictResponse)
def predict_with_latest_model(
    model_name: str,
    request: PredictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Use the latest trained model of the given name to make a single prediction.
    Requires PREDICT_TOKENS_COST tokens.
    """
    require_tokens(current_user, needed=PREDICT_TOKENS_COST, action=f"predict:{model_name}")

    record = get_latest_model_record(model_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No model found for name '{model_name}'")

    pipeline = load_model_from_record(record)

    df = pd.DataFrame([request.data])

    try:
        prediction = pipeline.predict(df)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    logger.info(
        f"User {current_user['username']} requested prediction with model "
        f"'{model_name}' (id={record['model_id']}): {request.data} -> {prediction}"
    )

    return PredictResponse(
        model_name=record["model_name"],
        model_id=record["model_id"],
        prediction=float(prediction),
    )
