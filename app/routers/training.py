from __future__ import annotations

import json
import logging
from typing import Optional, Dict, Any, List
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from app.model_service import train_model
from app.schemas import TrainResponse, TrainResponseModelInfo, TrainMultiResponse, TrainMultiItem
from app.auth_service import get_current_user, require_tokens
from app.config import TRAIN_TOKENS_COST, TRAIN_MULTI_TOKENS_COST

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger("training")


@router.post("/train", response_model=TrainResponse)
async def train_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_params: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Train a single model on the uploaded CSV file.
    Requires TRAIN_TOKENS_COST tokens.
    """
    require_tokens(current_user, needed=TRAIN_TOKENS_COST, action="train")

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}")

    params_dict: Dict[str, Any] = {}
    if model_params:
        try:
            params_dict = json.loads(model_params)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="model_params must be valid JSON")

    try:
        result = train_model(df=df, model_name=model_name, model_params=params_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    model_record = result["model_info"]
    metrics = result["metrics"]

    logger.info(
        f"User {current_user['username']} trained model '{model_name}' "
        f"(id={model_record['model_id']}) with metrics: {metrics}"
    )

    response_model_info = TrainResponseModelInfo(
        model_id=model_record["model_id"],
        model_name=model_record["model_name"],
        model_type=model_record["model_type"],
        trained_at=model_record["trained_at"],
        features_used=model_record["features_used"],
        label_column=model_record["label_column"],
        n_samples=model_record["n_samples"],
        categorical_columns=model_record["categorical_columns"],
        numerical_columns=model_record["numerical_columns"],
        metrics=metrics,
        model_path=model_record["model_path"],
    )

    response = TrainResponse(
        status="success",
        message="Model was trained successfully and is ready for predictions.",
        model_info=response_model_info,
    )

    return response


@router.post("/train_multi", response_model=TrainMultiResponse)
async def train_multi_endpoint(
    file: UploadFile = File(...),
    model_names: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Train multiple models on the same dataset.
    model_names is a JSON list of strings.
    Requires TRAIN_MULTI_TOKENS_COST tokens.
    """
    require_tokens(current_user, needed=TRAIN_MULTI_TOKENS_COST, action="train_multi")

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}")

    try:
        names_list: List[str] = json.loads(model_names)
        if not isinstance(names_list, list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="model_names must be a JSON list of strings")

    trained_items: List[TrainMultiItem] = []
    best_model_name: Optional[str] = None
    best_r2: float = float("-inf")

    for name in names_list:
        try:
            result = train_model(df=df, model_name=name, model_params={})
            metrics = result["metrics"]
            trained_items.append(
                TrainMultiItem(
                    model_name=name,
                    metrics=metrics,
                )
            )
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_model_name = name
        except Exception as exc:
            logger.warning(f"Training for model '{name}' failed: {exc}")
            continue

    logger.info(
        f"User {current_user['username']} trained multiple models: "
        f"{[item.model_name for item in trained_items]}. Best: {best_model_name}"
    )

    return TrainMultiResponse(
        status="success",
        trained_models=trained_items,
        best_model=best_model_name,
    )
