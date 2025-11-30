from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class TrainResponseModelInfo(BaseModel):
    model_id: int
    model_name: str
    model_type: str
    trained_at: str
    features_used: List[str]
    label_column: str
    n_samples: int
    categorical_columns: List[str]
    numerical_columns: List[str]
    metrics: Dict[str, float]
    model_path: str


class TrainResponse(BaseModel):
    status: str
    message: str
    model_info: TrainResponseModelInfo


class TrainMultiItem(BaseModel):
    model_name: str
    metrics: Dict[str, float]


class TrainMultiResponse(BaseModel):
    status: str
    trained_models: List[TrainMultiItem]
    best_model: Optional[str]


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    model_name: str
    model_id: int
    prediction: float


class ModelSummary(BaseModel):
    model_id: int
    model_name: str
    model_type: str
    trained_at: str
    r2: float
    mae: float
    mse: float


class ModelsListResponse(BaseModel):
    models: List[ModelSummary]


# ----- Auth & tokens -----


class UserSignupRequest(BaseModel):
    username: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokensInfoResponse(BaseModel):
    username: str
    tokens: int


class AddTokensRequest(BaseModel):
    amount: int


class DeleteUserRequest(BaseModel):
    username: str
    password: str

