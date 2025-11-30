from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status

from app.auth_service import (
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
)
from app.database import update_tokens, delete_user
from app.schemas import (
    UserSignupRequest,
    UserLoginRequest,
    TokenResponse,
    TokensInfoResponse,
    AddTokensRequest,
    DeleteUserRequest,
)

router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger("auth_router")


@router.post("/signup", response_model=TokensInfoResponse)
def signup(request: UserSignupRequest):
    """
    Register a new user with username and password.
    """
    try:
        user = register_user(username=request.username, password=request.password)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not create user: {exc}",
        )

    return TokensInfoResponse(username=user["username"], tokens=user["tokens"])


@router.post("/login", response_model=TokenResponse)
def login(request: UserLoginRequest):
    """
    Login with username and password, return JWT access token.
    """
    user = authenticate_user(username=request.username, password=request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(data={"sub": user["username"]})
    logger.info(f"User logged in: {user['username']}")

    return TokenResponse(access_token=access_token)


@router.get("/tokens", response_model=TokensInfoResponse)
def get_tokens(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Return the current token balance for the logged-in user.
    """
    return TokensInfoResponse(username=current_user["username"], tokens=current_user["tokens"])


@router.post("/add_tokens", response_model=TokensInfoResponse)
def add_tokens(
    request: AddTokensRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Add tokens to the current user's account.
    (In a real system, this would be tied to a payment or credit card.)
    """
    if request.amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be positive.",
        )

    new_balance = update_tokens(current_user["username"], request.amount)
    if new_balance is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update token balance.",
        )

    logger.info(
        f"User {current_user['username']} purchased {request.amount} tokens. "
        f"New balance: {new_balance}"
    )

    return TokensInfoResponse(username=current_user["username"], tokens=new_balance)


@router.delete("/remove_user")
def remove_user(
    request: DeleteUserRequest,
):
    """
    Delete a user by username and password.
    """
    user = authenticate_user(username=request.username, password=request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    ok = delete_user(request.username)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or already deleted.",
        )

    logger.info(f"User deleted: {request.username}")
    return {"status": "success", "message": f"User '{request.username}' deleted."}
