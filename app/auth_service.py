from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.database import get_user_by_username, create_user, update_tokens

# Use pbkdf2_sha256 instead of bcrypt to avoid 72-byte password limit
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)

security = HTTPBearer()
logger = logging.getLogger("auth")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    Verify that a plain password matches the stored hash.
    """
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str) -> str:
    """
    Hash a plain password using pbkdf2_sha256.
    """
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token with an expiration time.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def register_user(username: str, password: str) -> Dict[str, Any]:
    """
    Create a new user in the database with a hashed password.
    """
    password_hash = get_password_hash(password)
    user = create_user(username=username, password_hash=password_hash)
    logger.info(f"User registered: {username}")
    return user


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by username and password.
    """
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Extract and validate the current user from the JWT token.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


def require_tokens(user: Dict[str, Any], needed: int, action: str) -> None:
    """
    Ensure the user has enough tokens for a specific action
    and deduct them from the database.
    """
    username = user["username"]
    current_tokens = user["tokens"]

    if current_tokens < needed:
        logger.warning(
            f"User {username} tried to perform '{action}' without enough tokens "
            f"(has {current_tokens}, needs {needed})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough tokens for this action.",
        )

    new_balance = update_tokens(username, -needed)
    if new_balance is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update token balance.",
        )

    logger.info(
        f"User {username} performed '{action}', "
        f"tokens deducted: {needed}, new balance: {new_balance}"
    )
    user["tokens"] = new_balance
