"""
JWT token utilities for authentication.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from pydantic import BaseModel
import os

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    email: str
    exp: datetime


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> Token:
    """
    Create a new JWT access token.

    Args:
        user_id: User's unique identifier
        email: User's email address
        expires_delta: Optional custom expiration time

    Returns:
        Token object with access_token and metadata
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return Token(
        access_token=encoded_jwt,
        expires_in=int((expire - datetime.utcnow()).total_seconds())
    )


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        exp: datetime = datetime.fromtimestamp(payload.get("exp"))

        if user_id is None or email is None:
            return None

        return TokenData(user_id=user_id, email=email, exp=exp)

    except JWTError:
        return None


def get_current_user(token: str) -> Optional[TokenData]:
    """
    Get current user from token.
    Alias for verify_token for clarity.
    """
    return verify_token(token)
