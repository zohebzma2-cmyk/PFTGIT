"""
Authentication module for FunGen Web API.
JWT-based authentication with password hashing.
"""

from .jwt import create_access_token, verify_token, get_current_user
from .password import hash_password, verify_password
from .dependencies import get_current_active_user, get_optional_user

__all__ = [
    "create_access_token",
    "verify_token",
    "get_current_user",
    "hash_password",
    "verify_password",
    "get_current_active_user",
    "get_optional_user",
]
