import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("HEALTH_CLAIM_API_KEY")
    
    # If no key is configured in .env, we allow access (for easier dev setup)
    # But if a key IS configured, we must match it.
    if not expected_key:
        return api_key

    if api_key == expected_key:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )
