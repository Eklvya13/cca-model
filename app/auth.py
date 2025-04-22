from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from app.config.settings import settings  # Adjust if your config path differs

api_key_scheme = HTTPBearer()

def get_api_key(authorization: str = Depends(api_key_scheme)) -> str:
    api_key = authorization.credentials
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    return api_key