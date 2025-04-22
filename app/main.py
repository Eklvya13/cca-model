from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from app.router import inference
from app.config import settings


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_scheme = HTTPBearer()

# Function to get API key from headers and validate it
def get_api_key(authorization: str = Depends(api_key_scheme)) -> str:
    api_key = authorization.credentials
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    return api_key

# inference router
app.include_router(inference.router)

