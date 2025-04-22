from fastapi import FastAPI
from app.router import inference
from fastapi.middleware.cors import CORSMiddleware
from app.auth import get_api_key

from app.config.settings import settings

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference.router)