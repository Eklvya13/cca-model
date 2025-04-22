SLICE_DURATION = 15
RMS_THRESHOLD = 0.01
SAMPLING_RATE = 16000

import os
from pydantic_settings import BaseSettings 

class Settings(BaseSettings):
    API_KEY: str
    CORS_ORIGIN: str
    huggingface_token: str

    class Config:
        env_file = ".env"

settings = Settings()