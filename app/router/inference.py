from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import HttpUrl
from app.services.audio_utils import slice_audio
from app.services.emotion_model import EmotionDetectorLocal
import requests
import shutil
from tempfile import NamedTemporaryFile
import os
import librosa
from app.logger import logger
from app.main import get_api_key  # Import the API key validation function

SLICE_DURATION = 15  # seconds
RMS_THRESHOLD = 0.01
SAMPLING_RATE = 16000

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/wake")
def wake_up():
    logger.info("Wake-up signal received.")
    return {"status": "awake"}

@router.get("/predict")
def predict_emotions(
    audio_url: HttpUrl,
    delete_after_process: bool = Query(True),
    api_key: str = Depends(get_api_key)  # API key validation
):
    logger.info(f"Prediction request received for audio URL: {audio_url}")

    try:
        # Download the audio file from the URL
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            local_path = tmp_file.name

        logger.info(f"Audio downloaded to: {local_path}")
    except Exception as e:
        logger.error(f"Failed to download audio: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to download audio")

    try:
        # Load and slice audio
        audio, sr = librosa.load(local_path, sr=SAMPLING_RATE, mono=True)
        logger.info(f"Audio loaded. Duration: {len(audio)/sr:.2f}s")
        audio_slices = slice_audio(audio, sr, SLICE_DURATION, RMS_THRESHOLD)
        logger.info(f"Extracted {len(audio_slices)} non-silent slices.")

        if not audio_slices:
            return {"result": [], "message": "No non-silent audio segments found."}

        # Run emotion detection
        detector = EmotionDetectorLocal()
        results = detector.predict_in_mini_batches(
            [slice["audio_slice"] for slice in audio_slices],
            sampling_rate=sr
        )

        # Create scorecard
        scorecard = []
        for i, result in enumerate(results):
            scorecard.append({
                "start_time": audio_slices[i]["start_time"],
                "end_time": audio_slices[i]["end_time"],
                "real_emotion": result["real_emotion"],
                "absolute_emotion": result["absolute_emotion"],
                "emotion_score": result["emotion_score"],
            })

        logger.info(f"Prediction complete. {len(scorecard)} entries.")

        return {"result": scorecard}

    finally:
        if delete_after_process and os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Temporary file deleted: {local_path}")
