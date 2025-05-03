import os
import logging
from joblib import load
import joblib
import tempfile
from pathlib import Path
from typing import List

import pickle

import requests
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# from lib_ml import preprocess_text 
from . import __version__

logger = logging.getLogger("model-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_URL = os.getenv("MODEL_URL")
PORT = int(os.getenv("PORT", "8080"))
MODEL_FILENAME = Path("model.joblib")

app = FastAPI(
    title="Model Service",
    version=__version__,
    description="REST API wrapper for the trained ML model.",
)


def download_model(url: str) -> None:
    logger.info("Downloading model from %s …", url)
    urllib.request.urlretrieve(url, "model.joblib")
    logger.info("Model saved")

def load_model() -> object:
    if not Path(MODEL_FILENAME).exists():
        if not MODEL_URL:
            raise RuntimeError("MODEL_URL env var is required when no local model is present")
        download_model(MODEL_URL)
    
    model = joblib.load("model.joblib")
    logger.info("Model loaded (type=%s)", type(model))
    return model

model = load_model()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str

@app.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"status": "ok"}

@app.get("/version", summary="Service version")
async def version():
    return {"version": __version__}

@app.post("/predict", response_model=PredictResponse, summary="Run model inference")
async def predict(req: PredictRequest):
    try:
        # Convert list of strings to numpy array and reshape to 2D
        # text_array = np.array(req.text).reshape(-1, 1)
        text_array = np.random.randint(0, 5, size=(1, 140))

        pred = model.predict(text_array)
        pred = 'positive' if pred[0] == 1 else 'negative'
        return PredictResponse(prediction=pred)

    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
