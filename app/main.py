import os
import logging
from joblib import load
import joblib
import tempfile
from pathlib import Path
from typing import List

import pickle
from lib_ml.preprocessing import preprocess

import requests
import urllib.request
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np

# from lib_ml import preprocess_text 
from . import __version__

logger = logging.getLogger("model-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_URL = os.getenv("MODEL_URL")
CV_URL = os.getenv("CV_URL")
PORT = int(os.getenv("PORT", "8080"))
MODEL_FILENAME = Path("model.joblib")

app = Flask(__name__)
api = Api(
    app,
    version=__version__,
    title="Model Service",
    description="REST API wrapper for the trained ML model.",
    doc="/docs"
)

ns = api.namespace('', description='Model Service operations')

predict_request = api.model('PredictRequest', {
    'text': fields.String(required=True, description='Text to analyze')
})

predict_response = api.model('PredictResponse', {
    'prediction': fields.String(description='Prediction result (positive/negative)')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Error message')
})

def download_model(url: str, cv_url: str) -> None:
    logger.info("Downloading model from %s …", url)
    urllib.request.urlretrieve(url, "model.joblib")
    logger.info("Model downloaded")
    logger.info("Downloading count vectorizer from %s …", cv_url)
    urllib.request.urlretrieve(cv_url, "count_vectorizer.joblib")
    logger.info("Count vectorizer downloaded")

def load_models() -> object:
    if not Path(MODEL_FILENAME).exists():
        if not MODEL_URL or not CV_URL:
            raise RuntimeError("MODEL_URL and CV_URL env vars are required when no local model is present")
        download_model(MODEL_URL, CV_URL)
    
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("count_vectorizer.joblib")
    logger.info("Model loaded (type=%s)", type(model))
    logger.info("Vectorizer loaded (type=%s)", type(vectorizer))
    return model, vectorizer

model, vectorizer = load_models()

@ns.route('/healthz')
class HealthCheck(Resource):
    @ns.doc('health_check')
    @ns.response(200, 'OK')
    def get(self):
        """Health check endpoint"""
        return {"status": "ok"}

@ns.route('/version')
class Version(Resource):
    @ns.doc('get_version')
    @ns.response(200, 'OK')
    def get(self):
        """Get service version"""
        return {"version": __version__}

@ns.route('/predict')
class Predict(Resource):
    @ns.doc('predict')
    @ns.expect(predict_request)
    @ns.response(200, 'Success', predict_response)
    @ns.response(400, 'Bad Request', error_response)
    @ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Run model inference"""
        try:
            data = api.payload
            if not data or "text" not in data:
                return {"error": "Missing 'text' field in request"}, 400

            text_array = preprocess([data["text"]])
            text_array = vectorizer.transform(text_array).toarray()

            pred = model.predict(text_array)
            pred = 'positive' if pred[0] == 1 else 'negative'
            return {"prediction": pred}

        except Exception as exc:
            logger.exception("Prediction failed")
            return {"error": str(exc)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
