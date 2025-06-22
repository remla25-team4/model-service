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

from . import __version__

logger = logging.getLogger("model-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_URL     = os.getenv("MODEL_URL")
CV_URL        = os.getenv("CV_URL")
HOST  = os.getenv("SERVICE_HOST", "0.0.0.0")
PORT  = int(os.getenv("PORT", "8080"))

MODEL_FILE  = Path("model.joblib")
CV_FILE     = Path("count_vectorizer.joblib")

app = Flask(__name__)
api = Api(
    app,
    version=__version__,
    title="Model Service",
    description="REST API wrapper for the trained ML model",
    doc="/docs"
)
ns = api.namespace('', description='Model Service operations')

health_resp = api.model('HealthCheckResponse', {
    'status': fields.String(description='Service health status, e.g., "ok"')
})
version_resp = api.model('VersionResponse', {
    'version': fields.String(description='Service version string')
})
predict_req = api.model('PredictRequest', {
    'text': fields.String(required=True, description='Text to analyze')
})
predict_resp = api.model('PredictResponse', {
    'prediction': fields.String(description='"positive" or "negative"')
})
error_resp = api.model('ErrorResponse', {
    'error': fields.String(description='Error message detail')
})

def download_model(model_url: str, cv_url: str) -> None:
    logger.info("Downloading model from %s …", model_url)
    urllib.request.urlretrieve(model_url, MODEL_FILE.name)
    logger.info("Downloading vectorizer from %s …", cv_url)
    urllib.request.urlretrieve(cv_url, CV_FILE.name)

def load_models():
    if not MODEL_FILE.exists() or not CV_FILE.exists():
        if not MODEL_URL or not CV_URL:
            raise RuntimeError("Both MODEL_URL and CV_URL must be set if no local artifacts exist")
        download_model(MODEL_URL, CV_URL)

    model = joblib.load(MODEL_FILE.name)
    vectorizer = joblib.load(CV_FILE.name)
    logger.info("Loaded model %s and vectorizer %s", type(model), type(vectorizer))
    return model, vectorizer

model, vectorizer = load_models()

@ns.route('/healthz')
class HealthCheck(Resource):
    @ns.doc(
        summary="Health check endpoint",
        description="Returns service health status."
    )
    @ns.response(200, 'Service is healthy', health_resp)
    def get(self):
        return {'status': 'ok'}

@ns.route('/version')
class Version(Resource):
    @ns.doc(
        summary="Get service version",
        description="Returns the current version of the service."
    )
    @ns.response(200, 'Version retrieved', version_resp)
    def get(self):
        return {'version': __version__}

@ns.route('/predict')
class Predict(Resource):
    @ns.doc(
        summary="Run model inference",
        description="Returns sentiment prediction for provided text.",
        params={'text': 'Text to analyze'}
    )
    @ns.expect(predict_req, validate=True)
    @ns.response(200, 'Prediction successful', predict_resp)
    @ns.response(400, 'Bad request', error_resp)
    @ns.response(500, 'Internal server error', error_resp)
    def post(self):
        try:
            payload = api.payload
            if not payload or 'text' not in payload:
                return {'error': "Missing 'text' field"}, 400

            processed = preprocess([payload['text']])
            features = vectorizer.transform(processed).toarray()
            label = model.predict(features)[0]
            result = 'positive' if label == 1 else 'negative'
            return {'prediction': result}

        except Exception as exc:
            logger.exception("Prediction error")
            return {'error': str(exc)}, 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
