# model-service

A lightweight FastAPI microservice that serves a trained ML model via REST.

---

## Build and Run Docker

### 1. Build the image

```bash
docker build -t model-service .
```

### 2. Run the container

```bash
docker run -p 8080:8080 \
  -e MODEL_URL="https://github.com/remla25-team4/model-training/releases/download/v1.0.5/naive_bayes.joblib" \
  -e CV_URL="https://github.com/remla25-team4/model-training/releases/download/v1.0.5/count_vectorizer.joblib" \
  model-service
```
---

## Run Pre-built Image from GHCR

### 1. Pull the image

```bash
docker pull ghcr.io/remla25-team4/model-service:1.0.2
```

### 2. Run the container

```bash
docker run -p 8080:8080 \
  -e MODEL_URL="https://github.com/remla25-team4/model-training/releases/download/v1.0.5/naive_bayes.joblib" \
  -e CV_URL="https://github.com/remla25-team4/model-training/releases/download/v1.0.5/count_vectorizer.joblib" \
  ghcr.io/remla25-team4/model-service:1.0.2
```

### 3. Access the service

Once the container is running, you can access the service endpoints at http://localhost:8080.

## 🔍 Endpoints

| Method | Endpoint     | Description        |
|--------|--------------|--------------------|
| GET    | `/healthz`   | Liveness check     |
| GET    | `/version`   | Service version    |
| POST   | `/predict`   | Run model inference|

### 3. Interactive qurying

If you want to query the app using an UI, please visit:

```bash
http://0.0.0.0:8080/docs
```