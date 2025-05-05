# model-service

A lightweight FastAPI microservice that serves a trained ML model via REST.

---

## Run with Docker

### 1. Build the image

```bash
docker build -t model-service .
```

### 2. Run the container

```bash
docker run -p 8080:8080 \
  -e MODEL_URL="https://github.com/remla25-team4/model-training/raw/main/models/naive_bayes.joblib" \
  model-service
```

---

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