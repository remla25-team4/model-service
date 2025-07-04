FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords

COPY app/ app/

# Configurable at runtime
ENV MODEL_URL=""
ENV CV_URL=""
ENV HOST="0.0.0.0"
ENV PORT=8080

EXPOSE ${PORT}

ENTRYPOINT ["sh", "-c", "exec python -m flask --app app.main run --host=${HOST} --port=${PORT}"]