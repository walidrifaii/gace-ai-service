FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    INSIGHTFACE_HOME=/data/insightface \
    ARGOS_TRANSLATE_PACKAGES_DIR=/data/argos-translate

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data/insightface /data/argos-translate /data/faiss

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
