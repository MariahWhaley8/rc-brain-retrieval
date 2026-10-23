FROM python:3.12-slim

WORKDIR /app

# ZoneInfo("America/Chicago") needs the system timezone database.
# python:3.12-slim does not ship it, and the Monday scheduler depends on it.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY indexer.py .

ENV PORT=8000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
