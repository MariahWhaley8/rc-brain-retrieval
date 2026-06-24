FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY indexer.py .

ENV PORT=8000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
