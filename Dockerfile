FROM python:3.12-slim

WORKDIR /app

# Системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Python залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir fastapi uvicorn websockets \
     deepmultilingualpunctuation onnxruntime

# Код проекту
COPY src/ src/
COPY server.py .
COPY main.py .
COPY models/ models/

EXPOSE 8000

CMD ["python", "server.py"]