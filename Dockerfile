FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Modelleri indir
RUN bash download_models.sh

# Python bağımlılıkları
RUN pip install --no-cache-dir -r requirements.txt

# Spaces için hafif HTTP server (FastAPI)
RUN pip install fastapi uvicorn

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
