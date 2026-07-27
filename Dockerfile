FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid pulling the 2+ GB GPU build
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data

ENV DATABASE_URL=sqlite:////app/data/recommendation.db
ENV DEFAULT_EXCEL_PATH=/app/data/map_2026.xlsx

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "--timeout", "300", "--keep-alive", "5", "flask_app:app"]
