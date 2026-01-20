# =========================
# Render-safe Dockerfile
# FastAPI + LightGBM
# =========================

FROM python:3.10-slim

# Avoid Python buffering, keep logs visible in Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (LightGBM needs libgomp; build-essential helps with some wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Render will set PORT. We also keep a default for local runs.
EXPOSE 8000

# IMPORTANT: shell form enables ${PORT} expansion (JSON form won't)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}





