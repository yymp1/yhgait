FROM --platform=$TARGETPLATFORM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip uv \
    && uv pip install --system --torch-backend cpu -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/samples /app/outputs

ENTRYPOINT ["python", "app.py"]
