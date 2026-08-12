# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
FROM python:3.11-slim AS cpu
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap \
 && rm -rf /var/lib/apt/lists/*
# Resolve ordinary dependencies from PyPI; only CPU PyTorch wheels use the
# vendor index. Using that index as the sole source makes unrelated downloads
# slow and brittle.
RUN pip install --no-cache-dir torch==2.13.0+cpu torchvision==0.28.0+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app \
 && mkdir -p /var/lib/computefield-machine \
 && chown -R app:app /app /var/lib/computefield-machine
USER app
CMD ["python", "main.py"]

FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime AS cuda
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir nvidia-ml-py==12.560.30
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN groupadd --system app && useradd --system --gid app --uid 10001 app \
 && mkdir -p /var/lib/computefield-machine \
 && chown -R app:app /app /var/lib/computefield-machine
USER app
CMD ["python", "main.py"]
