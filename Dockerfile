# TscForge - reproducible container image
# Author: 晨星 (CJX0712)
# Build:   docker build -t tscforge:0.1.0 .
# Run:     docker run --rm -v "$PWD":/work tscforge:0.1.0 tscforge benchmark
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building numpy / scipy wheels if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements.lock.txt ./
COPY tscforge ./tscforge
COPY scripts ./scripts
COPY examples ./examples

# Install runtime deps; optional Tier-0 backends are allowed to fail (offline-safe)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt || pip install numpy scikit-learn

RUN pip install -e . --no-deps || true

ENTRYPOINT ["tscforge"]
CMD ["--help"]
