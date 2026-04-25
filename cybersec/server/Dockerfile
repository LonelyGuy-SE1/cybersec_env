# Multi-stage build for the Cybersec OpenEnv environment.
#
# Build context is the env folder (cybersec/), not the repo root, matching
# the layout produced by `openenv init`. Run from inside cybersec/:
#
#     docker build -t cybersec-env:latest -f server/Dockerfile .
#
# `openenv push` builds the same context when shipping to Hugging Face Spaces.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app/env

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CYBERSEC_HOST=0.0.0.0 \
    CYBERSEC_PORT=8000 \
    CYBERSEC_MAX_CONCURRENT_ENVS=8

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy the env folder (build context == cybersec/).
COPY . /app/env

# Install the env package itself from pyproject.toml.
RUN pip install --upgrade pip && pip install .

ENV PYTHONPATH="/app/env:${PYTHONPATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${CYBERSEC_PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host ${CYBERSEC_HOST:-0.0.0.0} --port ${CYBERSEC_PORT:-8000}"]
