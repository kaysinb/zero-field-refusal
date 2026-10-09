# syntax=docker/dockerfile:1
# The base supplies Runpod's SSH startup; project dependencies live in an isolated environment.
FROM ghcr.io/astral-sh/uv:0.9.28 AS uv
FROM runpod/base:1.0.2-ubuntu2404@sha256:963056222f8edbe7a1103d31cea21df027b4d25dbc6e2fe65167a5040fca0ee8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev build-essential git rsync libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/project-deps
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --python /usr/bin/python3.12 \
    && uv pip check --python /opt/venv/bin/python

COPY docker/project-run.py /usr/local/bin/project-run
RUN chmod 755 /usr/local/bin/project-run
ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/workspace/cache/huggingface \
    XDG_CACHE_HOME=/workspace/cache

# Source snapshots, secrets, datasets and outputs are supplied at runtime, not baked into the image.
WORKDIR /workspace
EXPOSE 22
# Inherit Runpod's /start.sh command so public-key SSH works on Pods.
