# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1: builder — install Python deps via uv into a self-contained .venv
# ============================================================================
FROM python:3.12-slim AS builder

# uv installed via official static binary; pin version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Layer cache: deps change rarely, source changes often
COPY pyproject.toml uv.lock ./

# --frozen: fail if lockfile out of sync (catches accidental drift)
# --no-dev: skip pytest, ruff, jupyter, etc.
# --no-install-project: install deps only, not the local package yet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# copy source and install the local package
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ============================================================================
# Stage 2: runtime — copy only what's needed
# ============================================================================
FROM python:3.12-slim AS runtime

# libgomp1: required by scipy/numpy for OpenMP at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# no need to reinstall deps
COPY --from=builder /app/.venv /app/.venv

# Source code
COPY src/ ./src/
COPY configs/ ./configs/

# required for mmap to work at runtime, in dockerignore exclude all but latest
# the run is exposed for testability, ideally should be handled via dockerignore and you copy artifacts/runs
COPY artifacts/runs/2026-04-29-152605 ./artifacts/runs/latest

# Make .venv binaries the default
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV LOAD_ARTIFACTS=1

# Render injects PORT; default for local docker run
ENV PORT=8000
EXPOSE 8000

# Direct uvicorn invocation — bypasses run.py for clarity in production
# (run.py still works via `uv run serve` for local; CMD here pins the prod path)
# workers for 0.1 CPU
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]