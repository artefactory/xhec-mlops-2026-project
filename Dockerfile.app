FROM python:3.11-slim

# uv installs the exact dependency versions pinned in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /bin/uv

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Code and trained objects (run the training flow before building the image)
COPY src/ src/
COPY bin/ bin/

# 8001: FastAPI, 4201: Prefect server (see bin/run_services.sh)
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
