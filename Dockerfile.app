FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

WORKDIR /app

# Install the dependencies first, so this layer is cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group app --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ src/
COPY bin/ bin/

# 8001: API, 4201: Prefect UI (bound to 8000 and 4200 on the host with `docker run -p`)
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
