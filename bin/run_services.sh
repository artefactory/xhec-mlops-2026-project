#!/bin/bash

# Starts the Prefect server and the API inside the Docker container (see Dockerfile.app).

# The Prefect UI is opened from the host on port 4200, so it must call the API there
export PREFECT_UI_API_URL=http://localhost:4200/api

prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
