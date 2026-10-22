#!/bin/sh
set -eu

# Replace the shell so Docker stop reaches Uvicorn directly.
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
