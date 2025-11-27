#!/bin/bash
# Startup script for Railway deployment
# Reads PORT from environment variable (Railway sets this)

PORT=${PORT:-8000}
echo "Starting server on port $PORT"
exec uvicorn app:app --host 0.0.0.0 --port $PORT

