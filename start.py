#!/usr/bin/env python3
"""Startup script that reads PORT from environment variable."""
import os
import sys

port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")

print(f"Starting server on {host}:{port}")

# Import and run uvicorn
import uvicorn
uvicorn.run("app:app", host=host, port=port)

