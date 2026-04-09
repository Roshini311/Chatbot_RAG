#!/bin/bash
# Exit script in case of error
set -e

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to initialize
sleep 5

# Start the Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
