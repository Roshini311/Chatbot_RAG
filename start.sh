#!/bin/bash
# Exit script in case of error
set -e

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to initialize
sleep 5

# Port assigned by Render or default to 8501
STREAMLIT_PORT=${PORT:-8501}

# Start the Streamlit frontend
echo "Starting Streamlit frontend on port $STREAMLIT_PORT..."
streamlit run frontend/app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
