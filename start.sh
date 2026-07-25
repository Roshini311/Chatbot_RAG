#!/bin/bash
set -e

# Port assigned by platform or default to 8501
STREAMLIT_PORT=${PORT:-8501}

echo "Starting FastAPI Backend API..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

sleep 5

echo "Starting Streamlit Frontend UI on port $STREAMLIT_PORT..."
streamlit run frontend/app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
