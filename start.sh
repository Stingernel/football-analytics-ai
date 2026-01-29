#!/bin/bash
# Startup script for monolithic deployment (API + Streamlit)

echo "🚀 Starting Football Analytics System..."

# 1. Start FastAPI Backend in Background
# We bind to 0.0.0.0:8000 (Internal Localhost Port)
echo "Starting API on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!

# Wait for API to warm up
echo "Waiting for API startup..."
sleep 5

# 2. Start Streamlit Frontend in Foreground
# We bind to 0.0.0.0:$PORT (Port assigned by Railway/Render)
# If PORT is not set, default to 8501
PORT=${PORT:-8501}
echo "Starting Dashboard on port $PORT..."

# Ensure API_URL points to localhost for internal communication
export API_URL="http://localhost:8000/api"

streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
