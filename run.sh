#!/bin/bash
# Run the Nithin Kamath Post Generator

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting Nithin Kamath Post Generator..."
echo "Open http://localhost:8000 in your browser"
echo ""
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
