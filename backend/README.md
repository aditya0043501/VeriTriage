# VeriTriage Backend

FastAPI backend for clinical pre-visit intake tool.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --reload
```

## API Endpoints
- POST /api/chat - Main chat endpoint for patient interaction
- GET /api/health - Health check