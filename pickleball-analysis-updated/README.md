# Pickleball Analysis — Local Run

## prerequisites
- Docker & Docker Compose (recommended) OR
- Node 18+ and npm, Python 3.11+ and pip

## Run with Docker (recommended)
1. From repo root: `docker-compose up --build`
2. Open frontend at `http://localhost:5173`
3. Backend health at `http://localhost:8000/health`
4. Upload a pickleball video using the frontend UI. The app will send frames to the backend for detection and show overlays and tips.

## Run without Docker
### Backend
1. `cd backend`
2. `python -m venv .venv && source .venv/bin/activate` (on Windows use `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run start`
4. Open `http://localhost:5173`

## Troubleshooting
- If the backend returns no ball detections often, try increasing lighting, ensure the ball color is bright orange/yellow, and camera not too far away.
- To change detection thresholds, edit `backend/app/detection.py` HSV ranges and area filters.
- If CORS errors appear, confirm frontend fetch URL is `http://localhost:8000` and backend is running.
