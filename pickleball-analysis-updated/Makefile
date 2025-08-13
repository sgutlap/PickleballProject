.PHONY: up build backend frontend

up:
	docker-compose up --build

build:
	docker-compose build

backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run start
