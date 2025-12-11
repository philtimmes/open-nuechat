# =============================================================================
# Nexus Chat - Makefile
# =============================================================================
# Alternative to control.sh for those who prefer make
# =============================================================================

.PHONY: help build start stop down restart logs status shell test clean dev db-migrate db-seed db-reset

# Default target
help:
	@echo "Nexus Chat - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build Docker images"
	@echo "  start          Start containers in background"
	@echo "  start-fg       Start containers in foreground"
	@echo "  stop           Stop running containers"
	@echo "  down           Stop and remove containers"
	@echo "  down-v         Stop, remove containers and volumes"
	@echo "  restart        Restart containers"
	@echo "  logs           View all logs"
	@echo "  logs-f         Follow all logs"
	@echo "  logs-backend   View backend logs"
	@echo "  logs-frontend  View frontend logs"
	@echo "  status         Show container status"
	@echo "  shell          Shell into backend container"
	@echo "  shell-frontend Shell into frontend container"
	@echo "  test           Run all tests"
	@echo "  test-backend   Run backend tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  clean          Remove containers and volumes"
	@echo "  clean-all      Remove everything including images"
	@echo "  dev            Start in development mode"
	@echo "  db-migrate     Run database migrations"
	@echo "  db-seed        Seed database"
	@echo "  db-reset       Reset database (destructive)"
	@echo ""

# =============================================================================
# Build Commands
# =============================================================================

build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

build-backend:
	docker compose build backend

build-frontend:
	docker compose build frontend

# =============================================================================
# Start/Stop Commands
# =============================================================================

start:
	docker compose up -d
	@echo ""
	@echo "Nexus Chat is running!"
	@echo "  Frontend: http://localhost"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo ""

start-fg:
	docker compose up

start-build:
	docker compose up -d --build

stop:
	docker compose stop

down:
	docker compose down

down-v:
	docker compose down -v

restart:
	docker compose restart

restart-backend:
	docker compose restart backend

restart-frontend:
	docker compose restart frontend

# =============================================================================
# Logs Commands
# =============================================================================

logs:
	docker compose logs --tail=100

logs-f:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

# =============================================================================
# Status Commands
# =============================================================================

status:
	@echo "Container Status:"
	@docker compose ps
	@echo ""

ps: status

# =============================================================================
# Shell Commands
# =============================================================================

shell:
	docker compose exec backend /bin/bash

shell-backend: shell

shell-frontend:
	docker compose exec frontend /bin/sh

exec:
	docker compose exec backend $(CMD)

# =============================================================================
# Test Commands
# =============================================================================

test:
	docker compose exec -T backend pytest tests/ -v

test-backend: test

test-frontend:
	docker compose exec -T frontend npm test -- --run

test-cov:
	docker compose exec -T backend pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-watch:
	docker compose exec backend pytest tests/ -v --watch

# =============================================================================
# Clean Commands
# =============================================================================

clean:
	docker compose down -v
	docker volume ls --filter "name=nexus-chat" -q | xargs -r docker volume rm 2>/dev/null || true

clean-all: clean
	docker images --filter "reference=nexus-chat*" -q | xargs -r docker rmi -f 2>/dev/null || true
	docker system prune -f

# =============================================================================
# Development Commands
# =============================================================================

dev:
	@if [ ! -f docker-compose.dev.yml ]; then \
		echo "Creating development compose file..."; \
		./control.sh dev --create-only; \
	fi
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-backend:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up backend

dev-frontend:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up frontend

# =============================================================================
# Database Commands
# =============================================================================

db-migrate:
	docker compose exec backend python -c "\
		from app.db.database import engine; \
		from app.models.models import Base; \
		import asyncio; \
		async def migrate(): \
			async with engine.begin() as conn: \
				await conn.run_sync(Base.metadata.create_all); \
			print('Migrations completed'); \
		asyncio.run(migrate())"

db-seed:
	docker compose exec backend python -c "\
		from app.db.seed import seed_database; \
		import asyncio; \
		asyncio.run(seed_database()); \
		print('Database seeded')"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose exec backend rm -f /app/data/nexus_chat.db
	docker compose restart backend
	@sleep 5
	@$(MAKE) db-migrate
	@echo "Database reset completed"

# =============================================================================
# Utility Commands
# =============================================================================

# Check if .env exists, copy from example if not
check-env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "Please edit .env with your configuration"; \
	fi

# Install local development dependencies (outside Docker)
install-local:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

# Format code
format:
	docker compose exec backend python -m black app/
	docker compose exec backend python -m isort app/

# Lint code
lint:
	docker compose exec backend python -m flake8 app/
	docker compose exec frontend npm run lint

# Generate API documentation
docs:
	@echo "API documentation available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc"
