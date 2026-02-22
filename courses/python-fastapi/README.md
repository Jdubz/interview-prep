# Python & FastAPI

Advanced Python backend development with FastAPI. Covers async patterns, Pydantic data modeling, production API design, and testing strategies for senior engineers coming from TypeScript/Node.js.

> **Perspective**: You already know how to build APIs. This course focuses on what's different (and better) about Python's async ecosystem and FastAPI's approach compared to Express/NestJS.

---

## Modules

### 01 — FastAPI Foundations
- Request/response lifecycle vs Express middleware
- Path operations, dependency injection, and the DI container
- Pydantic v2 models: validation, serialization, computed fields
- Settings management with `pydantic-settings` (replaces dotenv)
- Automatic OpenAPI/Swagger generation

### 02 — Async Python Deep Dive
- `asyncio` event loop internals (vs Node.js event loop)
- `async`/`await` patterns: gather, TaskGroup, semaphores
- Async context managers and generators
- Connection pooling with async database drivers
- Structured concurrency with `anyio`

### 03 — Database Patterns with SQLAlchemy
- SQLAlchemy 2.0 async ORM (vs Prisma/TypeORM/Drizzle)
- Session management, unit of work pattern
- Alembic migrations (vs Prisma migrate)
- Raw SQL with `text()` and hybrid approaches
- Connection pooling and read replicas

### 04 — Authentication & Authorization
- OAuth2 with Password/Bearer flow in FastAPI
- JWT handling with `python-jose`
- Dependency-based auth (middleware equivalent)
- Role-based and permission-based access control
- API key management

### 05 — Advanced API Patterns
- Background tasks and task queues (Celery, ARQ)
- WebSocket support in FastAPI
- Server-Sent Events (SSE) for streaming
- File upload/download handling
- Rate limiting and throttling
- API versioning strategies

### 06 — Testing & Quality
- `pytest` + `pytest-asyncio` for async tests
- `httpx.AsyncClient` for integration testing (vs supertest)
- Fixtures, factories, and dependency overrides
- Coverage, mutation testing
- Type checking with `mypy` (strict mode)
- Linting with `ruff` (replaces flake8 + isort + black)

### 07 — Production Deployment
- Docker multi-stage builds for Python
- Uvicorn/Gunicorn worker configuration
- Structured logging with `structlog`
- Health checks, readiness probes
- OpenTelemetry instrumentation
- Error tracking (Sentry integration)

### 08 — Interview Prep
- Common Python backend interview questions
- System design: designing a FastAPI service at scale
- Code review exercises
- Live coding patterns (build an API endpoint from scratch)

---

## Prerequisites
- Solid understanding of REST API design
- Experience with at least one ORM (Prisma, TypeORM, etc.)
- Familiarity with Python basics (see `courses/llms/00-python-quickstart/` if needed)

## Status
**Scaffolded** — module outlines complete, content to be written.
