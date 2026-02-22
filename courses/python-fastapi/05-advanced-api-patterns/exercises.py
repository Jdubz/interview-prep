"""
Advanced API Patterns Exercises

Skeleton functions with TODOs. Each exercise builds a real FastAPI pattern.
Uses only fastapi, pydantic, starlette, and the standard library.
"""
from __future__ import annotations

import asyncio, time, uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator

from fastapi import (
    BackgroundTasks, FastAPI, File, HTTPException,
    Request, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# ============================================================================
# EXERCISE 1: WebSocket Echo Server with Uppercase
# ============================================================================
# Accept connections, echo messages back UPPERCASED, track active connections.
# Test: websocat ws://localhost:8000/ws/echo

app_ws = FastAPI(title="Exercise 1: WebSocket Echo")
active_connections: list[WebSocket] = []

@app_ws.websocket("/ws/echo")
async def websocket_echo(ws: WebSocket):
    """
    1. Accept connection, add ws to active_connections
    2. Send welcome: {"type": "welcome", "message": "Connected!"}
    3. Loop: receive text, respond {"type": "echo", "original": ..., "uppercased": ...}
    4. On WebSocketDisconnect: remove ws from active_connections
    """
    # TODO: Accept the WebSocket connection
    # TODO: Add ws to active_connections
    # TODO: Send a JSON welcome message
    try:
        while True:
            pass
            # TODO: Receive text (ws.receive_text)
            # TODO: Send back JSON with original and uppercased text
    except WebSocketDisconnect:
        pass
        # TODO: Remove ws from active_connections

@app_ws.get("/ws/connections")
async def get_connection_count():
    """Return {"active_connections": <count>}."""
    # TODO: Return the count
    pass

# ============================================================================
# EXERCISE 2: SSE Progress Stream
# ============================================================================
# Stream progress 0-100% as Server-Sent Events (5% increments, 0.2s apart).
# Test: curl -N http://localhost:8000/progress/my_task

app_sse = FastAPI(title="Exercise 2: SSE Progress")

async def progress_generator(task_name: str) -> AsyncGenerator[str, None]:
    """
    Yield 21 SSE events (0%, 5%, ... 100%), then one "complete" event.
    Format per event:
        event: progress\nid: <n>\ndata: {"task_name":"...","percent":0}\n\n
    Sleep 0.2s between events.
    """
    # TODO: Loop 0..100 step 5, yield SSE-formatted strings
    # TODO: Yield final "complete" event
    pass

@app_sse.get("/progress/{task_name}")
async def stream_progress(task_name: str):
    """Return StreamingResponse with media_type="text/event-stream"."""
    # TODO: Return StreamingResponse using progress_generator
    pass

# ============================================================================
# EXERCISE 3: Background Task System
# ============================================================================
# POST /tasks -> 202 with job_id.  GET /tasks/{id} -> status + progress.

app_bg = FastAPI(title="Exercise 3: Background Tasks")

class TaskStatus(str, Enum):
    pending = "pending"; running = "running"
    completed = "completed"; failed = "failed"

class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    duration_seconds: int = Field(default=3, ge=1, le=30)

class TaskResponse(BaseModel):
    task_id: str; name: str; status: TaskStatus
    progress: int = 0; result: dict | None = None

_tasks: dict[str, dict] = {}

async def execute_task(task_id: str, duration: int) -> None:
    """
    1. Set status="running"
    2. 10 steps, sleep duration/10 each, update progress 10..100
    3. Set status="completed", result={"message":..., "steps":10}
    4. On exception: status="failed", store error
    """
    # TODO: Implement
    pass

@app_bg.post("/tasks", status_code=202, response_model=TaskResponse)
async def create_task(task: TaskCreate, background_tasks: BackgroundTasks):
    """Generate UUID, store initial state, schedule execute_task, return 202."""
    # TODO: Implement
    pass

@app_bg.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Return task status or 404."""
    # TODO: Implement
    pass

# ============================================================================
# EXERCISE 4: File Upload with Type/Size Validation
# ============================================================================
# Accept JPEG, PNG, PDF only. Reject > 5 MB. Stream to disk in 64 KB chunks.

app_upload = FastAPI(title="Exercise 4: File Upload")
ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_SIZE_BYTES = 5 * 1024 * 1024
DEST_DIR = Path("/tmp/exercise_uploads")

class FileUploadResponse(BaseModel):
    filename: str; size_bytes: int; content_type: str; saved_path: str

@app_upload.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    1. Validate content_type (422 if bad)
    2. Stream to disk in 64 KB chunks, track total bytes
    3. If total > MAX_SIZE_BYTES: delete partial file, raise 413
    4. Return FileUploadResponse
    """
    # TODO: Validate content type
    # TODO: Prepare destination (DEST_DIR / f"{uuid4().hex}_{file.filename}")
    # TODO: Stream to disk, enforce size limit, clean up on failure
    pass

# ============================================================================
# EXERCISE 5: In-Memory Rate Limiter Middleware
# ============================================================================
# Token bucket per client IP. 429 + Retry-After when empty.

app_rate = FastAPI(title="Exercise 5: Rate Limiter")

@dataclass
class RateBucket:
    tokens: float
    last_refill: float

@dataclass
class InMemoryRateLimiter:
    capacity: float
    refill_rate: float
    _buckets: dict[str, RateBucket] = field(default_factory=dict)

    def _get_or_create_bucket(self, key: str) -> RateBucket:
        """Get/create bucket, refill tokens based on elapsed time, cap at capacity."""
        # TODO: Implement
        pass

    def try_consume(self, key: str) -> bool:
        """Consume one token. True if allowed, False if rate-limited."""
        # TODO: Implement
        pass

    def get_retry_after(self, key: str) -> float:
        """Seconds until the next token is available."""
        # TODO: Implement
        pass

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_per_second: float = 5.0, burst: float = 10.0):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(capacity=burst, refill_rate=rate_per_second)

    async def dispatch(self, request: Request, call_next):
        """Extract IP, try_consume, return 429+Retry-After or call_next."""
        # TODO: Implement
        pass

# Uncomment to apply: app_rate.add_middleware(RateLimiterMiddleware, rate_per_second=2.0, burst=5.0)

@app_rate.get("/limited")
async def limited_endpoint():
    return {"message": "You got through the rate limiter!"}

# ============================================================================
# TESTS
# ============================================================================

def test_exercise_1():
    print("\n=== EX 1: WebSocket Echo ===")
    routes = [r.path for r in app_ws.routes]
    assert "/ws/echo" in routes and "/ws/connections" in routes
    print("Routes exist: PASS")

def test_exercise_2():
    print("\n=== EX 2: SSE Progress ===")
    import inspect
    assert inspect.isasyncgenfunction(progress_generator)
    print("Async generator: PASS")

def test_exercise_3():
    print("\n=== EX 3: Background Tasks ===")
    t = TaskResponse(task_id="x", name="test", status=TaskStatus.pending)
    assert t.status == TaskStatus.pending
    print("Model valid: PASS")

def test_exercise_4():
    print("\n=== EX 4: File Upload ===")
    assert "image/jpeg" in ALLOWED_TYPES and MAX_SIZE_BYTES == 5 * 1024 * 1024
    print("Constants correct: PASS")

def test_exercise_5():
    print("\n=== EX 5: Rate Limiter ===")
    lim = InMemoryRateLimiter(capacity=3.0, refill_rate=1.0)
    if lim.try_consume("t") is None:
        print("Not yet implemented: SKIP"); return
    assert all(lim.try_consume("t") for _ in range(2))  # 2nd and 3rd
    assert not lim.try_consume("t"), "4th should fail"
    assert lim.get_retry_after("t") > 0
    assert lim.try_consume("other"), "Separate client should pass"
    print("Token bucket: PASS")

if __name__ == "__main__":
    test_exercise_1(); test_exercise_2(); test_exercise_3()
    test_exercise_4(); test_exercise_5()
    print("\nStructure tests passed. Implement TODOs, then run:")
    for name in ["app_ws", "app_sse", "app_bg", "app_upload", "app_rate"]:
        print(f"  uvicorn exercises:{name} --reload")

"""
LEARNING OBJECTIVES

- [ ] WebSocket accept/send/receive and disconnect handling
- [ ] Async generators yielding SSE-formatted event streams
- [ ] BackgroundTasks with job status tracking and 202 Accepted
- [ ] File upload validation (type/size) with chunked streaming
- [ ] Token bucket rate limiting as BaseHTTPMiddleware
"""
