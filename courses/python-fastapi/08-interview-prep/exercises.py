"""
Interview Prep Exercises: Timed Coding Challenges

Six interview-style exercises with skeleton code, clear requirements,
and test functions. Time yourself — these simulate real interview pacing.
Uses only: fastapi, pydantic, and Python standard library.
"""
from __future__ import annotations
import hashlib, time, uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from fastapi import FastAPI, HTTPException, Query, Path, Request, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# EXERCISE 1: TODO API with CRUD + Filtering (15 min)
# ============================================================================
# POST /todos          -> Create (title 1-200 chars, description optional max 1000)
# GET  /todos          -> List, filterable by ?status=pending|completed
# GET  /todos/{id}     -> Get single todo by ID
# PATCH /todos/{id}    -> Update title, description, or completed (bool)
# DELETE /todos/{id}   -> Delete (204)
# TodoResponse: id (int), title, description, completed (bool), created_at (str)
# In-memory dict, auto-incrementing int IDs. Return 404/201/204 as appropriate.

app1 = FastAPI(title="TODO API")
_todos: dict[int, dict[str, Any]] = {}
_todo_counter: int = 0

class TodoCreate(BaseModel):
    pass  # TODO: title (required 1-200), description (optional, max 1000)

class TodoUpdate(BaseModel):
    pass  # TODO: title, description, completed — all optional

class TodoResponse(BaseModel):
    pass  # TODO: id, title, description, completed, created_at

# TODO: Implement all five endpoints on app1

# ============================================================================
# EXERCISE 2: User Registration with Validation (15 min)
# ============================================================================
# POST /register -> username (3-30 alphanumeric), email (must have @),
#                   password (8+ chars, needs digit + uppercase). Hash with sha256.
# POST /login    -> Validate creds, return {"token": "..."}. 401 on failure.
# GET  /me       -> Read Authorization: Bearer <token> header. Return user info.
# Duplicate username/email -> 409. Never expose password_hash in responses.

app2 = FastAPI(title="User Registration")
_users: dict[str, dict[str, Any]] = {}
_tokens: dict[str, str] = {}

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _make_token(username: str) -> str:
    return hashlib.sha256(f"{username}:{datetime.utcnow()}:{uuid.uuid4()}".encode()).hexdigest()

class RegisterRequest(BaseModel):
    pass  # TODO: username, email, password with validators

class LoginRequest(BaseModel):
    pass  # TODO: username, password

class UserResponse(BaseModel):
    pass  # TODO: id, username, email, created_at (no password!)

# TODO: Implement three endpoints on app2

# ============================================================================
# EXERCISE 3: Rate-Limited API Endpoint (20 min)
# ============================================================================
# Implement TokenBucket(capacity, refill_rate_per_second):
#   consume() -> (allowed: bool, tokens_remaining: float)
#   Tokens refill continuously based on elapsed time, capped at capacity.
# Middleware: per-client (by IP), 10 capacity, 1 tok/sec. Skip /health.
# Blocked -> 429 + Retry-After. Success -> X-RateLimit-Remaining header.

app3 = FastAPI(title="Rate Limited API")

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        # TODO: init tokens + last_refill

    def consume(self) -> tuple[bool, float]:
        pass  # TODO: refill, consume, return (allowed, remaining)

_buckets: dict[str, TokenBucket] = {}
def _get_bucket(ip: str) -> TokenBucket:
    pass  # TODO: get or create (capacity=10, rate=1.0)

# TODO: Add middleware to app3 (skip /health)

@app3.get("/resource")
async def get_resource():
    return {"data": "protected resource", "timestamp": datetime.utcnow().isoformat()}

@app3.get("/health")
async def health_check():
    return {"status": "ok"}

# ============================================================================
# EXERCISE 4: Debug Broken Code (10 min)
# ============================================================================
# Bookmark manager with 5 bugs. Find and fix them all.

app4_buggy = FastAPI(title="Bookmark Manager (BUGGY)")
_bookmarks: dict[int, dict[str, Any]] = {}
_bm_counter: int = 0

class BookmarkCreate(BaseModel):
    url: str = Field(min_length=1)       # BUG 1: no http/https validation
    title: str = Field(min_length=1, max_length=200)
    tags: list[str] = []

class BookmarkResponse(BaseModel):
    id: int; url: str; title: str; tags: list[str]; created_at: str

@app4_buggy.post("/bookmarks", response_model=BookmarkResponse)  # BUG 2: should be status_code=201
async def create_bookmark(bm: BookmarkCreate):
    global _bm_counter
    _bm_counter += 1
    _bookmarks[_bm_counter] = {"id": _bm_counter, "url": bm.url, "title": bm.title,
                                "tags": bm.tags, "created_at": datetime.utcnow().isoformat()}
    return BookmarkResponse(**_bookmarks[_bm_counter])

@app4_buggy.get("/bookmarks", response_model=list[BookmarkResponse])
async def list_bookmarks(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    all_bm = list(_bookmarks.values())
    return [BookmarkResponse(**b) for b in all_bm[skip:limit]]  # BUG 3: should be [skip:skip+limit]

@app4_buggy.get("/bookmarks/{bm_id}", response_model=BookmarkResponse)
async def get_bookmark(bm_id: int):
    bm = _bookmarks.get(bm_id)
    if bm == None:  # BUG 4: should be 'bm is None'
        raise HTTPException(404, detail="Bookmark not found")
    return BookmarkResponse(**bm)

@app4_buggy.delete("/bookmarks/{bm_id}", status_code=204)
async def delete_bookmark(bm_id: int):
    _bookmarks.pop(bm_id)  # BUG 5: KeyError if missing — needs default or guard

# ============================================================================
# EXERCISE 5: Code Review (10 min)
# ============================================================================
# Working but poorly written. List at least 8 improvements.

app5_review = FastAPI()
data = {}
id = 0

@app5_review.post("/user")
async def make_user(name: str, email: str, password: str, age: int):
    global id
    id = id + 1
    data[id] = {"id": id, "name": name, "email": email, "password": password,
                "age": age, "created": str(datetime.utcnow())}
    return data[id]

@app5_review.get("/user")
async def get_users():
    result = []
    for key in data: result.append(data[key])
    return result

@app5_review.get("/user/{id}")
async def get_user(id): return data[id]

@app5_review.post("/user/{id}")
async def update_user(id, name: str = None, email: str = None):
    if name: data[int(id)]["name"] = name
    if email: data[int(id)]["email"] = email
    return data[int(id)]

@app5_review.get("/user/search")
async def search_users(q: str):
    return [data[k] for k in data if q.lower() in data[k]["name"].lower()]

# IMPROVEMENTS TO IDENTIFY:
# 1. [Security] password stored & returned in plaintext
# 2. [Security] password as query param — visible in logs/URLs
# 3. [Naming] 'id' and 'data' shadow Python builtins
# 4. [Validation] no Pydantic models — zero input validation
# 5. [API design] POST for update should be PATCH or PUT
# 6. [Errors] get_user raises KeyError on missing ID (no 404)
# 7. [Routing] /user/search unreachable — /user/{id} matches first
# 8. [Types] id param has no type hint — no automatic validation

# ============================================================================
# EXERCISE 6: Webhook Delivery System with Retry (20 min)
# ============================================================================
# POST /webhooks             -> Register (url http/https, event_type, secret?)
# DELETE /webhooks/{id}      -> Unregister
# GET  /webhooks             -> List (?event_type filter)
# POST /events               -> Deliver to matching webhooks with retry
# GET  /deliveries/{eid}     -> Delivery report
# Retry: max 3 attempts, backoff [0, 1, 2]s (simulated). Stop on success.
# Delivery succeeds if "fail" NOT in webhook URL.

app6 = FastAPI(title="Webhook Delivery System")
_webhooks: dict[str, dict[str, Any]] = {}
_deliveries: dict[str, dict[str, Any]] = {}

class WebhookCreate(BaseModel):
    pass  # TODO: url (http/https), event_type, secret (optional)

class WebhookResponse(BaseModel):
    pass  # TODO: id, url, event_type, created_at

class EventTrigger(BaseModel):
    pass  # TODO: event_type (str), payload (dict)

class DeliveryAttempt(BaseModel):
    webhook_id: str; webhook_url: str; attempt: int
    status: str; scheduled_at: str; error: str | None = None

class DeliveryReport(BaseModel):
    event_id: str; event_type: str; deliveries: list[DeliveryAttempt]
    total_webhooks: int; successful: int; failed: int

def _simulate_delivery(url: str) -> tuple[bool, str | None]:
    pass  # TODO: succeed if 'fail' not in url

def _deliver_with_retry(wh: dict, payload: dict, max_attempts: int = 3) -> list[DeliveryAttempt]:
    pass  # TODO: attempt with backoff [0,1,2]s, stop on success

# TODO: Implement five endpoints on app6

# ============================================================================
# TESTS (run: python exercises.py)
# ============================================================================
def test_exercise_1():
    print("\n=== EXERCISE 1: TODO API ===")
    c = TestClient(app1)
    r = c.post("/todos", json={"title": "Buy milk", "description": "2% milk"})
    if r.status_code == 201:
        t = r.json(); tid = t["id"]
        print(f"  Created: {t}")
        print(f"  Get: {c.get(f'/todos/{tid}').json()}")
        print(f"  Updated: {c.patch(f'/todos/{tid}', json={'completed': True}).json()}")
        print(f"  Completed: {len(c.get('/todos?status=completed').json())} items")
        print(f"  Deleted: {c.delete(f'/todos/{tid}').status_code}")
    else: print(f"  Not implemented yet ({r.status_code})")

def test_exercise_2():
    print("\n=== EXERCISE 2: User Registration ===")
    c = TestClient(app2)
    r = c.post("/register", json={"username": "alice", "email": "alice@ex.com", "password": "Secret123"})
    if r.status_code == 201:
        print(f"  Registered: {r.json()}")
        r = c.post("/login", json={"username": "alice", "password": "Secret123"})
        tok = r.json().get("token", "")
        print(f"  Token: {tok[:16]}...")
        print(f"  Profile: {c.get('/me', headers={'Authorization': f'Bearer {tok}'}).json()}")
    else: print(f"  Not implemented yet ({r.status_code})")

def test_exercise_3():
    print("\n=== EXERCISE 3: Rate Limiter ===")
    b = TokenBucket(capacity=3, refill_rate=1.0)
    out = []
    for i in range(5):
        res = b.consume()
        if res: out.append(f"req{i+1}={'OK' if res[0] else 'BLOCKED'}")
    print(f"  Bucket: {', '.join(out)}" if out else "  Not implemented yet")

def test_exercise_4():
    print("\n=== EXERCISE 4: Debug ===")
    print("  5 bugs: 1) no URL validation  2) missing 201  3) wrong slice")
    print("          4) == None  5) pop() KeyError")

def test_exercise_5():
    print("\n=== EXERCISE 5: Code Review ===")
    print("  8+ improvements to find (see hints in code)")

def test_exercise_6():
    print("\n=== EXERCISE 6: Webhook Delivery ===")
    c = TestClient(app6)
    r = c.post("/webhooks", json={"url": "https://example.com/hook", "event_type": "order.created"})
    if r.status_code == 201:
        print(f"  Webhook: {r.json()}")
        c.post("/webhooks", json={"url": "https://fail.example.com/hook", "event_type": "order.created"})
        r = c.post("/events", json={"event_type": "order.created", "payload": {"order_id": 42}})
        ev = r.json(); eid = ev.get("event_id", "")
        if eid:
            rpt = c.get(f"/deliveries/{eid}").json()
            print(f"  {rpt.get('successful', '?')} ok, {rpt.get('failed', '?')} failed")
    else: print(f"  Not implemented yet ({r.status_code})")

if __name__ == "__main__":
    test_exercise_1(); test_exercise_2(); test_exercise_3()
    test_exercise_4(); test_exercise_5(); test_exercise_6()
    print("\n" + "=" * 60)
    print("Implement each exercise, then re-run to verify.")
    print("=" * 60)
