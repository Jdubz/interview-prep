"""
Module 06 — Testing & Quality: Exercises
=========================================

A mini FastAPI app is provided below. Write the test functions.

Run with:  pytest exercises.py -v
Requires:  fastapi, httpx, pytest, pytest-asyncio
Configure: [tool.pytest.ini_options] asyncio_mode = "auto"

Replace `pass` with your implementation. Use only fastapi, httpx, pytest, and stdlib.
"""
from __future__ import annotations

import uuid
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

# =============================================================================
# THE APP UNDER TEST -- do NOT modify this section
# =============================================================================

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    price: float = Field(..., gt=0, le=10_000)
    category: str = Field(default="general")
    tags: list[str] = Field(default_factory=list)

class ItemResponse(BaseModel):
    id: str
    name: str
    price: float
    category: str
    tags: list[str]

items_db: dict[str, dict] = {}

def get_items_db() -> dict[str, dict]:
    return items_db

class WeatherService:
    """External weather API -- raises if called unoverridden in tests."""
    async def get_temperature(self, city: str) -> dict:
        raise RuntimeError("Override this dependency in your test!")

def get_weather_service() -> WeatherService:
    return WeatherService()

exercise_app = FastAPI(title="Exercise App")

@exercise_app.get("/items/", response_model=list[ItemResponse])
async def list_items(db: Annotated[dict, Depends(get_items_db)], category: str | None = None,
                     min_price: float | None = Query(None, ge=0),
                     max_price: float | None = Query(None, ge=0)):
    results = list(db.values())
    if category:
        results = [i for i in results if i["category"] == category]
    if min_price is not None:
        results = [i for i in results if i["price"] >= min_price]
    if max_price is not None:
        results = [i for i in results if i["price"] <= max_price]
    return [ItemResponse(**i) for i in results]

@exercise_app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str, db: Annotated[dict, Depends(get_items_db)]):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(**db[item_id])

@exercise_app.post("/items/", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate, db: Annotated[dict, Depends(get_items_db)]):
    for existing in db.values():
        if existing["name"].lower() == item.name.lower():
            raise HTTPException(status_code=409, detail="Item name already exists")
    item_id = str(uuid.uuid4())
    record = {"id": item_id, "name": item.name, "price": item.price,
              "category": item.category, "tags": item.tags}
    db[item_id] = record
    return ItemResponse(**record)

@exercise_app.get("/weather/{city}")
async def get_city_weather(city: str,
                           weather_svc: Annotated[WeatherService, Depends(get_weather_service)]):
    temp_data = await weather_svc.get_temperature(city)
    return {"city": city, "temperature": temp_data["temperature"],
            "unit": temp_data.get("unit", "celsius")}

@exercise_app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """On connect: {"type":"system","message":"Connected"}.
    On {"text":"X"}: {"type":"message","content":"X"}.
    On {"command":"ping"}: {"type":"pong"}."""
    await websocket.accept()
    await websocket.send_json({"type": "system", "message": "Connected"})
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "ping":
                await websocket.send_json({"type": "pong"})
            elif "text" in data:
                await websocket.send_json({"type": "message", "content": data["text"]})
    except WebSocketDisconnect:
        pass

# =============================================================================
# SHARED FIXTURES -- provided for you
# =============================================================================

@pytest.fixture
def test_db() -> dict[str, dict]:
    return {}

@pytest.fixture
def seeded_db() -> dict[str, dict]:
    """Keyboard $75 electronics, Notebook $4.50 stationery,
    Monitor $350 electronics, Pen $1.25 stationery."""
    return {
        "item-1": {"id": "item-1", "name": "Keyboard", "price": 75.00,
                    "category": "electronics", "tags": ["input", "peripheral"]},
        "item-2": {"id": "item-2", "name": "Notebook", "price": 4.50,
                    "category": "stationery", "tags": ["paper"]},
        "item-3": {"id": "item-3", "name": "Monitor", "price": 350.00,
                    "category": "electronics", "tags": ["display", "peripheral"]},
        "item-4": {"id": "item-4", "name": "Pen", "price": 1.25,
                    "category": "stationery", "tags": []},
    }

@pytest.fixture
async def client(test_db) -> AsyncClient:
    exercise_app.dependency_overrides[get_items_db] = lambda: test_db
    transport = ASGITransport(app=exercise_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    exercise_app.dependency_overrides.clear()

@pytest.fixture
async def seeded_client(seeded_db) -> AsyncClient:
    exercise_app.dependency_overrides[get_items_db] = lambda: seeded_db
    transport = ASGITransport(app=exercise_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    exercise_app.dependency_overrides.clear()

def build_item(**overrides) -> dict:
    defaults = {"name": f"Item-{uuid.uuid4().hex[:6]}", "price": 9.99,
                "category": "general", "tags": []}
    defaults.update(overrides)
    return defaults

# =============================================================================
# EXERCISE 1: Test GET /items/ (list, filter, 404)
# =============================================================================
# Use `seeded_client` (Keyboard $75, Notebook $4.50, Monitor $350, Pen $1.25).

class TestListItems:
    async def test_list_all_items(self, seeded_client: AsyncClient):
        """GET /items/ -> status 200, 4 items."""
        # YOUR CODE HERE
        pass

    async def test_filter_by_category(self, seeded_client: AsyncClient):
        """GET /items/?category=electronics -> 2 items (Keyboard, Monitor)."""
        # YOUR CODE HERE
        pass

    async def test_filter_by_price_range(self, seeded_client: AsyncClient):
        """GET /items/?min_price=5&max_price=100 -> 1 item (Keyboard $75)."""
        # YOUR CODE HERE
        pass

    async def test_filter_returns_empty_not_404(self, seeded_client: AsyncClient):
        """GET /items/?category=nonexistent -> status 200, empty list."""
        # YOUR CODE HERE
        pass

    async def test_get_item_by_id(self, seeded_client: AsyncClient):
        """GET /items/item-1 -> status 200, name "Keyboard"."""
        # YOUR CODE HERE
        pass

    async def test_get_item_not_found(self, seeded_client: AsyncClient):
        """GET /items/nonexistent -> status 404, detail "Item not found"."""
        # YOUR CODE HERE
        pass

# =============================================================================
# EXERCISE 2: Test POST /items/ with validation (valid, invalid, duplicate)
# =============================================================================

class TestCreateItem:
    async def test_create_valid_item(self, client: AsyncClient):
        """POST valid payload -> 201, response has id and matching name."""
        # YOUR CODE HERE
        pass

    async def test_create_item_with_tags(self, client: AsyncClient):
        """POST with tags=["sale","new"] -> 201, tags persisted in response."""
        # YOUR CODE HERE
        pass

    async def test_empty_name_rejected(self, client: AsyncClient):
        """POST name="" -> 422, error references 'name'."""
        # YOUR CODE HERE
        pass

    async def test_negative_price_rejected(self, client: AsyncClient):
        """POST price=-5 -> 422, error references 'price'."""
        # YOUR CODE HERE
        pass

    async def test_price_over_limit_rejected(self, client: AsyncClient):
        """POST price=10001 -> 422, error references 'price'."""
        # YOUR CODE HERE
        pass

    async def test_duplicate_name_rejected(self, client: AsyncClient):
        """Create "Widget", then create "widget" (case-insensitive) -> 409."""
        # YOUR CODE HERE
        pass

# =============================================================================
# EXERCISE 3: Override a dependency to mock an external service
# =============================================================================
# GET /weather/{city} depends on WeatherService (raises RuntimeError if real).
# Override get_weather_service with a fake, make request, verify response.

class TestWeatherEndpoint:
    async def test_get_weather_success(self, test_db):
        """Override WeatherService to return {"temperature": 22, "unit": "celsius"}.
        GET /weather/london -> {"city":"london","temperature":22,"unit":"celsius"}.
        Hint: create FakeWeatherService, set dependency_overrides, make AsyncClient."""
        # YOUR CODE HERE
        pass

    async def test_get_weather_city_specific(self, test_db):
        """Override so london->15, cairo->35. Make two requests, verify each."""
        # YOUR CODE HERE
        pass

# =============================================================================
# EXERCISE 4: Test WebSocket endpoint communication
# =============================================================================
# Use TestClient(exercise_app) -- httpx has no WS support.
# /ws/chat sends welcome, echoes messages, responds to ping.

class TestWebSocketChat:
    def test_connect_receives_welcome(self):
        """Connect -> first message is {"type":"system","message":"Connected"}."""
        # YOUR CODE HERE
        pass

    def test_echo_message(self):
        """Send {"text":"hello"} -> {"type":"message","content":"hello"}.
        Remember to receive the welcome message first."""
        # YOUR CODE HERE
        pass

    def test_ping_pong(self):
        """Send {"command":"ping"} -> {"type":"pong"}."""
        # YOUR CODE HERE
        pass

    def test_multiple_messages(self):
        """1. Receive welcome. 2. Send text->echo. 3. Send ping->pong. 4. Send text->echo."""
        # YOUR CODE HERE
        pass

# =============================================================================
# EXERCISE 5: Parametrized tests for input validation edge cases
# =============================================================================
# @pytest.mark.parametrize is the pytest equivalent of Vitest's test.each().

# 5a: Valid item creation -- all should return 201
@pytest.mark.parametrize("payload", [
    # Fill in test cases with pytest.param(..., id="..."):
    #   - name="A", price=0.01            (minimum valid values)
    #   - name="X"*50, price=10000         (maximum valid values)
    #   - name="Widget", price=49.99, category="tools"
    #   - name="Sale Item", price=1, tags=["sale"]
])
async def test_valid_item_creation(client: AsyncClient, payload: dict):
    """Each payload should succeed with 201."""
    # YOUR CODE HERE
    pass

# 5b: Invalid item creation -- all should return 422
@pytest.mark.parametrize("payload,error_field", [
    # Fill in test cases with pytest.param(..., id="..."):
    #   - empty name ""                    -> "name"
    #   - name too long "X"*51             -> "name"
    #   - zero price 0                     -> "price"
    #   - negative price -5                -> "price"
    #   - price over limit 10001           -> "price"
    #   - missing name field               -> "name"
    #   - wrong type for price "not a num" -> "price"
])
async def test_invalid_item_creation(client: AsyncClient, payload: dict, error_field: str):
    """Each payload should be rejected with 422 mentioning the error field."""
    # YOUR CODE HERE
    pass
