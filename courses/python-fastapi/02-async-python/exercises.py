"""
Module 02: Async Python Deep Dive -- Exercises
===============================================

Skeleton functions with TODOs. Implement each exercise, then run:
    python exercises.py

All exercises use only stdlib asyncio (no external packages).
Assumes Python 3.11+ for TaskGroup and asyncio.timeout.

Each exercise includes:
- A docstring explaining expected behavior
- Stub code with TODO markers
- A test function that validates your implementation
"""

import asyncio
import time


# ---------------------------------------------------------------------------
# Helpers (used by multiple exercises -- do NOT modify)
# ---------------------------------------------------------------------------

async def _simulated_fetch(url: str, *, delay: float | None = None) -> dict:
    """Simulate an async HTTP GET. Returns dict with url, status, data."""
    d = delay if delay is not None else 0.1
    await asyncio.sleep(d)
    # Simulate occasional failures for testing error handling
    if "fail" in url:
        raise ConnectionError(f"Failed to fetch {url}")
    return {"url": url, "status": 200, "data": f"Response from {url}"}


def _sync_fetch(url: str) -> dict:
    """A deliberately BLOCKING fetch (uses time.sleep). Do NOT use in async code."""
    time.sleep(0.1)
    return {"url": url, "status": 200, "data": f"Sync response from {url}"}


# ===========================================================================
# EXERCISE 1: Convert Synchronous Code to Async
# ===========================================================================

def fetch_all_sync(urls: list[str]) -> list[dict]:
    """REFERENCE: synchronous version -- fetches URLs one at a time.
    This takes ~N * 0.1s for N URLs. Your async version should be faster.
    """
    results = []
    for url in urls:
        results.append(_sync_fetch(url))
    return results


async def fetch_all_async(urls: list[str]) -> list[dict]:
    """Convert the synchronous fetch_all_sync to async.

    Requirements:
    - Use _simulated_fetch (the async helper) instead of _sync_fetch
    - Fetch ALL urls concurrently using asyncio.gather
    - Return results in the same order as the input urls
    - Should complete in ~0.1s regardless of how many URLs (concurrent!)

    Expected behavior:
        >>> results = await fetch_all_async(["https://a.com", "https://b.com"])
        >>> len(results)
        2
        >>> results[0]["url"]
        'https://a.com'
    """
    # TODO: Replace this with a concurrent async implementation.
    # Hint: use asyncio.gather() with _simulated_fetch()
    raise NotImplementedError("Implement fetch_all_async")


# ===========================================================================
# EXERCISE 2: Fan-Out/Fan-In -- Fetch and Aggregate
# ===========================================================================

async def fan_out_fan_in(urls: list[str], max_concurrent: int = 5) -> dict:
    """Fetch multiple URLs concurrently with limited concurrency, aggregate results.

    Requirements:
    - Use asyncio.Semaphore to limit concurrency to max_concurrent
    - Use asyncio.gather with return_exceptions=True
    - Return a dict with:
        "successes": list of successful response dicts
        "failures":  list of dicts like {"url": url, "error": str(exception)}
    - Order does not matter within successes/failures

    Expected behavior:
        >>> result = await fan_out_fan_in(
        ...     ["https://a.com", "https://fail.com/x", "https://b.com"],
        ...     max_concurrent=2,
        ... )
        >>> len(result["successes"])
        2
        >>> len(result["failures"])
        1
        >>> result["failures"][0]["url"]
        'https://fail.com/x'
    """
    # TODO: Implement fan-out/fan-in with semaphore-limited concurrency.
    #
    # Step 1: Create an asyncio.Semaphore with max_concurrent
    # Step 2: Write a helper coroutine that acquires the semaphore, then
    #         calls _simulated_fetch. Wrap in try/except to catch errors
    #         and return either the result or an error dict.
    # Step 3: Use asyncio.gather to run all helpers concurrently.
    # Step 4: Separate successes from failures and return the aggregated dict.
    raise NotImplementedError("Implement fan_out_fan_in")


# ===========================================================================
# EXERCISE 3: Async Rate Limiter (Token Bucket)
# ===========================================================================

class AsyncRateLimiter:
    """Token bucket rate limiter for async code.

    Args:
        rate: number of tokens added per second
        capacity: maximum tokens the bucket can hold

    Requirements:
    - Tokens refill continuously based on elapsed time
    - acquire() should block (await) until a token is available
    - Must be safe for concurrent callers (use asyncio.Lock)
    - Bucket starts full (tokens == capacity)

    Expected behavior:
        >>> limiter = AsyncRateLimiter(rate=10, capacity=10)
        >>> # First 10 calls return immediately (bucket is full)
        >>> # 11th call blocks until a token refills (~0.1s)
    """

    def __init__(self, rate: float, capacity: int):
        # TODO: Initialize the rate limiter state.
        # Store rate, capacity, current token count, last refill timestamp.
        # Use asyncio.Lock for thread-safe refill + consume.
        raise NotImplementedError("Implement __init__")

    def _refill(self) -> None:
        """Add tokens based on time elapsed since last refill.

        Tokens = min(capacity, current_tokens + elapsed * rate)
        Update the last-refill timestamp.
        """
        # TODO: Calculate elapsed time, add tokens, cap at capacity.
        raise NotImplementedError("Implement _refill")

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until the requested number of tokens are available, then consume them.

        Algorithm:
        1. Lock
        2. Refill
        3. If enough tokens: consume and return
        4. Otherwise: release lock, sleep briefly, retry
        """
        # TODO: Implement the acquire loop.
        # Hint: use a while True loop. Inside the lock, call _refill(),
        # check if self._tokens >= tokens. If yes, subtract and return.
        # If no, release the lock and await asyncio.sleep(tokens / self._rate).
        raise NotImplementedError("Implement acquire")


# ===========================================================================
# EXERCISE 4: Timeout Wrapper
# ===========================================================================

async def with_timeout(coro, *, timeout_seconds: float, fallback=None):
    """Run a coroutine with a timeout. Return fallback if it times out.

    Requirements:
    - Use asyncio.timeout (Python 3.11+) or asyncio.wait_for
    - If the coroutine completes before the timeout, return its result
    - If it times out, return the fallback value (do NOT raise)
    - The timed-out coroutine must be cancelled (not left running)

    Expected behavior:
        >>> async def slow():
        ...     await asyncio.sleep(10)
        ...     return "done"
        >>> result = await with_timeout(slow(), timeout_seconds=0.1, fallback="timed out")
        >>> result
        'timed out'

        >>> async def fast():
        ...     await asyncio.sleep(0.01)
        ...     return "done"
        >>> result = await with_timeout(fast(), timeout_seconds=1.0)
        >>> result
        'done'
    """
    # TODO: Implement timeout wrapper.
    # Option A (Python 3.11+):
    #   try:
    #       async with asyncio.timeout(timeout_seconds):
    #           return await coro
    #   except TimeoutError:
    #       return fallback
    #
    # Option B (older Python):
    #   try:
    #       return await asyncio.wait_for(coro, timeout=timeout_seconds)
    #   except asyncio.TimeoutError:
    #       return fallback
    raise NotImplementedError("Implement with_timeout")


# ===========================================================================
# EXERCISE 5: Producer/Consumer Pipeline with Error Handling
# ===========================================================================

async def producer_consumer_pipeline(
    items: list[str],
    num_consumers: int = 3,
    max_queue_size: int = 5,
) -> dict:
    """Process items through an async producer/consumer pipeline.

    Architecture:
    - One producer: puts items into an asyncio.Queue
    - N consumers: take items from the queue, process them via _simulated_fetch
    - After all items are produced, send shutdown signals (None sentinels)
    - Collect successes and failures separately

    Requirements:
    - Queue must have max size for backpressure (max_queue_size)
    - Each consumer processes items in a loop until it receives None
    - Failures (items containing "fail") should be caught, not crash the pipeline
    - Use asyncio.TaskGroup for structured concurrency
    - Call queue.task_done() for every item (including sentinels)
    - Return {"successes": [...results...], "failures": [...error dicts...]}

    Expected behavior:
        >>> result = await producer_consumer_pipeline(
        ...     ["https://a.com", "https://fail.com/x", "https://b.com"],
        ...     num_consumers=2,
        ... )
        >>> len(result["successes"])
        2
        >>> len(result["failures"])
        1
    """
    # TODO: Implement the pipeline.
    #
    # 1. Create an asyncio.Queue with maxsize=max_queue_size
    # 2. Create shared lists for successes and failures
    # 3. Write an async producer(queue) that:
    #    - Puts each item into the queue
    #    - Then puts None once per consumer (shutdown signals)
    # 4. Write an async consumer(queue, cid) that:
    #    - Loops: gets item from queue
    #    - If item is None: call task_done() and break
    #    - Otherwise: try _simulated_fetch(item), append to successes
    #    - On exception: append {"url": item, "error": str(e)} to failures
    #    - Always call task_done()
    # 5. Use asyncio.TaskGroup to run 1 producer + num_consumers consumers
    # 6. Return the results dict
    raise NotImplementedError("Implement producer_consumer_pipeline")


# ===========================================================================
# Test Suite -- validates all exercises
# ===========================================================================

async def test_exercise_1() -> None:
    print("--- Exercise 1: Convert Sync to Async ---")
    urls = [f"https://example.com/{i}" for i in range(10)]

    t0 = time.perf_counter()
    results = await fetch_all_async(urls)
    elapsed = time.perf_counter() - t0

    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    assert all(r["status"] == 200 for r in results), "All should have status 200"
    assert results[0]["url"] == urls[0], "Results must preserve input order"
    assert elapsed < 0.5, f"Should be concurrent (<0.5s), took {elapsed:.2f}s"
    print(f"  PASS -- fetched {len(results)} URLs in {elapsed:.2f}s (concurrent)")


async def test_exercise_2() -> None:
    print("--- Exercise 2: Fan-Out/Fan-In ---")
    urls = [
        "https://api.example.com/a",
        "https://api.example.com/b",
        "https://fail.example.com/c",   # will fail
        "https://api.example.com/d",
        "https://fail.example.com/e",   # will fail
    ]

    result = await fan_out_fan_in(urls, max_concurrent=2)

    assert "successes" in result and "failures" in result, "Must return successes and failures"
    assert len(result["successes"]) == 3, f"Expected 3 successes, got {len(result['successes'])}"
    assert len(result["failures"]) == 2, f"Expected 2 failures, got {len(result['failures'])}"
    assert all("url" in f and "error" in f for f in result["failures"]), (
        "Each failure must have 'url' and 'error' keys"
    )
    print(f"  PASS -- {len(result['successes'])} successes, {len(result['failures'])} failures")


async def test_exercise_3() -> None:
    print("--- Exercise 3: Async Rate Limiter ---")
    limiter = AsyncRateLimiter(rate=10, capacity=5)

    # First 5 should be instant (bucket starts full)
    t0 = time.perf_counter()
    for _ in range(5):
        await limiter.acquire()
    burst_time = time.perf_counter() - t0
    assert burst_time < 0.1, f"Initial burst should be instant, took {burst_time:.2f}s"

    # Next 5 should take ~0.5s (10 tokens/sec, need 5 tokens)
    t1 = time.perf_counter()
    for _ in range(5):
        await limiter.acquire()
    throttled_time = time.perf_counter() - t1
    assert throttled_time > 0.3, f"Throttled requests should take >0.3s, took {throttled_time:.2f}s"
    print(f"  PASS -- burst: {burst_time:.3f}s, throttled: {throttled_time:.3f}s")


async def test_exercise_4() -> None:
    print("--- Exercise 4: Timeout Wrapper ---")

    async def fast_op():
        await asyncio.sleep(0.01)
        return "fast result"

    async def slow_op():
        await asyncio.sleep(10)
        return "slow result"

    # Test 1: fast operation completes
    coro = fast_op()
    try:
        result = await with_timeout(coro, timeout_seconds=1.0, fallback="timed out")
    except NotImplementedError:
        coro.close()
        raise
    assert result == "fast result", f"Expected 'fast result', got {result!r}"

    # Test 2: slow operation times out
    t0 = time.perf_counter()
    result = await with_timeout(slow_op(), timeout_seconds=0.1, fallback="timed out")
    elapsed = time.perf_counter() - t0
    assert result == "timed out", f"Expected 'timed out', got {result!r}"
    assert elapsed < 0.5, f"Should timeout quickly, took {elapsed:.2f}s"

    # Test 3: fallback defaults to None
    result = await with_timeout(slow_op(), timeout_seconds=0.1)
    assert result is None, f"Default fallback should be None, got {result!r}"
    print(f"  PASS -- fast op returned result, slow op returned fallback")


async def test_exercise_5() -> None:
    print("--- Exercise 5: Producer/Consumer Pipeline ---")
    items = [
        "https://api.example.com/1",
        "https://api.example.com/2",
        "https://fail.example.com/3",   # will fail
        "https://api.example.com/4",
        "https://api.example.com/5",
        "https://fail.example.com/6",   # will fail
        "https://api.example.com/7",
    ]

    result = await producer_consumer_pipeline(items, num_consumers=3, max_queue_size=3)

    assert "successes" in result and "failures" in result
    assert len(result["successes"]) == 5, f"Expected 5 successes, got {len(result['successes'])}"
    assert len(result["failures"]) == 2, f"Expected 2 failures, got {len(result['failures'])}"
    assert all("url" in f and "error" in f for f in result["failures"])
    print(f"  PASS -- {len(result['successes'])} successes, {len(result['failures'])} failures")


# ===========================================================================
# Main runner
# ===========================================================================

async def main() -> None:
    print("=" * 60)
    print("  Module 02: Async Python Exercises")
    print("  Implement the TODO sections, then run this file.")
    print("=" * 60)
    print()

    exercises = [
        ("Exercise 1", test_exercise_1),
        ("Exercise 2", test_exercise_2),
        ("Exercise 3", test_exercise_3),
        ("Exercise 4", test_exercise_4),
        ("Exercise 5", test_exercise_5),
    ]

    passed = 0
    failed = 0

    for name, test_fn in exercises:
        try:
            await test_fn()
            passed += 1
        except NotImplementedError:
            print(f"--- {name}: NOT IMPLEMENTED (skipped) ---")
        except AssertionError as e:
            print(f"--- {name}: FAILED -- {e} ---")
            failed += 1
        except Exception as e:
            print(f"--- {name}: ERROR -- {type(e).__name__}: {e} ---")
            failed += 1
        print()

    print("=" * 60)
    total = passed + failed
    not_impl = len(exercises) - total
    print(f"  Results: {passed} passed, {failed} failed, {not_impl} not implemented")
    if passed == len(exercises):
        print("  All exercises complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
