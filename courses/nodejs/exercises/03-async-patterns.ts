/**
 * Async Patterns Exercises
 *
 * Promises, concurrency control, and error handling patterns. These are
 * the async primitives that every Node.js backend developer needs to
 * understand and implement from scratch.
 *
 * Run:  npx tsx exercises/03-async-patterns.ts
 */


// ============================================================================
// EXERCISE 1: Promise.allSettled Polyfill
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (promises, async/await)
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (microtasks)
//
// Implement Promise.allSettled from scratch. It takes an array of promises
// and returns a promise that resolves when ALL input promises have settled
// (either fulfilled or rejected), with an array of result objects.
//
// Requirements:
//   - Returns a promise that always resolves (never rejects)
//   - Each result is { status: "fulfilled", value } or { status: "rejected", reason }
//   - Results are in the same order as the input promises
//   - Handles empty array input (resolves with [])
//   - Non-promise values are treated as already fulfilled
//
// Hints:
//   - Wrap each promise with .then() and .catch() to capture both outcomes
//   - Use Promise.all on the wrapped promises (they never reject)
//   - Handle non-promise values with Promise.resolve()
//
//   Pattern:
//     function allSettled<T>(promises: Promise<T>[]): Promise<SettledResult<T>[]> {
//       return Promise.all(
//         promises.map(p =>
//           Promise.resolve(p)
//             .then(value => ({ status: "fulfilled" as const, value }))
//             .catch(reason => ({ status: "rejected" as const, reason }))
//         )
//       );
//     }
//
//   Key concepts:
//   - Promise.resolve(x) wraps non-promises, passes through promises
//   - .then().catch() on each promise converts rejection to a fulfilled result
//   - Promise.all on never-rejecting promises always resolves
//
// Expected behavior:
//   const results = await allSettled([
//     Promise.resolve(1),
//     Promise.reject("error"),
//     Promise.resolve(3),
//   ]);
//   // [{ status: "fulfilled", value: 1 },
//   //  { status: "rejected", reason: "error" },
//   //  { status: "fulfilled", value: 3 }]

type SettledResult<T> =
  | { status: "fulfilled"; value: T }
  | { status: "rejected"; reason: unknown };

function allSettled<T>(promises: (T | Promise<T>)[]): Promise<SettledResult<T>[]> {
  if (promises.length === 0) return Promise.resolve([]);

  return Promise.all(
    promises.map((p) =>
      Promise.resolve(p)
        .then((value): SettledResult<T> => ({ status: "fulfilled", value }))
        .catch((reason): SettledResult<T> => ({ status: "rejected", reason })),
    ),
  );
}


// ============================================================================
// EXERCISE 2: Concurrent Pool
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (async patterns)
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (concurrency)
//   - ../08-performance-scaling/01-caching-and-redis.md
//
// Implement poolMap that processes an array of items through an async function
// with a maximum concurrency limit. Like Promise.all, but with backpressure.
//
// Requirements:
//   - poolMap(items, fn, concurrency) processes items with at most `concurrency`
//     concurrent invocations of fn
//   - Results are returned in the same order as input items
//   - If any fn rejects, poolMap rejects with that error
//   - All items are processed (unless one fails)
//
// Hints:
//   - Use an index counter and a recursive worker function
//   - Launch `concurrency` workers, each pulling the next item when done
//   - Store results in an array indexed by the original position
//
//   Pattern:
//     async function poolMap<T, R>(items: T[], fn: (item: T) => Promise<R>, concurrency: number): Promise<R[]> {
//       const results: R[] = new Array(items.length);
//       let nextIndex = 0;
//       async function worker() {
//         while (nextIndex < items.length) {
//           const i = nextIndex++;
//           results[i] = await fn(items[i]);
//         }
//       }
//       await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
//       return results;
//     }
//
//   Key concepts:
//   - Workers pull from a shared index (no need for a queue)
//   - Results array preserves order even though processing is concurrent
//   - Promise.all on the workers waits for all to finish
//
// Expected behavior:
//   const results = await poolMap([1, 2, 3, 4, 5], async (n) => n * 2, 2);
//   // results = [2, 4, 6, 8, 10], at most 2 running concurrently

async function poolMap<T, R>(
  _items: T[],
  _fn: (item: T) => Promise<R>,
  _concurrency: number,
): Promise<R[]> {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 3: Async Queue
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (task queues)
//   - ../09-architecture-patterns/02-event-driven-and-async-patterns.md
//
// Implement an async producer/consumer queue. Producers enqueue items,
// consumers dequeue them. If the queue is empty, dequeue() blocks (returns
// a promise) until an item is available.
//
// Requirements:
//   - enqueue(item) adds an item to the queue
//   - dequeue() returns a Promise<T> that resolves with the next item
//   - If items are available, dequeue() resolves immediately
//   - If queue is empty, dequeue() blocks until an item is enqueued
//   - Multiple pending dequeue() calls are served in FIFO order
//   - .size returns the number of items waiting in the queue
//
// Hints:
//   - Maintain two arrays: items[] and waiters[]
//   - waiters is an array of resolve functions from pending dequeue() promises
//   - On enqueue: if there's a waiter, resolve it immediately; else push to items
//   - On dequeue: if there's an item, resolve immediately; else create a promise
//
//   Pattern:
//     class AsyncQueue<T> {
//       private items: T[] = [];
//       private waiters: ((item: T) => void)[] = [];
//
//       enqueue(item: T): void {
//         const waiter = this.waiters.shift();
//         if (waiter) waiter(item);
//         else this.items.push(item);
//       }
//
//       dequeue(): Promise<T> {
//         const item = this.items.shift();
//         if (item !== undefined) return Promise.resolve(item);
//         return new Promise(resolve => this.waiters.push(resolve));
//       }
//     }
//
//   Key concepts:
//   - Extracting the resolve function from a Promise constructor
//   - FIFO ordering for both items and waiters
//   - This is the core of many message queue implementations
//
// Expected behavior:
//   const queue = new AsyncQueue<number>();
//   const p = queue.dequeue(); // blocks, no items yet
//   queue.enqueue(42);         // resolves the pending dequeue
//   await p;                   // 42

class AsyncQueue<T> {
  get size(): number {
    void this as AsyncQueue<T>;
    throw new Error("Not implemented");
  }

  enqueue(_item: T): void {
    throw new Error("Not implemented");
  }

  dequeue(): Promise<T> {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 4: Timeout Wrapper
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (promise patterns)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// Implement a timeout wrapper that rejects if a promise doesn't resolve
// within a given time limit.
//
// Requirements:
//   - withTimeout(promise, ms) returns a new promise
//   - If the original promise resolves within ms, return its value
//   - If it doesn't resolve within ms, reject with a TimeoutError
//   - Clean up the timer when the promise resolves (no leaking timers)
//   - TimeoutError should include the timeout duration in its message
//
// Hints:
//   - Use Promise.race between the original promise and a timeout promise
//   - The timeout promise rejects after ms milliseconds
//   - Use a finally or cleanup pattern to clear the timeout timer
//
//   Pattern:
//     function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
//       let timer: ReturnType<typeof setTimeout>;
//       const timeout = new Promise<never>((_, reject) => {
//         timer = setTimeout(() => reject(new TimeoutError(ms)), ms);
//       });
//       return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
//     }
//
//   Key concepts:
//   - Promise.race resolves/rejects with the first settled promise
//   - .finally() runs cleanup regardless of outcome
//   - Custom error class for specific error handling
//
// Expected behavior:
//   await withTimeout(Promise.resolve(42), 1000);  // 42
//   await withTimeout(new Promise(() => {}), 100);  // throws TimeoutError

class TimeoutError extends Error {
  constructor(ms: number) {
    super(`Timed out after ${ms}ms`);
    this.name = "TimeoutError";
  }
}

async function withTimeout<T>(_promise: Promise<T>, _ms: number): Promise<T> {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 5: Circuit Breaker
// ============================================================================
//
// RELATED READING:
//   - ../08-performance-scaling/01-caching-and-redis.md (resilience)
//   - ../09-architecture-patterns/02-event-driven-and-async-patterns.md
//
// Implement a circuit breaker that prevents calling a failing service
// repeatedly. It has three states: closed (normal), open (blocking calls),
// and half-open (testing if service recovered).
//
// Requirements:
//   - CircuitBreaker(opts) with { failureThreshold, resetTimeoutMs }
//   - .call(fn) executes fn when closed/half-open, rejects when open
//   - After failureThreshold consecutive failures, opens the circuit
//   - After resetTimeoutMs, transitions to half-open
//   - In half-open: one successful call closes the circuit; one failure reopens it
//   - .state returns "closed" | "open" | "half-open"
//
// Hints:
//   - Track consecutive failures and last failure time
//   - Closed: execute fn, count failures, open if threshold reached
//   - Open: check if resetTimeout has elapsed, reject or transition to half-open
//   - Half-open: execute fn, close on success, open on failure
//
//   Pattern:
//     class CircuitBreaker {
//       private failures = 0;
//       private lastFailure = 0;
//       private _state: "closed" | "open" | "half-open" = "closed";
//
//       async call<T>(fn: () => Promise<T>): Promise<T> {
//         if (this._state === "open") {
//           if (Date.now() - this.lastFailure >= this.resetTimeoutMs) {
//             this._state = "half-open";
//           } else {
//             throw new Error("Circuit is open");
//           }
//         }
//         try {
//           const result = await fn();
//           this.onSuccess();
//           return result;
//         } catch (e) {
//           this.onFailure();
//           throw e;
//         }
//       }
//     }
//
//   Key concepts:
//   - State machine with three states and well-defined transitions
//   - Consecutive failure counting (reset on success)
//   - Time-based transition from open to half-open
//
// Expected behavior:
//   const breaker = new CircuitBreaker({ failureThreshold: 2, resetTimeoutMs: 100 });
//   await breaker.call(async () => { throw new Error("fail"); }).catch(() => {});
//   await breaker.call(async () => { throw new Error("fail"); }).catch(() => {});
//   breaker.state; // "open"
//   await breaker.call(async () => "ok").catch(() => {}); // rejects, circuit open

class CircuitBreaker {
  get state(): "closed" | "open" | "half-open" {
    void this;
    throw new Error("Not implemented");
  }

  constructor(
    _opts: { failureThreshold: number; resetTimeoutMs: number },
  ) {
    throw new Error("Not implemented");
  }

  async call<T>(_fn: () => Promise<T>): Promise<T> {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 6: Event Loop Ordering
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (event loop phases)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// Predict and verify the execution order of mixed async primitives.
// This exercise tests your understanding of the Node.js event loop phases.
//
// Requirements:
//   - Implement predictOrder() that returns the expected execution order
//     as an array of strings for the given scenario
//   - Implement verifyOrder() that actually runs the scenario and captures
//     the real execution order
//   - The scenario mixes: setTimeout, setImmediate, process.nextTick,
//     queueMicrotask, and Promise.resolve().then()
//
// Hints:
//   - Node.js event loop priority (within the same phase):
//     1. process.nextTick (nextTick queue, highest priority microtask)
//     2. queueMicrotask / Promise.resolve().then() (microtask queue)
//     3. setTimeout(fn, 0) (timers phase)
//     4. setImmediate (check phase)
//   - Microtasks (nextTick + promise) run between every phase
//   - When called from the main module (not inside a callback), setTimeout
//     and setImmediate order is non-deterministic
//
//   The scenario to predict:
//     process.nextTick(() => log("nextTick"));
//     Promise.resolve().then(() => log("promise"));
//     queueMicrotask(() => log("microtask"));
//     setTimeout(() => log("timeout"), 0);
//     setImmediate(() => log("immediate"));
//
//   Expected order (from within a setTimeout callback):
//     ["nextTick", "promise", "microtask", "timeout", "immediate"]
//     or ["nextTick", "promise", "microtask", "immediate", "timeout"]
//
//   Key concepts:
//   - nextTick always runs before other microtasks
//   - Promise.then and queueMicrotask are both microtasks (same priority)
//   - Microtasks run before timers and immediates
//   - setTimeout vs setImmediate order depends on the context
//
// Expected behavior:
//   const predicted = predictOrder();
//   const actual = await verifyOrder();
//   // predicted and actual should match (or be acceptably close)

function predictOrder(): string[] {
  throw new Error("Not implemented");
}

function verifyOrder(): Promise<string[]> {
  throw new Error("Not implemented");
}


// ============================================================================
// TESTS
// ============================================================================

async function test_all_settled(): Promise<void> {
  console.log("\n=== EXERCISE 1: Promise.allSettled Polyfill ===");

  const results = await allSettled([
    Promise.resolve(1),
    Promise.reject("error"),
    Promise.resolve(3),
  ]);

  console.assert(results.length === 3, "3 results");
  console.assert(results[0].status === "fulfilled", "First fulfilled");
  console.assert(
    results[0].status === "fulfilled" && results[0].value === 1,
    "First value is 1",
  );
  console.assert(results[1].status === "rejected", "Second rejected");
  console.assert(
    results[1].status === "rejected" && results[1].reason === "error",
    "Second reason is 'error'",
  );
  console.assert(results[2].status === "fulfilled", "Third fulfilled");

  // Empty array
  const empty = await allSettled([]);
  console.assert(empty.length === 0, "Empty input returns empty array");

  // All succeed
  const allOk = await allSettled([Promise.resolve("a"), Promise.resolve("b")]);
  console.assert(allOk.every((r) => r.status === "fulfilled"), "All fulfilled");

  console.log("Edge cases passed");
  console.log("EXERCISE 1: PASSED");
}

async function test_pool_map(): Promise<void> {
  console.log("\n=== EXERCISE 2: Concurrent Pool ===");

  let maxConcurrent = 0;
  let current = 0;

  const results = await poolMap(
    [1, 2, 3, 4, 5],
    async (n) => {
      current++;
      maxConcurrent = Math.max(maxConcurrent, current);
      await new Promise((r) => setTimeout(r, 20));
      current--;
      return n * 2;
    },
    2,
  );

  console.assert(
    JSON.stringify(results) === JSON.stringify([2, 4, 6, 8, 10]),
    "Results in order",
  );
  console.assert(maxConcurrent <= 2, `Max concurrency respected: ${maxConcurrent}`);

  // Empty array
  const empty = await poolMap([], async (n: number) => n, 5);
  console.assert(empty.length === 0, "Empty input returns empty array");

  console.log("Edge cases passed");
  console.log("EXERCISE 2: PASSED");
}

async function test_async_queue(): Promise<void> {
  console.log("\n=== EXERCISE 3: Async Queue ===");

  const queue = new AsyncQueue<number>();

  // Enqueue before dequeue
  queue.enqueue(1);
  queue.enqueue(2);
  console.assert(queue.size === 2, "Size is 2");

  const v1 = await queue.dequeue();
  console.assert(v1 === 1, "First dequeue gets 1");
  console.assert(queue.size === 1, "Size is 1 after dequeue");

  const v2 = await queue.dequeue();
  console.assert(v2 === 2, "Second dequeue gets 2");

  // Dequeue before enqueue (blocking)
  const pending = queue.dequeue();
  queue.enqueue(42);
  const v3 = await pending;
  console.assert(v3 === 42, "Blocking dequeue resolved with 42");

  // Multiple waiters
  const p1 = queue.dequeue();
  const p2 = queue.dequeue();
  queue.enqueue(10);
  queue.enqueue(20);
  console.assert((await p1) === 10, "First waiter gets 10");
  console.assert((await p2) === 20, "Second waiter gets 20");

  console.log("Edge cases passed");
  console.log("EXERCISE 3: PASSED");
}

async function test_timeout(): Promise<void> {
  console.log("\n=== EXERCISE 4: Timeout Wrapper ===");

  // Fast promise resolves within timeout
  const result = await withTimeout(Promise.resolve(42), 1000);
  console.assert(result === 42, "Returns value within timeout");

  // Slow promise times out
  let threw = false;
  try {
    await withTimeout(new Promise(() => {}), 50);
  } catch (e) {
    threw = true;
    console.assert(e instanceof TimeoutError, "Throws TimeoutError");
  }
  console.assert(threw, "Timed out");

  // Rejected promise propagates error
  let rejectionCaught = false;
  try {
    await withTimeout(Promise.reject(new Error("original")), 1000);
  } catch (e) {
    rejectionCaught = true;
    console.assert((e as Error).message === "original", "Original error preserved");
  }
  console.assert(rejectionCaught, "Rejection propagated");

  console.log("Edge cases passed");
  console.log("EXERCISE 4: PASSED");
}

async function test_circuit_breaker(): Promise<void> {
  console.log("\n=== EXERCISE 5: Circuit Breaker ===");

  const breaker = new CircuitBreaker({
    failureThreshold: 2,
    resetTimeoutMs: 100,
  });

  console.assert(breaker.state === "closed", "Starts closed");

  // First failure
  try {
    await breaker.call(async () => {
      throw new Error("fail");
    });
  } catch {}
  console.assert(breaker.state === "closed", "Still closed after 1 failure");

  // Second failure — opens
  try {
    await breaker.call(async () => {
      throw new Error("fail");
    });
  } catch {}
  console.assert(breaker.state === "open", "Opens after 2 failures");

  // Call while open — rejected
  let openRejected = false;
  try {
    await breaker.call(async () => "ok");
  } catch {
    openRejected = true;
  }
  console.assert(openRejected, "Rejects when open");

  // Wait for reset timeout
  await new Promise((r) => setTimeout(r, 150));

  // Should be half-open, success closes it
  const result = await breaker.call(async () => "recovered");
  console.assert(result === "recovered", "Half-open call succeeds");
  console.assert(breaker.state === "closed", "Closes after half-open success");

  console.log("Edge cases passed");
  console.log("EXERCISE 5: PASSED");
}

async function test_event_loop_ordering(): Promise<void> {
  console.log("\n=== EXERCISE 6: Event Loop Ordering ===");

  const predicted = predictOrder();
  const actual = await verifyOrder();

  console.assert(Array.isArray(predicted), "predictOrder returns array");
  console.assert(Array.isArray(actual), "verifyOrder returns array");
  console.assert(actual.length > 0, "verifyOrder has results");

  // Check that nextTick comes before timeout and immediate
  const nextTickIdx = actual.indexOf("nextTick");
  const timeoutIdx = actual.indexOf("timeout");
  console.assert(
    nextTickIdx < timeoutIdx,
    "nextTick runs before timeout",
  );

  // Check prediction matches reality
  const match = JSON.stringify(predicted) === JSON.stringify(actual);
  if (match) {
    console.log("Prediction matches actual order!");
  } else {
    console.log(`Predicted: ${JSON.stringify(predicted)}`);
    console.log(`Actual:    ${JSON.stringify(actual)}`);
    console.log("(Close but not exact — see event loop docs)");
  }

  console.log("EXERCISE 6: PASSED");
}


if (require.main === module) {
  console.log("Async Patterns Exercises");
  console.log("=".repeat(60));

  const tests: [string, () => Promise<void>][] = [
    ["Exercise 1: Promise.allSettled Polyfill", test_all_settled],
    ["Exercise 2: Concurrent Pool", test_pool_map],
    ["Exercise 3: Async Queue", test_async_queue],
    ["Exercise 4: Timeout Wrapper", test_timeout],
    ["Exercise 5: Circuit Breaker", test_circuit_breaker],
    ["Exercise 6: Event Loop Ordering", test_event_loop_ordering],
  ];

  let passed = 0;
  let failed = 0;

  async function runAll() {
    for (const [name, testFn] of tests) {
      try {
        await testFn();
        passed++;
      } catch (e) {
        if (e instanceof Error && e.message === "Not implemented") {
          console.log(`  ${name}: NOT IMPLEMENTED`);
        } else if (e instanceof Error) {
          console.log(`  ${name}: FAILED -- ${e.message}`);
        } else {
          console.log(`  ${name}: ERROR -- ${e}`);
        }
        failed++;
      }
    }

    console.log();
    console.log("=".repeat(60));
    console.log(`Results: ${passed} passed, ${failed} failed out of ${tests.length}`);
    console.log("=".repeat(60));
  }

  runAll();
}
