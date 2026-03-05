/**
 * Closures & Functions Exercises
 *
 * Higher-order functions, closures, memoization, and functional patterns.
 * These are the building blocks behind middleware, decorators, and every
 * utility library in the Node.js ecosystem.
 *
 * Run:  npx tsx exercises/02-closures-and-functions.ts
 */


// ============================================================================
// EXERCISE 1: Memoize
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (closures, higher-order functions)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// Implement a generic memoize function that caches results of single-argument
// functions. Uses a Map as the cache and returns the cached value on subsequent
// calls with the same argument.
//
// Requirements:
//   - Works with any single-argument function
//   - Caches results keyed by the argument
//   - Returns cached value without calling fn again on cache hit
//   - Preserves the return type of the original function
//
// Hints:
//   - Use a closure to capture a Map<A, R> cache
//   - Check cache.has(arg) before calling fn
//   - The returned function has the same signature as the input
//
//   Pattern — closure-based caching:
//     function memoize<A, R>(fn: (arg: A) => R): (arg: A) => R {
//       const cache = new Map<A, R>();
//       return (arg: A): R => {
//         if (cache.has(arg)) return cache.get(arg)!;
//         const result = fn(arg);
//         cache.set(arg, result);
//         return result;
//       };
//     }
//
//   Key concepts:
//   - The cache Map lives in the closure, persisting across calls
//   - Map.has() + Map.get() for cache lookup
//   - Generic types <A, R> preserve argument and return types
//
// Expected behavior:
//   let calls = 0;
//   const expensive = memoize((n: number) => { calls++; return n * n; });
//   expensive(5);  // calls = 1, returns 25
//   expensive(5);  // calls = 1 (cached), returns 25
//   expensive(3);  // calls = 2, returns 9

function memoize<A, R>(fn: (arg: A) => R): (arg: A) => R {
  const cache = new Map<A, R>();
  return (arg: A): R => {
    if (cache.has(arg)) return cache.get(arg)!;
    const result = fn(arg);
    cache.set(arg, result);
    return result;
  };
}


// ============================================================================
// EXERCISE 2: Debounce
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (closures, timers)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// Implement a debounce function that delays invoking fn until after `ms`
// milliseconds have elapsed since the last invocation. Each new call resets
// the timer. Returns a wrapper with a .cancel() method.
//
// Requirements:
//   - Delays execution by `ms` milliseconds
//   - Resets the timer on each subsequent call
//   - Returns a debounced function with a .cancel() method
//   - .cancel() prevents the pending invocation
//   - Uses the latest arguments when finally invoked
//
// Hints:
//   - Use setTimeout/clearTimeout in a closure
//   - Store the timer ID, clear it on each call, set a new one
//   - Attach .cancel() as a property on the returned function
//
//   Pattern:
//     function debounce<A extends unknown[]>(fn: (...args: A) => void, ms: number) {
//       let timer: ReturnType<typeof setTimeout> | null = null;
//       const debounced = (...args: A) => {
//         if (timer) clearTimeout(timer);
//         timer = setTimeout(() => { fn(...args); timer = null; }, ms);
//       };
//       debounced.cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
//       return debounced;
//     }
//
//   Key concepts:
//   - ReturnType<typeof setTimeout> handles Node vs browser timer types
//   - Rest parameters (...args: A) preserve argument types
//   - The closure captures both the timer ID and the latest args
//
// Expected behavior:
//   const results: number[] = [];
//   const debounced = debounce((n: number) => results.push(n), 100);
//   debounced(1); debounced(2); debounced(3);
//   // After 100ms: results = [3] (only the last call fires)
//   debounced.cancel(); // prevents any pending call

function debounce<A extends unknown[]>(
  _fn: (...args: A) => void,
  _ms: number,
): ((...args: A) => void) & { cancel: () => void } {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 3: Retry with Backoff
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (async patterns)
//   - ../08-performance-scaling/01-caching-and-redis.md (resilience patterns)
//
// Implement a retry function that calls an async function up to maxRetries
// times with configurable backoff between attempts.
//
// Requirements:
//   - Calls fn(), retries on failure up to maxRetries times
//   - Supports "exponential" and "linear" backoff strategies
//   - Exponential: delay = baseDelay * 2^attempt (with optional jitter)
//   - Linear: delay = baseDelay * (attempt + 1)
//   - Optional jitter adds random 0-50% extra delay
//   - Returns the successful result or throws the last error
//
// Hints:
//   - Use a for loop with try/catch
//   - Calculate delay based on strategy and attempt number
//   - Use a promise-based sleep: new Promise(r => setTimeout(r, ms))
//   - Jitter: delay *= (1 + Math.random() * 0.5)
//
//   Pattern:
//     async function retry<T>(fn: () => Promise<T>, opts: RetryOpts): Promise<T> {
//       let lastError: Error;
//       for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
//         try { return await fn(); }
//         catch (e) {
//           lastError = e as Error;
//           if (attempt < opts.maxRetries) {
//             let delay = opts.backoff === "exponential"
//               ? opts.baseDelay * 2 ** attempt
//               : opts.baseDelay * (attempt + 1);
//             if (opts.jitter) delay *= (1 + Math.random() * 0.5);
//             await new Promise(r => setTimeout(r, delay));
//           }
//         }
//       }
//       throw lastError!;
//     }
//
// Expected behavior:
//   let attempts = 0;
//   const result = await retry(
//     async () => { attempts++; if (attempts < 3) throw new Error("fail"); return "ok"; },
//     { maxRetries: 5, baseDelay: 10, backoff: "exponential" }
//   );
//   // result = "ok", attempts = 3

interface RetryOpts {
  maxRetries: number;
  baseDelay: number;
  backoff: "exponential" | "linear";
  jitter?: boolean;
}

async function retry<T>(_fn: () => Promise<T>, _opts: RetryOpts): Promise<T> {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 4: Pipe / Compose
// ============================================================================
//
// RELATED READING:
//   - ../00-ts-node-fundamentals.md (functional patterns)
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (function types)
//
// Implement a type-safe pipe function that chains functions together,
// passing the output of each as the input to the next.
//
// Requirements:
//   - pipe(f, g, h)(x) === h(g(f(x)))
//   - Type-safe: return type of each function matches input of the next
//   - Works with 1 to N functions
//   - Returns a single composed function
//
// Hints:
//   - Overload signatures for 1-5 functions give the best type inference
//   - Implementation uses Array.reduce
//   - Each function is (arg: any) => any at runtime
//
//   Pattern — overloaded pipe:
//     function pipe<A, B>(f1: (a: A) => B): (a: A) => B;
//     function pipe<A, B, C>(f1: (a: A) => B, f2: (b: B) => C): (a: A) => C;
//     function pipe<A, B, C, D>(f1: (a: A) => B, f2: (b: B) => C, f3: (c: C) => D): (a: A) => D;
//     function pipe(...fns: Function[]): Function {
//       return (x: unknown) => fns.reduce((acc, fn) => fn(acc), x);
//     }
//
//   Key concepts:
//   - Function overloads let TypeScript infer types through the chain
//   - reduce accumulates: fn3(fn2(fn1(x)))
//   - Each overload adds one more function to the chain
//
// Expected behavior:
//   const transform = pipe(
//     (s: string) => s.length,      // string -> number
//     (n: number) => n * 2,          // number -> number
//     (n: number) => n.toString(),   // number -> string
//   );
//   transform("hello"); // "10"

function pipe(..._fns: Function[]): Function {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 5: Rate Limiter (Token Bucket)
// ============================================================================
//
// RELATED READING:
//   - ../08-performance-scaling/01-caching-and-redis.md (rate limiting)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// Implement a token bucket rate limiter. The bucket has a maximum number of
// tokens and refills at a constant rate. Each request consumes one token.
//
// Requirements:
//   - createRateLimiter({ maxTokens, refillRate }) returns a limiter object
//   - .tryAcquire(): boolean — returns true if a token is available, false otherwise
//   - Tokens refill over time: refillRate tokens per second
//   - Bucket never exceeds maxTokens
//   - Uses Date.now() for timing (no setInterval)
//
// Hints:
//   - Store { tokens, lastRefill } in the closure
//   - On each tryAcquire(), calculate elapsed time and add tokens
//   - tokens = Math.min(maxTokens, tokens + elapsed * refillRate)
//   - If tokens >= 1, decrement and return true; else return false
//
//   Pattern:
//     function createRateLimiter(opts: { maxTokens: number; refillRate: number }) {
//       let tokens = opts.maxTokens;
//       let lastRefill = Date.now();
//       return {
//         tryAcquire(): boolean {
//           const now = Date.now();
//           const elapsed = (now - lastRefill) / 1000;
//           tokens = Math.min(opts.maxTokens, tokens + elapsed * opts.refillRate);
//           lastRefill = now;
//           if (tokens >= 1) { tokens--; return true; }
//           return false;
//         }
//       };
//     }
//
//   Key concepts:
//   - Lazy refill: calculate tokens on each call, not with a timer
//   - Date.now() returns milliseconds; divide by 1000 for seconds
//   - Math.min caps the bucket at maxTokens
//
// Expected behavior:
//   const limiter = createRateLimiter({ maxTokens: 3, refillRate: 1 });
//   limiter.tryAcquire(); // true (3 -> 2)
//   limiter.tryAcquire(); // true (2 -> 1)
//   limiter.tryAcquire(); // true (1 -> 0)
//   limiter.tryAcquire(); // false (0 tokens)
//   // After 1 second: limiter.tryAcquire() -> true (refilled 1 token)

function createRateLimiter(
  _opts: { maxTokens: number; refillRate: number },
): { tryAcquire: () => boolean } {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 6: LRU Cache
// ============================================================================
//
// RELATED READING:
//   - ../08-performance-scaling/01-caching-and-redis.md (caching strategies)
//   - ../10-interview-prep/01-interview-fundamentals.md (data structure design)
//
// Implement an LRU (Least Recently Used) cache with a fixed capacity.
// When the cache is full, evict the least recently used item.
//
// Requirements:
//   - LRUCache<K, V> with a capacity set in the constructor
//   - .get(key): V | undefined — returns value and marks as recently used
//   - .set(key, value) — adds/updates entry, evicts LRU if at capacity
//   - .delete(key): boolean — removes an entry
//   - .size: number — current number of entries
//   - O(1) for all operations (use Map's insertion-order iteration)
//
// Hints:
//   - JavaScript Map iterates in insertion order
//   - To "refresh" an entry: delete then re-insert it
//   - Map.keys().next().value gives the oldest (LRU) key
//   - This is the "Map trick" — no need for a separate doubly-linked list
//
//   Pattern:
//     class LRUCache<K, V> {
//       private cache = new Map<K, V>();
//       constructor(private capacity: number) {}
//       get(key: K): V | undefined {
//         if (!this.cache.has(key)) return undefined;
//         const value = this.cache.get(key)!;
//         this.cache.delete(key);     // remove
//         this.cache.set(key, value); // re-insert at end (most recent)
//         return value;
//       }
//       set(key: K, value: V): void {
//         this.cache.delete(key);     // remove if exists (to refresh order)
//         if (this.cache.size >= this.capacity) {
//           const lruKey = this.cache.keys().next().value!; // oldest entry
//           this.cache.delete(lruKey);
//         }
//         this.cache.set(key, value);
//       }
//     }
//
//   Key concepts:
//   - Map preserves insertion order (ES2015 spec)
//   - Delete + re-insert moves an entry to the "most recent" position
//   - .keys().next().value gets the first (oldest) key — O(1) amortized
//
// Expected behavior:
//   const cache = new LRUCache<string, number>(2);
//   cache.set("a", 1);
//   cache.set("b", 2);
//   cache.get("a");     // 1 (now "a" is most recent)
//   cache.set("c", 3);  // evicts "b" (least recently used)
//   cache.get("b");     // undefined

class LRUCache<K, V> {
  constructor(private _capacity: number) {
    void _capacity;
    throw new Error("Not implemented");
  }

  get(_key: K): V | undefined {
    throw new Error("Not implemented");
  }

  set(_key: K, _value: V): void {
    throw new Error("Not implemented");
  }

  delete(_key: K): boolean {
    throw new Error("Not implemented");
  }

  get size(): number {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// TESTS
// ============================================================================

function test_memoize(): void {
  console.log("\n=== EXERCISE 1: Memoize ===");

  let calls = 0;
  const expensive = memoize((n: number) => {
    calls++;
    return n * n;
  });

  console.assert(expensive(5) === 25, "First call computes");
  console.assert(calls === 1, "One call made");
  console.assert(expensive(5) === 25, "Second call returns cached");
  console.assert(calls === 1, "No additional call (cached)");
  console.assert(expensive(3) === 9, "Different arg computes");
  console.assert(calls === 2, "New call for new arg");

  // String keys
  const upper = memoize((s: string) => s.toUpperCase());
  console.assert(upper("hello") === "HELLO", "String memoize works");
  console.assert(upper("hello") === "HELLO", "String memoize cached");

  console.log("EXERCISE 1: PASSED");
}

async function test_debounce(): Promise<void> {
  console.log("\n=== EXERCISE 2: Debounce ===");

  const results: number[] = [];
  const debounced = debounce((n: number) => results.push(n), 50);

  debounced(1);
  debounced(2);
  debounced(3);

  // Should not have fired yet
  console.assert(results.length === 0, "Not fired immediately");

  await new Promise((r) => setTimeout(r, 80));
  console.assert(results.length === 1, "Fired once after delay");
  console.assert(results[0] === 3, "Used latest args");

  // Test cancel
  debounced(4);
  debounced.cancel();
  await new Promise((r) => setTimeout(r, 80));
  console.assert(results.length === 1, "Cancelled call did not fire");

  console.log("Edge cases passed");
  console.log("EXERCISE 2: PASSED");
}

async function test_retry(): Promise<void> {
  console.log("\n=== EXERCISE 3: Retry with Backoff ===");

  let attempts = 0;
  const result = await retry(
    async () => {
      attempts++;
      if (attempts < 3) throw new Error("fail");
      return "ok";
    },
    { maxRetries: 5, baseDelay: 10, backoff: "exponential" },
  );

  console.assert(result === "ok", "Returns successful result");
  console.assert(attempts === 3, "Took 3 attempts");

  // Should throw after exhausting retries
  let threw = false;
  try {
    await retry(async () => { throw new Error("always fails"); }, {
      maxRetries: 2,
      baseDelay: 10,
      backoff: "linear",
    });
  } catch {
    threw = true;
  }
  console.assert(threw, "Throws after max retries exhausted");

  console.log("Edge cases passed");
  console.log("EXERCISE 3: PASSED");
}

function test_pipe(): void {
  console.log("\n=== EXERCISE 4: Pipe / Compose ===");

  const transform = pipe(
    (s: string) => s.length,
    (n: number) => n * 2,
    (n: number) => n.toString(),
  );

  console.assert(transform("hello") === "10", "pipe chains 3 functions");

  // Single function
  const identity = pipe((x: number) => x + 1);
  console.assert(identity(5) === 6, "Single function pipe");

  // Two functions
  const double = pipe(
    (n: number) => n * 2,
    (n: number) => `Result: ${n}`,
  );
  console.assert(double(21) === "Result: 42", "Two function pipe");

  console.log("Edge cases passed");
  console.log("EXERCISE 4: PASSED");
}

function test_rate_limiter(): void {
  console.log("\n=== EXERCISE 5: Rate Limiter ===");

  const limiter = createRateLimiter({ maxTokens: 3, refillRate: 100 });

  console.assert(limiter.tryAcquire() === true, "Token 1 available");
  console.assert(limiter.tryAcquire() === true, "Token 2 available");
  console.assert(limiter.tryAcquire() === true, "Token 3 available");
  console.assert(limiter.tryAcquire() === false, "No tokens left");
  console.assert(limiter.tryAcquire() === false, "Still no tokens");

  console.log("Edge cases passed");
  console.log("EXERCISE 5: PASSED");
}

function test_lru_cache(): void {
  console.log("\n=== EXERCISE 6: LRU Cache ===");

  const cache = new LRUCache<string, number>(2);
  cache.set("a", 1);
  cache.set("b", 2);

  console.assert(cache.get("a") === 1, "Get existing key");
  console.assert(cache.size === 2, "Size is 2");

  cache.set("c", 3); // evicts "b" (LRU)
  console.assert(cache.get("b") === undefined, "b was evicted");
  console.assert(cache.get("c") === 3, "c exists");
  console.assert(cache.get("a") === 1, "a still exists (was refreshed)");

  // Update existing
  cache.set("a", 10);
  console.assert(cache.get("a") === 10, "Updated value");
  console.assert(cache.size === 2, "Size unchanged after update");

  // Delete
  console.assert(cache.delete("a") === true, "Delete returns true");
  console.assert(cache.get("a") === undefined, "Deleted key gone");
  console.assert(cache.size === 1, "Size decreased");

  console.log("Edge cases passed");
  console.log("EXERCISE 6: PASSED");
}


if (require.main === module) {
  console.log("Closures & Functions Exercises");
  console.log("=".repeat(60));

  const syncTests: [string, () => void][] = [
    ["Exercise 1: Memoize", test_memoize],
    ["Exercise 4: Pipe / Compose", test_pipe],
    ["Exercise 5: Rate Limiter", test_rate_limiter],
    ["Exercise 6: LRU Cache", test_lru_cache],
  ];

  const asyncTests: [string, () => Promise<void>][] = [
    ["Exercise 2: Debounce", test_debounce],
    ["Exercise 3: Retry with Backoff", test_retry],
  ];

  let passed = 0;
  let failed = 0;

  async function runAll() {
    // Run sync tests
    for (const [name, testFn] of syncTests) {
      try {
        testFn();
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

    // Run async tests
    for (const [name, testFn] of asyncTests) {
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
    console.log(`Results: ${passed} passed, ${failed} failed out of ${syncTests.length + asyncTests.length}`);
    console.log("=".repeat(60));
  }

  runAll();
}
