/**
 * Node.js Runtime Exercises
 *
 * Node.js-specific patterns: EventEmitter, streams, graceful shutdown,
 * AsyncLocalStorage, and worker threads. These show you understand the
 * platform, not just the language.
 *
 * Run:  npx tsx exercises/05-node-runtime.ts
 */

import { EventEmitter } from "node:events";


// ============================================================================
// EXERCISE 1: EventEmitter from Scratch
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (event system)
//   - ../02-node-runtime/03-memory-streams-and-runtime-internals.md
//
// Implement a basic EventEmitter from scratch (without extending Node's
// EventEmitter). This tests your understanding of the observer pattern
// and how Node's event system works under the hood.
//
// Requirements:
//   - on(event, listener) registers a listener for an event
//   - off(event, listener) removes a specific listener
//   - emit(event, ...args) calls all listeners for that event with args
//   - once(event, listener) registers a listener that fires only once
//   - Supports multiple listeners per event
//   - Listeners are called in registration order
//   - emit returns true if there were listeners, false otherwise
//
// Hints:
//   - Use a Map<string, Function[]> to store listeners
//   - once() wraps the listener in a function that calls off() after firing
//   - Be careful with off() inside once() — the wrapper, not the original
//
//   Pattern:
//     class MyEventEmitter {
//       private events = new Map<string, ((...args: unknown[]) => void)[]>();
//
//       on(event: string, listener: (...args: unknown[]) => void): this {
//         if (!this.events.has(event)) this.events.set(event, []);
//         this.events.get(event)!.push(listener);
//         return this;
//       }
//
//       once(event: string, listener: (...args: unknown[]) => void): this {
//         const wrapper = (...args: unknown[]) => {
//           this.off(event, wrapper);
//           listener(...args);
//         };
//         return this.on(event, wrapper);
//       }
//
//       emit(event: string, ...args: unknown[]): boolean {
//         const listeners = this.events.get(event);
//         if (!listeners || listeners.length === 0) return false;
//         for (const listener of [...listeners]) { listener(...args); }
//         return true;
//       }
//     }
//
//   Key concepts:
//   - Spreading [...listeners] before iterating prevents issues if a
//     listener modifies the array (e.g., once() calling off())
//   - Returning `this` enables method chaining
//   - This is the core of Node.js's EventEmitter
//
// Expected behavior:
//   const ee = new MyEventEmitter();
//   ee.on("data", (msg) => console.log(msg));
//   ee.once("data", (msg) => console.log("once:", msg));
//   ee.emit("data", "hello");  // logs "hello" and "once: hello"
//   ee.emit("data", "world");  // logs "world" only (once listener removed)

type Listener = (...args: unknown[]) => void;

class MyEventEmitter {
  private events = new Map<string, Listener[]>();

  on(event: string, listener: Listener): this {
    if (!this.events.has(event)) this.events.set(event, []);
    this.events.get(event)!.push(listener);
    return this;
  }

  off(event: string, listener: Listener): this {
    const listeners = this.events.get(event);
    if (listeners) {
      const idx = listeners.indexOf(listener);
      if (idx !== -1) listeners.splice(idx, 1);
    }
    return this;
  }

  once(event: string, listener: Listener): this {
    const wrapper: Listener = (...args) => {
      this.off(event, wrapper);
      listener(...args);
    };
    return this.on(event, wrapper);
  }

  emit(event: string, ...args: unknown[]): boolean {
    const listeners = this.events.get(event);
    if (!listeners || listeners.length === 0) return false;
    for (const listener of [...listeners]) {
      listener(...args);
    }
    return true;
  }
}


// ============================================================================
// EXERCISE 2: Transform Stream
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/03-memory-streams-and-runtime-internals.md (streams)
//   - ../02-node-runtime/01-event-loop-and-task-queues.md
//
// Implement a transform stream that uppercases text chunks. This tests your
// understanding of Node.js streams and backpressure handling.
//
// Requirements:
//   - Extend Transform from 'node:stream'
//   - Override _transform(chunk, encoding, callback) to uppercase the chunk
//   - Handle both string and Buffer inputs
//   - Properly call callback() to signal completion
//   - Demonstrate usage by piping data through the transform
//
// Hints:
//   - import { Transform } from 'node:stream'
//   - chunk.toString() converts Buffer to string
//   - this.push(data) sends data downstream
//   - callback() signals this chunk is processed (backpressure signal)
//
//   Pattern:
//     import { Transform, TransformCallback } from 'node:stream';
//
//     class UppercaseTransform extends Transform {
//       _transform(chunk: Buffer | string, encoding: string, callback: TransformCallback): void {
//         const text = chunk.toString().toUpperCase();
//         this.push(text);
//         callback();
//       }
//     }
//
//   Key concepts:
//   - _transform is called for each chunk of data
//   - this.push() sends transformed data to the readable side
//   - callback() MUST be called or the stream will stall (backpressure)
//   - Streams process data incrementally, not all at once
//
// Expected behavior:
//   const transform = new UppercaseTransform();
//   transform.write("hello");
//   transform.read().toString(); // "HELLO"

// TODO: Import Transform from 'node:stream' and implement UppercaseTransform

class UppercaseTransform {
  constructor() {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 3: Graceful Shutdown
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/02-threading-and-process-management.md (signals)
//   - ../09-architecture-patterns/01-clean-architecture-and-ddd.md
//
// Implement a graceful shutdown manager that coordinates cleanup when the
// process receives SIGTERM or SIGINT. This is essential for production
// Node.js services.
//
// Requirements:
//   - register(name, cleanupFn) adds a cleanup handler
//   - shutdown() runs all cleanup handlers with a hard deadline
//   - Hooks into SIGTERM and SIGINT signals
//   - Cleanup handlers run in parallel
//   - If cleanup doesn't complete within deadlineMs, force exit
//   - Tracks shutdown state to prevent double-shutdown
//   - Returns cleanup results (which succeeded, which failed/timed out)
//
// Hints:
//   - Store handlers in a Map<string, () => Promise<void>>
//   - Use Promise.allSettled to run handlers in parallel
//   - Race the allSettled against a deadline timeout
//   - process.on('SIGTERM', () => this.shutdown()) for signal handling
//
//   Pattern:
//     class GracefulShutdown {
//       private handlers = new Map<string, () => Promise<void>>();
//       private isShuttingDown = false;
//
//       constructor(private deadlineMs: number = 10000) {
//         process.on('SIGTERM', () => this.shutdown());
//         process.on('SIGINT', () => this.shutdown());
//       }
//
//       register(name: string, handler: () => Promise<void>): void {
//         this.handlers.set(name, handler);
//       }
//
//       async shutdown(): Promise<{ succeeded: string[]; failed: string[] }> {
//         if (this.isShuttingDown) return { succeeded: [], failed: [] };
//         this.isShuttingDown = true;
//         // Run all handlers with deadline...
//       }
//     }
//
//   Key concepts:
//   - SIGTERM: "please stop" (sent by orchestrators like Kubernetes)
//   - SIGINT: Ctrl+C from the terminal
//   - Hard deadline prevents hanging on stuck cleanup
//   - Promise.allSettled + Promise.race for parallel with timeout
//
// Expected behavior:
//   const shutdown = new GracefulShutdown(5000);
//   shutdown.register("db", async () => { /* close connections */ });
//   shutdown.register("cache", async () => { /* flush cache */ });
//   const result = await shutdown.shutdown();
//   // result = { succeeded: ["db", "cache"], failed: [] }

class GracefulShutdown {
  constructor(_deadlineMs?: number) {
    throw new Error("Not implemented");
  }

  register(_name: string, _handler: () => Promise<void>): void {
    throw new Error("Not implemented");
  }

  async shutdown(): Promise<{ succeeded: string[]; failed: string[] }> {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 4: AsyncLocalStorage Context
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/01-event-loop-and-task-queues.md (async context)
//   - ../02-node-runtime/02-threading-and-process-management.md
//
// Implement a request context module using AsyncLocalStorage. This propagates
// context (like request IDs) through async call chains without passing it
// explicitly as a parameter.
//
// Requirements:
//   - RequestContext.run(ctx, fn) executes fn with the given context
//   - RequestContext.get() returns the current context (or undefined)
//   - Context propagates through async/await, setTimeout, Promise chains
//   - Each concurrent request has its own isolated context
//   - Context includes at least { requestId: string, startTime: number }
//
// Hints:
//   - import { AsyncLocalStorage } from 'node:async_hooks'
//   - AsyncLocalStorage.run(store, fn) executes fn with the given store
//   - AsyncLocalStorage.getStore() retrieves the current store
//
//   Pattern:
//     import { AsyncLocalStorage } from 'node:async_hooks';
//
//     interface RequestCtx {
//       requestId: string;
//       startTime: number;
//       [key: string]: unknown;
//     }
//
//     const storage = new AsyncLocalStorage<RequestCtx>();
//
//     const RequestContext = {
//       run<T>(ctx: RequestCtx, fn: () => T): T {
//         return storage.run(ctx, fn);
//       },
//       get(): RequestCtx | undefined {
//         return storage.getStore();
//       },
//     };
//
//   Key concepts:
//   - AsyncLocalStorage provides "thread-local" storage for async contexts
//   - It follows the async execution chain automatically
//   - No need to pass context through every function parameter
//   - Essential for logging, tracing, and request-scoped state
//
// Expected behavior:
//   RequestContext.run({ requestId: "abc", startTime: Date.now() }, async () => {
//     await someAsyncWork();
//     const ctx = RequestContext.get();
//     ctx.requestId; // "abc" — propagated through async calls
//   });

interface RequestCtx {
  requestId: string;
  startTime: number;
  [key: string]: unknown;
}

// TODO: Implement RequestContext using AsyncLocalStorage

const RequestContext = {
  run<T>(_ctx: RequestCtx, _fn: () => T): T {
    throw new Error("Not implemented");
  },
  get(): RequestCtx | undefined {
    throw new Error("Not implemented");
  },
};


// ============================================================================
// EXERCISE 5: Worker Thread Pool
// ============================================================================
//
// RELATED READING:
//   - ../02-node-runtime/02-threading-and-process-management.md (workers)
//   - ../08-performance-scaling/03-profiling-and-advanced-performance.md
//
// Implement a fixed-size worker thread pool that distributes CPU-bound tasks
// to worker threads and returns results via promises.
//
// Requirements:
//   - WorkerPool(size, workerScript) creates a pool of worker threads
//   - .exec(data) sends data to an available worker, returns Promise<result>
//   - If all workers are busy, queue the task until one is available
//   - Handle worker crashes: reject the pending promise, replace the worker
//   - .destroy() terminates all workers
//
// Hints:
//   - import { Worker } from 'node:worker_threads'
//   - Workers communicate via postMessage / on('message')
//   - Maintain a queue of pending tasks and a set of idle workers
//   - On 'exit' event: reject pending task, spawn replacement worker
//
//   Pattern:
//     import { Worker } from 'node:worker_threads';
//
//     class WorkerPool {
//       private workers: Worker[] = [];
//       private idle: Worker[] = [];
//       private queue: Array<{ data: unknown; resolve: Function; reject: Function }> = [];
//
//       constructor(private size: number, private script: string) {
//         for (let i = 0; i < size; i++) this.addWorker();
//       }
//
//       exec<T>(data: unknown): Promise<T> {
//         return new Promise((resolve, reject) => {
//           const worker = this.idle.pop();
//           if (worker) this.runTask(worker, data, resolve, reject);
//           else this.queue.push({ data, resolve, reject });
//         });
//       }
//     }
//
//   Key concepts:
//   - Worker threads have their own V8 instance and event loop
//   - postMessage serializes data (structured clone algorithm)
//   - Pool pattern prevents spawning too many threads
//   - Error recovery: replace crashed workers automatically
//
// Expected behavior:
//   const pool = new WorkerPool(2, './worker.js');
//   const results = await Promise.all([
//     pool.exec({ type: 'fibonacci', n: 40 }),
//     pool.exec({ type: 'fibonacci', n: 35 }),
//     pool.exec({ type: 'fibonacci', n: 30 }), // queued until a worker is free
//   ]);
//   pool.destroy();

class WorkerPool {
  constructor(_size: number, _workerScript: string) {
    throw new Error("Not implemented");
  }

  exec<T>(_data: unknown): Promise<T> {
    throw new Error("Not implemented");
  }

  destroy(): void {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 6: In-Memory Pub/Sub
// ============================================================================
//
// RELATED READING:
//   - ../09-architecture-patterns/02-event-driven-and-async-patterns.md
//   - ../02-node-runtime/01-event-loop-and-task-queues.md
//
// Implement an in-memory publish/subscribe event bus with topic subscriptions,
// wildcard matching, and async handler support.
//
// Requirements:
//   - subscribe(topic, handler) returns an unsubscribe function
//   - publish(topic, data) sends data to all matching subscribers
//   - Wildcard topics: "user.*" matches "user.created", "user.deleted"
//   - Double wildcard: "**" matches everything
//   - Handlers can be async — publish waits for all handlers to complete
//   - Multiple handlers per topic
//   - publish returns the number of handlers that were called
//
// Hints:
//   - Store subscriptions as { pattern: string, handler: Function, id: number }
//   - Convert pattern to regex: "user.*" -> /^user\.[^.]+$/
//   - "**" matches any topic -> /^.*$/
//   - Use Promise.allSettled to handle async handlers
//
//   Pattern:
//     class EventBus {
//       private subs: Array<{ pattern: RegExp; handler: Function; id: number }> = [];
//       private nextId = 0;
//
//       subscribe(topic: string, handler: (data: unknown) => void | Promise<void>) {
//         const id = this.nextId++;
//         const pattern = this.topicToRegex(topic);
//         this.subs.push({ pattern, handler, id });
//         return () => { this.subs = this.subs.filter(s => s.id !== id); };
//       }
//
//       async publish(topic: string, data: unknown): Promise<number> {
//         const matching = this.subs.filter(s => s.pattern.test(topic));
//         await Promise.allSettled(matching.map(s => s.handler(data)));
//         return matching.length;
//       }
//     }
//
//   Key concepts:
//   - Returning unsubscribe function avoids needing to track references
//   - Wildcard matching via regex is simple and flexible
//   - Promise.allSettled ensures one failing handler doesn't break others
//
// Expected behavior:
//   const bus = new EventBus();
//   const unsub = bus.subscribe("user.*", (data) => console.log(data));
//   await bus.publish("user.created", { id: 1 }); // handler called
//   await bus.publish("order.created", { id: 2 }); // handler NOT called
//   unsub(); // removes the subscription

class EventBus {
  subscribe(
    _topic: string,
    _handler: (data: unknown) => void | Promise<void>,
  ): () => void {
    throw new Error("Not implemented");
  }

  async publish(_topic: string, _data: unknown): Promise<number> {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// TESTS
// ============================================================================

function test_my_event_emitter(): void {
  console.log("\n=== EXERCISE 1: EventEmitter from Scratch ===");

  const ee = new MyEventEmitter();
  const received: unknown[] = [];

  ee.on("data", (msg) => received.push(msg));
  ee.emit("data", "hello");
  console.assert(received.length === 1, "One listener called");
  console.assert(received[0] === "hello", "Correct arg");

  // Multiple listeners
  ee.on("data", (msg) => received.push(`copy:${msg}`));
  ee.emit("data", "world");
  console.assert(received.length === 3, "Both listeners called");

  // off
  const handler = (msg: unknown) => received.push(`removable:${msg}`);
  ee.on("test", handler);
  ee.off("test", handler);
  console.assert(ee.emit("test", "x") === false, "No listeners after off");

  // once
  const onceResults: unknown[] = [];
  ee.once("once-event", (msg) => onceResults.push(msg));
  ee.emit("once-event", "first");
  ee.emit("once-event", "second");
  console.assert(onceResults.length === 1, "Once fires only once");
  console.assert(onceResults[0] === "first", "Once got first event");

  // emit return value
  console.assert(ee.emit("data", "x") === true, "emit returns true with listeners");
  console.assert(ee.emit("nonexistent") === false, "emit returns false without listeners");

  console.log("Edge cases passed");
  console.log("EXERCISE 1: PASSED");
}

function test_uppercase_transform(): void {
  console.log("\n=== EXERCISE 2: Transform Stream ===");

  const transform = new UppercaseTransform();

  // Write and read
  (transform as any).write("hello world");
  const result = (transform as any).read();
  console.assert(result !== null, "Got output");
  console.assert(result.toString() === "HELLO WORLD", "Uppercased");

  // Multiple chunks
  (transform as any).write("foo");
  (transform as any).write("bar");
  const r1 = (transform as any).read();
  const r2 = (transform as any).read();
  console.assert(r1.toString() === "FOO", "First chunk");
  console.assert(r2.toString() === "BAR", "Second chunk");

  console.log("Edge cases passed");
  console.log("EXERCISE 2: PASSED");
}

async function test_graceful_shutdown(): Promise<void> {
  console.log("\n=== EXERCISE 3: Graceful Shutdown ===");

  const shutdown = new GracefulShutdown(5000);
  const cleanedUp: string[] = [];

  shutdown.register("db", async () => {
    await new Promise((r) => setTimeout(r, 10));
    cleanedUp.push("db");
  });
  shutdown.register("cache", async () => {
    cleanedUp.push("cache");
  });

  const result = await shutdown.shutdown();

  console.assert(cleanedUp.includes("db"), "DB cleaned up");
  console.assert(cleanedUp.includes("cache"), "Cache cleaned up");
  console.assert(result.succeeded.length === 2, "2 succeeded");
  console.assert(result.failed.length === 0, "0 failed");

  // Double shutdown is a no-op
  const result2 = await shutdown.shutdown();
  console.assert(
    result2.succeeded.length === 0 && result2.failed.length === 0,
    "Double shutdown is no-op",
  );

  console.log("Edge cases passed");
  console.log("EXERCISE 3: PASSED");
}

async function test_request_context(): Promise<void> {
  console.log("\n=== EXERCISE 4: AsyncLocalStorage Context ===");

  // Outside of run, get() returns undefined
  console.assert(RequestContext.get() === undefined, "No context outside run");

  // Inside run, context is available
  const result = RequestContext.run(
    { requestId: "abc-123", startTime: Date.now() },
    () => {
      const ctx = RequestContext.get();
      return ctx?.requestId;
    },
  );
  console.assert(result === "abc-123", "Context available in run");

  // Context propagates through async
  await RequestContext.run(
    { requestId: "async-456", startTime: Date.now() },
    async () => {
      await new Promise((r) => setTimeout(r, 10));
      const ctx = RequestContext.get();
      console.assert(ctx?.requestId === "async-456", "Context survives async");
    },
  );

  // Concurrent contexts are isolated
  const results: string[] = [];
  await Promise.all([
    RequestContext.run({ requestId: "req-1", startTime: 0 }, async () => {
      await new Promise((r) => setTimeout(r, 20));
      results.push(RequestContext.get()!.requestId);
    }),
    RequestContext.run({ requestId: "req-2", startTime: 0 }, async () => {
      await new Promise((r) => setTimeout(r, 10));
      results.push(RequestContext.get()!.requestId);
    }),
  ]);
  console.assert(results.includes("req-1"), "Context 1 isolated");
  console.assert(results.includes("req-2"), "Context 2 isolated");

  console.log("Edge cases passed");
  console.log("EXERCISE 4: PASSED");
}

function test_worker_pool(): void {
  console.log("\n=== EXERCISE 5: Worker Thread Pool ===");

  // We can't easily test actual worker threads without a worker script file,
  // so we test the pool's API surface and queuing behavior.
  // In a real scenario, you'd create a worker.js file and test end-to-end.

  // For now, just verify construction doesn't crash
  // (the actual test would require a worker script)
  console.log("  Note: Worker pool requires a worker script for full testing.");
  console.log("  See the hints in the exercise for implementation guidance.");
  console.log("  Verifying class exists and has correct API...");

  const pool = new WorkerPool(2, "nonexistent.js");
  console.assert(typeof pool.exec === "function", "exec method exists");
  console.assert(typeof pool.destroy === "function", "destroy method exists");
  pool.destroy();

  console.log("EXERCISE 5: PASSED");
}

async function test_event_bus(): Promise<void> {
  console.log("\n=== EXERCISE 6: In-Memory Pub/Sub ===");

  const bus = new EventBus();
  const received: unknown[] = [];

  // Exact topic match
  const unsub1 = bus.subscribe("user.created", (data) => received.push(data));
  let count = await bus.publish("user.created", { id: 1 });
  console.assert(count === 1, "1 handler called");
  console.assert((received[0] as any).id === 1, "Correct data");

  // Wildcard match
  const unsub2 = bus.subscribe("user.*", (data) =>
    received.push({ wildcard: data }),
  );
  count = await bus.publish("user.deleted", { id: 2 });
  console.assert(count === 1, "Wildcard matched user.deleted");

  count = await bus.publish("user.created", { id: 3 });
  console.assert(count === 2, "Both exact and wildcard matched");

  // Non-matching topic
  count = await bus.publish("order.created", { id: 4 });
  console.assert(count === 0, "No match for order.created");

  // Double wildcard matches everything
  const allEvents: unknown[] = [];
  const unsub3 = bus.subscribe("**", (data) => allEvents.push(data));
  await bus.publish("anything.here", "test");
  console.assert(allEvents.length === 1, "** matches everything");

  // Unsubscribe
  unsub1();
  unsub2();
  unsub3();
  count = await bus.publish("user.created", { id: 5 });
  console.assert(count === 0, "No handlers after unsubscribe");

  console.log("Edge cases passed");
  console.log("EXERCISE 6: PASSED");
}


if (require.main === module) {
  console.log("Node.js Runtime Exercises");
  console.log("=".repeat(60));

  const syncTests: [string, () => void][] = [
    ["Exercise 1: EventEmitter from Scratch", test_my_event_emitter],
    ["Exercise 2: Transform Stream", test_uppercase_transform],
    ["Exercise 5: Worker Thread Pool", test_worker_pool],
  ];

  const asyncTests: [string, () => Promise<void>][] = [
    ["Exercise 3: Graceful Shutdown", test_graceful_shutdown],
    ["Exercise 4: AsyncLocalStorage Context", test_request_context],
    ["Exercise 6: In-Memory Pub/Sub", test_event_bus],
  ];

  let passed = 0;
  let failed = 0;

  async function runAll() {
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
