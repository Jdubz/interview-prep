/**
 * Types & Generics Exercises
 *
 * TypeScript's type system as a language — generics, mapped types, branded
 * types, and conditional types. These patterns show up in library design
 * and senior-level TypeScript interviews.
 *
 * Run:  npx tsx exercises/01-types-and-generics.ts
 */


// ============================================================================
// EXERCISE 1: Type-Safe Event Emitter
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md
//   - ../01-typescript-advanced/02-advanced-type-patterns.md
//
// Implement a generic TypedEmitter<Events> where Events is a record mapping
// event names to their payload types. The .on() and .emit() methods should
// enforce correct event names and payload types at compile time.
//
// Requirements:
//   - TypedEmitter<Events> is generic over an event map type
//   - .on(event, handler) only accepts known event names
//   - Handler receives the correct payload type for that event
//   - .emit(event, payload) only accepts the correct payload type
//   - .off(event, handler) removes a specific listener
//   - Multiple listeners per event
//
// Hints:
//   - Define Events as Record<string, unknown> constraint
//   - Use keyof Events to restrict event names
//   - Store listeners in a Map<string, Set<Function>>
//
//   Type pattern — constraining event names and payloads:
//     class TypedEmitter<Events extends Record<string, unknown>> {
//       on<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void
//       emit<K extends keyof Events>(event: K, payload: Events[K]): void
//     }
//
//   Key concepts:
//   - K extends keyof Events restricts the event parameter to known keys
//   - Events[K] looks up the payload type for that specific event
//   - This is called "indexed access types" or "lookup types"
//
// Expected behavior:
//   type MyEvents = { login: { userId: string }; logout: void };
//   const emitter = new TypedEmitter<MyEvents>();
//   emitter.on("login", (payload) => console.log(payload.userId)); // payload is { userId: string }
//   emitter.emit("login", { userId: "abc" }); // OK
//   emitter.emit("login", { wrong: true });   // Type error!

class TypedEmitter<Events extends Record<string, unknown>> {
  private listeners = new Map<keyof Events, Set<Function>>();

  on<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
  }

  off<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void {
    this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    for (const handler of this.listeners.get(event) ?? []) {
      (handler as (payload: Events[K]) => void)(payload);
    }
  }
}


// ============================================================================
// EXERCISE 2: Deep Readonly
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md (mapped types)
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (recursive types)
//
// Implement a DeepReadonly<T> type that recursively makes all properties
// readonly, including nested objects and arrays.
//
// Requirements:
//   - All properties at every level become readonly
//   - Arrays become ReadonlyArray (readonly T[])
//   - Primitives are returned as-is
//   - Works with nested objects of arbitrary depth
//   - Provide a deepFreeze<T>(obj: T): DeepReadonly<T> runtime function
//     that uses Object.freeze recursively
//
// Hints:
//   - Use a conditional type: if T is primitive, return T
//   - If T extends Array<infer U>, return ReadonlyArray<DeepReadonly<U>>
//   - Otherwise, map over properties: { readonly [K in keyof T]: DeepReadonly<T[K]> }
//
//   Type pattern — recursive mapped type:
//     type DeepReadonly<T> =
//       T extends primitive ? T :
//       T extends Array<infer U> ? ReadonlyArray<DeepReadonly<U>> :
//       { readonly [K in keyof T]: DeepReadonly<T[K]> };
//
//   Runtime implementation:
//     function deepFreeze<T>(obj: T): DeepReadonly<T> {
//       Object.freeze(obj);
//       // recursively freeze all object/array properties
//       return obj as DeepReadonly<T>;
//     }
//
// Expected behavior:
//   type Nested = { a: { b: { c: number }; d: string[] } };
//   type Result = DeepReadonly<Nested>;
//   // Result = { readonly a: { readonly b: { readonly c: number }; readonly d: ReadonlyArray<string> } }
//
//   const frozen = deepFreeze({ a: { b: [1, 2] } });
//   frozen.a.b.push(3); // Runtime error! Object is frozen

type Primitive = string | number | boolean | bigint | symbol | undefined | null;

// TODO: Implement DeepReadonly<T>
type DeepReadonly<T> = T;

// TODO: Implement deepFreeze
function deepFreeze<T extends Record<string, unknown>>(obj: T): DeepReadonly<T> {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 3: Type-Safe Builder Pattern
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (phantom types)
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md (conditional types)
//
// Implement a generic Builder that tracks which required fields have been set
// at the type level. The .build() method should only be available when ALL
// required fields have been set.
//
// Requirements:
//   - Builder<Shape, Set> where Shape is the target type, Set tracks filled fields
//   - .set(key, value) returns a new builder type with that key marked as set
//   - .build() only available when all required fields in Shape are set
//   - Type errors if you call .build() before setting all required fields
//
// Hints:
//   - Use a phantom type parameter to track which fields have been set
//   - The trick: Builder<Shape, SetKeys extends keyof Shape>
//   - .set<K>(key, value) returns Builder<Shape, SetKeys | K>
//   - .build() is conditionally available: SetKeys extends keyof Shape ? ...
//
//   Type pattern:
//     class Builder<T, Set extends keyof T = never> {
//       private data: Partial<T> = {};
//       set<K extends keyof T>(key: K, value: T[K]): Builder<T, Set | K> { ... }
//       // build() only when Set covers all keys:
//       build(this: Builder<T, keyof T>): T { return this.data as T; }
//     }
//
//   Key concepts:
//   - `this` parameter typing constrains when a method can be called
//   - Builder<T, Set | K> adds K to the set of filled fields
//   - Builder<T, keyof T> means "all fields set"
//
// Expected behavior:
//   type Config = { host: string; port: number; debug: boolean };
//   const config = new Builder<Config>()
//     .set("host", "localhost")
//     .set("port", 3000)
//     .set("debug", true)
//     .build(); // OK: all fields set
//
//   new Builder<Config>()
//     .set("host", "localhost")
//     .build(); // Type error! port and debug not set

class Builder<T extends Record<string, unknown>, _Set extends keyof T = never> {
  // TODO: implement set() and build()

  set<K extends keyof T>(_key: K, _value: T[K]): Builder<T, _Set | K> {
    throw new Error("Not implemented");
  }

  build(this: Builder<T, keyof T>): T {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 4: Branded Types
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (nominal typing)
//   - ../10-interview-prep/01-interview-fundamentals.md
//
// TypeScript uses structural typing, so `type UserId = string` doesn't prevent
// passing an OrderId where a UserId is expected. Branded types add a phantom
// property to create nominal-like types.
//
// Requirements:
//   - Define Brand<T, B> that adds a phantom __brand property
//   - Create UserId and OrderId as Brand<string, "UserId"> and Brand<string, "OrderId">
//   - Implement createUserId(id: string): UserId and createOrderId(id: string): OrderId
//   - Implement getUser(id: UserId) and getOrder(id: OrderId) that only accept
//     their respective branded types
//   - Passing a UserId to getOrder should be a type error
//
// Hints:
//   - Brand<T, B> = T & { readonly __brand: B }
//   - The __brand property never exists at runtime — it's purely a type-level tag
//   - Creating a branded value: return id as UserId (type assertion)
//
//   Type pattern:
//     type Brand<T, B extends string> = T & { readonly __brand: B };
//     type UserId = Brand<string, "UserId">;
//     function createUserId(id: string): UserId { return id as UserId; }
//
//   Key concepts:
//   - Intersection type (T & { __brand }) adds a phantom property
//   - Type assertions (as UserId) bridge from unbranded to branded
//   - At runtime, branded values are just plain strings/numbers
//
// Expected behavior:
//   const userId = createUserId("user-123");
//   const orderId = createOrderId("order-456");
//   getUser(userId);    // OK
//   getOrder(orderId);  // OK
//   getUser(orderId);   // Type error!
//   getOrder(userId);   // Type error!

// TODO: Define Brand<T, B>, UserId, OrderId, and the helper functions

type Brand<T, B extends string> = T;  // Fix this
type UserId = string;    // Fix this
type OrderId = string;   // Fix this

function createUserId(_id: string): UserId {
  throw new Error("Not implemented");
}

function createOrderId(_id: string): OrderId {
  throw new Error("Not implemented");
}

function getUser(_id: UserId): string {
  throw new Error("Not implemented");
}

function getOrder(_id: OrderId): string {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 5: Result Type
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (discriminated unions)
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md
//
// Implement a Result<T, E> type inspired by Rust's Result. It represents
// either a success value (Ok) or an error (Err), with chainable operations.
//
// Requirements:
//   - Result<T, E> is a discriminated union of Ok<T> and Err<E>
//   - Ok<T> has { ok: true, value: T }
//   - Err<E> has { ok: false, error: E }
//   - .map(fn) transforms the success value, passes through errors
//   - .flatMap(fn) chains operations that return Results
//   - .unwrapOr(default) returns the value or a default
//   - Provide ok<T>(value: T) and err<E>(error: E) constructor functions
//
// Hints:
//   - Use a discriminated union with an `ok` boolean tag
//   - map: if ok, apply fn to value and wrap in Ok; otherwise return self
//   - flatMap: if ok, apply fn (which returns a Result); otherwise return self
//   - unwrapOr: if ok, return value; otherwise return default
//
//   Type pattern:
//     type Result<T, E> = Ok<T, E> | Err<T, E>;
//     class Ok<T, E> { ok = true as const; constructor(public value: T) {} }
//     class Err<T, E> { ok = false as const; constructor(public error: E) {} }
//
//   Key concepts:
//   - Discriminated union: `ok` field narrows the type
//   - Generic methods preserve type information through chains
//   - This pattern eliminates try/catch in favor of explicit error handling
//
// Expected behavior:
//   const result = ok<number, string>(42)
//     .map(x => x * 2)
//     .flatMap(x => x > 50 ? ok(x) : err("too small"));
//   result.unwrapOr(0); // 84

// TODO: Implement Result<T, E>, Ok, Err, ok(), err()

interface Result<T, E> {
  ok: boolean;
  map<U>(fn: (value: T) => U): Result<U, E>;
  flatMap<U>(fn: (value: T) => Result<U, E>): Result<U, E>;
  unwrapOr(defaultValue: T): T;
}

function ok<T, E = never>(_value: T): Result<T, E> {
  throw new Error("Not implemented");
}

function err<T = never, E = unknown>(_error: E): Result<T, E> {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 6: Type-Safe Router
// ============================================================================
//
// RELATED READING:
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md (template literal types)
//   - ../01-typescript-advanced/02-advanced-type-patterns.md (inference in conditionals)
//
// Extract path parameters from a string literal route type. Given a path like
// "/users/:id/posts/:postId", produce a type { id: string; postId: string }.
//
// Requirements:
//   - ExtractParams<"/users/:id"> → { id: string }
//   - ExtractParams<"/users/:id/posts/:postId"> → { id: string; postId: string }
//   - ExtractParams<"/static"> → {} (no params)
//   - Implement a createRoute<P>(path) function that returns a typedParams(url) method
//   - typedParams parses the actual URL string and returns the extracted params object
//
// Hints:
//   - Use template literal types with infer to parse the string
//   - Split on "/" and check each segment for ":" prefix
//   - Recursive conditional type to process each segment
//
//   Type pattern:
//     type ExtractParams<Path extends string> =
//       Path extends `${string}:${infer Param}/${infer Rest}`
//         ? { [K in Param]: string } & ExtractParams<`/${Rest}`>
//         : Path extends `${string}:${infer Param}`
//           ? { [K in Param]: string }
//           : {};
//
//   Key concepts:
//   - Template literal types can destructure string types
//   - `infer` captures parts of the string for use in the output type
//   - Recursive conditional types process the string segment by segment
//   - Intersection types (&) merge the params from each segment
//
// Expected behavior:
//   type Params = ExtractParams<"/users/:id/posts/:postId">;
//   // Params = { id: string } & { postId: string }
//
//   const route = createRoute("/users/:id/posts/:postId");
//   const params = route.typedParams("/users/42/posts/abc");
//   // params = { id: "42", postId: "abc" }

// TODO: Implement ExtractParams<Path> and createRoute

type ExtractParams<_Path extends string> = Record<string, string>;  // Fix this

function createRoute<Path extends string>(
  _path: Path,
): { typedParams(url: string): ExtractParams<Path> } {
  throw new Error("Not implemented");
}


// ============================================================================
// TESTS
// ============================================================================

function test_typed_emitter(): void {
  console.log("\n=== EXERCISE 1: Type-Safe Event Emitter ===");

  type Events = {
    login: { userId: string; timestamp: number };
    logout: { userId: string };
    error: { message: string; code: number };
  };

  const emitter = new TypedEmitter<Events>();
  const received: unknown[] = [];

  emitter.on("login", (payload) => received.push(payload));
  emitter.on("error", (payload) => received.push(payload));

  emitter.emit("login", { userId: "abc", timestamp: Date.now() });
  emitter.emit("error", { message: "oops", code: 500 });

  console.assert(received.length === 2, "Should receive 2 events");
  console.assert((received[0] as Events["login"]).userId === "abc", "Login payload correct");
  console.assert((received[1] as Events["error"]).code === 500, "Error payload correct");

  // Test off
  const handler = (p: Events["logout"]) => received.push(p);
  emitter.on("logout", handler);
  emitter.off("logout", handler);
  emitter.emit("logout", { userId: "abc" });
  console.assert(received.length === 2, "Handler removed, should still be 2");

  // Multiple listeners
  let count = 0;
  emitter.on("login", () => count++);
  emitter.on("login", () => count++);
  emitter.emit("login", { userId: "x", timestamp: 0 });
  console.assert(count === 2, "Both listeners called");

  console.log("EXERCISE 1: PASSED");
}

function test_deep_readonly(): void {
  console.log("\n=== EXERCISE 2: Deep Readonly ===");

  const obj = { a: { b: { c: 1 }, d: [1, 2, 3] }, e: "hello" };
  const frozen = deepFreeze(obj);

  // Verify the object is deeply frozen
  console.assert(Object.isFrozen(frozen), "Top level frozen");
  console.assert(
    Object.isFrozen((frozen as typeof obj).a),
    "Nested object frozen",
  );
  console.assert(
    Object.isFrozen((frozen as typeof obj).a.b),
    "Deeply nested frozen",
  );
  console.assert(
    Object.isFrozen((frozen as typeof obj).a.d),
    "Array frozen",
  );

  // Verify mutations throw in strict mode
  let threw = false;
  try {
    (frozen as any).a.b.c = 999;
  } catch {
    threw = true;
  }
  // In strict mode this throws, in sloppy mode it silently fails
  console.assert(
    threw || (frozen as typeof obj).a.b.c === 1,
    "Mutation prevented",
  );

  console.log("Edge cases passed");
  console.log("EXERCISE 2: PASSED");
}

function test_builder(): void {
  console.log("\n=== EXERCISE 3: Type-Safe Builder ===");

  type Config = { host: string; port: number; debug: boolean };

  const config = new Builder<Config>()
    .set("host", "localhost")
    .set("port", 3000)
    .set("debug", true)
    .build();

  console.assert(config.host === "localhost", "host set");
  console.assert(config.port === 3000, "port set");
  console.assert(config.debug === true, "debug set");

  // Order shouldn't matter
  const config2 = new Builder<Config>()
    .set("debug", false)
    .set("port", 8080)
    .set("host", "0.0.0.0")
    .build();

  console.assert(config2.host === "0.0.0.0", "host set (reordered)");
  console.assert(config2.port === 8080, "port set (reordered)");

  console.log("Edge cases passed");
  console.log("EXERCISE 3: PASSED");
}

function test_branded_types(): void {
  console.log("\n=== EXERCISE 4: Branded Types ===");

  const userId = createUserId("user-123");
  const orderId = createOrderId("order-456");

  const userResult = getUser(userId);
  console.assert(userResult.includes("user-123"), "getUser works with UserId");

  const orderResult = getOrder(orderId);
  console.assert(orderResult.includes("order-456"), "getOrder works with OrderId");

  // At runtime, branded values are still strings
  console.assert(typeof userId === "string", "UserId is still a string at runtime");
  console.assert(typeof orderId === "string", "OrderId is still a string at runtime");

  console.log("Edge cases passed");
  console.log("EXERCISE 4: PASSED");
}

function test_result_type(): void {
  console.log("\n=== EXERCISE 5: Result Type ===");

  // ok path
  const r1 = ok<number, string>(42);
  console.assert(r1.ok === true, "ok result has ok=true");
  console.assert(r1.unwrapOr(0) === 42, "unwrapOr returns value for Ok");

  // err path
  const r2 = err<number, string>("bad");
  console.assert(r2.ok === false, "err result has ok=false");
  console.assert(r2.unwrapOr(0) === 0, "unwrapOr returns default for Err");

  // map
  const r3 = ok<number, string>(10).map((x) => x * 2);
  console.assert(r3.unwrapOr(0) === 20, "map transforms Ok value");

  const r4 = err<number, string>("fail").map((x) => x * 2);
  console.assert(r4.unwrapOr(0) === 0, "map passes through Err");

  // flatMap
  const r5 = ok<number, string>(84).flatMap((x) =>
    x > 50 ? ok<number, string>(x) : err<number, string>("too small"),
  );
  console.assert(r5.unwrapOr(0) === 84, "flatMap chains Ok");

  const r6 = ok<number, string>(10).flatMap((x) =>
    x > 50 ? ok<number, string>(x) : err<number, string>("too small"),
  );
  console.assert(r6.unwrapOr(0) === 0, "flatMap chains to Err");

  console.log("Edge cases passed");
  console.log("EXERCISE 5: PASSED");
}

function test_router(): void {
  console.log("\n=== EXERCISE 6: Type-Safe Router ===");

  const route = createRoute("/users/:id/posts/:postId");
  const params = route.typedParams("/users/42/posts/abc");

  console.assert(params.id === "42", "Extracted :id");
  console.assert(params.postId === "abc", "Extracted :postId");

  // Single param
  const route2 = createRoute("/items/:itemId");
  const params2 = route2.typedParams("/items/xyz");
  console.assert(params2.itemId === "xyz", "Single param extracted");

  // No params
  const route3 = createRoute("/static/page");
  const params3 = route3.typedParams("/static/page");
  console.assert(Object.keys(params3).length === 0, "No params for static route");

  console.log("Edge cases passed");
  console.log("EXERCISE 6: PASSED");
}


if (require.main === module) {
  console.log("Types & Generics Exercises");
  console.log("=".repeat(60));

  const tests: [string, () => void][] = [
    ["Exercise 1: Type-Safe Event Emitter", test_typed_emitter],
    ["Exercise 2: Deep Readonly", test_deep_readonly],
    ["Exercise 3: Type-Safe Builder", test_builder],
    ["Exercise 4: Branded Types", test_branded_types],
    ["Exercise 5: Result Type", test_result_type],
    ["Exercise 6: Type-Safe Router", test_router],
  ];

  let passed = 0;
  let failed = 0;

  for (const [name, testFn] of tests) {
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

  console.log();
  console.log("=".repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed out of ${tests.length}`);
  console.log("=".repeat(60));
}
