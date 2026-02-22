# Lesson 01: Hooks Deep Dive

> Core interview knowledge for senior-level React interviews.
> Assumes you already know what hooks are and have used them in production.

---

## Table of Contents

1. [useState](#usestate)
2. [useEffect](#useeffect)
3. [useRef](#useref)
4. [useReducer](#usereducer)
5. [useLayoutEffect vs useEffect](#uselayouteffect-vs-useeffect)
6. [useId](#useid)
7. [Rules of Hooks — The WHY](#rules-of-hooks--the-why)
8. [Common Interview Questions](#common-interview-questions)

---

## useState

### Lazy Initialization

The initializer argument can be a **function**. React calls it only on the first render. This matters when the initial value is expensive to compute.

```tsx
// BAD: computeExpensiveDefault() runs on EVERY render (return value is just ignored after the first)
const [value, setValue] = useState(computeExpensiveDefault());

// GOOD: function reference — React calls it once
const [value, setValue] = useState(() => computeExpensiveDefault());
```

- The lazy initializer receives **no arguments**.
- It runs synchronously during the first render — don't put side effects in here.
- This is commonly missed in code reviews and is a frequent interview probe.

### Functional Updates

When the next state depends on the previous state, always use the updater form.

```tsx
// WRONG in concurrent React — may use a stale snapshot
setCount(count + 1);

// CORRECT — guaranteed latest state
setCount((prev) => prev + 1);
```

Why it matters:
- In React 18+ with automatic batching and concurrent features, the closure-captured `count` can be stale.
- Multiple `setCount(count + 1)` calls in the same event handler collapse to a single +1.
- Multiple `setCount(prev => prev + 1)` calls correctly chain.

```tsx
function handleClick() {
  // Results in count + 1 (one update, applied twice with the same stale `count`)
  setCount(count + 1);
  setCount(count + 1);

  // Results in count + 2 (each updater sees the result of the previous)
  setCount((c) => c + 1);
  setCount((c) => c + 1);
}
```

### Object State Pitfalls (Reference Equality)

React uses `Object.is()` to decide whether to re-render. For objects, that means **reference identity**.

```tsx
const [user, setUser] = useState({ name: "Alice", age: 30 });

// BUG: mutating the existing object — same reference — React bails out, no re-render
user.age = 31;
setUser(user);

// CORRECT: new object reference
setUser((prev) => ({ ...prev, age: 31 }));
```

Interview gotcha: What about `useState([])`?

```tsx
const [items, setItems] = useState<string[]>([]);

// BUG: push mutates in place, same reference
items.push("new");
setItems(items); // No re-render

// CORRECT
setItems((prev) => [...prev, "new"]);
```

### Batching Behavior

**React 18+ batches all state updates automatically** — inside event handlers, timeouts, promises, and native event listeners. Before React 18, only React synthetic event handlers were batched.

```tsx
function handleClick() {
  setA(1);
  setB(2);
  setC(3);
  // ONE re-render, not three
}

// React 18: also batched (was NOT batched in React 17)
setTimeout(() => {
  setA(1);
  setB(2);
  // ONE re-render
}, 100);

// To opt out (rare):
import { flushSync } from "react-dom";
flushSync(() => setA(1)); // re-renders immediately
flushSync(() => setB(2)); // re-renders again
```

Key detail: batching means your state updates are **queued** and applied together before the next render. You will not see intermediate states.

---

## useEffect

### Dependency Array Nuances

The dependency array uses `Object.is()` for each element. Consequences:

```tsx
// Runs every render — new object reference each time
useEffect(() => { /* ... */ }, [{ id: 1 }]);

// Runs every render — new array reference each time
useEffect(() => { /* ... */ }, [[1, 2, 3]]);

// Runs every render — new function reference each time
useEffect(() => { /* ... */ }, [() => doSomething()]);
```

Stabilize with `useMemo`, `useCallback`, or extract the primitive values:

```tsx
// Extract primitives
useEffect(() => { /* ... */ }, [user.id, user.name]);

// Or memoize the object if you truly need it
const config = useMemo(() => ({ id, name }), [id, name]);
useEffect(() => { /* ... */ }, [config]);
```

**Missing dependencies** — the exhaustive-deps lint rule exists because stale closures are _silent_ bugs. Trust the lint rule. If it feels wrong, your abstraction is wrong.

### Cleanup Timing

1. Component renders with new props/state.
2. React **paints to the screen**.
3. React runs the **previous effect's cleanup** with the _previous_ closure values.
4. React runs the **new effect** with the _current_ closure values.

```tsx
useEffect(() => {
  const id = setInterval(() => console.log(count), 1000);
  // Cleanup runs BEFORE the next effect, with the `count` from THIS render
  return () => clearInterval(id);
}, [count]);
```

On unmount, the last cleanup runs and no new effect fires.

### Race Conditions

Classic interview problem: what happens when a fast response arrives after a slow one?

```tsx
// BUG: race condition
useEffect(() => {
  fetchUser(userId).then((data) => setUser(data));
}, [userId]);
```

If `userId` changes from 1 to 2 quickly, the response for user 2 might arrive before user 1's response. When user 1's response finally arrives, it overwrites the correct data.

**Fix with a cleanup boolean:**

```tsx
useEffect(() => {
  let cancelled = false;

  fetchUser(userId).then((data) => {
    if (!cancelled) setUser(data);
  });

  return () => {
    cancelled = true;
  };
}, [userId]);
```

**Fix with AbortController (preferred):**

```tsx
useEffect(() => {
  const controller = new AbortController();

  fetchUser(userId, { signal: controller.signal })
    .then((data) => setUser(data))
    .catch((err) => {
      if (err.name !== "AbortError") throw err;
    });

  return () => controller.abort();
}, [userId]);
```

### Async Effects Pattern

You cannot pass an async function directly to `useEffect` because it returns a Promise, not a cleanup function.

```tsx
// WRONG: returns a Promise, React ignores it (and the cleanup is lost)
useEffect(async () => {
  const data = await fetchData();
  setData(data);
}, []);

// CORRECT: define and immediately invoke an async function
useEffect(() => {
  async function load() {
    const data = await fetchData();
    setData(data);
  }
  load();
}, []);

// CORRECT (alternative): IIFE
useEffect(() => {
  (async () => {
    const data = await fetchData();
    setData(data);
  })();
}, []);
```

---

## useRef

### DOM Refs vs Mutable Containers

`useRef` serves two distinct purposes. Interviewers often test whether you understand both.

**1. DOM access:**

```tsx
const inputRef = useRef<HTMLInputElement>(null);

useEffect(() => {
  inputRef.current?.focus();
}, []);

return <input ref={inputRef} />;
```

**2. Mutable instance variable (survives re-renders, doesn't trigger them):**

```tsx
const intervalIdRef = useRef<ReturnType<typeof setInterval> | null>(null);
const renderCountRef = useRef(0);

useEffect(() => {
  renderCountRef.current += 1;
});
```

Key distinction: `useRef` is a **box** holding a mutable `.current` property. Changing `.current` does NOT cause a re-render. This is by design — it's escape hatch storage outside React's rendering model.

### Callback Refs Pattern

When you need to run logic _the moment_ a DOM node attaches or detaches, a ref object won't notify you. Use a callback ref instead.

```tsx
const [height, setHeight] = useState(0);

const measuredRef = useCallback((node: HTMLDivElement | null) => {
  if (node !== null) {
    setHeight(node.getBoundingClientRect().height);
  }
}, []);

return <div ref={measuredRef}>Hello</div>;
```

Why callback refs matter:
- Ref objects are assigned during commit but don't trigger re-renders or effects.
- Callback refs give you a **synchronous notification** when the ref value changes.
- Useful for: measuring DOM elements, integrating third-party libraries, conditional refs.

### Why Ref Changes Don't Trigger Re-renders

Refs are intentionally outside the React rendering cycle. Internally, `useRef` is essentially:

```tsx
function useRef<T>(initialValue: T) {
  const [ref] = useState(() => ({ current: initialValue }));
  return ref;
}
```

The object reference is stable across renders. Mutating `.current` is just a property assignment on a plain object — React has no way to know it happened and no reason to re-render.

---

## useReducer

### When to Prefer Over useState

Use `useReducer` when:
- Next state depends on previous state in **complex** ways (multiple fields, conditional logic).
- Multiple state values change together and you want **atomic, predictable updates**.
- You want to pass `dispatch` down instead of multiple setter callbacks (dispatch is **referentially stable**).
- State transitions are testable as pure functions.

```tsx
type State = { count: number; step: number; };
type Action =
  | { type: "increment" }
  | { type: "decrement" }
  | { type: "setStep"; payload: number }
  | { type: "reset" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "increment":
      return { ...state, count: state.count + state.step };
    case "decrement":
      return { ...state, count: state.count - state.step };
    case "setStep":
      return { ...state, step: action.payload };
    case "reset":
      return { count: 0, step: 1 };
  }
}

const [state, dispatch] = useReducer(reducer, { count: 0, step: 1 });
```

### Action Patterns

Discriminated unions are the standard TypeScript pattern for actions:

```tsx
type Action =
  | { type: "add"; item: Item }
  | { type: "remove"; id: string }
  | { type: "update"; id: string; changes: Partial<Item> };
```

Lazy initialization (third argument):

```tsx
const [state, dispatch] = useReducer(reducer, userId, (id) => {
  // Called once, receives the second argument
  return { user: loadFromCache(id), loading: false };
});
```

### Dispatch Stability

`dispatch` is **referentially stable** across re-renders. You never need to wrap it in `useCallback` or include it in dependency arrays (though including it is harmless).

This makes it ideal for passing to deeply nested children or context:

```tsx
const DispatchContext = createContext<Dispatch<Action>>(() => {});

function Parent() {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <DispatchContext.Provider value={dispatch}>
      {/* dispatch never changes — children don't re-render from it */}
      <DeepChild />
    </DispatchContext.Provider>
  );
}
```

---

## useLayoutEffect vs useEffect

### Rendering Timeline

```
1. React renders (calls your component function)
2. React updates the DOM
3. useLayoutEffect fires (synchronously, BLOCKS paint)
4. Browser paints to screen
5. useEffect fires (asynchronously, after paint)
```

### When to Use useLayoutEffect

- **Measuring DOM** before the user sees it (avoid flicker).
- **Synchronously mutating DOM** based on measurements.
- **Tooltip/popover positioning** — you need coordinates before paint.

```tsx
function Tooltip({ anchorEl, children }: Props) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [coords, setCoords] = useState({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!anchorEl || !tooltipRef.current) return;
    const rect = anchorEl.getBoundingClientRect();
    setCoords({ top: rect.bottom, left: rect.left });
  }, [anchorEl]);

  return (
    <div ref={tooltipRef} style={{ position: "fixed", ...coords }}>
      {children}
    </div>
  );
}
```

### Key Differences Summary

| Aspect | useEffect | useLayoutEffect |
|--------|-----------|-----------------|
| Timing | After paint | Before paint |
| Blocking | Non-blocking | Blocks paint |
| Use case | Data fetching, subscriptions, logging | DOM measurement, sync mutations |
| SSR | Warns (no DOM) | Warns (no DOM) |
| Performance | Preferred default | Use sparingly |

---

## useId

### SSR-Safe IDs

`useId` generates stable, unique IDs that are consistent between server and client renders.

```tsx
function FormField({ label }: { label: string }) {
  const id = useId();
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <input id={id} />
    </>
  );
}
```

Why not `Math.random()` or a counter?
- `Math.random()` produces different values on server vs client (hydration mismatch).
- A global counter produces different values if component render order differs between server and client.
- `useId` uses the component's position in the fiber tree to generate deterministic IDs.

### Accessibility

Use `useId` for `aria-describedby`, `aria-labelledby`, `htmlFor`, and any attribute that links elements by ID:

```tsx
function PasswordField() {
  const id = useId();
  const errorId = `${id}-error`;
  const hintId = `${id}-hint`;

  return (
    <div>
      <label htmlFor={id}>Password</label>
      <input id={id} type="password" aria-describedby={`${hintId} ${errorId}`} />
      <p id={hintId}>Must be 8+ characters</p>
      <p id={errorId} role="alert">{error}</p>
    </div>
  );
}
```

Key details:
- `useId` returns a string like `:r1:` — the colons make it safe as a CSS selector when escaped.
- You can derive multiple related IDs from a single `useId()` call with suffixes.
- Do NOT use `useId` for list keys — it generates the same ID on every render (that's the point).

---

## Rules of Hooks — The WHY

### The Real Reason: Linked List / Call Order

React stores hook state as a **linked list** on the fiber node. Each `useState`, `useEffect`, `useRef`, etc. call corresponds to a node in this list. React matches hooks to their state **by position (call order)**, not by name.

```
Fiber Node
  └── memoizedState -> Hook1 -> Hook2 -> Hook3 -> null
                       (useState) (useEffect) (useRef)
```

On re-render, React walks the list in order. If you call hooks in a different order, React assigns the wrong state to the wrong hook.

```tsx
// BROKEN: conditional hook changes call order
function Bad({ showName }: { showName: boolean }) {
  if (showName) {
    const [name, setName] = useState(""); // Hook 1 (sometimes)
  }
  const [age, setAge] = useState(0);      // Hook 1 or 2 (ambiguous!)
  // React can't tell which state belongs to which hook
}
```

### The Rules (and Why Each Exists)

1. **Only call hooks at the top level** — no conditions, loops, or nested functions.
   - Why: Ensures the hook call order is identical on every render so the linked list matches.

2. **Only call hooks from React functions** (components or custom hooks).
   - Why: Hooks need the fiber context (the "currently rendering component") to read/write state. Outside a component, there's no fiber.

3. **Custom hooks must start with `use`**.
   - Why: Convention that enables the linter to enforce rules 1 and 2 on custom hooks.

---

## Common Interview Questions

### Q1: What happens if you call useState inside a condition?

React stores hook state in a linked list indexed by call order. If a hook is conditionally skipped, all subsequent hooks shift position and receive the wrong state. React will likely throw an error: "Rendered fewer/more hooks than during the previous render."

### Q2: Why can't useEffect be async?

`useEffect` expects its callback to return either `undefined` or a cleanup function. An `async` function returns a `Promise`, which React would silently ignore — meaning your cleanup logic never runs. Define the async function inside the effect and call it.

### Q3: How does React know a state update should cause a re-render?

When you call `setState`, React enqueues an update on the fiber node and schedules a re-render. During re-render, it compares the new state with the current state using `Object.is()`. If they're the same, React may bail out (skip rendering children), though it still needs to render the component itself to confirm.

### Q4: When would you use useRef instead of useState?

When you need a mutable value that persists across renders but whose changes should **not** trigger re-renders. Common cases: timer IDs, previous values, DOM element references, tracking whether a component is mounted.

### Q5: Explain the stale closure problem.

When an effect or callback captures a variable from a render's closure, it sees the value _from that render_, not the latest value. If the effect doesn't re-run (because deps haven't changed), it reads an outdated value.

```tsx
const [count, setCount] = useState(0);

useEffect(() => {
  const id = setInterval(() => {
    console.log(count); // Always logs 0 — stale closure
  }, 1000);
  return () => clearInterval(id);
}, []); // Empty deps = effect never re-runs, `count` is always 0
```

Fix: add `count` to deps (interval restarts), use a ref, or use the functional updater `setCount(c => c + 1)`.

### Q6: What is the difference between useLayoutEffect and useEffect?

`useLayoutEffect` fires **synchronously after DOM mutations but before the browser paints**. Use it when you need to measure or mutate the DOM without the user seeing an intermediate state (e.g., positioning a tooltip). `useEffect` fires **after paint** and is non-blocking.

### Q7: Why is dispatch from useReducer referentially stable but setState from useState is too?

Both `dispatch` and `setState` are stable — React guarantees their identity doesn't change across re-renders. The practical difference is that `useReducer` centralizes state transitions in a pure function, making complex logic more testable and `dispatch` more ergonomic to pass down (one function vs. many setters).

### Q8: How does automatic batching work in React 18?

React 18 batches all state updates in a single synchronous execution context into one re-render — regardless of where they originate (event handlers, promises, timeouts, native events). Previously, only React event handlers were batched. Use `flushSync` to opt out when you need an immediate re-render between updates.

---

## Quick Mental Model

```
Component Render
  │
  ├── useState    → returns [state, setState] from fiber's hook list
  ├── useReducer  → returns [state, dispatch] from fiber's hook list
  ├── useRef      → returns { current } (stable object) from fiber's hook list
  ├── useMemo     → returns cached value if deps unchanged
  ├── useCallback → returns cached function if deps unchanged (sugar for useMemo(() => fn, deps))
  │
  ▼
DOM Update (commit phase)
  │
  ├── useLayoutEffect → fires sync, blocks paint
  │
  ▼
Browser Paint
  │
  ├── useEffect → fires async, after paint
  │
  ▼
User sees the screen
```
