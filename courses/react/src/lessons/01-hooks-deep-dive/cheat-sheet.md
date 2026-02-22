# Hooks Cheat Sheet

> Quick-reference card. Scannable tables, TypeScript signatures, patterns.

---

## Every Built-in Hook

### State & Refs

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `useState` | `<S>(initial: S \| (() => S)) => [S, Dispatch<SetStateAction<S>>]` | Simple state that triggers re-renders |
| `useReducer` | `<S, A>(reducer: (s: S, a: A) => S, init: S) => [S, Dispatch<A>]` | Complex state transitions, testable logic, stable dispatch |
| `useRef` | `<T>(initial: T) => MutableRefObject<T>` | Mutable value that persists without re-rendering; DOM access |
| `useState` (lazy) | `useState(() => expensiveCompute())` | Expensive initial value computed once |

### Effects

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `useEffect` | `(effect: () => void \| (() => void), deps?: any[]) => void` | Side effects after paint (fetch, subscribe, log) |
| `useLayoutEffect` | Same as useEffect | DOM measurement/mutation before paint (avoid flicker) |
| `useInsertionEffect` | Same as useEffect | CSS-in-JS library injection (before DOM reads) |

### Memoization

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `useMemo` | `<T>(factory: () => T, deps: any[]) => T` | Cache expensive computations; stabilize object references |
| `useCallback` | `<T extends Function>(fn: T, deps: any[]) => T` | Stabilize function references for child props or effect deps |

### Context & External State

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `useContext` | `<T>(context: Context<T>) => T` | Read nearest context value |
| `useSyncExternalStore` | `<T>(sub, getSnap, getServerSnap?) => T` | Subscribe to external stores without tearing |

### Identity & Transitions

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `useId` | `() => string` | SSR-safe unique IDs for accessibility attributes |
| `useTransition` | `() => [boolean, (cb: () => void) => void]` | Mark state updates as non-urgent (keep UI responsive) |
| `useDeferredValue` | `<T>(value: T) => T` | Defer a value to avoid blocking urgent updates |
| `useDebugValue` | `(value: any, format?: (v: any) => any) => void` | Label custom hooks in React DevTools |

### React 19

| Hook | Signature | When to Use |
|------|-----------|-------------|
| `use` | `<T>(resource: Promise<T> \| Context<T>) => T` | Read promises/context in render (can use conditionally) |
| `useOptimistic` | `<S, A>(passthrough: S, reducer?: (s: S, a: A) => S) => [S, (a: A) => void]` | Optimistic UI during async actions |
| `useActionState` | `<S>(action, initialState, permalink?) => [S, formAction, isPending]` | Form actions with returned state and pending flag |
| `useFormStatus` | `() => { pending, data, method, action }` | Read parent form's submission status |

---

## Common Gotchas (One Line Each)

| Gotcha | Fix |
|--------|-----|
| `useState(expensiveFn())` runs every render | Use `useState(() => expensiveFn())` — lazy initializer |
| `setCount(count + 1)` called twice = one increment | Use `setCount(c => c + 1)` — functional update |
| Mutating an object/array and calling setState = no re-render | Spread into a new reference: `setState(prev => ({...prev, key: val}))` |
| `useEffect(async () => ...)` — cleanup is lost | Define async fn inside effect, call it |
| `useEffect` with object dep reruns every render | Extract primitives or `useMemo` the object |
| Stale closure in `setInterval` inside `useEffect([])` | Use a ref to hold the latest value, or add deps and restart the interval |
| `useLayoutEffect` warning during SSR | Conditionally use `useEffect` on server, or suppress with `useIsomorphicLayoutEffect` |
| `useSyncExternalStore` infinite loop | `getSnapshot` is returning a new object reference each call — return cached/primitive |
| `useId` used for list keys | Never — `useId` is for accessibility IDs, not keys |
| Calling hooks inside conditions/loops | Move hook to top level; restructure the condition to be inside the hook's callback |
| `useCallback` without deps = stale closure | Always list captured variables in deps; trust `eslint-plugin-react-hooks` |

---

## Quick Patterns

### Async in useEffect

```tsx
useEffect(() => {
  const controller = new AbortController();
  async function load() {
    try {
      const res = await fetch(url, { signal: controller.signal });
      const data = await res.json();
      setData(data);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e);
    }
  }
  load();
  return () => controller.abort();
}, [url]);
```

### Previous Value

```tsx
function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined);
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}
```

### Debounced Value

```tsx
function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}
```

### Isomorphic useLayoutEffect

```tsx
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;
```

### Latest Ref (Solve Stale Closures)

```tsx
function useLatest<T>(value: T): React.RefObject<T> {
  const ref = useRef(value);
  useLayoutEffect(() => {
    ref.current = value;
  });
  return ref;
}
```

### Stable Callback (DIY useEffectEvent)

```tsx
function useStableCallback<T extends (...args: any[]) => any>(fn: T): T {
  const ref = useRef(fn);
  useLayoutEffect(() => {
    ref.current = fn;
  });
  return useCallback(
    ((...args: any[]) => ref.current(...args)) as T,
    []
  );
}
```

### Interval with Latest Callback

```tsx
function useInterval(callback: () => void, delayMs: number | null) {
  const savedCallback = useRef(callback);
  useLayoutEffect(() => {
    savedCallback.current = callback;
  });
  useEffect(() => {
    if (delayMs === null) return;
    const id = setInterval(() => savedCallback.current(), delayMs);
    return () => clearInterval(id);
  }, [delayMs]);
}
```

### Window Event Listener

```tsx
function useWindowEvent<K extends keyof WindowEventMap>(
  event: K,
  handler: (e: WindowEventMap[K]) => void,
  options?: AddEventListenerOptions
) {
  const handlerRef = useRef(handler);
  useLayoutEffect(() => {
    handlerRef.current = handler;
  });
  useEffect(() => {
    const listener = (e: WindowEventMap[K]) => handlerRef.current(e);
    window.addEventListener(event, listener, options);
    return () => window.removeEventListener(event, listener, options);
  }, [event]);
}
```

### Media Query

```tsx
function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", cb);
      return () => mql.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false
  );
}
```

---

## Hook Dependency Quick Rules

| Scenario | What goes in deps |
|----------|-------------------|
| Run once on mount | `[]` |
| Run when specific values change | `[val1, val2]` |
| Run every render | Omit the array entirely |
| `setState` / `dispatch` | Safe to omit (stable), safe to include (no harm) |
| Refs from `useRef` | Safe to omit (stable object), `.current` changes aren't tracked |
| Props | Always include if read inside the effect |
| Values from custom hooks | Always include |
