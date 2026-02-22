# Lesson 03: Performance Optimization

## Why React Re-Renders: The Complete Mental Model

A React component re-renders in exactly three situations:

1. **Its state changes** (`useState` setter, `useReducer` dispatch)
2. **Its parent re-renders** (regardless of whether props changed)
3. **A context it consumes changes** (any consumer re-renders when the provider value changes)

That second point is the one most developers get wrong. Props changing does **not** trigger a re-render. The parent re-rendering triggers the child re-render, and new props are computed as a side effect of that process.

```tsx
function Parent() {
  const [count, setCount] = useState(0);

  // Child re-renders every time Parent re-renders,
  // even though "hello" never changes.
  return (
    <>
      <button onClick={() => setCount(c => c + 1)}>Increment</button>
      <Child greeting="hello" />
    </>
  );
}

function Child({ greeting }: { greeting: string }) {
  console.log("Child rendered"); // logs on every Parent state change
  return <p>{greeting}</p>;
}
```

### The Render Phase vs. The Commit Phase

Understanding the two-phase model is critical:

- **Render phase**: React calls your component functions, builds a new virtual DOM tree, and diffs it against the previous tree. This is pure computation — no DOM mutations.
- **Commit phase**: React applies the minimal set of DOM mutations needed. This is where the browser actually updates.

A "wasted render" means the render phase ran but the commit phase found nothing to change. The render phase itself has a cost — function calls, hook evaluations, JSX allocation — but it is often cheaper than developers assume.

### State Batching

React 18+ batches all state updates automatically, including those inside `setTimeout`, promises, and native event handlers. This was a significant change from React 17, which only batched inside React event handlers.

```tsx
function BatchingDemo() {
  const [a, setA] = useState(0);
  const [b, setB] = useState(0);

  const handleClick = () => {
    // React 18: single re-render (batched)
    // React 17: two re-renders
    setA(1);
    setB(2);
  };

  const handleAsync = async () => {
    const data = await fetchSomething();
    // React 18: still batched into a single re-render
    // React 17: two re-renders
    setA(data.a);
    setB(data.b);
  };

  return <div onClick={handleClick}>...</div>;
}
```

To opt out of batching (rare), use `flushSync`:

```tsx
import { flushSync } from "react-dom";

function handleClick() {
  flushSync(() => setA(1)); // commits immediately
  flushSync(() => setB(2)); // commits immediately
}
```

---

## React.memo

`React.memo` is a higher-order component that skips re-rendering when props are shallowly equal to the previous render's props.

```tsx
const ExpensiveList = React.memo(function ExpensiveList({
  items,
  onSelect,
}: {
  items: Item[];
  onSelect: (id: string) => void;
}) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={() => onSelect(item.id)}>
          {item.name}
        </li>
      ))}
    </ul>
  );
});
```

### How Shallow Comparison Works

`React.memo` uses `Object.is` for each prop. For primitives this is a value comparison. For objects, arrays, and functions, this is a **referential identity** check.

```tsx
// This memo is USELESS because `style` is a new object every render:
function Parent() {
  return <MemoizedChild style={{ color: "red" }} />;
  // { color: "red" } !== { color: "red" } (different references)
}

// Fix: stable reference
const style = { color: "red" }; // module-level constant
function Parent() {
  return <MemoizedChild style={style} />;
}
```

### Custom Comparison Function

```tsx
const Chart = React.memo(
  function Chart({ data, config }: ChartProps) {
    // expensive rendering
    return <canvas />;
  },
  (prevProps, nextProps) => {
    // Return true to SKIP re-render (opposite of shouldComponentUpdate)
    return (
      prevProps.data.length === nextProps.data.length &&
      prevProps.data.every((d, i) => d.id === nextProps.data[i].id) &&
      prevProps.config.theme === nextProps.config.theme
    );
  }
);
```

### When React.memo Hurts

1. **Comparison cost exceeds render cost**: If the component is cheap to render but has many props, the shallow comparison on every render may cost more than just re-rendering.
2. **Props always change**: If a parent always passes new object/array/function references, memo does the comparison and then re-renders anyway — strictly worse than no memo.
3. **Premature memoization**: Adds cognitive overhead and code complexity for no measurable gain.

**Rule of thumb**: Profile first. Apply `React.memo` to components that are expensive to render AND receive stable (or stabilizable) props.

---

## useMemo

`useMemo` caches a computed value between re-renders when dependencies haven't changed.

```tsx
function SearchResults({ query, items }: { query: string; items: Item[] }) {
  // Only recomputes when `query` or `items` changes (by reference)
  const filtered = useMemo(
    () => items.filter(item =>
      item.name.toLowerCase().includes(query.toLowerCase())
    ),
    [query, items]
  );

  return <ItemList items={filtered} />;
}
```

### Two Distinct Use Cases

**1. Expensive computation caching**

```tsx
const sortedData = useMemo(() => {
  // O(n log n) sort — worth memoizing if `data` is large
  return [...data].sort((a, b) => complexComparator(a, b));
}, [data]);
```

**2. Referential stability for downstream memoization**

```tsx
function Parent({ userId }: { userId: string }) {
  const [count, setCount] = useState(0);

  // Without useMemo, this is a new object every render,
  // breaking React.memo on MemoizedChild
  const config = useMemo(
    () => ({ userId, theme: "dark" }),
    [userId]
  );

  return (
    <>
      <button onClick={() => setCount(c => c + 1)}>{count}</button>
      <MemoizedChild config={config} />
    </>
  );
}
```

### Dependency Gotchas

```tsx
// BUG: `options` is a new object every render, so this useMemo
// recomputes every render — completely defeating the purpose
function Component({ id }: { id: string }) {
  const options = { id, limit: 10 };

  const result = useMemo(() => expensiveComputation(options), [options]);
  // `options` is a new reference every render -> memo never caches
}

// FIX: depend on primitive values
function Component({ id }: { id: string }) {
  const result = useMemo(
    () => expensiveComputation({ id, limit: 10 }),
    [id] // primitive — stable between renders when value is the same
  );
}
```

### useMemo Is Not a Semantic Guarantee

React's docs explicitly state that `useMemo` is a performance optimization, not a semantic guarantee. React may discard cached values to free memory (e.g., offscreen components). Your code must work correctly even if `useMemo` recomputes on every render.

---

## useCallback

`useCallback` is syntactic sugar for `useMemo(() => fn, deps)`. It memoizes a function's referential identity.

```tsx
// These are equivalent:
const handleClick = useCallback((id: string) => {
  selectItem(id);
}, [selectItem]);

const handleClick = useMemo(() => {
  return (id: string) => { selectItem(id); };
}, [selectItem]);
```

### When useCallback Actually Matters

**Scenario 1: Passing callbacks to memoized children**

```tsx
function TodoList({ todos }: { todos: Todo[] }) {
  const [selected, setSelected] = useState<string | null>(null);

  // Without useCallback: new function every render
  // -> MemoizedTodoItem's memo check fails on `onSelect` prop
  const handleSelect = useCallback((id: string) => {
    setSelected(id);
  }, []); // no deps — setSelected is stable

  return (
    <ul>
      {todos.map(todo => (
        <MemoizedTodoItem
          key={todo.id}
          todo={todo}
          onSelect={handleSelect}
        />
      ))}
    </ul>
  );
}
```

**Scenario 2: Stable dependency for useEffect**

```tsx
function useDataFetcher(url: string) {
  // If fetchData is not stabilized, the effect re-runs on every render
  const fetchData = useCallback(async () => {
    const res = await fetch(url);
    return res.json();
  }, [url]);

  useEffect(() => {
    fetchData().then(setData);
  }, [fetchData]);
}
```

### Common Misuse

```tsx
// POINTLESS: useCallback without a memoized consumer
function Form() {
  const handleSubmit = useCallback((e: FormEvent) => {
    e.preventDefault();
    // submit logic
  }, []);

  // <form> is a native element — it doesn't use React.memo.
  // This useCallback adds overhead for zero benefit.
  return <form onSubmit={handleSubmit}>...</form>;
}
```

**Rule of thumb**: `useCallback` is only useful if the function is either (a) passed to a `React.memo`-wrapped child, (b) used as a dependency of `useEffect`/`useMemo`/`useCallback`, or (c) used in a context value.

---

## React Compiler (React 19)

The React Compiler (formerly React Forget) is an ahead-of-time compiler that automatically inserts memoization during the build step.

### What It Auto-Memoizes

- Component return values (equivalent to wrapping every component in `React.memo`)
- Expensive expressions (equivalent to `useMemo`)
- Callback functions (equivalent to `useCallback`)
- Hook dependency arrays

### How It Works

The compiler analyzes your component code at build time using a custom Babel transform. It tracks value dependencies through assignments, function calls, and control flow, then inserts cache slots that check dependencies and return cached values when inputs are unchanged.

```tsx
// What you write:
function ProductCard({ product, onAddToCart }) {
  const discountedPrice = product.price * (1 - product.discount);
  const handleClick = () => onAddToCart(product.id);

  return (
    <div>
      <span>{discountedPrice}</span>
      <button onClick={handleClick}>Add</button>
    </div>
  );
}

// Conceptually what the compiler produces (simplified):
function ProductCard({ product, onAddToCart }) {
  const $ = useMemoCache(4);

  let discountedPrice;
  if ($[0] !== product.price || $[1] !== product.discount) {
    discountedPrice = product.price * (1 - product.discount);
    $[0] = product.price;
    $[1] = product.discount;
    $[2] = discountedPrice;
  } else {
    discountedPrice = $[2];
  }

  let handleClick;
  if ($[3] !== product.id || $[4] !== onAddToCart) {
    handleClick = () => onAddToCart(product.id);
    $[3] = product.id;
    $[4] = onAddToCart;
    $[5] = handleClick;
  } else {
    handleClick = $[5];
  }

  // JSX memoized similarly...
}
```

### Rules of React

The compiler relies on you following the Rules of React:

- Components and hooks must be pure (same inputs -> same output)
- No mutating values after rendering
- Hooks must be called at the top level, in the same order

Code that violates these rules will either be skipped by the compiler (with a diagnostic) or produce incorrect behavior.

### Current Status (as of early 2026)

- Shipped in React 19 as an opt-in Babel plugin
- Available via `babel-plugin-react-compiler`
- Adopted by Meta in production across Instagram and Facebook
- ESLint plugin (`eslint-plugin-react-compiler`) available for detecting violations
- Not yet the default — you must explicitly enable it

### Interview Implications

When discussing memoization in interviews, acknowledge the compiler:

> "I'd use `useMemo`/`useCallback` here for now, but with the React Compiler in React 19, this manual memoization becomes unnecessary. The compiler handles it automatically as long as you follow the Rules of React."

---

## Keys and Reconciliation

### Why Keys Matter

During reconciliation, React uses keys to match children in the old tree with children in the new tree. Without keys (or with index keys), React relies on position, which breaks down when the order or count of children changes.

```tsx
// BAD: index as key for a reorderable list
{todos.map((todo, index) => (
  <TodoItem key={index} todo={todo} />
))}
// If you insert an item at index 0, React thinks the item at index 0
// changed (it didn't — it moved to index 1). Every item remounts
// or receives wrong props.

// GOOD: stable, unique key
{todos.map(todo => (
  <TodoItem key={todo.id} todo={todo} />
))}
```

### Index-as-Key: When It's Actually Fine

Index keys are acceptable when **all three** conditions hold:

1. The list is static (no adds, removes, or reorders)
2. Items have no local state or uncontrolled inputs
3. Items have no stable unique identifier

### The Key Reset Pattern

Changing a component's key forces React to unmount and remount it, resetting all internal state. This is a deliberate use of the reconciliation algorithm.

```tsx
function UserProfile({ userId }: { userId: string }) {
  // When userId changes, the entire form remounts with fresh state.
  // No need for useEffect cleanup or state synchronization.
  return <ProfileForm key={userId} userId={userId} />;
}

function ProfileForm({ userId }: { userId: string }) {
  const [name, setName] = useState(""); // resets when key changes
  const [email, setEmail] = useState(""); // resets when key changes

  useEffect(() => {
    fetchUser(userId).then(user => {
      setName(user.name);
      setEmail(user.email);
    });
  }, [userId]);

  return (
    <form>
      <input value={name} onChange={e => setName(e.target.value)} />
      <input value={email} onChange={e => setEmail(e.target.value)} />
    </form>
  );
}
```

### Keys Outside of Lists

Keys work on any component, not just list items. The key prop is not forwarded to the component — it is consumed by React's reconciler.

```tsx
// Force re-initialize an animation when `step` changes
<AnimatedPanel key={step} content={steps[step]} />
```

---

## Code Splitting

### React.lazy + Suspense

```tsx
// Route-based splitting (most common and highest impact)
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Settings = lazy(() => import("./pages/Settings"));
const Analytics = lazy(() => import("./pages/Analytics"));

function App() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  );
}
```

### Component-Based Splitting

```tsx
// Split a heavy component that's conditionally rendered
const MarkdownEditor = lazy(() => import("./MarkdownEditor"));

function CommentBox({ editing }: { editing: boolean }) {
  return editing ? (
    <Suspense fallback={<TextareaSkeleton />}>
      <MarkdownEditor />
    </Suspense>
  ) : (
    <CommentDisplay />
  );
}
```

### Named Exports with lazy

`React.lazy` requires a default export. For named exports:

```tsx
// Option 1: re-export as default in a barrel file
// chartHelpers.ts
export { BarChart as default } from "./charts";

// Option 2: inline wrapper
const BarChart = lazy(() =>
  import("./charts").then(mod => ({ default: mod.BarChart }))
);
```

### Preloading

```tsx
// Preload on hover — start the download before the user clicks
function NavLink({ to, component }: { to: string; component: () => Promise<any> }) {
  const prefetch = () => component(); // trigger the dynamic import

  return (
    <Link
      to={to}
      onMouseEnter={prefetch}
      onFocus={prefetch}
    >
      {to}
    </Link>
  );
}

// Usage
const Dashboard = lazy(() => import("./pages/Dashboard"));
<NavLink to="/dashboard" component={() => import("./pages/Dashboard")} />
```

### Route-Based vs. Component-Based: Decision Framework

| Factor | Route-based | Component-based |
|--------|-------------|-----------------|
| Impact | High (separate page bundles) | Moderate (deferred heavy widgets) |
| Complexity | Low (natural split point) | Medium (manage loading states) |
| UX risk | Low (users expect page transitions) | Higher (inline loading spinners) |
| When to use | Almost always | Large modals, editors, charts, admin panels |

---

## Virtualization

Virtualization (windowing) renders only the visible items in a long list, plus a small overscan buffer. Instead of mounting 10,000 DOM nodes, you mount ~30.

### react-window (Lightweight)

```tsx
import { FixedSizeList } from "react-window";

interface RowProps {
  index: number;
  style: React.CSSProperties;
  data: Item[];
}

const Row = memo(function Row({ index, style, data }: RowProps) {
  const item = data[index];
  return (
    <div style={style}>
      {item.name} — {item.description}
    </div>
  );
});

function VirtualizedList({ items }: { items: Item[] }) {
  return (
    <FixedSizeList
      height={600}
      width="100%"
      itemCount={items.length}
      itemSize={50}
      itemData={items}
    >
      {Row}
    </FixedSizeList>
  );
}
```

### TanStack Virtual (Framework-Agnostic, More Flexible)

```tsx
import { useVirtualizer } from "@tanstack/react-virtual";

function VirtualList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,
  });

  return (
    <div ref={parentRef} style={{ height: 600, overflow: "auto" }}>
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div
            key={virtualRow.key}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            {items[virtualRow.index].name}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### When to Virtualize

- **Do virtualize**: Lists with >100 items, tables with >50 rows, any scroll container where DOM node count causes jank
- **Don't virtualize**: Short lists (<50 items), lists that need to be fully searchable by browser Ctrl+F, cases where the implementation complexity outweighs the performance gain

---

## React Profiler

### DevTools Flame Graph

The React DevTools Profiler records commit-by-commit render information:

1. Open React DevTools -> Profiler tab
2. Click Record
3. Interact with your app
4. Click Stop
5. Analyze the flame graph:
   - **Gray bars**: component did not render in this commit
   - **Blue/green bars**: component rendered (color intensity = render time)
   - **Yellow/red bars**: component was slow to render

Key metrics per commit:
- **Render duration**: time React spent rendering
- **Commit duration**: time React spent committing DOM changes
- **"Why did this render?"**: enable in Profiler settings (gear icon)

### Profiler API (Programmatic)

```tsx
import { Profiler, ProfilerOnRenderCallback } from "react";

const onRender: ProfilerOnRenderCallback = (
  id,           // the "id" prop of the Profiler tree
  phase,        // "mount" | "update" | "nested-update"
  actualDuration,   // time spent rendering the committed update
  baseDuration,     // estimated time to render the entire subtree without memoization
  startTime,        // when React began rendering this update
  commitTime,       // when React committed this update
) => {
  // Send to analytics, log, etc.
  if (actualDuration > 16) {
    console.warn(`Slow render in ${id}: ${actualDuration.toFixed(2)}ms`);
  }
};

function App() {
  return (
    <Profiler id="Dashboard" onRender={onRender}>
      <Dashboard />
    </Profiler>
  );
}
```

### Identifying Wasted Renders

A "wasted render" is one where the component re-rendered but its output didn't change. Detection strategies:

1. **React DevTools "Highlight updates"**: Settings -> General -> "Highlight updates when components render." Flashing borders show which components re-rendered.
2. **"Why did this render?" in Profiler**: Shows whether the render was caused by state change, parent re-render, context change, or hook change.
3. **`baseDuration` vs. `actualDuration`**: If `actualDuration` is close to `baseDuration`, memoization is not helping much.

---

## startTransition and useDeferredValue

### startTransition

Marks a state update as non-urgent. React can interrupt the rendering of transition updates to handle urgent updates (like typing) first.

```tsx
import { useState, useTransition } from "react";

function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Item[]>([]);
  const [isPending, startTransition] = useTransition();

  const handleSearch = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;

    // Urgent: update the input immediately
    setQuery(value);

    // Non-urgent: filter/search can be deferred
    startTransition(() => {
      const filtered = expensiveFilter(allItems, value);
      setResults(filtered);
    });
  };

  return (
    <div>
      <input value={query} onChange={handleSearch} />
      {isPending && <Spinner />}
      <ResultsList items={results} />
    </div>
  );
}
```

### useDeferredValue

Creates a deferred version of a value. When the original value changes, the deferred value "lags behind," allowing React to prioritize rendering with the old deferred value while computing the new one in the background.

```tsx
import { useDeferredValue, useMemo } from "react";

function FilteredList({ query, items }: { query: string; items: Item[] }) {
  // `deferredQuery` lags behind `query` during rapid updates
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;

  const filtered = useMemo(
    () => items.filter(item =>
      item.name.toLowerCase().includes(deferredQuery.toLowerCase())
    ),
    [deferredQuery, items]
  );

  return (
    <div style={{ opacity: isStale ? 0.7 : 1 }}>
      {filtered.map(item => (
        <ListItem key={item.id} item={item} />
      ))}
    </div>
  );
}
```

### startTransition vs. useDeferredValue

| Aspect | startTransition | useDeferredValue |
|--------|----------------|------------------|
| Controls | The state update itself | A derived value |
| Use when | You own the state setter | You receive data as a prop |
| Works with | `useState` / `useReducer` dispatches | Any value |
| Interrupts render | Yes | Yes |
| Shows stale UI | Via `isPending` | Via comparing original vs deferred |

### When to Use Concurrent Features

- **Large list filtering**: Defer the filtered list while keeping the input responsive
- **Tab switching**: Wrap `setActiveTab` in `startTransition` to keep the old tab visible while the new tab renders
- **Data-heavy dashboards**: Defer expensive chart re-renders
- **Search-as-you-type**: Separate the input update from the results update

---

## Common Interview Questions

### Q1: "A component is re-rendering too often. Walk me through your debugging process."

**Answer**: I'd follow a systematic approach:

1. **Identify the problem**: Use React DevTools Profiler to record and find which components are rendering unnecessarily. Enable "Why did this render?" to see the cause.
2. **Check the render trigger**: Is it a state change, parent re-render, or context change?
3. **For parent re-renders**: Consider whether the child needs to re-render. If it's expensive, wrap it in `React.memo`. Ensure props have stable references — use `useMemo` for objects/arrays and `useCallback` for functions.
4. **For context changes**: Check if the context value is a new object every render. Memoize the context value. Consider splitting the context into smaller contexts so consumers only subscribe to what they need.
5. **For state changes**: Check if state is lifted too high. Consider colocating state closer to where it's used.
6. **Measure the impact**: Use the Profiler to confirm the optimization actually improved performance. Don't optimize blindly.

### Q2: "When would you NOT use React.memo?"

**Answer**: I'd skip `React.memo` when:

- The component is cheap to render (simple UI, few children). The overhead of shallow comparison may exceed the cost of rendering.
- Props change on virtually every render (unstable references that can't easily be stabilized). Memo would run the comparison and then re-render anyway.
- The component is a leaf node with primitive props only and renders in <1ms. Not worth the code complexity.
- The React Compiler is enabled — it handles memoization automatically.
- During early development when the component structure is still evolving. Premature memoization creates maintenance burden.

### Q3: "Explain the difference between useMemo and useCallback."

**Answer**: They serve the same underlying mechanism — caching a value between renders when dependencies haven't changed. `useCallback(fn, deps)` is literally `useMemo(() => fn, deps)`. The difference is intent:

- `useMemo` caches the **return value** of a function (computed data, derived objects, JSX)
- `useCallback` caches the **function itself** (stable identity for callbacks)

Both are for performance. Neither should affect correctness — your code must work identically if React drops the cache.

### Q4: "You have a list of 10,000 items. How do you make it performant?"

**Answer**: Layered approach:

1. **Virtualize first**: Use TanStack Virtual or react-window to render only visible items (~30-50 DOM nodes instead of 10,000).
2. **Memoize row components**: Wrap list items in `React.memo` with stable keys so rows outside the viewport change don't cause re-renders of visible rows.
3. **Stabilize callbacks**: `useCallback` for any handlers passed to row components.
4. **Defer filtering/sorting**: If the list is filterable, use `useDeferredValue` or `startTransition` to keep the UI responsive during expensive recomputation.
5. **Paginate on the server**: If feasible, don't send 10,000 items to the client at all. Use cursor-based pagination.

### Q5: "What is the React Compiler and how does it change performance optimization?"

**Answer**: The React Compiler is a build-time tool that analyzes your components and automatically inserts memoization. It uses a custom Babel plugin to track data dependencies through your component code and inserts fine-grained caching similar to what `useMemo`, `useCallback`, and `React.memo` provide manually.

It ships with React 19 as opt-in. The practical impact is that manual memoization (`useMemo`, `useCallback`, `React.memo`) becomes unnecessary for most cases. The compiler does it better than humans because it can memoize at a granularity that would be impractical to do by hand (individual JSX expressions, intermediate computations).

The key requirement is following the Rules of React — pure components, no mutation during render, hooks at the top level. Code that violates these rules gets skipped by the compiler.

### Q6: "How does startTransition differ from debouncing?"

**Answer**: They solve similar UX problems but work at fundamentally different levels:

- **Debouncing** delays the state update entirely. The user types, and after N ms of inactivity, you update state. The downside is added latency — results always appear N ms after the user stops typing.
- **startTransition** updates state immediately but tells React the render is interruptible. React starts rendering the transition but will abandon that work if a higher-priority update arrives (like another keystroke). There's no artificial delay — if the render completes before the next keystroke, results appear instantly.

`startTransition` also integrates with Suspense boundaries. A transition that triggers a Suspense fallback will keep showing the previous UI (with `isPending`) instead of flashing a loading spinner.

### Q7: "When should you use the key reset pattern vs. useEffect for resetting component state?"

**Answer**: The key reset pattern (`<Component key={id} />`) is preferable when:

- You want to reset **all** state inside a component tree
- The component has complex internal state that's tedious to reset manually
- You want to guarantee a clean slate (equivalent to unmount + remount)

`useEffect` is better when:

- You want to reset only **some** state while preserving others
- The component has expensive initialization that you want to avoid repeating (subscriptions, DOM measurements)
- You need to react to the change with side effects beyond state reset

The key pattern is more declarative and less error-prone. The `useEffect` approach risks missing state variables or running into stale closure issues.

### Q8: "How would you optimize a React app that feels sluggish on initial load?"

**Answer**: Systematic approach, measuring each step:

1. **Analyze the bundle**: Use webpack-bundle-analyzer or equivalent to find the largest chunks. Look for accidentally bundled dev tools, duplicate dependencies, and tree-shaking failures.
2. **Code split routes**: `React.lazy` + `Suspense` for route-level splitting. This is the highest-impact change for initial load.
3. **Defer non-critical JS**: Dynamic imports for features not visible on the initial viewport (modals, below-fold content, admin tools).
4. **Optimize server-side**: If using SSR (Next.js / Remix), ensure streaming is enabled. Use selective hydration to prioritize interactive elements.
5. **Prefetch likely routes**: On hover or during idle time, start loading chunks the user is likely to navigate to.
6. **Audit third-party scripts**: Analytics, chat widgets, and ads often dominate load time. Load them after the main app.
7. **Measure with Lighthouse and Web Vitals**: Focus on LCP, FID/INP, and CLS. These are what users actually perceive.

---

## Summary

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| `React.memo` | Skip re-rendering when props unchanged | Expensive components with stable props |
| `useMemo` | Cache computed values | Expensive derivations, referential stability |
| `useCallback` | Stable function identity | Callbacks passed to memo'd children |
| React Compiler | Automatic memoization | React 19 projects (opt-in) |
| `React.lazy` | Code splitting | Route-level and heavy component splitting |
| Virtualization | Render only visible items | Lists > 100 items |
| `startTransition` | Mark non-urgent updates | Keeping input responsive during heavy renders |
| `useDeferredValue` | Deferred derived values | Filtering/searching with received props |
| Key reset | Force remount | Resetting all component state |
| Profiler | Measure render performance | Before and after optimization |
