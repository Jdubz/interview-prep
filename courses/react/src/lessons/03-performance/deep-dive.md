# Performance Deep Dive

## Fiber Architecture

### What Is a Fiber?

A Fiber is a JavaScript object that represents a unit of work. Every React element (component instance, DOM node, fragment) has a corresponding Fiber node. The Fiber tree is React's internal mutable working copy of your component tree.

Key Fiber node fields:

```
{
  tag: FunctionComponent | HostComponent | ...,  // what kind of element
  type: MyComponent | "div" | ...,               // the component function or DOM tag
  key: string | null,
  stateNode: DOM node | null,                    // for host components
  return: Fiber | null,                          // parent
  child: Fiber | null,                           // first child
  sibling: Fiber | null,                         // next sibling
  memoizedState: Hook | null,                    // linked list of hooks
  memoizedProps: Props,
  pendingProps: Props,
  flags: number,                                 // side effects (Placement, Update, Deletion)
  lanes: number,                                 // priority bitmask
  alternate: Fiber | null,                       // the "other" tree (current <-> workInProgress)
}
```

### Double Buffering

React maintains two Fiber trees:

1. **Current tree**: Represents what's currently on screen. React reads from this during the commit phase.
2. **Work-in-progress (WIP) tree**: Built during the render phase. Each Fiber in the current tree has an `alternate` pointing to its WIP counterpart (and vice versa).

When a render completes, React swaps the pointers: the WIP tree becomes the current tree, and the old current tree becomes available for reuse as the next WIP tree. This is analogous to double buffering in graphics rendering.

### The Work Loop

The core of React's render phase is a loop that walks the Fiber tree:

```
function workLoopConcurrent() {
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}
```

`performUnitOfWork` does two things:

1. **Begin work**: Calls the component function (or processes the host element), reconciles children, and returns the next child to process.
2. **Complete work**: When a subtree is fully processed, bubbles up — creating DOM nodes, collecting effects.

The traversal is depth-first: go down via `child`, then across via `sibling`, then up via `return`.

### Time Slicing and Interruptible Rendering

The `shouldYield()` check is what makes concurrent rendering possible. It checks whether the browser needs the main thread back (typically using a 5ms deadline via `MessageChannel`). If the deadline has passed:

1. React pauses the work loop
2. Control returns to the browser (for painting, input handling, etc.)
3. React schedules a new task to resume where it left off

This is only possible because the render phase is pure — no DOM mutations, no side effects. React can safely abandon or restart render work.

**Critical constraint**: This means component functions may be called multiple times for a single committed update. Never put side effects in the render path.

---

## Reconciliation Algorithm Details

### Tree Diffing Heuristics

React's diffing algorithm operates with two key heuristics that reduce O(n^3) tree diffing to O(n):

1. **Different types produce different trees**: If the element type changes (e.g., `<div>` to `<span>`, or `ComponentA` to `ComponentB`), React tears down the old subtree entirely and builds a new one. No attempt is made to reuse nodes across type boundaries.

2. **Keys identify stable elements across renders**: Within a list of children at the same level, keys tell React which elements correspond to each other.

### Type Comparison

```tsx
// Render 1:
<div>
  <Counter />
</div>

// Render 2:
<span>      // <-- type changed from "div" to "span"
  <Counter />
</span>
```

React does not try to move `Counter` from the old `<div>` to the new `<span>`. It unmounts the entire old subtree (destroying `Counter`'s state) and mounts a fresh subtree.

This is why you must never define components inside other components:

```tsx
function Parent() {
  // BUG: This creates a new component type every render.
  // React sees a different type each time and remounts.
  function Child() {
    return <input />;
  }

  return <Child />;
  // The <input> loses focus and state on every Parent re-render
  // because React unmounts and remounts it each time.
}
```

### Key-Based Matching

Without keys, React matches children by index:

```
Old: [A, B, C]
New: [X, A, B, C]

Index-based diff:
  0: A -> X (update)
  1: B -> A (update)
  2: C -> B (update)
  3: (none) -> C (insert)
  // 4 operations, all children updated
```

With keys:

```
Old: [A:a, B:b, C:c]
New: [X:x, A:a, B:b, C:c]

Key-based diff:
  x: (none) -> X (insert)
  a: A -> A (keep)
  b: B -> B (keep)
  c: C -> C (keep)
  // 1 insert, 3 no-ops — much cheaper
```

React builds a map of `key -> Fiber` from the old children, then iterates the new children, looking up each key. Matched Fibers are reused; unmatched old Fibers are deleted; unmatched new elements are inserted.

---

## Lane Model

### What Are Lanes?

Lanes are a bitmask-based priority system introduced in React 18 to replace the older `expirationTime` model. Each update is assigned one or more lane bits, and React processes lanes in priority order.

Key lane levels (from highest to lowest priority):

| Lane | Decimal | Purpose |
|------|---------|---------|
| `SyncLane` | 1 | Discrete user events (click, keypress), `flushSync` |
| `InputContinuousLane` | 4 | Continuous user events (mousemove, scroll) |
| `DefaultLane` | 16 | Normal updates (setState inside setTimeout, fetch callbacks) |
| `TransitionLane1..16` | 64..524288 | `startTransition` updates |
| `IdleLane` | 536870912 | `useDeferredValue`, offscreen updates |

### Batching Semantics

Updates within the same event are batched into the same lane. React processes all updates in a lane together in a single render pass.

```tsx
function handleClick() {
  setA(1); // SyncLane
  setB(2); // SyncLane (same event -> same lane -> batched)
}

function handleClick() {
  setA(1); // SyncLane

  startTransition(() => {
    setB(2); // TransitionLane (different lane -> separate render pass)
  });
}
```

### SyncLane vs. TransitionLane

| Aspect | SyncLane | TransitionLane |
|--------|----------|----------------|
| Interruptible | No | Yes |
| Shows intermediate state | No (committed synchronously) | Yes (can show stale UI with `isPending`) |
| Triggers Suspense fallback | Yes | No (keeps previous UI) |
| Used by | Click, type, flushSync | startTransition, useDeferredValue |

### Lane Entanglement

When React processes a TransitionLane and encounters a SyncLane update mid-render, it abandons the transition render and processes the SyncLane update first. After the sync update commits, React restarts the transition render. This is the mechanism behind `startTransition` keeping inputs responsive.

---

## Concurrent Rendering

### What "Concurrent" Actually Means

Concurrent rendering does **not** mean multi-threaded. JavaScript is single-threaded. "Concurrent" in React means:

1. **Rendering is interruptible**: React can pause a render in progress and resume it later.
2. **Multiple versions of UI can be "in progress"**: React can prepare a new UI in memory without committing it while the old UI remains on screen.
3. **Renders can be abandoned**: If a higher-priority update arrives, React can discard an in-progress render and start over.

### The Scheduler

React's scheduler (`react-reconciler/src/Scheduler.js`) manages work units:

1. Work is enqueued with a priority (lane).
2. The scheduler uses `MessageChannel` (not `requestIdleCallback`) to schedule tasks. `MessageChannel` fires after microtasks but before the next paint, giving predictable 5ms chunks.
3. Higher-priority work preempts lower-priority work.
4. The scheduler tracks multiple pending tasks and interleaves them based on priority and deadlines.

### Practical Implications

- **Component functions may be called but never committed**: If React starts rendering a transition but abandons it, your component ran for nothing. This is fine if your components are pure. It breaks if you have side effects in the render path.
- **State is consistent within a render**: Even though rendering is interruptible, a single component always sees a consistent snapshot of state. React does not mix states from different updates within one render.
- **StrictMode double-invokes**: In development, React intentionally double-invokes component functions, `useMemo`, and `useState` initializers to help you detect impure renders. This is a development-only behavior that does not happen in production.

---

## Selective Hydration

### Streaming SSR + Progressive Hydration

With React 18's `renderToPipeableStream` (Node) or `renderToReadableStream` (Web Streams), the server can stream HTML as it becomes available:

```tsx
// server.ts
import { renderToPipeableStream } from "react-dom/server";

app.get("/", (req, res) => {
  const { pipe } = renderToPipeableStream(<App />, {
    bootstrapScripts: ["/client.js"],
    onShellReady() {
      res.statusCode = 200;
      res.setHeader("Content-Type", "text/html");
      pipe(res);
    },
    onError(error) {
      console.error(error);
      res.statusCode = 500;
    },
  });
});
```

Suspense boundaries define the streaming chunks:

```tsx
function App() {
  return (
    <Layout>
      {/* Shell: streamed immediately */}
      <Header />
      <Suspense fallback={<NavSkeleton />}>
        {/* Streamed when data resolves */}
        <Navigation />
      </Suspense>
      <Suspense fallback={<ContentSkeleton />}>
        {/* Streamed when data resolves */}
        <MainContent />
      </Suspense>
    </Layout>
  );
}
```

### Priority-Based Hydration

Selective hydration allows React to prioritize which parts of the page to hydrate first based on user interaction:

1. HTML arrives from the server (static, not yet interactive)
2. React starts hydrating from the top
3. If the user clicks on an unhydrated Suspense boundary, React **reprioritizes** that boundary's hydration above others
4. The clicked component hydrates first and handles the event

This means a user clicking a button inside a Suspense boundary that hasn't hydrated yet won't experience a dead click — React fast-tracks that region's hydration.

---

## Performance Profiling Workflow

### Combined Chrome DevTools + React Profiler

For production-grade performance analysis, use both tools together:

**Step 1: Chrome Performance Tab (macro-level)**

1. Open Chrome DevTools -> Performance tab
2. Enable "Screenshots" and "Web Vitals"
3. Record the interaction
4. Analyze the main thread flame chart:
   - Long tasks (>50ms) highlighted with red corners
   - Look for "Recalculate Style" and "Layout" (forced reflow)
   - Identify if the bottleneck is JS execution, layout, or paint
5. Check the "Summary" tab for time distribution: Scripting / Rendering / Painting / System / Idle

**Step 2: React DevTools Profiler (component-level)**

1. Record the same interaction in React DevTools Profiler
2. Examine the flame graph:
   - Find the darkest (slowest) bars
   - Check "Why did this render?" for each slow component
   - Compare `actualDuration` to `baseDuration` to assess memoization effectiveness
3. Look at the "Ranked" view to sort components by render time

**Step 3: Correlate**

- A long task in Chrome DevTools that corresponds to a React commit tells you the problem is in render/reconciliation
- A long task after the commit (during layout/paint) suggests DOM-level issues (too many nodes, expensive CSS)
- Frequent short commits may indicate unnecessary re-renders (optimize at the React level)

### Measuring in Production

```tsx
// Report Web Vitals
import { onCLS, onFID, onLCP, onINP, onTTFB } from "web-vitals";

function reportMetric(metric: { name: string; value: number; id: string }) {
  // Send to your analytics backend
  navigator.sendBeacon("/api/metrics", JSON.stringify(metric));
}

onCLS(reportMetric);
onLCP(reportMetric);
onINP(reportMetric);
onTTFB(reportMetric);
```

---

## Real-World Optimization Case Studies

### Case Study 1: Large Form (200+ Fields)

**Problem**: A medical intake form with 200+ fields re-rendered the entire form on every keystroke. Each render took ~80ms, causing visible input lag.

**Root cause**: All form state lived in a single `useReducer` at the top level. Every field change dispatched an action, causing the entire form tree to re-render.

**Solution (layered)**:

1. **State colocation**: Moved each field's local state (value, touched, error) into the field component itself via `useState`. The top-level reducer only held submission-ready data.
2. **React.memo on field components**: Each `FormField` was wrapped in `React.memo`. Since field-level state was now local, parent re-renders didn't cause field re-renders.
3. **Debounced validation**: Cross-field validation (which required the full form state) was debounced to 300ms and wrapped in `startTransition`.
4. **Sectioned context**: Split the single form context into per-section contexts so updating Section A didn't re-render Section B's consumers.

**Result**: Per-keystroke render dropped from ~80ms to ~2ms.

### Case Study 2: Infinite Scroll Feed

**Problem**: A social media-style feed with infinite scroll. After loading ~500 posts, scrolling became janky. Memory usage grew linearly.

**Root cause**: All 500+ post components remained mounted in the DOM. Each post contained images, interactive buttons, and nested comment previews.

**Solution**:

1. **Virtualization with TanStack Virtual**: Only rendered ~15 posts (viewport + overscan). Used `estimateSize` with a measurement cache for variable-height posts.
2. **Image lazy loading**: Used `loading="lazy"` on images and `IntersectionObserver` for eager preloading of the next ~5 images.
3. **Memoized post components**: `React.memo` on `PostCard` with `useCallback` for interaction handlers.
4. **Stale data eviction**: Kept only the most recent ~200 posts in state. Older posts were evicted from the client and re-fetched if the user scrolled back.
5. **Optimistic updates**: Likes, bookmarks, and other interactions updated the local cache immediately via optimistic mutation, avoiding a full re-render from server response.

**Result**: Consistent 60fps scrolling at any feed depth. Memory capped at ~50MB regardless of session length.

### Case Study 3: Real-Time Dashboard

**Problem**: A monitoring dashboard receiving WebSocket updates every 100ms. 12 chart widgets, each with ~1000 data points. The entire page re-rendered on every update, causing ~120ms renders and dropped frames.

**Root cause**: All 12 charts shared a single `DashboardContext` containing the full data state. Every WebSocket message updated this context, triggering all 12 charts to re-render even if only one chart's data changed.

**Solution**:

1. **Per-metric subscriptions**: Replaced the single context with a pub/sub store (inspired by `useSyncExternalStore`). Each chart subscribed only to its own metric stream.

```tsx
function useMetricStream(metricId: string): DataPoint[] {
  return useSyncExternalStore(
    (cb) => metricsStore.subscribe(metricId, cb),
    () => metricsStore.getSnapshot(metricId),
    () => metricsStore.getServerSnapshot(metricId),
  );
}
```

2. **Throttled rendering**: Accumulated WebSocket messages in a buffer and flushed to state at 10fps (every 100ms) using `requestAnimationFrame` batching, rather than rendering on every message.
3. **Canvas-based charts**: Switched from SVG (DOM-based) to Canvas rendering for charts. This eliminated thousands of DOM nodes and moved the rendering bottleneck from the browser's layout engine to GPU-accelerated canvas.
4. **startTransition for non-visible charts**: Charts in collapsed/tabbed sections rendered with `startTransition` so they didn't block the visible charts.

**Result**: Render time dropped from ~120ms to ~8ms per update cycle. Smooth 60fps even with 12 active real-time charts.
