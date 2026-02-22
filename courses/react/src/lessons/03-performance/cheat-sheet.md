# Performance Optimization Cheat Sheet

## "Should I Memoize This?" Decision Flowchart

```
Is React Compiler enabled?
├── YES --> Do not manually memoize. The compiler handles it.
└── NO
    │
    Have you profiled and confirmed a performance problem?
    ├── NO --> Do not memoize. Optimize later when needed.
    └── YES
        │
        What is the bottleneck?
        ├── EXPENSIVE COMPUTATION (>2ms)
        │   └── useMemo with correct dependency array
        │
        ├── CHILD COMPONENT RE-RENDERING UNNECESSARILY
        │   ├── Is the child expensive to render?
        │   │   ├── NO --> Leave it. Cheap renders are fine.
        │   │   └── YES
        │   │       ├── Wrap child in React.memo
        │   │       ├── Stabilize object/array props with useMemo
        │   │       └── Stabilize callback props with useCallback
        │   │
        │   └── Is context causing it?
        │       ├── Split context into smaller pieces
        │       └── Memoize the provider value
        │
        ├── LARGE LIST (>100 items)
        │   └── Virtualize with TanStack Virtual or react-window
        │
        ├── INPUT LAG / JANKY UI
        │   └── Wrap non-urgent updates in startTransition
        │       or use useDeferredValue
        │
        └── LARGE INITIAL BUNDLE
            └── Code split with React.lazy + Suspense
```

## Performance API Quick Reference

### React.memo

```tsx
const Memoized = React.memo(Component);
const Memoized = React.memo(Component, (prev, next) => /* true to skip */);
```

- Shallow-compares all props using `Object.is`
- Returns `true` from custom comparator to **skip** re-render
- Does NOT prevent re-renders from internal state or context changes

### useMemo

```tsx
const value = useMemo(() => computeExpensive(a, b), [a, b]);
```

- Caches return value until dependencies change
- Dependencies compared with `Object.is`
- Not a semantic guarantee; React may discard cache

### useCallback

```tsx
const fn = useCallback((arg: T) => doSomething(arg, dep), [dep]);
```

- Equivalent to `useMemo(() => fn, deps)`
- Only useful when passed to memo'd children or used as a hook dependency

### React.lazy + Suspense

```tsx
const Component = lazy(() => import("./Component"));

<Suspense fallback={<Loading />}>
  <Component />
</Suspense>
```

- Accepts only default exports (wrap named exports)
- Suspense boundary catches the loading state
- Nest multiple Suspense boundaries for granular loading UI

### startTransition

```tsx
const [isPending, startTransition] = useTransition();

startTransition(() => {
  setExpensiveState(newValue);
});
```

- Marks state update as non-urgent (interruptible)
- `isPending` is `true` while the transition renders
- Does not delay the update; allows interruption

### useDeferredValue

```tsx
const deferred = useDeferredValue(value);
const isStale = value !== deferred;
```

- Returns a deferred version of the value that lags behind
- Use when you receive data as a prop (don't control the setter)
- Combine with `useMemo` to avoid recomputing with stale inputs

---

## Common Re-Render Causes and Fixes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Parent re-renders | Child re-renders despite unchanged props | `React.memo` on child |
| New object literal in props | `React.memo` never skips | `useMemo` on the object, or hoist to module scope |
| New function in props | `React.memo` never skips | `useCallback` on the function |
| Context value is new object | All consumers re-render | `useMemo` on provider value |
| Too-broad context | Unrelated consumers re-render | Split into focused contexts |
| State too high in tree | Large subtree re-renders | Colocate state closer to usage |
| Inline component definition | Component remounts every render | Extract to module-level definition |
| Index as key in dynamic list | Items remount on reorder/insert | Use stable unique IDs as keys |
| Uncontrolled to controlled switch | Input loses state | Pick one pattern and stick with it |
| Missing dependency in useMemo | Stale cached value | Add all dependencies; use lint rule |

---

## Profiling Steps

1. **Reproduce the problem** in a development build with React DevTools installed.
2. **Enable "Highlight updates"** in React DevTools settings to visually see which components re-render.
3. **Record with React Profiler**: Click Record, perform the slow interaction, click Stop.
4. **Analyze the flame graph**: Sort by render duration. Identify the slowest components.
5. **Check "Why did this render?"** for each slow component (enable in Profiler settings).
6. **Record with Chrome Performance tab**: Same interaction. Look for long tasks (>50ms).
7. **Correlate**: Match React commits to Chrome main thread activity.
8. **Apply targeted fix**: Based on the root cause (see table above).
9. **Re-profile**: Confirm the fix reduced render time. Compare before/after screenshots.
10. **Test in production build**: Development builds are 3-10x slower. Always validate in production mode.

---

## Bundle Size Reduction Checklist

- [ ] Analyze bundle with `webpack-bundle-analyzer` or `source-map-explorer`
- [ ] Code split all routes with `React.lazy`
- [ ] Dynamic import heavy libraries (chart libs, editors, date pickers)
- [ ] Replace moment.js with date-fns or dayjs (tree-shakeable)
- [ ] Check for duplicate dependencies (`npm ls <package>`)
- [ ] Enable tree shaking (ES modules, `sideEffects: false` in package.json)
- [ ] Use named imports (`import { debounce } from "lodash-es"` not `import _ from "lodash"`)
- [ ] Lazy load below-fold content and modals
- [ ] Compress with gzip/brotli at the CDN or server level
- [ ] Set appropriate `Cache-Control` headers for hashed assets
- [ ] Audit third-party scripts (analytics, chat, ads) for size and load timing
- [ ] Consider lighter alternatives (preact, million.js) for performance-critical widgets

---

## Rules of Thumb

| Rule | Rationale |
|------|-----------|
| Memoize computations that take >2ms | Below 2ms, the overhead of memoization and dependency checking approaches the cost of recomputing |
| Virtualize lists with >100 items | At 100+ DOM nodes with event handlers and children, layout and paint costs become noticeable |
| Code split at route boundaries first | Routes are natural async boundaries; users expect page transitions to have loading states |
| Profile before optimizing | Intuition about bottlenecks is wrong ~70% of the time; measure first |
| Prefer composition over memoization | Moving state down or lifting content up via `children` avoids re-renders without any API |
| Keep context values narrow | A context with 20 fields triggers all consumers on any single field change |
| Stable keys > index keys for dynamic lists | Index keys cause remounts on insert/delete/reorder, destroying state |
| `startTransition` for anything not directly typed | User input must be synchronous; everything else can be deferred |
| Production builds for benchmarking | Dev mode adds StrictMode double-renders, extra warnings, and disables compiler optimizations |
| Measure INP, not just render time | Interaction to Next Paint captures the full user-perceived delay, including browser work |
