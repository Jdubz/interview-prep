# Lesson 08: React Internals — Cheat Sheet

> Quick reference for interview prep. Skim before your interview.

---

## Re-Render Triggers Checklist

A component re-renders when **any** of these happen:

- [ ] `useState` setter called with a new value
- [ ] `useReducer` dispatch called
- [ ] Parent component re-rendered (regardless of prop changes)
- [ ] Context value changed (for any context consumed via `useContext`)
- [ ] `useSyncExternalStore` detects a snapshot change
- [ ] `forceUpdate()` called (class components only)
- [ ] Custom hook internally calls any of the above

**NOT a re-render trigger:**

- Prop changes alone (props changing without a parent re-render is impossible)
- `useRef.current` changing (refs do not trigger re-renders)
- Direct DOM mutation

---

## Rendering Phases Diagram

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         RENDER PHASE                                │
 │  Pure | Interruptible (concurrent) | No DOM writes                  │
 │                                                                     │
 │  1. Call component functions                                        │
 │  2. Diff returned elements against previous fiber children          │
 │  3. Create/update/delete fibers                                     │
 │  4. Tag fibers with effect flags (Placement, Update, Deletion)      │
 │                                                                     │
 │  Output: workInProgress fiber tree with effect flags                │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         COMMIT PHASE                                │
 │  Synchronous | NOT interruptible | Side effects happen here         │
 │                                                                     │
 │  ┌───────────────────┐                                              │
 │  │ Before Mutation    │  getSnapshotBeforeUpdate (class)            │
 │  └────────┬──────────┘                                              │
 │           ▼                                                         │
 │  ┌───────────────────┐                                              │
 │  │ Mutation           │  DOM insertions, updates, deletions         │
 │  └────────┬──────────┘                                              │
 │           ▼                                                         │
 │  ┌───────────────────┐                                              │
 │  │ Layout             │  useLayoutEffect callbacks run              │
 │  │                    │  componentDidMount/Update run               │
 │  │                    │  DOM is mutated, browser has NOT painted    │
 │  └────────┬──────────┘                                              │
 └───────────┼─────────────────────────────────────────────────────────┘
             │
             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Browser paints                                                     │
 └────────────────────────────┬────────────────────────────────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Passive Effects          │  useEffect callbacks run                │
 │  (async, after paint)     │  useEffect cleanups run                 │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Fiber Node Key Properties

| Property | Type | Description |
|---|---|---|
| `tag` | `number` | Fiber kind: 0=Function, 1=Class, 3=HostRoot, 5=HostComponent, 6=HostText, 13=Suspense |
| `type` | `any` | Component function, class, or DOM tag string (`'div'`) |
| `key` | `string \| null` | The key prop for reconciliation |
| `stateNode` | `any` | DOM node (host), class instance (class), or `null` (function) |
| `memoizedState` | `any` | Hooks linked list (function) or state object (class) |
| `memoizedProps` | `any` | Props from last completed render |
| `pendingProps` | `any` | Props for current in-progress render |
| `flags` | `number` | Bitwise effect flags: Placement, Update, Deletion, Passive, etc. |
| `subtreeFlags` | `number` | Bubbled-up flags from descendants (skip subtree if `NoFlags`) |
| `lanes` | `number` | Pending update priority bitmask |
| `childLanes` | `number` | Bubbled-up lanes from descendants |
| `child` | `Fiber \| null` | First child |
| `sibling` | `Fiber \| null` | Next sibling |
| `return` | `Fiber \| null` | Parent fiber |
| `alternate` | `Fiber \| null` | Counterpart in the other tree (current <-> workInProgress) |

---

## Concurrent API Quick Reference

### `startTransition`

```tsx
import { startTransition } from 'react';

startTransition(() => {
  setState(newValue); // marked as non-urgent (TransitionLane)
});
```

- No pending indicator. Use `useTransition` if you need one.
- Works outside of components (e.g., in event handler utilities).

### `useTransition`

```tsx
const [isPending, startTransition] = useTransition();

startTransition(() => {
  setState(newValue);
});
// isPending === true while the transition is rendering
```

- `isPending` flips to `true` immediately and `false` when the transition commits.
- Use for showing loading indicators during tab switches, navigation, etc.

### `useDeferredValue`

```tsx
const deferredValue = useDeferredValue(value);
const isStale = value !== deferredValue;
```

- Returns the old value during urgent renders, updates to the new value in a background render.
- Use when you receive a value as a prop and cannot wrap the update in `startTransition`.
- Combine with `React.memo` on the consuming component for best effect.

### When to Use Which

| Scenario | API |
|---|---|
| You control the `setState` call | `startTransition` / `useTransition` |
| You receive the value as a prop | `useDeferredValue` |
| You need a loading indicator | `useTransition` (provides `isPending`) |
| Updating from outside React (utility) | `startTransition` (module import) |

---

## Batching Rules

| Context | React 17 | React 18+ |
|---|---|---|
| React event handlers | Batched | Batched |
| `setTimeout` / `setInterval` | **NOT batched** | Batched |
| `Promise.then` / `async/await` | **NOT batched** | Batched |
| Native event listeners | **NOT batched** | Batched |
| `fetch` callbacks | **NOT batched** | Batched |

**Opt out of batching:**

```tsx
import { flushSync } from 'react-dom';

flushSync(() => setState1(a)); // commits immediately
flushSync(() => setState2(b)); // commits immediately
// Two separate renders + commits
```

**Prerequisite for React 18 batching:** Must use `createRoot` instead of `ReactDOM.render`.

---

## Hydration Error Causes and Fixes

| Cause | Fix |
|---|---|
| `Date.now()` / `Math.random()` in render | Move to `useEffect`; render `null` or placeholder on server |
| `typeof window !== 'undefined'` conditionals | Use `useEffect` + state for client-only rendering |
| Browser auto-correcting invalid HTML | Fix nesting: no `<div>` inside `<p>`, no `<p>` inside `<p>`, etc. |
| Different data on server vs client | Ensure same data source; pass server data via props/context |
| Non-deterministic CSS class names | Use deterministic CSS-in-JS config or CSS Modules |
| Browser extensions modifying DOM | Use `suppressHydrationWarning` on affected elements (sparingly) |
| Third-party scripts injecting elements | Render third-party content inside `useEffect` |

**Client-only rendering pattern:**

```tsx
const [mounted, setMounted] = useState(false);
useEffect(() => setMounted(true), []);
if (!mounted) return <Placeholder />;  // matches server
return <ClientOnlyContent />;
```

---

## "Why Did This Re-Render?" Debugging Steps

1. **Check the component itself.** Did it call `setState` / `dispatch`?

2. **Check the parent.** Did the parent re-render? (Most common cause.)
   - Use React DevTools Profiler: enable "Record why each component rendered."

3. **Check context.** Does the component consume a context whose value changed?
   - Look for `useContext` calls. Check if the provider value is referentially stable.

4. **Check if `React.memo` is being defeated.**
   - Are you passing new object/function references as props?
   - Log `prevProps === nextProps` in a custom comparison function.

5. **Use React DevTools Profiler.**
   - Click on a component in the flamegraph.
   - The "Why did this render?" section tells you: props changed, state changed, hooks changed, or parent rendered.

6. **Add temporary logging:**
   ```tsx
   // Temporary: log render count
   const renderCount = useRef(0);
   renderCount.current++;
   console.log(`MyComponent render #${renderCount.current}`);
   ```

7. **Check for state updates in effects.**
   - `useEffect` that calls `setState` causes a second render every commit.
   - `useLayoutEffect` that calls `setState` causes a synchronous re-render (before paint).

---

## Prevention Strategies

| Problem | Solution | Notes |
|---|---|---|
| Parent re-render causes child re-render | `React.memo(Child)` | Only helps if props are referentially stable |
| New function prop every render | `useCallback(fn, deps)` | Combine with `React.memo` on the child |
| New object prop every render | `useMemo(() => obj, deps)` | Combine with `React.memo` on the child |
| Context change re-renders all consumers | `useMemo` on provider value | `useMemo(() => ({ a, b }), [a, b])` |
| Context change re-renders consumers that don't use changed part | Split context into pieces | One context per independent value/group |
| Expensive computation every render | `useMemo(() => compute(x), [x])` | Only for genuinely expensive work |
| State in parent affects unrelated children | Move state down | Extract stateful part into its own component |
| State in parent affects unrelated children (alternative) | Children pattern | Pass expensive subtree as `children` prop |
| Component state needs full reset on prop change | `key={prop}` on the component | Forces unmount/mount — use deliberately |
| List re-renders on every item change | Stable `key` prop + `React.memo` on list items | Never use `key={index}` for reorderable lists |

---

## Reconciliation Rules (Quick Reference)

```
Same element type?
├── YES → Keep DOM node / component instance. Update props. Re-render children.
└── NO  → Destroy old subtree entirely. Mount new subtree. All state lost.

List children?
├── Has stable keys → Match by key. Reuse, move, create, delete as needed.
└── No keys (or index keys) → Match by position. Reorder = update content (bad).
```

---

## Priority Lanes Summary

```
Highest priority                                           Lowest priority
    ◄──────────────────────────────────────────────────────────────►
    SyncLane    InputContinuous    Default    Transition(1-16)    Idle
    (click,      (mousemove,      (setState    (startTransition)  (offscreen
     keypress)    scroll)          in events)                      prerender)
```

- Lanes are **bitwise flags** in a 31-bit integer.
- Higher priority lanes interrupt lower priority work in concurrent mode.
- Multiple transition lanes allow independent transitions to be distinguished.
- `childLanes` bubble up — if a subtree has no pending lanes, React skips it entirely.

---

> **Full coverage:** See [README.md](./README.md) for detailed explanations and interview Q&A.
> **Deep dive:** See [deep-dive.md](./deep-dive.md) for fiber internals, work loop, and Suspense mechanics.
