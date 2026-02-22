# Lesson 08: React Internals — Deep Dive

> Optional deeper content for senior engineers who want to understand React's implementation.
> This material goes beyond typical interview scope but is valuable for debugging complex issues,
> contributing to React, or answering "go deeper" follow-ups in staff-level interviews.

---

## Table of Contents

1. [Fiber Node Structure](#fiber-node-structure)
2. [The Work Loop](#the-work-loop)
3. [Lane Model Deep Dive](#lane-model-deep-dive)
4. [Suspense Internals](#suspense-internals)
5. [Error Boundary Internals](#error-boundary-internals)
6. [Effect Tracking and Flags](#effect-tracking-and-flags)
7. [Key-Based Reconciliation Edge Cases](#key-based-reconciliation-edge-cases)
8. [createElement vs JSX Transform](#createelement-vs-jsx-transform)
9. [React Without JSX](#react-without-jsx)

---

## Fiber Node Structure

A fiber is a plain JavaScript object. Here are its actual properties with explanations:

```ts
interface Fiber {
  // === Identity ===
  tag: WorkTag;
  // Numeric enum identifying the fiber kind:
  //   0 = FunctionComponent
  //   1 = ClassComponent
  //   3 = HostRoot (the root of the fiber tree)
  //   5 = HostComponent (DOM elements like 'div', 'span')
  //   6 = HostText (raw text nodes)
  //   7 = Fragment
  //  11 = ForwardRef
  //  12 = MemoComponent
  //  13 = SuspenseComponent
  //  15 = SimpleMemoComponent
  //  etc.

  key: string | null;
  // The `key` prop, used in reconciliation.

  type: any;
  // For function components: the function itself.
  // For class components: the class constructor.
  // For host elements: the tag name string ('div', 'span').
  // For fragments: null.

  elementType: any;
  // Usually the same as `type`. Differs for lazy components and memo wrappers
  // where `elementType` is the outer wrapper and `type` is the resolved inner.

  // === Instance ===
  stateNode: any;
  // For HostRoot: the FiberRootNode.
  // For HostComponent: the actual DOM node.
  // For ClassComponent: the class instance.
  // For FunctionComponent: null (no instance).

  // === Tree Structure ===
  return: Fiber | null;    // parent fiber
  child: Fiber | null;     // first child fiber
  sibling: Fiber | null;   // next sibling fiber
  index: number;           // position among siblings

  // === Props ===
  pendingProps: any;        // props for the current render (input)
  memoizedProps: any;       // props from the last completed render (for comparison)

  // === State ===
  memoizedState: any;
  // For class components: the state object.
  // For function components: the HEAD of the hooks linked list.
  //   Each hook is a node: { memoizedState, queue, next }
  //   useState: memoizedState = the state value
  //   useEffect: memoizedState = { create, destroy, deps, next }
  //   useRef: memoizedState = { current: value }
  //   useMemo: memoizedState = [computedValue, deps]

  updateQueue: any;
  // Queue of pending state updates. For function components,
  // this contains the dispatch queue for hooks.

  // === Effects ===
  flags: Flags;
  // Bitwise flags indicating side effects:
  //   Placement       = 0b0000000000000000000000010  (needs DOM insertion)
  //   Update          = 0b0000000000000000000000100  (needs DOM update)
  //   Deletion        = 0b0000000000000000000001000  (needs DOM removal)
  //   ChildDeletion   = 0b0000000000000000000010000
  //   Callback        = 0b0000000000000000000100000
  //   Ref             = 0b0000000000000000001000000
  //   Snapshot        = 0b0000000000000000010000000
  //   Passive         = 0b0000000000000100000000000  (has useEffect)
  //   LayoutMask      = Update | Callback | Ref | Visibility
  //   PassiveMask     = Passive | ChildDeletion

  subtreeFlags: Flags;
  // Bubbled-up flags from descendants. If a fiber has no subtreeFlags,
  // React can skip traversing its entire subtree during commit.
  // This is a major optimization over the old "effect list" approach.

  // === Scheduling ===
  lanes: Lanes;
  // Which priority lanes this fiber has pending work on.
  // A bitwise OR of all pending update lanes.

  childLanes: Lanes;
  // Bubbled-up lanes from descendants. If childLanes is empty,
  // React can skip traversing the subtree entirely.

  // === Double Buffering ===
  alternate: Fiber | null;
  // Points to the corresponding fiber in the other tree.
  // current.alternate === workInProgress
  // workInProgress.alternate === current

  // === Refs ===
  ref: Ref | null;
}
```

### The Hooks Linked List

For function components, `memoizedState` is a linked list of hook nodes. The order of hooks in the list corresponds to the order of hook calls in the component:

```
fiber.memoizedState
  ┌──────────────────────────────┐
  │ Hook 0 (useState)            │
  │ memoizedState: 42            │
  │ queue: { pending, dispatch } │
  │ next: ─────────────────────────┐
  └──────────────────────────────┘ │
  ┌──────────────────────────────┐◄┘
  │ Hook 1 (useEffect)           │
  │ memoizedState: { create,     │
  │   destroy, deps, next }      │
  │ next: ─────────────────────────┐
  └──────────────────────────────┘ │
  ┌──────────────────────────────┐◄┘
  │ Hook 2 (useMemo)             │
  │ memoizedState: [value, deps] │
  │ next: null                   │
  └──────────────────────────────┘
```

**This is why hooks cannot be called conditionally.** On re-renders, React walks this linked list in order, matching each hook call to its node by position. If you skip a hook call, every subsequent hook reads the wrong node.

---

## The Work Loop

The work loop is the heart of the reconciler. It is what drives the traversal of the fiber tree.

### `workLoopSync` vs `workLoopConcurrent`

```ts
// Synchronous — used for SyncLane updates and legacy mode
function workLoopSync() {
  while (workInProgress !== null) {
    performUnitOfWork(workInProgress);
  }
}

// Concurrent — used for transitions, deferred updates
function workLoopConcurrent() {
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}
```

The only difference: `workLoopConcurrent` checks `shouldYield()` after each unit of work. `shouldYield()` returns `true` when the current time slice (5ms) has elapsed, allowing the browser to handle events and paint.

### `performUnitOfWork`

Processes one fiber and advances the `workInProgress` pointer:

```ts
function performUnitOfWork(unitOfWork: Fiber): void {
  const current = unitOfWork.alternate;

  // Phase 1: Begin work — process this fiber, return its first child
  const next = beginWork(current, unitOfWork, renderLanes);

  unitOfWork.memoizedProps = unitOfWork.pendingProps;

  if (next !== null) {
    // This fiber has a child — descend into it
    workInProgress = next;
  } else {
    // No children — complete this fiber and move to sibling or parent
    completeUnitOfWork(unitOfWork);
  }
}
```

### `beginWork`

Called when entering a fiber (going "down" the tree). Its job:

1. Check if the fiber can bail out (props unchanged, no pending lanes).
2. If it cannot bail out, process the fiber based on its tag:
   - `FunctionComponent` → call the function, process hooks
   - `ClassComponent` → call `render()`, check `shouldComponentUpdate`
   - `HostComponent` → diff props
   - `SuspenseComponent` → check if the promise resolved
3. Reconcile children (diff old children vs new children).
4. Return the first child fiber (or `null` if no children).

The bailout check is critical for performance:

```ts
// Simplified bailout logic in beginWork
if (current !== null) {
  const oldProps = current.memoizedProps;
  const newProps = workInProgress.pendingProps;

  if (oldProps === newProps && !hasLegacyContextChanged() && !includesSomeLane(renderLanes, updateLanes)) {
    // Nothing changed — bail out of this subtree
    return bailoutOnAlreadyFinishedWork(current, workInProgress, renderLanes);
  }
}
```

Note: the props check is **referential equality** (`===`), not shallow comparison. This is why a parent re-render causes children to re-render — new JSX elements create new props objects.

### `completeWork`

Called when leaving a fiber (going "up" the tree). Its job:

1. For `HostComponent`: create the DOM node (if new) or diff the props (if updating). Build the list of prop changes.
2. Bubble up `subtreeFlags` — merge child flags into the parent so the commit phase knows which subtrees have effects.
3. Bubble up `childLanes` — merge child lanes into the parent so future renders can skip unchanged subtrees.

```ts
// Simplified completeWork for HostComponent
case HostComponent: {
  if (current !== null && workInProgress.stateNode !== null) {
    // Update — diff old props vs new props
    const updatePayload = diffProperties(oldProps, newProps);
    if (updatePayload !== null) {
      workInProgress.flags |= Update;
      workInProgress.updateQueue = updatePayload;
    }
  } else {
    // Mount — create the DOM node
    const instance = createInstance(type, newProps);
    appendAllChildren(instance, workInProgress);
    workInProgress.stateNode = instance;
  }

  // Bubble flags
  bubbleProperties(workInProgress);
}
```

### Traversal Order

The work loop traverses the tree in a depth-first manner:

```
        App
       / | \
      A   B   C
     / \     |
    D   E    F

beginWork order:  App → A → D → (complete D) → E → (complete E) → (complete A) → B → (complete B) → C → F → (complete F) → (complete C) → (complete App)

In other words:
  - Go down (child) as far as possible
  - When no more children, complete current fiber
  - Move to sibling
  - When no more siblings, complete parent
  - Repeat
```

---

## Lane Model Deep Dive

### Lanes as Bitwise Flags

React's priority system uses a 31-bit integer where each bit represents a "lane":

```ts
export const NoLanes: Lanes       = /*                          */ 0b0000000000000000000000000000000;
export const NoLane: Lane         = /*                          */ 0b0000000000000000000000000000000;
export const SyncLane: Lane       = /*                          */ 0b0000000000000000000000000000010;
export const SyncBatchedLane: Lane = /*                         */ 0b0000000000000000000000000000100;
export const InputContinuousLane: Lane = /*                     */ 0b0000000000000000000000000001000;
export const DefaultLane: Lane    = /*                          */ 0b0000000000000000000000000100000;
export const TransitionLane1: Lane = /*                         */ 0b0000000000000000000001000000000;
export const TransitionLane2: Lane = /*                         */ 0b0000000000000000000010000000000;
// ... TransitionLane3 through TransitionLane16
export const IdleLane: Lane       = /*                          */ 0b0100000000000000000000000000000;
export const OffscreenLane: Lane  = /*                          */ 0b1000000000000000000000000000000;
```

### Lane Operations

Because lanes are bitfields, operations are simple bitwise math:

```ts
// Merge lanes (union)
const merged = laneA | laneB;

// Check if a lane is included in a set
const isIncluded = (lanes & lane) !== NoLane;

// Remove a lane from a set
const removed = lanes & ~lane;

// Get the highest priority lane (rightmost set bit)
const highest = lanes & -lanes;
```

### Lane Entanglement

Some updates must be processed together even if they have different priorities. React tracks this with `entanglements`:

```ts
root.entanglements[index] |= otherLanes;
```

Example: if a `SyncLane` update reads state that was modified by a `TransitionLane` update, those lanes become entangled. React must process them in the same render to avoid inconsistency.

### How Transitions Get Their Lanes

When you call `startTransition`, React:

1. Sets a module-level flag: `ReactCurrentBatchConfig.transition = 1`.
2. Calls your callback, which calls `setState`.
3. Inside `setState`, React checks the transition flag and assigns a `TransitionLane` instead of the default lane.
4. React clears the transition flag.
5. The update is enqueued with the transition lane and scheduled at lower priority.

Multiple transitions may share the same `TransitionLane` if they happen close together. React has 16 transition lanes and cycles through them, which allows different transitions to be distinguished and processed independently when needed.

---

## Suspense Internals

### How Promise Throwing Works

When a component "suspends," it literally **throws a Promise**:

```tsx
// Simplified data fetching (this is what libraries like Relay do internally)
function use<T>(promise: Promise<T>): T {
  if (promise.status === 'fulfilled') {
    return promise.value;
  } else if (promise.status === 'rejected') {
    throw promise.reason;
  } else {
    // Suspend — throw the promise
    throw promise;
  }
}

function UserProfile({ userId }: { userId: string }) {
  const user = use(fetchUser(userId)); // throws if pending
  return <div>{user.name}</div>;
}
```

React catches this thrown Promise in the work loop:

```ts
// Simplified — inside the work loop's try/catch
try {
  beginWork(current, workInProgress, renderLanes);
} catch (thrownValue) {
  if (typeof thrownValue === 'object' && typeof thrownValue.then === 'function') {
    // It's a thenable — this is a Suspense suspension
    const wakeable = thrownValue;
    handleSuspense(workInProgress, wakeable);
  } else {
    // It's a real error — propagate to error boundary
    handleError(workInProgress, thrownValue);
  }
}
```

### Fallback Rendering

When a suspension is caught:

1. React walks up the fiber tree to find the nearest `Suspense` boundary.
2. It marks the `Suspense` fiber with a `DidCapture` flag.
3. React re-renders the `Suspense` component, but this time it renders the `fallback` prop instead of `children`.
4. React attaches a `.then()` callback to the thrown Promise.
5. When the Promise resolves, React schedules a "retry" update on the `Suspense` fiber.

### Retry Mechanics

When the Promise resolves:

1. The `.then()` callback fires: `ping(root, wakeable, pingedLanes)`.
2. React schedules a re-render at the pinged lanes.
3. During re-render, the component calls `use()` again. This time the Promise is fulfilled, so it returns the value instead of throwing.
4. React renders the actual children and removes the fallback.

### Nested Suspense Boundaries

Suspense boundaries nest. When a component suspends, React walks up to the **nearest** boundary:

```tsx
<Suspense fallback={<PageSkeleton />}>
  <Header />
  <Suspense fallback={<SidebarSkeleton />}>
    <Sidebar />  {/* if this suspends, only SidebarSkeleton shows */}
  </Suspense>
  <Content />    {/* if this suspends, PageSkeleton shows */}
</Suspense>
```

If a component suspends and there is no `Suspense` boundary in its ancestor tree, the root catches it and the entire app shows nothing (or an error in development). Always provide Suspense boundaries.

---

## Error Boundary Internals

### `getDerivedStateFromError` Lifecycle

Error boundaries are class components that implement `getDerivedStateFromError` and/or `componentDidCatch`. There is no hook equivalent (as of React 19 — the `use` hook handles Suspense errors differently).

```tsx
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError(_error: Error) {
    // Called during the render phase — must be pure
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Called during the commit phase — side effects allowed
    logErrorToService(error, errorInfo.componentStack);
  }

  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}
```

### Error Propagation Through the Fiber Tree

When an error is thrown during rendering:

1. React catches the error in the work loop.
2. React walks **up** the fiber tree (via `return` pointers) looking for a fiber that can handle the error.
3. A fiber can handle an error if it is a class component with `getDerivedStateFromError` or `componentDidCatch`.
4. If found, React calls `getDerivedStateFromError` to compute new state. This happens during the render phase.
5. React re-renders the error boundary with the new state, causing it to show its fallback UI.
6. During the commit phase, `componentDidCatch` is called for logging.

**What error boundaries do NOT catch:**

- Errors in event handlers (use try/catch directly).
- Errors in asynchronous code (`setTimeout`, `fetch` callbacks) — unless they happen during rendering via `use()`.
- Errors in the error boundary itself.
- Errors during server-side rendering (different mechanism).

---

## Effect Tracking and Flags

### The Old Effect List (Pre-React 18)

In React 17 and earlier, React maintained a **linked list of fibers with effects** called the "effect list." During the completeWork phase, any fiber with effect flags was appended to its parent's effect list. The commit phase only needed to traverse this list rather than the entire tree.

### The New Subtree Flags Approach (React 18+)

React 18 replaced the effect list with **subtreeFlags**. Each fiber bubbles its children's flags upward:

```ts
function bubbleProperties(completedWork: Fiber): void {
  let subtreeFlags = NoFlags;
  let child = completedWork.child;

  while (child !== null) {
    subtreeFlags |= child.subtreeFlags;
    subtreeFlags |= child.flags;
    child = child.sibling;
  }

  completedWork.subtreeFlags |= subtreeFlags;
}
```

During the commit phase, React traverses the tree but **skips entire subtrees** when `subtreeFlags === NoFlags`. This is more memory-efficient (no linked list allocation) and works better with concurrent features.

### Flag Combinations

Fibers can have multiple flags set simultaneously:

```ts
// A fiber that needs a DOM update AND has a useEffect
fiber.flags = Update | Passive;

// A fiber being moved in a list AND having a ref update
fiber.flags = Placement | Ref;
```

The commit phase processes flags in a specific order to ensure correctness:

1. Process `Snapshot` flags (getSnapshotBeforeUpdate)
2. Process `Deletion` flags (unmount children, cleanup effects)
3. Process `Placement` and `Update` flags (DOM mutations)
4. Process `LayoutMask` flags (useLayoutEffect)
5. Schedule `PassiveMask` flags (useEffect — runs asynchronously after paint)

---

## Key-Based Reconciliation Edge Cases

### Moving Elements in a List

React's list reconciliation algorithm processes children in two passes:

**Pass 1: Match from the start.** Walk old and new children simultaneously. If keys match, reuse the fiber. If keys diverge, break to pass 2.

**Pass 2: Handle remaining.** Build a map of remaining old children keyed by their key (or index if no key). For each remaining new child, look up the map. If found, reuse and mark as a "move." If not found, create new.

```tsx
// Old: [A, B, C, D, E]
// New: [A, C, E, B, D]

// Pass 1: A matches — reuse
// Pass 1: B vs C — keys diverge, break

// Pass 2: remaining old = {B, C, D, E}
// C: found in map → reuse, mark as moved
// E: found in map → reuse, mark as moved
// B: found in map → reuse, mark as moved
// D: found in map → reuse, mark as moved

// Result: A stays, C/E/B/D are moved
// DOM operations: 3 moves (insertBefore calls) — not 4, because React
// identifies that some elements are already in the right relative position
```

### Reordering Optimization Limitations

React does NOT compute the minimum number of moves (that requires solving the Longest Increasing Subsequence problem, which Vue 3 does). React uses a simpler heuristic:

1. Track the `lastPlacedIndex` — the highest index in the old list of a reused fiber.
2. For each reused fiber, if its old index is less than `lastPlacedIndex`, mark it as needing a move (Placement flag).

This means certain reorderings produce more DOM moves than theoretically necessary:

```tsx
// Old: [A(0), B(1), C(2), D(3)]
// New: [D, A, B, C]

// D: oldIndex=3, lastPlacedIndex=0 → 3 >= 0, no move. lastPlacedIndex=3
// A: oldIndex=0, lastPlacedIndex=3 → 0 < 3, MOVE
// B: oldIndex=1, lastPlacedIndex=3 → 1 < 3, MOVE
// C: oldIndex=2, lastPlacedIndex=3 → 2 < 3, MOVE

// Result: 3 moves. Optimal would be 1 move (just move D to front).
```

This is a deliberate tradeoff — the simpler algorithm is faster in the common case and only suboptimal for certain reordering patterns.

### The Key-Reset Pattern

Using `key` to deliberately destroy and recreate a subtree:

```tsx
// Reset form state when switching between editing different items
<ItemForm key={selectedItemId} item={selectedItem} />
```

When `selectedItemId` changes, React sees a new key, unmounts the old `ItemForm` (destroying all state including useState, useRef, etc.), and mounts a fresh one. This is cleaner than using `useEffect` to reset state when props change.

**Caution:** Overusing key-based resets can be expensive. Each reset destroys and recreates the entire subtree, including all DOM nodes. Use it deliberately, not as a default pattern.

---

## createElement vs JSX Transform

### The Classic Transform (Pre-React 17)

Before React 17, JSX compiled to `React.createElement`:

```tsx
// JSX
<div className="card">
  <h1>{title}</h1>
</div>

// Compiled to:
React.createElement('div', { className: 'card' },
  React.createElement('h1', null, title)
);
```

This required `import React from 'react'` in every file that used JSX, even if React was not explicitly referenced.

### The New JSX Transform (React 17+)

The new transform compiles to `_jsx` and `_jsxs` (for static children) from `react/jsx-runtime`:

```tsx
// JSX
<div className="card">
  <h1>{title}</h1>
</div>

// Compiled to:
import { jsx as _jsx, jsxs as _jsxs } from 'react/jsx-runtime';

_jsxs('div', {
  className: 'card',
  children: [
    _jsx('h1', { children: title }),
  ],
});
```

Key differences:

| | `React.createElement` | `_jsx` / `_jsxs` |
|---|---|---|
| Import | Manual `import React` | Auto-inserted by compiler |
| `key` | Part of props | Separate argument: `_jsx(type, props, key)` |
| `ref` | Part of props | Handled separately (React 19 makes `ref` a regular prop) |
| Children | Rest arguments | `children` prop (always a prop, not rest args) |
| `_jsxs` | N/A | Used for static children arrays (compiler optimization) |
| `defaultProps` | Applied inside `createElement` | NOT applied (deprecated path) |

### What `_jsx()` Returns

Both `createElement` and `_jsx` return a React element — a plain object:

```ts
{
  $$typeof: Symbol.for('react.element'),  // security: prevents injection
  type: 'div',                            // or a component function/class
  key: null,                              // or a string key
  ref: null,                              // or a ref object/callback
  props: { className: 'card', children: [...] },
  _owner: currentlyRenderingFiber,        // internal: tracks which component created this
}
```

The `$$typeof` field is a Symbol, which means it **cannot be created from JSON**. This is a security feature: if an attacker injects JSON into a server response, it cannot be mistaken for a React element because JSON cannot contain Symbols.

---

## React Without JSX

Understanding what the component tree looks like in memory clarifies how React actually works.

### A Complete Example

```tsx
// This component:
function App() {
  const [count, setCount] = useState(0);
  return (
    <main>
      <h1>Counter</h1>
      <button onClick={() => setCount(c => c + 1)}>
        Count: {count}
      </button>
      {count > 5 && <Warning message="Too high!" />}
    </main>
  );
}

function Warning({ message }: { message: string }) {
  return <p className="warning">{message}</p>;
}
```

After JSX compilation and evaluation, the return value of `App()` (when `count` is 7) is:

```ts
// The element tree (what App() returns):
{
  $$typeof: Symbol.for('react.element'),
  type: 'main',
  props: {
    children: [
      {
        $$typeof: Symbol.for('react.element'),
        type: 'h1',
        props: { children: 'Counter' },
      },
      {
        $$typeof: Symbol.for('react.element'),
        type: 'button',
        props: {
          onClick: [Function],
          children: ['Count: ', 7],
        },
      },
      {
        $$typeof: Symbol.for('react.element'),
        type: Warning,  // <-- the function reference itself
        props: { message: 'Too high!' },
      },
    ],
  },
}
```

### The Fiber Tree (in memory)

The corresponding fiber tree looks like:

```
FiberRootNode
  └── HostRoot (tag: 3)
        └── App (tag: 0, type: App)
              │  memoizedState: Hook { memoizedState: 7, queue: {...}, next: null }
              └── main (tag: 5, type: 'main')
                    ├── h1 (tag: 5, type: 'h1')
                    │     └── "Counter" (tag: 6, text node)
                    ├── button (tag: 5, type: 'button')
                    │     ├── "Count: " (tag: 6, text node)
                    │     └── "7" (tag: 6, text node)
                    └── Warning (tag: 0, type: Warning)
                          └── p (tag: 5, type: 'p')
                                └── "Too high!" (tag: 6, text node)
```

Note that:

- The fiber tree is **more granular** than the element tree — it includes text nodes as separate fibers.
- Function component fibers have `stateNode: null` — they have no instance.
- The `main`, `h1`, `button`, and `p` fibers have `stateNode` pointing to their actual DOM nodes.
- The `App` fiber's `memoizedState` is the hooks linked list — in this case, a single `useState` hook.

### Element Tree vs Fiber Tree

These are different structures with different lifetimes:

| | Element Tree | Fiber Tree |
|---|---|---|
| **Created** | Every render | Once (reused across renders) |
| **Mutated** | Never (immutable) | Yes (updated in place) |
| **Structure** | `children` arrays | `child`/`sibling`/`return` pointers |
| **Contains** | Description of UI | Instance data, state, effect info |
| **Lifetime** | Garbage collected after diffing | Persists for the component's lifetime |

The element tree is the **input** to reconciliation. The fiber tree is the **persistent working structure** that React mutates to track what is on screen.

---

> **Back to core material:** See [README.md](./README.md) for interview-focused coverage.
> **Quick reference:** See [cheat-sheet.md](./cheat-sheet.md) for tables, diagrams, and debugging checklists.
