# Lesson 08: React Internals

> Core interview knowledge for senior-level React interviews.
> Assumes production experience with React 18+ and familiarity with hooks, reconciliation at a surface level, and concurrent features.

---

## Table of Contents

1. [Virtual DOM](#virtual-dom)
2. [Fiber Architecture](#fiber-architecture)
3. [Rendering Phases](#rendering-phases)
4. [Reconciliation](#reconciliation)
5. [Why React Re-Renders](#why-react-re-renders)
6. [Preventing Unnecessary Re-Renders](#preventing-unnecessary-re-renders)
7. [Concurrent Features](#concurrent-features)
8. [The Scheduler and Priority Lanes](#the-scheduler-and-priority-lanes)
9. [Hydration](#hydration)
10. [Batching](#batching)
11. [Common Interview Questions](#common-interview-questions)

---

## Virtual DOM

### What It Actually Is

The "Virtual DOM" is not a shadow copy of the browser DOM. It is a **plain JavaScript object tree** that describes what the UI should look like. Each node in this tree is a React element — the return value of `React.createElement()` (or the JSX transform `_jsx()`).

```tsx
// JSX
<div className="card">
  <h1>Title</h1>
  <p>Body</p>
</div>

// Compiles to (simplified):
{
  type: 'div',
  props: {
    className: 'card',
    children: [
      { type: 'h1', props: { children: 'Title' } },
      { type: 'p',  props: { children: 'Body' } },
    ],
  },
}
```

Key properties of React elements:

- **Immutable** — once created, you cannot change their children or attributes.
- **Cheap to create** — they are plain objects, not DOM nodes.
- **Descriptive** — they describe what should be on screen, not how to get there.

### Why It Exists

Direct DOM manipulation is expensive because:

1. DOM nodes are heavyweight objects with hundreds of properties.
2. Touching the DOM triggers style recalculation, layout, paint, and compositing.
3. Batching and minimizing DOM writes is hard to do by hand at scale.

React's model: re-run the component function, produce a new element tree, diff it against the previous tree, compute the minimal set of DOM mutations, and apply them in a single batch. The "virtual DOM" is the intermediate representation that makes this diffing possible.

### Diffing Overview

React does not do a generic tree diff (which is O(n^3)). Instead it relies on two heuristics that reduce it to **O(n)**:

1. **Elements of different types produce different trees.** If a `<div>` becomes a `<span>`, React tears down the entire subtree and rebuilds it.
2. **The `key` prop hints at which children are stable across renders.** Without keys, React matches children by index.

This is the reconciliation algorithm, covered in detail below.

---

## Fiber Architecture

React 16 replaced the old recursive "stack reconciler" with **Fiber** — a complete rewrite of the reconciliation engine. The old reconciler was synchronous and recursive: once it started rendering a tree, it could not stop until it finished. This blocked the main thread and caused jank.

### Fiber Nodes as a Unit of Work

A fiber is a JavaScript object that corresponds to a component instance (or a host DOM element). Each fiber represents **one unit of work**. The reconciler processes fibers one at a time, and after each unit it can check whether the browser needs the main thread back.

```
Fiber = {
  tag,            // what kind of fiber (FunctionComponent, HostComponent, etc.)
  type,           // the component function/class, or the DOM tag string
  stateNode,      // the actual DOM node (for host fibers) or class instance
  memoizedState,  // the hook linked list (for function components)
  memoizedProps,  // props from the last completed render
  pendingProps,   // props for the current in-progress render
  lanes,          // priority bitmask
  flags,          // side-effect flags (Placement, Update, Deletion, etc.)
  // tree pointers:
  child,          // first child fiber
  sibling,        // next sibling fiber
  return,         // parent fiber
  alternate,      // pointer to the other tree's version of this fiber
}
```

### The Fiber Tree: Current and WorkInProgress

React maintains **two fiber trees** at all times:

- **`current`** — the tree that is currently committed to the DOM. What the user sees.
- **`workInProgress`** — the tree being built during the render phase.

Each fiber in `current` has an `alternate` pointer to its counterpart in `workInProgress`, and vice versa. When the render phase completes, React swaps the trees: `workInProgress` becomes `current`. This is called **double buffering** — the same technique used in graphics programming.

```
           current tree                     workInProgress tree
        ┌──────────────┐                  ┌──────────────┐
        │   FiberRoot   │  ───alternate──▶ │   FiberRoot   │
        └──────┬───────┘                  └──────┬───────┘
               │ child                           │ child
        ┌──────▼───────┐                  ┌──────▼───────┐
        │   App Fiber   │  ◀──alternate──  │   App Fiber   │
        └──────┬───────┘                  └──────┬───────┘
               │ child                           │ child
        ┌──────▼───────┐                  ┌──────▼───────┐
        │  Child Fiber  │  ◀──alternate──  │  Child Fiber  │
        └──────────────┘                  └──────────────┘
```

### Child / Sibling / Return Pointers

Fibers form a **singly-linked list tree**, not a traditional tree with a `children` array:

- `fiber.child` — points to the **first** child.
- `fiber.sibling` — points to the **next** sibling.
- `fiber.return` — points to the **parent** (called "return" because that is where work returns to after processing this fiber).

```
        Parent
          │ child
          ▼
        Child1  ──sibling──▶  Child2  ──sibling──▶  Child3
          │                      │
          │ child                │ child
          ▼                      ▼
       Grandchild            Grandchild
```

This structure allows the reconciler to traverse the tree with a simple while loop instead of recursion, which is essential for making rendering interruptible.

---

## Rendering Phases

React's work happens in two distinct phases with very different properties.

### Render Phase

| Property | Detail |
|---|---|
| **Purity** | Must be pure — no side effects |
| **Interruptible** | Yes (in concurrent mode) |
| **What happens** | Traverses the fiber tree, calls component functions, diffs elements, builds the workInProgress tree |
| **Result** | A list of fibers tagged with effect flags (Placement, Update, Deletion) |

During the render phase, React:

1. Calls your component function (or `render()` for class components).
2. Compares the returned elements with the previous fiber children.
3. Creates new fibers, reuses existing fibers, or marks fibers for deletion.
4. Records what DOM changes are needed as **flags** on the fiber.

**Critical interview point:** The render phase does NOT touch the DOM. Your component function can be called multiple times, thrown away, and restarted. This is why side effects in the render body are bugs — they may execute an unpredictable number of times.

```tsx
function MyComponent() {
  // This is the render phase. This code must be pure.
  // No fetch(), no subscriptions, no DOM manipulation.
  const [count, setCount] = useState(0);

  // WRONG: side effect in render phase
  document.title = `Count: ${count}`; // DO NOT DO THIS

  // CORRECT: side effect in an effect
  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]);

  return <div>{count}</div>;
}
```

### Commit Phase

| Property | Detail |
|---|---|
| **Purity** | Side effects happen here |
| **Interruptible** | No — always synchronous |
| **What happens** | Applies DOM mutations, runs lifecycle methods, runs effects |
| **Sub-phases** | Before mutation, Mutation, Layout, Passive effects |

The commit phase has multiple sub-phases:

1. **Before Mutation** — `getSnapshotBeforeUpdate` (class components). Read DOM before it changes.
2. **Mutation** — React walks the effect list and applies DOM insertions, updates, and deletions.
3. **Layout** — `useLayoutEffect` callbacks and `componentDidMount`/`componentDidUpdate` run synchronously. The DOM has been mutated but the browser has NOT painted yet. You can read layout and synchronously re-render.
4. **Passive Effects** — `useEffect` callbacks run asynchronously after the browser has painted.

```
 Render Phase          Commit Phase
 ─────────────    ──────────────────────────────────────────────
 Build fiber       Before     Mutation     Layout      Passive
 tree, diff,       Mutation   (DOM         (useLayout  (useEffect
 compute effects   (snapshot) writes)      Effect)     after paint)
```

---

## Reconciliation

Reconciliation is the algorithm React uses to diff two trees of elements and determine the minimal set of changes to apply to the DOM.

### Same Type: Update

When the new element has the **same type** as the old one, React keeps the same underlying DOM node (or component instance) and updates its props.

```tsx
// Old:
<div className="old" />
// New:
<div className="new" />
// React: keep the same <div> DOM node, call setAttribute('class', 'new')
```

For component elements of the same type, React keeps the same fiber, updates props, and re-renders the component. State is preserved.

```tsx
// Old:
<Counter initialCount={1} />
// New:
<Counter initialCount={5} />
// React: same fiber, same state, re-render with new props
```

### Different Type: Unmount and Mount

When the type changes, React destroys the old subtree entirely and builds a new one from scratch. All state in the old subtree is lost.

```tsx
// Old:
<div><Counter /></div>
// New:
<section><Counter /></section>
// React: unmount entire <div> subtree (Counter state lost),
//        mount new <section> subtree (Counter starts fresh)
```

This is why changing a component's wrapper element type can unexpectedly reset state — a common source of bugs.

### Key Matching in Lists

Without keys, React matches children by position index:

```tsx
// Old:
<ul>
  <li>Alice</li>
  <li>Bob</li>
</ul>

// New (prepend):
<ul>
  <li>Charlie</li>
  <li>Alice</li>
  <li>Bob</li>
</ul>

// Without keys: React thinks index 0 changed from Alice→Charlie,
// index 1 changed from Bob→Alice, and index 2 is new (Bob).
// It updates TWO nodes and inserts ONE — wasteful.
```

With keys, React matches by identity:

```tsx
<ul>
  <li key="charlie">Charlie</li>
  <li key="alice">Alice</li>
  <li key="bob">Bob</li>
</ul>

// With keys: React knows Alice and Bob are the same, just prepends Charlie.
// It inserts ONE node — optimal.
```

**Why `key={index}` is harmful for reorderable lists:** If items swap positions, their index keys swap too, causing React to update the content of existing DOM nodes instead of moving them. This breaks component state and causes visual bugs in uncontrolled inputs, animations, etc.

**When `key={index}` is acceptable:** Static lists that never reorder, filter, or have items added/removed in the middle.

### Forcing a Full Remount with Keys

You can use the `key` prop on any component to force React to unmount and remount it:

```tsx
// Changing the key resets ALL state inside UserProfile
<UserProfile key={userId} userId={userId} />
```

This is a deliberate tool, not a hack. It is the idiomatic way to reset a component's state when a prop changes.

---

## Why React Re-Renders

A re-render means React calls your component function again and diffs the output. This is the **complete list** of triggers:

### 1. `setState` / `useState` setter

The most common trigger. Calling a state setter schedules a re-render of that component.

```tsx
const [count, setCount] = useState(0);
setCount(1); // schedules a re-render
```

**Bailout optimization:** If the new state is the same as the current state (via `Object.is`), React may bail out early and skip the re-render. However, React may still call the component function once before bailing out — it just won't commit.

### 2. Parent Re-Render

When a parent component re-renders, **all of its children re-render by default**, regardless of whether their props changed. This is the single most misunderstood aspect of React performance.

```tsx
function Parent() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <button onClick={() => setCount(c => c + 1)}>+</button>
      {/* Child re-renders on EVERY parent re-render, even though it has no props */}
      <ExpensiveChild />
    </div>
  );
}
```

Props do NOT need to change for a child to re-render. The default behavior is: parent re-renders, children re-render. You must opt out explicitly.

### 3. Context Value Change

When the value provided to a Context changes (by reference), every component that calls `useContext` for that context re-renders.

```tsx
const ThemeContext = createContext<Theme>('light');

function App() {
  const [theme, setTheme] = useState<Theme>('light');
  // Every consumer of ThemeContext re-renders when theme changes
  return (
    <ThemeContext.Provider value={theme}>
      <DeepTree />
    </ThemeContext.Provider>
  );
}
```

Context re-renders **cannot be prevented by `React.memo`** — if a component consumes a context and the context value changes, it will re-render even if it is memoized and its props have not changed.

### 4. `forceUpdate` (Class Components)

Class components can call `this.forceUpdate()` to skip `shouldComponentUpdate` and force a re-render. There is no hook equivalent.

### 5. Hook State Changes (`useReducer`, `useSyncExternalStore`, etc.)

Any hook that manages state triggers re-renders the same way `useState` does. This includes:

- `useReducer` — when `dispatch` is called
- `useSyncExternalStore` — when the external store's snapshot changes
- Custom hooks — any hook that calls `useState` or `useReducer` internally

### 6. React Developer Tooling

React DevTools "highlight updates" and StrictMode double-invocations cause extra renders in development. These do not happen in production.

---

## Preventing Unnecessary Re-Renders

### `React.memo`

Wraps a component to skip re-rendering when props have not changed (shallow comparison by default).

```tsx
const ExpensiveChild = React.memo(function ExpensiveChild({ data }: { data: Item[] }) {
  return <VirtualizedList items={data} />;
});
```

**When it actually helps:**

- The component renders often with the same props.
- The component's render is expensive (large subtree, heavy computation).
- The parent re-renders frequently for unrelated state.

**When it hurts or is useless:**

- Props include new object/function references on every render (memo checks fail every time).
- The component is cheap to render — the overhead of the shallow comparison costs more than re-rendering.

### Stabilizing Props for `React.memo`

`React.memo` uses shallow comparison. If you pass a new object or function reference every render, the memo check will always fail.

```tsx
function Parent() {
  const [count, setCount] = useState(0);

  // BAD: new function reference every render — defeats React.memo
  // const handleClick = () => console.log('click');

  // GOOD: stable reference
  const handleClick = useCallback(() => console.log('click'), []);

  // BAD: new object every render
  // const style = { color: 'red' };

  // GOOD: stable reference
  const style = useMemo(() => ({ color: 'red' }), []);

  return <MemoizedChild onClick={handleClick} style={style} />;
}
```

### `useMemo` for Context Values

A context provider that passes an object literal creates a new reference on every render, causing all consumers to re-render:

```tsx
// BAD: new object on every render
<AuthContext.Provider value={{ user, permissions }}>

// GOOD: stable reference
const contextValue = useMemo(() => ({ user, permissions }), [user, permissions]);
<AuthContext.Provider value={contextValue}>
```

### The Children Pattern (Composition)

Moving state down or lifting content up via `children` is often the best optimization — no memoization needed.

```tsx
// PROBLEM: ScrollPosition changes cause ExpensiveTree to re-render
function App() {
  const [scrollY, setScrollY] = useState(0);
  return (
    <div onScroll={(e) => setScrollY(e.currentTarget.scrollTop)}>
      <ScrollIndicator y={scrollY} />
      <ExpensiveTree /> {/* re-renders on every scroll! */}
    </div>
  );
}

// SOLUTION: Extract the stateful part, pass ExpensiveTree as children
function ScrollTracker({ children }: { children: React.ReactNode }) {
  const [scrollY, setScrollY] = useState(0);
  return (
    <div onScroll={(e) => setScrollY(e.currentTarget.scrollTop)}>
      <ScrollIndicator y={scrollY} />
      {children} {/* children is the SAME reference — no re-render */}
    </div>
  );
}

function App() {
  return (
    <ScrollTracker>
      <ExpensiveTree />
    </ScrollTracker>
  );
}
```

Why this works: `<ExpensiveTree />` is created in `App`'s render. When `ScrollTracker` re-renders due to scroll state, the `children` prop is the same React element reference (because `App` did not re-render). React sees the same element and skips reconciling it.

### Moving State Down

If only a small part of a component needs some state, extract that part into its own component.

```tsx
// BEFORE: entire form re-renders on every keystroke in the search field
function ProductPage() {
  const [query, setQuery] = useState('');
  return (
    <div>
      <input value={query} onChange={(e) => setQuery(e.target.value)} />
      <ExpensiveProductList />
      <ExpensiveRecommendations />
    </div>
  );
}

// AFTER: only SearchInput re-renders on keystrokes
function SearchInput() {
  const [query, setQuery] = useState('');
  return <input value={query} onChange={(e) => setQuery(e.target.value)} />;
}

function ProductPage() {
  return (
    <div>
      <SearchInput />
      <ExpensiveProductList />
      <ExpensiveRecommendations />
    </div>
  );
}
```

---

## Concurrent Features

React 18 introduced concurrent rendering — the ability to prepare multiple versions of the UI simultaneously, interrupting and resuming work as needed.

### `startTransition`

Marks a state update as **non-urgent**. React will start rendering the new state, but will interrupt it if a higher-priority update arrives (like user input).

```tsx
import { startTransition } from 'react';

function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    // Urgent: update input immediately
    setQuery(e.target.value);

    // Non-urgent: rendering thousands of search results can be deferred
    startTransition(() => {
      setResults(search(e.target.value));
    });
  }

  return (
    <div>
      <input value={query} onChange={handleChange} />
      <ResultsList results={results} />
    </div>
  );
}
```

**How it works internally:** The state update inside `startTransition` is assigned a `TransitionLane` instead of `SyncLane`. The scheduler processes `SyncLane` work first, so the input update renders immediately while the results update is deferred.

### `useTransition`

The hook form of `startTransition`, which also provides an `isPending` flag:

```tsx
function TabContainer() {
  const [tab, setTab] = useState<'posts' | 'comments'>('posts');
  const [isPending, startTransition] = useTransition();

  function handleTabChange(nextTab: 'posts' | 'comments') {
    startTransition(() => {
      setTab(nextTab);
    });
  }

  return (
    <div>
      <TabBar activeTab={tab} onTabChange={handleTabChange} />
      {isPending && <Spinner />}
      <TabContent tab={tab} />
    </div>
  );
}
```

### `useDeferredValue`

Defers re-rendering a specific value. React will first render with the old value (keeping the UI responsive), then re-render with the new value in the background.

```tsx
function SearchResults({ query }: { query: string }) {
  // deferredQuery lags behind query during rapid updates
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;

  return (
    <div style={{ opacity: isStale ? 0.7 : 1 }}>
      <SlowList query={deferredQuery} />
    </div>
  );
}
```

**`useDeferredValue` vs `startTransition`:**

| | `startTransition` | `useDeferredValue` |
|---|---|---|
| **Wraps** | The state update (setter call) | The consuming value |
| **Use when** | You control the state update | You receive the value as a prop |
| **Mechanism** | Assigns transition lane to the update | Triggers a secondary render with updated value |

### Suspense and Concurrent Mode

Suspense allows React to **pause rendering** a component subtree and show a fallback while something is loading. In concurrent mode, this becomes especially powerful:

```tsx
function App() {
  return (
    <Suspense fallback={<Skeleton />}>
      <UserProfile />  {/* can suspend */}
    </Suspense>
  );
}
```

When combined with concurrent rendering:

1. A transition triggers a navigation to a new route.
2. React starts rendering the new route's components.
3. A component suspends (throws a Promise).
4. Instead of immediately showing the fallback, React can **keep showing the old UI** while the new one loads (because the update was a transition).
5. When the Promise resolves, React retries and commits the new UI.

This is the foundation of the streaming SSR architecture used by Next.js App Router.

---

## The Scheduler and Priority Lanes

### Priority Lanes

React uses a **lane model** to assign priorities to updates. Lanes are bitwise flags — a single 31-bit integer can represent multiple priority levels.

| Lane | Priority | Example |
|---|---|---|
| `SyncLane` | Highest | Discrete user events (click, keypress) |
| `InputContinuousLane` | High | Continuous events (mousemove, scroll) |
| `DefaultLane` | Normal | `setState` in event handlers, data fetching callbacks |
| `TransitionLane` (1-16) | Low | `startTransition` updates |
| `IdleLane` | Lowest | `requestIdleCallback`-style work |
| `OffscreenLane` | Background | Pre-rendering offscreen content |

### Time Slicing

In concurrent mode, React does not render the entire tree in one synchronous pass. Instead:

1. The scheduler picks the highest-priority pending lane.
2. React begins the work loop, processing one fiber at a time.
3. After each fiber, React checks: **has 5ms elapsed?** (the default time slice).
4. If yes, React **yields** to the browser — the browser can handle events, paint frames, etc.
5. React schedules a continuation via `MessageChannel` (not `setTimeout` — that has a minimum 4ms delay).
6. When the browser is idle again, React resumes where it left off.

```
 ┌──────────────────────────────────────────────────────────┐
 │  5ms work   │ yield │  browser  │  5ms work  │  yield   │
 │  (fibers)   │       │  (paint,  │  (fibers)  │          │
 │             │       │  events)  │            │          │
 └──────────────────────────────────────────────────────────┘
```

### Yielding to the Browser

React uses `MessageChannel` for scheduling rather than `requestAnimationFrame` or `setTimeout`:

- **`requestAnimationFrame`** — fires before paint but at the display refresh rate (usually 16ms). Too coarse for time slicing.
- **`setTimeout(fn, 0)`** — browsers clamp nested timeouts to a minimum of 4ms. Too slow.
- **`MessageChannel`** — fires as a macrotask with no minimum delay. The right tool for yielding and immediately getting back to work.

---

## Hydration

### How Server HTML Becomes Interactive

Hydration is the process of attaching React's event handlers and internal state to server-rendered HTML.

1. The server renders components to HTML strings and sends them to the client.
2. The browser displays the HTML immediately (fast First Contentful Paint).
3. React's JavaScript loads and runs.
4. React renders the component tree in memory (the render phase) but instead of creating new DOM nodes, it **walks the existing DOM** and attaches event listeners and fiber pointers to the existing nodes.
5. The page is now interactive.

```tsx
// Server: renders to HTML
const html = renderToString(<App />);

// Client: hydrates the existing HTML
hydrateRoot(document.getElementById('root')!, <App />);
```

### Hydration Mismatch Errors

A hydration mismatch occurs when the server-rendered HTML does not match what the client render produces. React 18 will attempt to recover by patching the DOM, but it logs a warning and the patching is expensive.

**Common causes of hydration mismatches:**

| Cause | Example |
|---|---|
| Using `Date.now()` or `Math.random()` in render | `<span>{Date.now()}</span>` |
| Browser-only APIs in render | `typeof window !== 'undefined'` conditional rendering |
| Different data on server vs client | Stale server cache |
| Browser auto-formatting | `<p><div>` — browsers fix invalid HTML nesting |
| CSS-in-JS with non-deterministic class names | Older styled-components versions |
| Extensions/toolbars modifying DOM | Browser extensions inserting elements |

**How to handle browser-only rendering:**

```tsx
function ClientOnly({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? <>{children}</> : null;
}

// Or use the built-in hook:
function TimestampDisplay() {
  const [timestamp, setTimestamp] = useState<number>();

  useEffect(() => {
    setTimestamp(Date.now());
  }, []);

  // Server and initial client render both show nothing — they match
  // After hydration, useEffect sets the timestamp
  return <span>{timestamp ?? ''}</span>;
}
```

### Selective Hydration (React 18)

With `Suspense` and streaming SSR, React can hydrate different parts of the page independently:

1. React starts hydrating the page.
2. It encounters a `<Suspense>` boundary with unloaded content — it skips it and hydrates the rest.
3. The skipped content streams in and hydrates later.
4. If the user clicks on an un-hydrated `<Suspense>` boundary, React **prioritizes** hydrating that boundary first.

This means the page becomes interactive progressively, and user interactions automatically bump hydration priority.

---

## Batching

### Automatic Batching (React 18+)

In React 18, **all state updates are batched** regardless of where they originate. This was a major change from React 17.

```tsx
// React 17: Only batched inside event handlers
// React 18: ALL of these are batched into a single re-render

// Event handler — batched in both 17 and 18
function handleClick() {
  setCount(c => c + 1);
  setFlag(f => !f);
  // ONE re-render (batched)
}

// setTimeout — NOT batched in 17, BATCHED in 18
setTimeout(() => {
  setCount(c => c + 1);
  setFlag(f => !f);
  // React 17: TWO re-renders
  // React 18: ONE re-render (batched)
}, 1000);

// Promise — NOT batched in 17, BATCHED in 18
fetch('/api/data').then(() => {
  setCount(c => c + 1);
  setFlag(f => !f);
  // React 17: TWO re-renders
  // React 18: ONE re-render (batched)
});

// Native event listener — NOT batched in 17, BATCHED in 18
element.addEventListener('click', () => {
  setCount(c => c + 1);
  setFlag(f => !f);
  // React 17: TWO re-renders
  // React 18: ONE re-render (batched)
});
```

### `flushSync` for Synchronous Updates

Occasionally you need a state update to be applied immediately (for example, to read the resulting DOM). Use `flushSync`:

```tsx
import { flushSync } from 'react-dom';

function handleClick() {
  flushSync(() => {
    setCount(c => c + 1);
  });
  // DOM is updated HERE — you can read it

  flushSync(() => {
    setFlag(f => !f);
  });
  // DOM is updated AGAIN — two separate commits
}
```

**`flushSync` is an escape hatch.** It opts out of batching and forces synchronous rendering. It should be rare in application code. Common legitimate uses:

- Reading DOM measurements immediately after a state update.
- Integrating with non-React libraries that need synchronous DOM state.
- Workarounds for third-party library timing issues.

### How Batching Works Internally

When you call a state setter:

1. React enqueues the update on the fiber's update queue.
2. React schedules a render (but does not start it immediately).
3. If another state setter is called before the render starts, it is enqueued on the same (or another fiber's) update queue.
4. When the microtask queue flushes (or the event handler returns), React processes all queued updates in a single render pass.

The key insight: `setState` does not immediately update state. It **schedules** an update. Batching is just the natural consequence of processing all scheduled updates together.

---

## Common Interview Questions

### Q1: "What is the Virtual DOM, and is it faster than the real DOM?"

**Answer:** The Virtual DOM is a plain JavaScript object tree describing the UI. It is NOT inherently faster than direct DOM manipulation — a hand-crafted, surgical DOM update will always be faster than React's diffing plus DOM update. The Virtual DOM's value is that it provides a **declarative programming model** where you describe what the UI should look like, and React figures out the minimal DOM changes. This is a productivity and correctness tradeoff, not a raw performance win. It makes it practical to build complex UIs that would be unmaintainable with manual DOM manipulation.

### Q2: "Why can React call my component function multiple times?"

**Answer:** In the render phase, React is building a virtual tree and diffing it. This phase is **interruptible** in concurrent mode — React can start rendering, get interrupted by a higher-priority update, throw away the in-progress work, and restart. Additionally, React StrictMode intentionally double-invokes component functions, reducers, and initializers in development to surface impurity bugs. This is why component functions must be pure — no side effects in the render body.

### Q3: "A parent component re-renders but its child's props haven't changed. Does the child re-render?"

**Answer:** **Yes.** By default, when a parent re-renders, all children re-render regardless of whether their props changed. React does not do a shallow comparison of props by default — that would add overhead for the common case where props do change. To opt into prop comparison, wrap the child in `React.memo`. Alternatively, use the children composition pattern or move state down to avoid the re-render entirely.

### Q4: "Explain the difference between the render phase and the commit phase."

**Answer:** The render phase is where React calls component functions, builds the workInProgress fiber tree, and computes diffs. It is **pure** (no side effects) and **interruptible** (can be paused and restarted). The commit phase is where React applies the computed DOM mutations, runs `useLayoutEffect`, and schedules `useEffect`. It is **synchronous** and **cannot be interrupted** — once it starts, it runs to completion to avoid showing an inconsistent UI.

### Q5: "What happens when you call setState inside useEffect?"

**Answer:** It schedules another render. The sequence is: initial render, commit, browser paint, `useEffect` fires, `setState` is called, a second render is scheduled, commit, browser paint. The user may briefly see the first render's output before the second render updates it. For layout-dependent updates that should not flash, use `useLayoutEffect` instead — it fires synchronously after DOM mutation but before browser paint, so the second render happens before the user sees anything.

### Q6: "How does React's key prop work, and when should you use something other than index?"

**Answer:** The `key` prop is a hint to the reconciler about element identity across renders. When reconciling a list, React matches old and new children by their keys. If a key persists, React reuses the fiber (preserving state). If a key disappears, React destroys the fiber. If a new key appears, React creates a new fiber. Using index as key is problematic for lists that reorder, filter, or have insertions — it causes React to update existing nodes with wrong data instead of moving them, leading to state corruption (e.g., input values appearing in the wrong row). Use a stable, unique identifier from your data as the key.

### Q7: "How does automatic batching in React 18 differ from React 17?"

**Answer:** In React 17, batching only worked inside React event handlers. State updates in `setTimeout`, `Promise` callbacks, native event listeners, and other async contexts were NOT batched — each `setState` triggered a separate re-render. In React 18, all state updates are automatically batched regardless of their origin. This is possible because React 18 uses `createRoot` instead of `ReactDOM.render`, which enables the new concurrent-capable scheduler that handles batching universally. Use `flushSync` to opt out when you need synchronous updates.

### Q8: "What is selective hydration and why does it matter?"

**Answer:** Selective hydration (React 18) allows different parts of the page to hydrate independently using `Suspense` boundaries. Without it, hydration is all-or-nothing: the entire page must hydrate before any part is interactive. With selective hydration, React can skip `Suspense` boundaries whose content has not arrived yet, hydrate the rest, and come back to the skipped parts later. If a user interacts with an un-hydrated region, React bumps its hydration priority. This dramatically improves Time to Interactive for large pages because users can start interacting with hydrated parts while other parts are still loading.

---

> **Next:** See [deep-dive.md](./deep-dive.md) for fiber node internals, the work loop, lane model details, and Suspense mechanics.
> **Quick reference:** See [cheat-sheet.md](./cheat-sheet.md) for tables, diagrams, and debugging checklists.
