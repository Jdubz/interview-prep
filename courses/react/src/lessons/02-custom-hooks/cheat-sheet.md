# Custom Hooks Cheat Sheet

## Custom Hook Template (TypeScript)

```tsx
import { useState, useEffect, useCallback, useRef } from 'react';

interface UseMyHookOptions {
  enabled?: boolean;
  // ...options
}

interface UseMyHookReturn {
  data: SomeType | null;
  isActive: boolean;
  reset: () => void;
}

export function useMyHook(
  input: string,
  options: UseMyHookOptions = {},
): UseMyHookReturn {
  const { enabled = true } = options;
  const [data, setData] = useState<SomeType | null>(null);
  const [isActive, setIsActive] = useState(false);

  // Stable ref for latest callback / value
  const inputRef = useRef(input);
  inputRef.current = input;

  // Effects with cleanup
  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    setIsActive(true);

    doAsyncWork(input).then(result => {
      if (!cancelled) {
        setData(result);
        setIsActive(false);
      }
    });

    return () => { cancelled = true; };
  }, [input, enabled]);

  // Stable action
  const reset = useCallback(() => {
    setData(null);
    setIsActive(false);
  }, []);

  return { data, isActive, reset };
}
```

---

## Common Hooks Catalog

| Hook | Signature | Description |
|---|---|---|
| `useToggle` | `(initial?: boolean) => [boolean, () => void]` | Boolean state with a stable toggle function |
| `usePrevious` | `<T>(value: T) => T \| undefined` | Returns the value from the previous render |
| `useDebouncedValue` | `<T>(value: T, ms: number) => T` | Delays updating the returned value until input is stable |
| `useLocalStorage` | `<T>(key: string, init: T) => [T, SetState<T>]` | Persistent state synced to localStorage with SSR safety |
| `useMediaQuery` | `(query: string) => boolean` | Reactive CSS media query match |
| `useIntersectionObserver` | `(opts?) => [RefCallback, Entry \| null]` | Observes element visibility via IntersectionObserver |
| `useEventListener` | `(event, handler, element?, opts?) => void` | Declarative event listener with automatic cleanup |
| `useOnClickOutside` | `(ref, handler) => void` | Fires handler when a click occurs outside the ref element |
| `useInterval` | `(callback, delayMs \| null) => void` | Declarative `setInterval`; pass `null` to pause |
| `useTimeout` | `(callback, delayMs \| null) => void` | Declarative `setTimeout` with auto-cleanup |
| `useAsync` | `<T>(fn: () => Promise<T>) => AsyncState<T>` | Tracks loading/success/error for a promise |
| `useFetch` | `<T>(url: string) => { data, error, isLoading }` | Fetch with caching, abort, and race condition handling |
| `useLatestRef` | `<T>(value: T) => RefObject<T>` | Always-current ref that does not trigger re-renders |
| `useStableCallback` | `<T extends Function>(fn: T) => T` | Stable function identity that always calls the latest closure |
| `useIsomorphicLayoutEffect` | Same as `useLayoutEffect` | Uses `useLayoutEffect` in browser, `useEffect` on server |

---

## Testing Recipe

### Setup, Act, Assert

```tsx
import { renderHook, act, waitFor } from '@testing-library/react';

// 1. SETUP: render the hook with initial props
const { result, rerender, unmount } = renderHook(
  ({ value }) => useMyHook(value),
  { initialProps: { value: 'initial' } },
);

// 2. ACT: trigger state changes
act(() => {
  result.current.someAction();
});

// or change inputs:
rerender({ value: 'updated' });

// or advance timers:
jest.useFakeTimers();
act(() => jest.advanceTimersByTime(500));

// or wait for async:
await waitFor(() => {
  expect(result.current.isLoading).toBe(false);
});

// 3. ASSERT: check result
expect(result.current.data).toEqual(expected);

// 4. CLEANUP: verify teardown
unmount();
expect(cleanupSpy).toHaveBeenCalled();
```

### Providing Context in Tests

```tsx
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider value={mockAuth}>
    <ThemeProvider value={mockTheme}>
      {children}
    </ThemeProvider>
  </AuthProvider>
);

const { result } = renderHook(() => useMyHook(), { wrapper });
```

---

## Do's and Don'ts

| Do | Don't |
|---|---|
| Start hook names with `use` | Name a regular function `useSomething` |
| Return stable references (memoize objects/callbacks) | Return a new object/array literal every render |
| Use `useRef` for mutable values that should not trigger re-renders | Store mutable values in `useState` when you do not need re-renders |
| Clean up subscriptions, timers, and observers in effect cleanup | Assume the component will never unmount or re-render |
| Guard `window`/`document` access for SSR safety | Access browser APIs at module scope or in render |
| Use discriminated unions for multi-state returns | Use separate `isLoading` + `isError` + `isSuccess` booleans that can conflict |
| Compose small hooks into larger ones | Build a single hook that handles 5+ unrelated concerns |
| Accept options via an object for extensibility | Add positional parameters beyond 2-3 arguments |
| Use `useReducer` when state transitions are interdependent | Use multiple `useState` calls with manual synchronization |
| Use the `enabled` pattern to conditionally skip work | Call hooks conditionally (`if (x) useThing()`) |
| Write explicit TypeScript return types for public hooks | Rely solely on inference for complex return types |
| Test hooks via `renderHook` in isolation | Only test hooks indirectly through component tests |

---

## Decision: Should I Extract This Into a Hook?

```
Is there stateful logic (useState, useEffect, useRef working together)?
  No  --> Probably a plain function, not a hook.
  Yes |
      v
Does this logic represent a single, nameable concept?
  No  --> Split into smaller hooks first.
  Yes |
      v
Will extracting it make the component easier to read?
  No  --> Leave it inline. One useState + one useEffect is fine inline.
  Yes |
      v
Is the logic > ~15 lines or does it obscure the component's render logic?
  No  --> Extraction is optional. Consider readability vs indirection tradeoff.
  Yes |
      v
Extract it. Name it after what it provides, not what it does internally.
  useOnlineStatus, not useAddWindowEventListenerForOnlineAndOffline.
```

### Quick Sniff Tests

- **If you have to pass 5+ config options:** the hook may be doing too much. Split it.
- **If the hook returns values that callers never use together:** it is a god hook. Decompose it.
- **If testing the hook requires mocking more than 2 externals:** the hook has too many responsibilities.
- **If you cannot describe the hook in one sentence:** it needs to be smaller.
