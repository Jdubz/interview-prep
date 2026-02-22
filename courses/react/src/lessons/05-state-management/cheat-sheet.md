# Lesson 05: State Management -- Cheat Sheet

Quick reference for interview prep. Skim before the call.

---

## State Library Comparison

| Library | API Style | Bundle Size | Learning Curve | Best For |
|---------|-----------|-------------|----------------|----------|
| useState / useReducer | Built-in hooks | 0 KB | None | Local component state |
| Context API | Built-in provider/consumer | 0 KB | Low | Infrequent global values (theme, auth, locale) |
| Zustand | External store + hook | ~1 KB | Low | General-purpose global client state |
| Jotai | Atomic (bottom-up) | ~3 KB | Low-Medium | Fine-grained, composable, graph-like state |
| Redux Toolkit | Flux (top-down, single store) | ~11 KB | Medium | Large teams, strict conventions, RTK Query |
| TanStack Query | Cache manager + hooks | ~12 KB | Medium | Server/remote state (any data from an API) |
| XState | State machines / statecharts | ~15 KB | High | Complex workflows, impossible-state prevention |
| React Hook Form | Uncontrolled + subscription | ~9 KB | Low-Medium | Form state, validation, performance |

---

## Decision Tree: What State Solution Should I Use?

```
START: What kind of state is this?
|
+-- Server data (fetched from API)?
|   -> TanStack Query (or SWR / RTK Query)
|
+-- URL-serializable (filters, pagination, tabs)?
|   -> useSearchParams / nuqs
|
+-- Form input (validation, submission)?
|   -> React Hook Form + Zod
|
+-- Local to one component?
|   +-- Simple (toggle, counter)?  -> useState
|   +-- Coupled transitions?       -> useReducer
|
+-- Shared across components?
    +-- Changes infrequently (theme, auth)?  -> Context (split + memoize)
    +-- Changes frequently?
        +-- Many independent atoms?  -> Jotai
        +-- Single store preferred?  -> Zustand
        +-- Large team, strict patterns needed?  -> Redux Toolkit
```

---

## Context Performance Pattern: Split + Memoize

```tsx
// 1. Separate state and dispatch contexts
const StateCtx = createContext<AppState>(initialState);
const DispatchCtx = createContext<Dispatch<AppAction>>(() => {});

// 2. Memoize the state value
function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const memoizedState = useMemo(() => state, [state]);
  // dispatch is already referentially stable from useReducer

  return (
    <DispatchCtx.Provider value={dispatch}>
      <StateCtx.Provider value={memoizedState}>
        {children}
      </StateCtx.Provider>
    </DispatchCtx.Provider>
  );
}

// 3. Custom hooks for clean consumption
function useAppState() {
  return useContext(StateCtx);
}

function useAppDispatch() {
  return useContext(DispatchCtx);
}

// Components that only dispatch never re-render on state changes
function ActionButton() {
  const dispatch = useAppDispatch();
  return <button onClick={() => dispatch({ type: 'INCREMENT' })}>+1</button>;
}
```

---

## Zustand Store Template

```tsx
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface MyState {
  items: Item[];
  isLoading: boolean;
}

interface MyActions {
  addItem: (item: Item) => void;
  removeItem: (id: string) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

const initialState: MyState = {
  items: [],
  isLoading: false,
};

export const useMyStore = create<MyState & MyActions>()(
  devtools(
    persist(
      immer((set) => ({
        ...initialState,

        addItem: (item) =>
          set((state) => {
            state.items.push(item);
          }),

        removeItem: (id) =>
          set((state) => {
            state.items = state.items.filter((i) => i.id !== id);
          }),

        setLoading: (loading) => set({ isLoading: loading }),

        reset: () => set(initialState),
      })),
      { name: 'my-store' },
    ),
    { name: 'MyStore' },
  ),
);

// Usage with selectors (component only re-renders when selected slice changes)
function ItemCount() {
  const count = useMyStore((s) => s.items.length);
  return <span>{count}</span>;
}
```

---

## TanStack Query Setup Template

```tsx
// lib/query-client.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60,      // 1 min
      gcTime: 1000 * 60 * 5,     // 5 min
      retry: 2,
      refetchOnWindowFocus: true,
    },
  },
});

// app.tsx
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

// hooks/use-users.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => api.getUsers(),
  });
}

export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (newUser: CreateUserInput) => api.createUser(newUser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

// Optimistic update variant
export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (user: UpdateUserInput) => api.updateUser(user),
    onMutate: async (updatedUser) => {
      await queryClient.cancelQueries({ queryKey: ['users', updatedUser.id] });
      const previous = queryClient.getQueryData(['users', updatedUser.id]);
      queryClient.setQueryData(['users', updatedUser.id], updatedUser);
      return { previous };
    },
    onError: (_err, variables, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['users', variables.id], context.previous);
      }
    },
    onSettled: (_data, _err, variables) => {
      queryClient.invalidateQueries({ queryKey: ['users', variables.id] });
    },
  });
}
```

---

## React Hook Form + Zod Template

```tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// 1. Schema = single source of truth
const FormSchema = z.object({
  name: z.string().min(1, 'Required').max(100),
  email: z.string().email('Invalid email'),
  role: z.enum(['admin', 'user', 'viewer']),
  age: z.coerce.number().int().min(18, 'Must be 18+').optional(),
});

type FormData = z.infer<typeof FormSchema>;

// 2. Hook setup
function MyForm() {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting, isDirty },
  } = useForm<FormData>({
    resolver: zodResolver(FormSchema),
    defaultValues: { name: '', email: '', role: 'user' },
  });

  const onSubmit = async (data: FormData) => {
    await api.submitForm(data);
    reset(); // reset form after successful submission
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('name')} />
      {errors.name && <p>{errors.name.message}</p>}

      <input {...register('email')} />
      {errors.email && <p>{errors.email.message}</p>}

      <select {...register('role')}>
        <option value="admin">Admin</option>
        <option value="user">User</option>
        <option value="viewer">Viewer</option>
      </select>
      {errors.role && <p>{errors.role.message}</p>}

      <input type="number" {...register('age')} />
      {errors.age && <p>{errors.age.message}</p>}

      <button type="submit" disabled={isSubmitting || !isDirty}>
        {isSubmitting ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}
```

---

## Common Gotchas

| Gotcha | What Happens | Fix |
|--------|-------------|-----|
| Context mega-object | All consumers re-render on any field change | Split into focused contexts |
| New context value every render | `{ a, b }` is a new reference each render, so all consumers re-render | Wrap in `useMemo` |
| Zustand selector returns new object | `(s) => ({ a: s.a, b: s.b })` creates a new object, defeats selector optimization | Use `useShallow` from `zustand/react/shallow` |
| TanStack Query as global state | Putting client-only state in query cache | Use queries for server data only; use Zustand/Jotai for client state |
| staleTime: 0 (default) | Every mount triggers a refetch, even if data was just fetched | Set an appropriate `staleTime` (e.g., 60s) |
| Mutating state directly | `state.items.push(x)` without Immer silently corrupts state | Use Immer middleware or always return new objects |
| Form re-renders on keystroke | Controlled inputs (`value={state}`) re-render parent on every change | Use React Hook Form (uncontrolled) or isolate with `useWatch` |
| Forgetting to cancel queries before optimistic update | Background refetch overwrites optimistic data | Always call `queryClient.cancelQueries()` in `onMutate` |
| Putting server data in Zustand | Manual loading/error/stale management, no caching | Use TanStack Query -- it handles the cache lifecycle |
| Redux for a small app | Boilerplate overhead with no benefit | Zustand or Jotai for small-to-medium apps |
