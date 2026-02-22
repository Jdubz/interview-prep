# Lesson 05: State Management

## Table of Contents

1. [State Categories](#state-categories)
2. [useState vs useReducer](#usestate-vs-usereducer)
3. [Context API](#context-api)
4. [Zustand](#zustand)
5. [Jotai](#jotai)
6. [Redux Toolkit (RTK)](#redux-toolkit-rtk)
7. [Server State with TanStack Query](#server-state-with-tanstack-query)
8. [URL State](#url-state)
9. [Form State](#form-state)
10. [State Colocation](#state-colocation)
11. [Common Interview Questions](#common-interview-questions)

---

## State Categories

Not all state is created equal. Choosing the wrong tool for a state category is the single most
common architectural mistake in React applications. Before reaching for any library, classify
the state you are dealing with.

### Local / Component State

State that belongs to a single component and has no consumers elsewhere in the tree.

**When it applies:** Toggle visibility, input focus, animation flags, ephemeral UI bits.

```tsx
const [isOpen, setIsOpen] = useState(false);
```

Rule of thumb: if you can delete the component and no other component cares, it is local state.

### Global / Shared State

State consumed by multiple unrelated components across different subtrees.

**When it applies:** Authenticated user, feature flags, theme, shopping cart, notification queue.

```tsx
// Zustand store -- consumed anywhere in the tree
const useCartStore = create<CartState>((set) => ({
  items: [],
  addItem: (item) => set((s) => ({ items: [...s.items, item] })),
}));
```

### Server / Remote State

Data that lives on a backend and is fetched, cached, and synchronized with the server.

**When it applies:** Any data you GET from an API. This is *not* global state -- it is a cache
of a remote data source with its own lifecycle (stale, refetching, error, loading).

```tsx
const { data, isLoading } = useQuery({
  queryKey: ['users', userId],
  queryFn: () => api.getUser(userId),
});
```

A defining characteristic: the source of truth is the server, not the client.

### URL State

State serialized into the URL so that it survives page refreshes and can be shared via link.

**When it applies:** Search filters, pagination, sort order, selected tab, modal open state when
it needs to be linkable.

```tsx
const [searchParams, setSearchParams] = useSearchParams();
const page = Number(searchParams.get('page') ?? '1');
```

### Form State

Transient state tied to user input that ultimately produces a submission payload.

**When it applies:** Any form -- login, checkout, multi-step wizard, inline editing.

```tsx
const { register, handleSubmit } = useForm<CheckoutForm>();
```

Form state has unique concerns: validation, dirty tracking, touched fields, submit count,
and performance (avoiding re-renders on every keystroke).

### Decision Matrix

| Category | Source of truth | Lifecycle | Typical tool |
|----------|----------------|-----------|-------------|
| Local | Component | Mount/unmount | `useState`, `useReducer` |
| Global | Client | App session | Zustand, Jotai, Redux |
| Server | Backend DB | Cache TTL | TanStack Query, SWR |
| URL | Address bar | Navigation | `useSearchParams`, nuqs |
| Form | User input | Form mount to submit | React Hook Form |

---

## useState vs useReducer

Both are built-in. The choice between them is not about complexity -- it is about the *shape*
of state transitions.

### useState: When Transitions Are Independent

```tsx
const [count, setCount] = useState(0);
const [name, setName] = useState('');
```

Fine when each piece of state changes independently. Multiple `useState` calls are preferable
to a single object when the fields are unrelated.

### useReducer: When Transitions Are Coupled

When a single user action must update multiple fields consistently, a reducer centralizes
the transition logic.

```tsx
type State = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data: User[] | null;
  error: string | null;
};

type Action =
  | { type: 'FETCH_START' }
  | { type: 'FETCH_SUCCESS'; payload: User[] }
  | { type: 'FETCH_ERROR'; error: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'FETCH_START':
      return { status: 'loading', data: null, error: null };
    case 'FETCH_SUCCESS':
      return { status: 'success', data: action.payload, error: null };
    case 'FETCH_ERROR':
      return { status: 'error', data: null, error: action.error };
  }
}

function UserList() {
  const [state, dispatch] = useReducer(reducer, {
    status: 'idle',
    data: null,
    error: null,
  });

  // dispatch is referentially stable -- safe to pass as prop or dependency
  useEffect(() => {
    dispatch({ type: 'FETCH_START' });
    fetchUsers()
      .then((users) => dispatch({ type: 'FETCH_SUCCESS', payload: users }))
      .catch((e) => dispatch({ type: 'FETCH_ERROR', error: e.message }));
  }, [dispatch]); // dispatch identity never changes

  // ...
}
```

### Dispatch Stability

`dispatch` from `useReducer` is referentially stable across renders. This matters when you
pass it to memoized children or include it in effect dependency arrays. With `useState`, the
setter is also stable, but composite update functions you build on top of multiple setters
are not -- you end up wrapping them in `useCallback` chains.

### Decision Criteria

| Criterion | useState | useReducer |
|-----------|----------|------------|
| Independent primitives | Preferred | Overkill |
| Coupled transitions (one event, many fields) | Fragile | Preferred |
| Next state depends on previous state | Works (functional update) | Natural |
| Complex validation / guard logic | Messy | Clean |
| Testability of transitions | Harder | Pure function, easy |
| Stable callback to pass down | Need useCallback | dispatch is stable |

---

## Context API

### Proper Use Cases

Context is a **dependency injection** mechanism, not a state management library. It excels at
broadcasting *infrequently changing* values to a deep subtree:

- **Theme** (light/dark)
- **Authenticated user** (changes on login/logout)
- **Locale / i18n** (changes rarely)
- **Feature flags** (loaded once, read everywhere)

### The Performance Trap

Every component that calls `useContext(SomeContext)` re-renders when the context value changes.
There is no selector mechanism. This means:

```tsx
// ANTI-PATTERN: single mega-context
const AppContext = createContext<{
  user: User;
  theme: Theme;
  cart: CartItem[];
  notifications: Notification[];
}>({ /* ... */ });

function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  // Every consumer re-renders when ANY field changes
  return (
    <AppContext.Provider value={state}>
      {children}
    </AppContext.Provider>
  );
}
```

When `notifications` updates, every component reading `theme` also re-renders. This is the
number one reason Context gets a bad reputation for performance.

### Splitting Contexts

Separate concerns into distinct contexts so that consumers only subscribe to what they need.

```tsx
const ThemeContext = createContext<Theme>('light');
const UserContext = createContext<User | null>(null);
const CartContext = createContext<CartState>(initialCartState);

function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <UserProvider>
        <CartProvider>
          {children}
        </CartProvider>
      </UserProvider>
    </ThemeProvider>
  );
}
```

Now a `notifications` update does not touch `ThemeContext` consumers.

### Memoizing Context Value

Even with split contexts, a common mistake is creating a new object reference on every render
of the provider.

```tsx
// BUG: value is a new object every render
function CartProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([]);

  return (
    <CartContext.Provider value={{ items, addItem, removeItem }}>
      {children}
    </CartContext.Provider>
  );
}
```

Fix: memoize the value object.

```tsx
function CartProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([]);

  const addItem = useCallback((item: CartItem) => {
    setItems((prev) => [...prev, item]);
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
  }, []);

  const value = useMemo(
    () => ({ items, addItem, removeItem }),
    [items, addItem, removeItem],
  );

  return (
    <CartContext.Provider value={value}>
      {children}
    </CartContext.Provider>
  );
}
```

### Separate State and Dispatch Contexts

An advanced pattern: provide state and dispatch in two contexts. Components that only fire
actions never re-render when state changes.

```tsx
const CartStateContext = createContext<CartItem[]>([]);
const CartDispatchContext = createContext<Dispatch<CartAction>>(() => {});

function CartProvider({ children }: { children: ReactNode }) {
  const [items, dispatch] = useReducer(cartReducer, []);

  return (
    <CartStateContext.Provider value={items}>
      <CartDispatchContext.Provider value={dispatch}>
        {children}
      </CartDispatchContext.Provider>
    </CartStateContext.Provider>
  );
}

// Component that displays items -- subscribes to state
function CartBadge() {
  const items = useContext(CartStateContext);
  return <span>{items.length}</span>;
}

// Component that adds items -- subscribes to dispatch only (never re-renders on cart change)
function AddToCartButton({ item }: { item: CartItem }) {
  const dispatch = useContext(CartDispatchContext);
  return <button onClick={() => dispatch({ type: 'ADD', item })}>Add</button>;
}
```

---

## Zustand

Zustand is a minimal, unopinionated state management library. It is the most popular
alternative to Redux as of 2025/2026 and a strong default choice for global client state.

### Minimal API

A store is a hook. No providers, no boilerplate.

```tsx
import { create } from 'zustand';

interface BearState {
  bears: number;
  increasePopulation: () => void;
  removeAllBears: () => void;
}

const useBearStore = create<BearState>((set) => ({
  bears: 0,
  increasePopulation: () => set((state) => ({ bears: state.bears + 1 })),
  removeAllBears: () => set({ bears: 0 }),
}));

function BearCounter() {
  const bears = useBearStore((state) => state.bears);
  return <h1>{bears} bears around here...</h1>;
}

function Controls() {
  const increasePopulation = useBearStore((state) => state.increasePopulation);
  return <button onClick={increasePopulation}>one up</button>;
}
```

### Selector-Based Re-Renders

The key performance advantage: passing a selector to the hook means the component only
re-renders when the *selected slice* changes (shallow equality by default).

```tsx
// Only re-renders when `bears` changes, not when other fields change
const bears = useBearStore((state) => state.bears);

// Multiple selectors with useShallow for object slices
import { useShallow } from 'zustand/react/shallow';

const { bears, fish } = useBearStore(
  useShallow((state) => ({ bears: state.bears, fish: state.fish })),
);
```

### Middleware

Zustand uses a composable middleware pattern.

```tsx
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface TodoState {
  todos: Todo[];
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
}

const useTodoStore = create<TodoState>()(
  devtools(
    persist(
      immer((set) => ({
        todos: [],
        addTodo: (text) =>
          set((state) => {
            // immer allows direct mutation syntax
            state.todos.push({ id: crypto.randomUUID(), text, done: false });
          }),
        toggleTodo: (id) =>
          set((state) => {
            const todo = state.todos.find((t) => t.id === id);
            if (todo) todo.done = !todo.done;
          }),
      })),
      { name: 'todo-storage' }, // persist key
    ),
    { name: 'TodoStore' }, // devtools label
  ),
);
```

**Middleware stack (inner to outer):**

| Middleware | Purpose |
|-----------|---------|
| `immer` | Write mutations instead of spread-heavy immutable updates |
| `persist` | Serialize to localStorage/sessionStorage/AsyncStorage |
| `devtools` | Redux DevTools integration |
| `subscribeWithSelector` | Subscribe to slices outside React |

### Zustand vs Redux

| Dimension | Zustand | Redux Toolkit |
|-----------|---------|--------------|
| Boilerplate | Minimal (one `create` call) | Moderate (configureStore, createSlice) |
| Provider required | No | Yes (`<Provider store={store}>`) |
| Middleware | Composable functions | Redux middleware chain |
| Selectors | Built into the hook | `useSelector` + reselect |
| DevTools | Via middleware | Built-in |
| Bundle size | ~1 KB | ~11 KB (RTK) |
| Learning curve | Low | Moderate |
| Best for | Most apps, especially small-to-medium | Large teams needing strict conventions |

---

## Jotai

Jotai implements a bottom-up (atomic) state model inspired by Recoil but with a simpler API
and no string keys.

### Atomic State Model

Instead of a single store, state is defined as independent atoms. Components subscribe to
exactly the atoms they need.

```tsx
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';

// Primitive atom
const countAtom = atom(0);

// Read-only derived atom
const doubleCountAtom = atom((get) => get(countAtom) * 2);

function Counter() {
  const [count, setCount] = useAtom(countAtom);
  const doubled = useAtomValue(doubleCountAtom); // read-only hook

  return (
    <div>
      <p>{count} (doubled: {doubled})</p>
      <button onClick={() => setCount((c) => c + 1)}>+1</button>
    </div>
  );
}

function ResetButton() {
  const setCount = useSetAtom(countAtom); // write-only hook, no re-render on read
  return <button onClick={() => setCount(0)}>Reset</button>;
}
```

### Derived Atoms

Derived atoms are the composability primitive. They can depend on multiple atoms and the
dependency graph is tracked automatically.

```tsx
const usersAtom = atom<User[]>([]);
const searchAtom = atom('');

const filteredUsersAtom = atom((get) => {
  const users = get(usersAtom);
  const search = get(searchAtom).toLowerCase();
  if (!search) return users;
  return users.filter((u) => u.name.toLowerCase().includes(search));
});
```

When `searchAtom` changes, only components subscribed to `filteredUsersAtom` re-render --
not those subscribed to `usersAtom` alone.

### Async Atoms

```tsx
const userAtom = atom(async () => {
  const res = await fetch('/api/user');
  return res.json() as Promise<User>;
});

// Write-read async atom
const userWithRefetchAtom = atom(
  async (get) => {
    // triggers Suspense
    const res = await fetch('/api/user');
    return res.json() as Promise<User>;
  },
  async (_get, set) => {
    // write function = refetch
    const res = await fetch('/api/user');
    const user = await res.json();
    set(userWithRefetchAtom, user);
  },
);
```

Async atoms integrate with React Suspense out of the box.

### When Atomic State Shines

- You have many **independent but composable** pieces of state (think spreadsheet cells).
- You want **fine-grained** re-renders without manually writing selectors.
- You are building something with a **graph-like** dependency structure (editors, dashboards).
- You want **code splitting** -- atoms can be defined in any module, no single store file.

---

## Redux Toolkit (RTK)

Redux is not dead. RTK dramatically reduced boilerplate and remains the right choice in
specific scenarios.

### createSlice

```tsx
import { createSlice, configureStore, type PayloadAction } from '@reduxjs/toolkit';

interface CounterState {
  value: number;
}

const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 } satisfies CounterState as CounterState,
  reducers: {
    incremented(state) {
      state.value += 1; // Immer under the hood
    },
    amountAdded(state, action: PayloadAction<number>) {
      state.value += action.payload;
    },
  },
});

export const { incremented, amountAdded } = counterSlice.actions;

const store = configureStore({
  reducer: {
    counter: counterSlice.reducer,
  },
});

type RootState = ReturnType<typeof store.getState>;
type AppDispatch = typeof store.dispatch;
```

### RTK Query for Server State

RTK Query is Redux's answer to TanStack Query. It auto-generates hooks from an API definition.

```tsx
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

const pokemonApi = createApi({
  reducerPath: 'pokemonApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'https://pokeapi.co/api/v2/' }),
  tagTypes: ['Pokemon'],
  endpoints: (builder) => ({
    getPokemonByName: builder.query<Pokemon, string>({
      query: (name) => `pokemon/${name}`,
      providesTags: (result, error, name) => [{ type: 'Pokemon', id: name }],
    }),
    updatePokemon: builder.mutation<Pokemon, Partial<Pokemon> & Pick<Pokemon, 'id'>>({
      query: ({ id, ...patch }) => ({
        url: `pokemon/${id}`,
        method: 'PATCH',
        body: patch,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Pokemon', id }],
    }),
  }),
});

export const { useGetPokemonByNameQuery, useUpdatePokemonMutation } = pokemonApi;
```

### When Redux Is Still the Right Choice

- **Large team with strict architectural needs:** Redux enforces a single direction of data
  flow with a well-defined pattern. Onboarding 50 engineers? Redux conventions help.
- **Already using RTK Query:** If your server state is in RTK Query, keeping client state
  in Redux slices keeps everything in one devtools panel.
- **Complex cross-cutting middleware:** Logging, analytics, undo/redo -- Redux middleware
  chain is battle-tested.
- **Time-travel debugging is critical:** Redux DevTools time-travel is the most mature.

---

## Server State with TanStack Query

Server state is fundamentally different from client state. It is:

- Persisted remotely
- Asynchronous
- Shared ownership (other users can change it)
- Potentially stale

TanStack Query (formerly React Query) treats server data as a **cache** with lifecycle.

### Stale-While-Revalidate Mental Model

```
1. Component mounts -> cache miss -> fetch -> loading state -> data arrives -> fresh
2. staleTime elapses -> data marked stale
3. Component re-mounts or window focuses -> stale data shown immediately -> background refetch
4. Refetch succeeds -> cache updated -> component re-renders with fresh data
```

This gives the user instant UI while silently keeping data up to date.

### Cache Invalidation

```tsx
const queryClient = useQueryClient();

// Invalidate all queries with key starting with 'todos'
queryClient.invalidateQueries({ queryKey: ['todos'] });

// Invalidate a specific query
queryClient.invalidateQueries({ queryKey: ['todos', todoId] });

// Invalidate and refetch immediately
queryClient.refetchQueries({ queryKey: ['todos'] });
```

### Optimistic Updates

```tsx
const queryClient = useQueryClient();

const mutation = useMutation({
  mutationFn: updateTodo,
  onMutate: async (newTodo) => {
    // Cancel outgoing refetches so they don't overwrite our optimistic update
    await queryClient.cancelQueries({ queryKey: ['todos', newTodo.id] });

    // Snapshot previous value
    const previousTodo = queryClient.getQueryData(['todos', newTodo.id]);

    // Optimistically update
    queryClient.setQueryData(['todos', newTodo.id], newTodo);

    // Return context with snapshot for rollback
    return { previousTodo };
  },
  onError: (_err, _newTodo, context) => {
    // Rollback on error
    if (context?.previousTodo) {
      queryClient.setQueryData(
        ['todos', context.previousTodo.id],
        context.previousTodo,
      );
    }
  },
  onSettled: (_data, _error, variables) => {
    // Always refetch to ensure server truth
    queryClient.invalidateQueries({ queryKey: ['todos', variables.id] });
  },
});
```

### Infinite Queries

```tsx
const {
  data,
  fetchNextPage,
  hasNextPage,
  isFetchingNextPage,
} = useInfiniteQuery({
  queryKey: ['projects'],
  queryFn: ({ pageParam }) => fetchProjects(pageParam),
  initialPageParam: 0,
  getNextPageParam: (lastPage, allPages) => lastPage.nextCursor ?? undefined,
});

// Flatten pages for rendering
const allProjects = data?.pages.flatMap((page) => page.projects) ?? [];
```

### Prefetching

```tsx
// Prefetch on hover for instant navigation
function ProjectLink({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();

  const prefetch = () => {
    queryClient.prefetchQuery({
      queryKey: ['project', projectId],
      queryFn: () => fetchProject(projectId),
      staleTime: 5 * 60 * 1000, // don't refetch if already fresh
    });
  };

  return (
    <Link to={`/projects/${projectId}`} onMouseEnter={prefetch}>
      View Project
    </Link>
  );
}
```

### Key Configuration Options

```tsx
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 min before data is considered stale
      gcTime: 1000 * 60 * 5, // 5 min before inactive cache is garbage collected
      retry: 3,
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
    },
  },
});
```

---

## URL State

### useSearchParams (React Router)

```tsx
import { useSearchParams } from 'react-router-dom';

function ProductList() {
  const [searchParams, setSearchParams] = useSearchParams();

  const category = searchParams.get('category') ?? 'all';
  const sort = searchParams.get('sort') ?? 'newest';
  const page = Number(searchParams.get('page') ?? '1');

  const setPage = (p: number) => {
    setSearchParams((prev) => {
      prev.set('page', String(p));
      return prev;
    });
  };

  const setFilters = (filters: { category?: string; sort?: string }) => {
    setSearchParams((prev) => {
      if (filters.category) prev.set('category', filters.category);
      if (filters.sort) prev.set('sort', filters.sort);
      prev.set('page', '1'); // reset page on filter change
      return prev;
    });
  };

  // ...
}
```

### Encoding Complex State

For complex objects, use a serialization strategy.

```tsx
import { z } from 'zod';

const FiltersSchema = z.object({
  categories: z.array(z.string()).default([]),
  priceRange: z.tuple([z.number(), z.number()]).default([0, 1000]),
  inStock: z.boolean().default(false),
});

type Filters = z.infer<typeof FiltersSchema>;

function useFiltersFromURL(): Filters {
  const [searchParams] = useSearchParams();
  const raw = searchParams.get('filters');

  if (!raw) return FiltersSchema.parse({});

  try {
    const decoded = JSON.parse(atob(raw));
    return FiltersSchema.parse(decoded);
  } catch {
    return FiltersSchema.parse({});
  }
}

function setFiltersToURL(filters: Filters, setSearchParams: SetURLSearchParams) {
  setSearchParams((prev) => {
    prev.set('filters', btoa(JSON.stringify(filters)));
    return prev;
  });
}
```

### Syncing URL with App State

The nuqs library provides type-safe URL state hooks that feel like `useState`:

```tsx
import { useQueryState, parseAsInteger, parseAsString } from 'nuqs';

function ProductList() {
  const [page, setPage] = useQueryState('page', parseAsInteger.withDefault(1));
  const [sort, setSort] = useQueryState('sort', parseAsString.withDefault('newest'));

  // setPage(2) updates the URL to ?page=2&sort=newest
  // Typing is fully inferred: page is number, sort is string
}
```

---

## Form State

### React Hook Form

React Hook Form uses uncontrolled inputs by default, which means the DOM owns the input
values. This avoids re-rendering the entire form on every keystroke.

```tsx
import { useForm } from 'react-hook-form';

interface LoginForm {
  email: string;
  password: string;
}

function Login() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginForm>();

  const onSubmit = async (data: LoginForm) => {
    await api.login(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        {...register('email', {
          required: 'Email is required',
          pattern: {
            value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            message: 'Invalid email address',
          },
        })}
      />
      {errors.email && <span>{errors.email.message}</span>}

      <input
        type="password"
        {...register('password', {
          required: 'Password is required',
          minLength: { value: 8, message: 'Minimum 8 characters' },
        })}
      />
      {errors.password && <span>{errors.password.message}</span>}

      <button type="submit" disabled={isSubmitting}>
        Log In
      </button>
    </form>
  );
}
```

### Controlled Fields with `control`

For complex inputs (date pickers, rich text, custom selects) that cannot use `register`,
use `Controller`:

```tsx
import { useForm, Controller } from 'react-hook-form';
import { DatePicker } from '@/components/DatePicker';

function EventForm() {
  const { control, handleSubmit } = useForm<EventFormData>();

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Controller
        name="startDate"
        control={control}
        rules={{ required: 'Start date is required' }}
        render={({ field, fieldState }) => (
          <DatePicker
            value={field.value}
            onChange={field.onChange}
            error={fieldState.error?.message}
          />
        )}
      />
    </form>
  );
}
```

### Zod Integration

Zod schemas are the single source of truth for validation. The `@hookform/resolvers` package
bridges Zod and React Hook Form.

```tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const SignupSchema = z
  .object({
    email: z.string().email('Invalid email'),
    password: z.string().min(8, 'Minimum 8 characters'),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type SignupForm = z.infer<typeof SignupSchema>;

function Signup() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SignupForm>({
    resolver: zodResolver(SignupSchema),
  });

  return (
    <form onSubmit={handleSubmit((data) => console.log(data))}>
      <input {...register('email')} />
      {errors.email && <span>{errors.email.message}</span>}

      <input type="password" {...register('password')} />
      {errors.password && <span>{errors.password.message}</span>}

      <input type="password" {...register('confirmPassword')} />
      {errors.confirmPassword && <span>{errors.confirmPassword.message}</span>}

      <button type="submit">Sign Up</button>
    </form>
  );
}
```

### Performance Advantages

| Approach | Re-renders on keystroke | Re-renders on submit |
|----------|----------------------|---------------------|
| Controlled (useState per field) | Every field re-renders parent | 1 |
| React Hook Form (uncontrolled) | 0 (DOM handles input) | 1 |
| React Hook Form (watched field) | Only watched component | 1 |

React Hook Form isolates re-renders using a subscription model internally. Only components
that `watch` a specific field re-render when that field changes.

```tsx
// Only this component re-renders when 'email' changes
function EmailPreview() {
  const email = useWatch({ name: 'email' });
  return <p>Preview: {email}</p>;
}
```

---

## State Colocation

State colocation is a principle, not a library. The idea: keep state as close to where it is
used as possible.

### The Anti-Pattern: Everything at the Top

```tsx
// DO NOT DO THIS
function App() {
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  // 30 more pieces of state...

  return (
    <Dashboard
      modalOpen={modalOpen}
      setModalOpen={setModalOpen}
      selectedRow={selectedRow}
      setSelectedRow={setSelectedRow}
      tooltipVisible={tooltipVisible}
      // ...
    />
  );
}
```

### Colocation in Practice

```tsx
// State lives where it is used
function Dashboard() {
  return (
    <>
      <SearchBar /> {/* owns searchQuery */}
      <DataTable /> {/* owns selectedRow, tooltipVisible */}
      <Modal />     {/* owns modalOpen */}
    </>
  );
}

function DataTable() {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);

  return (
    <table>
      {rows.map((row) => (
        <Row
          key={row.id}
          row={row}
          isSelected={row.id === selectedRow}
          onSelect={() => setSelectedRow(row.id)}
        />
      ))}
    </table>
  );
}
```

### Lifting State Up vs Composition

When two siblings need the same state, lift it to their common parent -- but *only* that
state. Do not lift unrelated state along with it.

```tsx
// Two siblings need `selectedId`
function Page() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  return (
    <>
      <Sidebar selectedId={selectedId} onSelect={setSelectedId} />
      <Detail selectedId={selectedId} />
    </>
  );
}
```

If you find yourself lifting state multiple levels, consider:

1. **Composition** -- pass children as props to avoid prop drilling.
2. **Context** -- but only if the state is needed by many deeply nested components.
3. **External store** (Zustand/Jotai) -- if it crosses route boundaries or persists across navigations.

### The Colocation Ladder

```
1. useState inside component           (closest)
2. Lift to parent component
3. Composition (children / render props)
4. Context (subtree injection)
5. External store (Zustand / Jotai)
6. URL state (survives navigation)
7. Server state (TanStack Query)       (furthest -- source of truth is the server)
```

Move state down this ladder only as requirements demand.

---

## Common Interview Questions

### Q1: How do you decide between Context, Zustand, and Redux for global state?

**A:** Context is best for low-frequency values like theme or locale -- it has no selector
mechanism so every consumer re-renders on any change. Zustand is the modern default for
client-side global state: zero boilerplate, selector-based subscriptions, excellent
performance. Redux (via RTK) is justified on large teams that need strict conventions,
mature middleware, and time-travel debugging, or when RTK Query is already handling server
state. For most apps in 2025/2026, Zustand is the pragmatic choice.

### Q2: What is the stale-while-revalidate pattern and why does TanStack Query use it?

**A:** Stale-while-revalidate serves cached (potentially stale) data immediately while
silently refetching in the background. This gives users instant UI instead of loading
spinners, while still converging to fresh data. TanStack Query implements this via
`staleTime` (how long data is considered fresh) and automatic background refetches on
window focus, reconnect, or interval.

### Q3: When should you use useReducer instead of useState?

**A:** Use `useReducer` when a single user action updates multiple related fields (coupled
transitions), when the next state depends on the previous in complex ways, when you want
testable pure-function transitions, or when you need a stable `dispatch` reference to
pass to memoized children. Use `useState` for simple, independent pieces of state.

### Q4: How do you prevent Context from causing unnecessary re-renders?

**A:** Three techniques: (1) Split a mega-context into smaller, focused contexts so
consumers only subscribe to what they need. (2) Memoize the context value with `useMemo` to
avoid new object references on every provider render. (3) Separate state and dispatch into
two contexts so components that only dispatch actions do not re-render on state changes.

### Q5: What is the difference between server state and client state?

**A:** Server state is data whose source of truth is a remote server -- it is asynchronous,
can be stale, and has shared ownership (other users/systems can modify it). Client state
is data created and owned by the browser session (UI state, form state, user preferences).
Conflating the two leads to bugs: treating server data as client state means stale caches,
no background refetching, and manual cache invalidation. Use TanStack Query / SWR for
server state and useState / Zustand / Jotai for client state.

### Q6: How does Zustand avoid re-renders compared to Context?

**A:** Zustand uses an external store with a `subscribe` + `selector` pattern. When state
changes, Zustand runs each component's selector against the new state and compares the
result (shallow equality by default) to the previous result. If the selected slice has not
changed, the component does not re-render. Context has no such mechanism -- every consumer
re-renders when the provider value changes.

### Q7: Why use React Hook Form over controlled inputs?

**A:** Controlled inputs re-render the form component on every keystroke because each
change flows through React state. React Hook Form uses uncontrolled inputs where the DOM
holds the values, resulting in zero re-renders during typing. It only triggers re-renders
on validation events or submission. For large forms (10+ fields), this eliminates the
cumulative re-render cost that makes controlled forms feel sluggish.

### Q8: What is state colocation and why does it matter?

**A:** State colocation means keeping state as close to where it is consumed as possible.
It matters because state at the wrong level causes either unnecessary prop drilling (too
high) or duplicated state (too low). Start with `useState` in the component, lift only
when a sibling needs it, use Context or an external store only when the state crosses
deep or disparate parts of the tree. This keeps components self-contained and minimizes
the blast radius of state changes.
