# Lesson 07: Testing React

## Testing Philosophy

### Test Behavior, Not Implementation

The single most important principle in React testing: **your tests should resemble the way your software is used**. Users don't know about state variables, effect hooks, or re-render cycles. They see buttons, text, and forms. Test from that perspective.

```tsx
// BAD: testing implementation details
it('sets isOpen state to true when clicked', () => {
  const { result } = renderHook(() => useState(false));
  act(() => result.current[1](true));
  expect(result.current[0]).toBe(true);
});

// GOOD: testing behavior
it('opens the dropdown when the trigger is clicked', async () => {
  const user = userEvent.setup();
  render(<Dropdown items={['Apple', 'Banana']} />);

  await user.click(screen.getByRole('button', { name: /select fruit/i }));

  expect(screen.getByRole('listbox')).toBeInTheDocument();
  expect(screen.getByRole('option', { name: /apple/i })).toBeInTheDocument();
});
```

Implementation detail tests break when you refactor. Behavior tests break when the behavior changes. That is exactly what you want.

### The Testing Trophy

Kent C. Dodds' testing trophy (not pyramid) prioritizes:

```
        ╱  E2E  ╲           Few — high confidence, slow, expensive
       ╱──────────╲
      ╱ Integration ╲       Most tests here — best confidence/cost ratio
     ╱────────────────╲
    ╱    Unit Tests     ╲    Some — pure logic, utilities, hooks
   ╱──────────────────────╲
  ╱      Static Analysis    ╲  TypeScript, ESLint — cheapest, always on
 ╱────────────────────────────╲
```

**Integration tests** are the sweet spot for React. They render a component with its real children, real hooks, and mocked network calls. They give you the highest confidence-to-cost ratio because they exercise how components actually compose together.

### Core Principles

1. **The more your tests resemble the way your software is used, the more confidence they give you.**
2. **Write tests. Not too many. Mostly integration.**
3. **Avoid testing implementation details** — no querying by class name, no asserting internal state.
4. **If it's hard to test, the component design is probably wrong.** Tests act as a design pressure.
5. **One assertion per behavioral claim**, not one assertion per test. A test that verifies "user can submit a form" might assert several things.

---

## React Testing Library (RTL)

### Query Priority

RTL deliberately makes you use accessible queries. The priority order reflects what users and assistive technology actually interact with:

| Priority | Query | Use When |
|----------|-------|----------|
| 1 | `getByRole` | Almost always. Buttons, headings, textboxes, etc. |
| 2 | `getByLabelText` | Form fields with proper labels |
| 3 | `getByPlaceholderText` | When label is absent (not ideal) |
| 4 | `getByText` | Non-interactive elements, paragraphs, spans |
| 5 | `getByDisplayValue` | Filled-in form inputs |
| 6 | `getByAltText` | Images |
| 7 | `getByTitle` | Title attribute (less accessible) |
| 8 | `getByTestId` | Last resort. Not visible to users. |

```tsx
// Prefer role-based queries — they enforce accessibility
screen.getByRole('button', { name: /submit/i });
screen.getByRole('heading', { level: 2 });
screen.getByRole('textbox', { name: /email/i });
screen.getByRole('checkbox', { name: /agree to terms/i });
screen.getByRole('combobox', { name: /country/i });

// Good for form fields with visible labels
screen.getByLabelText(/email address/i);

// Acceptable for non-interactive content
screen.getByText(/no results found/i);

// Last resort — use data-testid only when no accessible query works
screen.getByTestId('complex-svg-chart');
```

### The `screen` Object

Always use `screen` instead of destructuring from `render()`. It makes tests more readable and ensures you are querying the full document.

```tsx
// AVOID: destructuring queries
const { getByRole, getByText } = render(<MyComponent />);

// PREFER: screen
render(<MyComponent />);
const button = screen.getByRole('button', { name: /save/i });
```

The only exception is when you need `container` for something truly not queryable via accessible means, or `rerender` / `unmount`:

```tsx
const { rerender, unmount } = render(<Counter count={0} />);
rerender(<Counter count={5} />);
expect(screen.getByText('5')).toBeInTheDocument();
unmount();
```

### `within()` for Scoped Queries

When the same role or text appears multiple times, scope your queries:

```tsx
import { within } from '@testing-library/react';

render(<Dashboard />);

const sidebar = screen.getByRole('navigation', { name: /sidebar/i });
const mainContent = screen.getByRole('main');

// Query within specific regions
const sidebarLinks = within(sidebar).getAllByRole('link');
const mainHeading = within(mainContent).getByRole('heading', { level: 1 });
```

This is especially useful for tables, lists, and multi-section layouts:

```tsx
render(<UserTable users={mockUsers} />);

const rows = screen.getAllByRole('row');
// Skip header row
const firstDataRow = rows[1];

expect(within(firstDataRow).getByText('jane@example.com')).toBeInTheDocument();
expect(within(firstDataRow).getByRole('button', { name: /edit/i })).toBeEnabled();
```

---

## User Events

### `userEvent` vs `fireEvent`

`fireEvent` dispatches a single DOM event. `userEvent` simulates the full interaction sequence a real user triggers — including focus, pointer events, keyboard events, and input events in the correct order.

```tsx
import userEvent from '@testing-library/user-event';

// AVOID: fireEvent — skips intermediate events
fireEvent.click(button);
fireEvent.change(input, { target: { value: 'hello' } });

// PREFER: userEvent — fires the full realistic event chain
const user = userEvent.setup();
await user.click(button);  // pointerdown, mousedown, pointerup, mouseup, click
await user.type(input, 'hello');  // focus, keydown, keypress, input, keyup per char
```

**Always call `userEvent.setup()` before rendering.** This creates a user instance with its own internal state (pointer position, keyboard state, clipboard).

```tsx
it('handles multi-step interaction', async () => {
  const user = userEvent.setup();
  render(<TextEditor />);

  const editor = screen.getByRole('textbox');
  await user.click(editor);
  await user.keyboard('Hello{Enter}World');

  expect(editor).toHaveValue('Hello\nWorld');
});
```

### Async User Interactions

All `userEvent` methods return promises. This is required because they may trigger state updates:

```tsx
it('shows autocomplete suggestions after typing', async () => {
  const user = userEvent.setup();
  render(<Autocomplete options={countries} />);

  const input = screen.getByRole('combobox');
  await user.type(input, 'Uni');

  // Options appear after debounce + state update
  const options = await screen.findAllByRole('option');
  expect(options).toHaveLength(2); // United States, United Kingdom
});
```

### Keyboard Navigation Testing

Testing keyboard interactions is critical for accessibility and a frequent interview topic:

```tsx
it('supports full keyboard navigation in dropdown', async () => {
  const user = userEvent.setup();
  render(<Select options={['Red', 'Green', 'Blue']} label="Color" />);

  // Tab to focus the trigger
  await user.tab();
  expect(screen.getByRole('combobox', { name: /color/i })).toHaveFocus();

  // Open with Enter
  await user.keyboard('{Enter}');
  expect(screen.getByRole('listbox')).toBeInTheDocument();

  // Navigate options with arrow keys
  await user.keyboard('{ArrowDown}');
  expect(screen.getByRole('option', { name: /red/i })).toHaveAttribute(
    'aria-selected',
    'true'
  );

  await user.keyboard('{ArrowDown}');
  expect(screen.getByRole('option', { name: /green/i })).toHaveAttribute(
    'aria-selected',
    'true'
  );

  // Select with Enter
  await user.keyboard('{Enter}');
  expect(screen.getByRole('combobox')).toHaveTextContent('Green');
  expect(screen.queryByRole('listbox')).not.toBeInTheDocument();

  // Escape closes without selecting
  await user.keyboard('{Enter}');
  await user.keyboard('{Escape}');
  expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  expect(screen.getByRole('combobox')).toHaveTextContent('Green');
});

it('supports typeahead in listbox', async () => {
  const user = userEvent.setup();
  render(<Select options={['Apple', 'Banana', 'Avocado']} label="Fruit" />);

  await user.click(screen.getByRole('combobox'));
  // Typing "av" should jump to "Avocado"
  await user.keyboard('av');
  expect(screen.getByRole('option', { name: /avocado/i })).toHaveAttribute(
    'aria-selected',
    'true'
  );
});
```

---

## Async Testing

### `waitFor`

Use `waitFor` when you need to wait for an assertion to pass. It retries the callback on an interval until it passes or times out.

```tsx
it('shows success message after save', async () => {
  const user = userEvent.setup();
  render(<ProfileForm />);

  await user.type(screen.getByLabelText(/name/i), 'Jane');
  await user.click(screen.getByRole('button', { name: /save/i }));

  await waitFor(() => {
    expect(screen.getByRole('alert')).toHaveTextContent(/saved successfully/i);
  });
});
```

**Rules for `waitFor`:**

1. Put only the assertion inside `waitFor`, not the action:

```tsx
// BAD: side effects inside waitFor get called multiple times
await waitFor(() => {
  fireEvent.click(button);
  expect(result).toBeInTheDocument();
});

// GOOD: action outside, assertion inside
await user.click(button);
await waitFor(() => {
  expect(result).toBeInTheDocument();
});
```

2. Prefer `findBy` queries over `waitFor` + `getBy`:

```tsx
// Verbose
await waitFor(() => {
  expect(screen.getByText(/loaded/i)).toBeInTheDocument();
});

// Concise — findBy = getBy + waitFor
expect(await screen.findByText(/loaded/i)).toBeInTheDocument();
```

### `findBy` Queries

`findBy` queries are sugar for `waitFor(() => getBy(...))`. They return a promise that resolves when the element appears. Use them for anything async:

```tsx
it('fetches and displays user data', async () => {
  render(<UserProfile userId="123" />);

  // Loading state is immediate
  expect(screen.getByText(/loading/i)).toBeInTheDocument();

  // Data appears asynchronously
  const heading = await screen.findByRole('heading', { name: /jane doe/i });
  expect(heading).toBeInTheDocument();

  // Loading state is gone
  expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
});
```

### `act()` — When It Is Needed and When It Is Not

`act()` ensures state updates are flushed and the DOM is updated before you make assertions. RTL wraps `render`, `userEvent`, `fireEvent`, `waitFor`, and `findBy` in `act()` for you.

**You almost never need to call `act()` directly.** If you see the "not wrapped in act" warning, the solution is usually:

1. Use `await findBy...` or `await waitFor(...)` to wait for the async update.
2. Await the user event call.
3. Fix the component (maybe a missing cleanup in useEffect).

```tsx
// WARNING-PRODUCING CODE:
it('updates on timer', () => {
  render(<AutoRefresh />);
  jest.advanceTimersByTime(5000);
  // Warning: An update was not wrapped in act(...)
  expect(screen.getByText(/refreshed/i)).toBeInTheDocument();
});

// FIXED: wrap timer advancement in act
it('updates on timer', async () => {
  vi.useFakeTimers();
  render(<AutoRefresh />);

  await act(() => {
    vi.advanceTimersByTime(5000);
  });

  expect(screen.getByText(/refreshed/i)).toBeInTheDocument();
  vi.useRealTimers();
});
```

The legitimate use cases for manual `act()`:

- Advancing fake timers (`vi.advanceTimersByTime`, `vi.runAllTimers`)
- Manually resolving promises in tests
- Testing `renderHook` results where you trigger state updates directly

---

## Mocking Strategies

### MSW (Mock Service Worker)

MSW intercepts requests at the network level. Your components use their real `fetch`/`axios` calls; MSW intercepts them before they leave the process. This is the gold standard for API mocking because it tests the full request/response pipeline.

#### Server Setup

```tsx
// src/test/mocks/handlers.ts
import { http, HttpResponse, delay } from 'msw';
import type { User } from '@/types';

export const handlers = [
  http.get('/api/users', async () => {
    return HttpResponse.json<User[]>([
      { id: '1', name: 'Alice', email: 'alice@test.com' },
      { id: '2', name: 'Bob', email: 'bob@test.com' },
    ]);
  }),

  http.get('/api/users/:id', async ({ params }) => {
    const { id } = params;
    return HttpResponse.json<User>({
      id: id as string,
      name: 'Alice',
      email: 'alice@test.com',
    });
  }),

  http.post('/api/users', async ({ request }) => {
    const body = (await request.json()) as Partial<User>;
    return HttpResponse.json<User>(
      { id: '3', name: body.name!, email: body.email! },
      { status: 201 }
    );
  }),

  http.delete('/api/users/:id', () => {
    return new HttpResponse(null, { status: 204 });
  }),
];
```

```tsx
// src/test/mocks/server.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
```

```tsx
// src/test/setup.ts (vitest setup file)
import { server } from './mocks/server';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

Setting `onUnhandledRequest: 'error'` is critical. It catches unhandled API calls — if your component fetches an endpoint you did not mock, the test fails immediately with a clear error instead of silently hanging.

#### Per-Test Overrides

Override handlers inside a specific test to simulate errors, edge cases, or different data:

```tsx
import { http, HttpResponse, delay } from 'msw';
import { server } from '@/test/mocks/server';

it('shows error state when API fails', async () => {
  server.use(
    http.get('/api/users', () => {
      return HttpResponse.json(
        { message: 'Internal Server Error' },
        { status: 500 }
      );
    })
  );

  render(<UserList />);

  expect(await screen.findByRole('alert')).toHaveTextContent(
    /something went wrong/i
  );
});

it('shows loading skeleton during slow response', async () => {
  server.use(
    http.get('/api/users', async () => {
      await delay(2000);
      return HttpResponse.json([]);
    })
  );

  render(<UserList />);
  expect(screen.getByTestId('skeleton-loader')).toBeInTheDocument();
});

it('handles empty state', async () => {
  server.use(
    http.get('/api/users', () => {
      return HttpResponse.json([]);
    })
  );

  render(<UserList />);
  expect(await screen.findByText(/no users found/i)).toBeInTheDocument();
});
```

### `vi.mock` / `jest.mock`: Module Mocking

Use module mocking when MSW is not appropriate: routing, analytics, non-HTTP side effects, or browser APIs.

#### Basic Module Mock

```tsx
// Mock an entire module
vi.mock('@/services/analytics', () => ({
  trackEvent: vi.fn(),
  trackPageView: vi.fn(),
  identify: vi.fn(),
}));

import { trackEvent } from '@/services/analytics';

it('tracks button click', async () => {
  const user = userEvent.setup();
  render(<FeatureCard feature={mockFeature} />);

  await user.click(screen.getByRole('button', { name: /learn more/i }));

  expect(trackEvent).toHaveBeenCalledWith('feature_click', {
    featureId: mockFeature.id,
  });
});
```

#### Partial Mock (Keep Real Implementations)

```tsx
vi.mock('@/utils/date', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/utils/date')>();
  return {
    ...actual,
    // Override only what you need
    now: vi.fn(() => new Date('2025-06-15T12:00:00Z')),
  };
});
```

#### Mocking with Factory Pattern

```tsx
// For mocks that need per-test configuration
const mockNavigate = vi.fn();

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

afterEach(() => {
  mockNavigate.mockReset();
});

it('navigates to detail page on row click', async () => {
  const user = userEvent.setup();
  render(<UserTable users={mockUsers} />);

  await user.click(screen.getByRole('row', { name: /alice/i }));

  expect(mockNavigate).toHaveBeenCalledWith('/users/1');
});
```

### Component Mocking: Child Components and Hooks

Sometimes you need to isolate a component from heavy children or custom hooks.

#### Mocking a Child Component

```tsx
// Mock a heavy child that has its own tests
vi.mock('@/components/Chart', () => ({
  Chart: ({ data, title }: { data: unknown[]; title: string }) => (
    <div data-testid="mock-chart" aria-label={title}>
      Chart: {JSON.stringify(data).slice(0, 50)}
    </div>
  ),
}));

it('passes correct data to chart', async () => {
  render(<Dashboard />);

  const chart = await screen.findByTestId('mock-chart');
  expect(chart).toHaveAttribute('aria-label', 'Revenue Over Time');
});
```

#### Mocking a Custom Hook

```tsx
import * as useAuthModule from '@/hooks/useAuth';

it('shows admin controls for admin users', () => {
  vi.spyOn(useAuthModule, 'useAuth').mockReturnValue({
    user: { id: '1', name: 'Admin', role: 'admin' },
    isAuthenticated: true,
    login: vi.fn(),
    logout: vi.fn(),
  });

  render(<Settings />);

  expect(screen.getByRole('button', { name: /manage users/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /system config/i })).toBeInTheDocument();
});

it('hides admin controls for regular users', () => {
  vi.spyOn(useAuthModule, 'useAuth').mockReturnValue({
    user: { id: '2', name: 'User', role: 'viewer' },
    isAuthenticated: true,
    login: vi.fn(),
    logout: vi.fn(),
  });

  render(<Settings />);

  expect(screen.queryByRole('button', { name: /manage users/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /system config/i })).not.toBeInTheDocument();
});
```

---

## Testing Patterns

### Custom Render Wrapper

Every non-trivial React app wraps components in providers. Create a custom render that includes them all so tests do not break when you add a new global provider:

```tsx
// src/test/render.tsx
import { render, type RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, type MemoryRouterProps } from 'react-router-dom';
import { ThemeProvider } from '@/providers/ThemeProvider';
import { AuthProvider } from '@/providers/AuthProvider';
import type { ReactElement, ReactNode } from 'react';

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  routerProps?: MemoryRouterProps;
  queryClient?: QueryClient;
  initialUser?: User | null;
}

function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,       // Fail fast in tests
        gcTime: Infinity,   // Don't garbage collect during test
      },
      mutations: {
        retry: false,
      },
    },
    logger: {
      log: console.log,
      warn: console.warn,
      error: () => {},      // Silence expected errors in tests
    },
  });
}

export function renderWithProviders(
  ui: ReactElement,
  {
    routerProps = { initialEntries: ['/'] },
    queryClient = createTestQueryClient(),
    initialUser = null,
    ...renderOptions
  }: CustomRenderOptions = {}
) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AuthProvider initialUser={initialUser}>
          <ThemeProvider>
            <MemoryRouter {...routerProps}>{children}</MemoryRouter>
          </ThemeProvider>
        </AuthProvider>
      </QueryClientProvider>
    );
  }

  return {
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
    queryClient, // Expose for cache assertions
  };
}
```

Usage:

```tsx
import { renderWithProviders } from '@/test/render';

it('shows dashboard for authenticated user', async () => {
  renderWithProviders(<Dashboard />, {
    initialUser: { id: '1', name: 'Alice', role: 'admin' },
    routerProps: { initialEntries: ['/dashboard'] },
  });

  expect(await screen.findByRole('heading', { name: /dashboard/i })).toBeInTheDocument();
});
```

### Testing Forms

Forms are the most common interview testing scenario. Test the full user flow:

```tsx
it('validates and submits registration form', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  render(<RegistrationForm onSubmit={onSubmit} />);

  // Submit empty form -> validation errors
  await user.click(screen.getByRole('button', { name: /register/i }));

  expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
  expect(screen.getByText(/password is required/i)).toBeInTheDocument();
  expect(onSubmit).not.toHaveBeenCalled();

  // Fill in invalid email
  await user.type(screen.getByLabelText(/email/i), 'not-an-email');
  await user.tab(); // Trigger blur validation

  expect(await screen.findByText(/invalid email/i)).toBeInTheDocument();

  // Fix email, fill password
  await user.clear(screen.getByLabelText(/email/i));
  await user.type(screen.getByLabelText(/email/i), 'alice@example.com');
  await user.type(screen.getByLabelText(/^password$/i), 'Str0ng!Pass');
  await user.type(screen.getByLabelText(/confirm password/i), 'Str0ng!Pass');

  // Validation errors should be gone
  expect(screen.queryByText(/invalid email/i)).not.toBeInTheDocument();

  // Submit valid form
  await user.click(screen.getByRole('button', { name: /register/i }));

  await waitFor(() => {
    expect(onSubmit).toHaveBeenCalledWith({
      email: 'alice@example.com',
      password: 'Str0ng!Pass',
    });
  });
});

it('disables submit button while submitting', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn(
    () => new Promise((resolve) => setTimeout(resolve, 1000))
  );
  render(<RegistrationForm onSubmit={onSubmit} />);

  // Fill valid data
  await user.type(screen.getByLabelText(/email/i), 'test@test.com');
  await user.type(screen.getByLabelText(/^password$/i), 'Str0ng!Pass');
  await user.type(screen.getByLabelText(/confirm password/i), 'Str0ng!Pass');
  await user.click(screen.getByRole('button', { name: /register/i }));

  // Button disabled during submission
  expect(screen.getByRole('button', { name: /registering/i })).toBeDisabled();
});
```

### Testing Async Flows (Loading, Error, Data)

The canonical pattern for testing data-fetching components:

```tsx
describe('UserProfile', () => {
  it('shows loading state then user data', async () => {
    render(<UserProfile userId="1" />);

    // Assert loading state
    expect(screen.getByRole('status')).toHaveTextContent(/loading/i);

    // Wait for data
    expect(
      await screen.findByRole('heading', { name: /alice/i })
    ).toBeInTheDocument();

    // Assert loading state is removed
    expect(screen.queryByRole('status')).not.toBeInTheDocument();

    // Assert full content
    expect(screen.getByText('alice@test.com')).toBeInTheDocument();
  });

  it('shows error state on failure', async () => {
    server.use(
      http.get('/api/users/:id', () => {
        return HttpResponse.json(
          { message: 'Not Found' },
          { status: 404 }
        );
      })
    );

    render(<UserProfile userId="999" />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /user not found/i
    );
  });

  it('refetches when userId changes', async () => {
    const { rerender } = render(<UserProfile userId="1" />);
    expect(await screen.findByText(/alice/i)).toBeInTheDocument();

    rerender(<UserProfile userId="2" />);
    expect(await screen.findByText(/bob/i)).toBeInTheDocument();
  });
});
```

### Testing Navigation

```tsx
import { renderWithProviders } from '@/test/render';

describe('Navigation', () => {
  it('navigates from list to detail page', async () => {
    const user = userEvent.setup();

    renderWithProviders(<App />, {
      routerProps: { initialEntries: ['/users'] },
    });

    // Wait for user list to load
    const aliceLink = await screen.findByRole('link', { name: /alice/i });
    await user.click(aliceLink);

    // Should be on detail page
    expect(
      await screen.findByRole('heading', { name: /alice/i })
    ).toBeInTheDocument();
    expect(screen.getByText('alice@test.com')).toBeInTheDocument();
  });

  it('shows 404 for unknown routes', async () => {
    renderWithProviders(<App />, {
      routerProps: { initialEntries: ['/nonexistent'] },
    });

    expect(screen.getByRole('heading', { name: /not found/i })).toBeInTheDocument();
  });

  it('redirects unauthenticated users to login', async () => {
    renderWithProviders(<App />, {
      initialUser: null,
      routerProps: { initialEntries: ['/dashboard'] },
    });

    expect(
      await screen.findByRole('heading', { name: /sign in/i })
    ).toBeInTheDocument();
  });
});
```

---

## Snapshot Testing

### When Appropriate

Snapshots work well for small, stable, presentational components:

```tsx
it('renders icon button correctly', () => {
  const { container } = render(
    <IconButton icon="trash" label="Delete" variant="danger" />
  );
  expect(container.firstChild).toMatchSnapshot();
});
```

### When to Avoid

Avoid snapshots for:

- **Large components**: Snapshots become unreadable noise. Reviewers click "update snapshots" without reading.
- **Components with dynamic content**: IDs, dates, random values cause constant snapshot churn.
- **Anything where you care about behavior**: Snapshots test structure, not behavior. A button can be perfectly snapshotted and completely broken.

**Inline snapshots** are often better because the expected output lives in the test file:

```tsx
it('renders badge with correct class', () => {
  const { container } = render(<Badge status="active" />);
  expect(container.firstChild).toMatchInlineSnapshot(`
    <span
      class="badge badge--active"
    >
      Active
    </span>
  `);
});
```

---

## Accessibility Testing

### jest-axe

`jest-axe` runs axe-core accessibility checks against your rendered component:

```tsx
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

it('has no accessibility violations', async () => {
  const { container } = render(<LoginForm />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

Run this on every major component. It catches missing labels, poor contrast ratios, invalid ARIA attributes, and more.

### Role-Based Queries as Implicit A11y Tests

Every time you use `getByRole`, you are implicitly asserting accessibility. If `getByRole('button', { name: /submit/i })` fails, your component has an accessibility problem — the element is either not a button or it lacks an accessible name.

```tsx
// This test ALSO verifies:
// - The form has proper label associations
// - The button is a real button (not a div with onClick)
// - The heading uses proper semantic HTML
it('renders accessible form', () => {
  render(<ContactForm />);

  expect(screen.getByRole('heading', { name: /contact us/i })).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: /your name/i })).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: /message/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
});
```

---

## Test Organization

### Describe Blocks and Naming Conventions

Structure tests around user-facing behaviors, not component internals:

```tsx
describe('TransferForm', () => {
  // Group by feature or user story
  describe('validation', () => {
    it('requires a recipient', async () => { /* ... */ });
    it('requires a positive amount', async () => { /* ... */ });
    it('rejects amounts exceeding balance', async () => { /* ... */ });
  });

  describe('submission', () => {
    it('submits valid transfer and shows confirmation', async () => { /* ... */ });
    it('shows error when transfer fails', async () => { /* ... */ });
    it('prevents double submission', async () => { /* ... */ });
  });

  describe('accessibility', () => {
    it('has no axe violations', async () => { /* ... */ });
    it('supports keyboard-only submission', async () => { /* ... */ });
  });
});
```

**Naming convention**: Test names should read as sentences. `it('shows error when transfer fails')` is clear. `it('error state')` is not.

### Test Data Factories

Avoid repetitive mock data. Use factories:

```tsx
// src/test/factories.ts
import type { User, Post, Comment } from '@/types';

let idCounter = 0;

export function createUser(overrides: Partial<User> = {}): User {
  idCounter++;
  return {
    id: `user-${idCounter}`,
    name: `User ${idCounter}`,
    email: `user${idCounter}@test.com`,
    role: 'viewer',
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

export function createPost(overrides: Partial<Post> = {}): Post {
  idCounter++;
  return {
    id: `post-${idCounter}`,
    title: `Post ${idCounter}`,
    body: 'Lorem ipsum dolor sit amet.',
    authorId: `user-1`,
    publishedAt: new Date().toISOString(),
    tags: [],
    ...overrides,
  };
}

export function createComment(overrides: Partial<Comment> = {}): Comment {
  idCounter++;
  return {
    id: `comment-${idCounter}`,
    postId: `post-1`,
    authorId: `user-1`,
    body: 'Great post!',
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}
```

Usage in tests:

```tsx
it('renders user list', async () => {
  const users = [
    createUser({ name: 'Alice', role: 'admin' }),
    createUser({ name: 'Bob' }),
    createUser({ name: 'Charlie' }),
  ];

  server.use(
    http.get('/api/users', () => HttpResponse.json(users))
  );

  render(<UserList />);

  for (const user of users) {
    expect(await screen.findByText(user.name)).toBeInTheDocument();
  }
});
```

---

## Common Interview Questions

### 1. "What is the difference between `getBy`, `queryBy`, and `findBy`?"

| Variant | Returns | When Element Missing | Async |
|---------|---------|---------------------|-------|
| `getBy` | Element | Throws error | No |
| `queryBy` | Element \| `null` | Returns `null` | No |
| `findBy` | Promise\<Element\> | Rejects after timeout | Yes |

- Use `getBy` when the element should be there right now.
- Use `queryBy` when you are asserting the element is NOT there: `expect(screen.queryByText(/error/i)).not.toBeInTheDocument()`.
- Use `findBy` when the element appears asynchronously.

### 2. "When would you use `fireEvent` over `userEvent`?"

Almost never. `userEvent` should be your default because it simulates realistic user interactions (fires the full browser event sequence). The only edge case for `fireEvent` is when you need to dispatch a specific event that `userEvent` does not support, like a custom event, `scroll`, or `resize`:

```tsx
fireEvent.scroll(container, { target: { scrollTop: 500 } });
fireEvent(element, new CustomEvent('my-event', { detail: { key: 'val' } }));
```

### 3. "How do you test a component that uses `useContext`?"

Wrap it in the provider during render. The custom render wrapper pattern handles this automatically. For one-off cases:

```tsx
render(
  <ThemeContext.Provider value={{ theme: 'dark', toggleTheme: vi.fn() }}>
    <ThemeToggle />
  </ThemeContext.Provider>
);
```

### 4. "How do you handle the 'not wrapped in act' warning?"

The warning means a state update happened outside of React's test utilities. Solutions in order of preference:

1. **Await the user event** — you may have forgotten `await`.
2. **Use `findBy` or `waitFor`** — the update is async and you need to wait for it.
3. **Wrap timer advancement in `act()`** — if you are using fake timers.
4. **Check for missing cleanup** — an effect may be updating state after unmount. Add a cleanup return to `useEffect`.

Reaching for raw `act()` as a first response is usually wrong.

### 5. "Why MSW over mocking `fetch` or `axios` directly?"

- **MSW tests the real request pipeline.** Your interceptors, request transforms, and error handling all run. Mocking `fetch` skips all of that.
- **MSW is library-agnostic.** Switch from `fetch` to `axios` and your tests still pass.
- **MSW handlers are reusable** across tests and even in development (browser service worker).
- **MSW gives you error handling coverage.** You can simulate network errors, timeouts, and specific HTTP status codes easily.

### 6. "How do you test a custom hook?"

Use `renderHook` from `@testing-library/react`:

```tsx
import { renderHook, act } from '@testing-library/react';
import { useCounter } from '@/hooks/useCounter';

it('increments and decrements', () => {
  const { result } = renderHook(() => useCounter(0));

  expect(result.current.count).toBe(0);

  act(() => result.current.increment());
  expect(result.current.count).toBe(1);

  act(() => result.current.decrement());
  expect(result.current.count).toBe(0);
});
```

But prefer testing hooks through a component when possible. If the hook is always used by one component, test the component, not the hook in isolation.

### 7. "What is the role of `screen.debug()` in testing?"

`screen.debug()` prints the current DOM tree to the console. It is invaluable during development:

```tsx
screen.debug();                           // Full DOM
screen.debug(screen.getByRole('form'));   // Scoped to a subtree
screen.logTestingPlaygroundURL();          // Opens Testing Playground
```

Never leave `screen.debug()` in committed tests. It is a development-only tool.

### 8. "How do you avoid flaky async tests?"

1. **Never use arbitrary timeouts** (`setTimeout` in tests, hardcoded `waitFor` timeouts).
2. **Wait for specific DOM changes**, not time: `await screen.findByText(...)`.
3. **Use `waitForElementToBeRemoved`** for loading spinners instead of checking absence immediately.
4. **Ensure proper MSW handler setup** — unhandled requests cause hanging tests.
5. **Reset state between tests** — `server.resetHandlers()`, clearing query caches, resetting mocks.
6. **Use `vi.useFakeTimers` carefully** — always restore with `vi.useRealTimers` in cleanup.

```tsx
// FLAKY: race condition
expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();

// SOLID: explicitly wait for removal
await waitForElementToBeRemoved(() => screen.queryByText(/loading/i));
```
