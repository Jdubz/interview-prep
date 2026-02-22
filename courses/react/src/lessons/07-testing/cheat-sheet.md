# Lesson 07 Cheat Sheet: Testing React

## RTL Query Priority

| Priority | Query | Use When | Example |
|----------|-------|----------|---------|
| 1 | `getByRole` | Interactive elements, headings, regions | `screen.getByRole('button', { name: /save/i })` |
| 2 | `getByLabelText` | Form fields with `<label>` | `screen.getByLabelText(/email address/i)` |
| 3 | `getByPlaceholderText` | Input with placeholder, no label | `screen.getByPlaceholderText(/search/i)` |
| 4 | `getByText` | Non-interactive text content | `screen.getByText(/no results/i)` |
| 5 | `getByDisplayValue` | Input with current value | `screen.getByDisplayValue('alice@test.com')` |
| 6 | `getByAltText` | Images | `screen.getByAltText(/user avatar/i)` |
| 7 | `getByTitle` | Elements with `title` attr | `screen.getByTitle(/close/i)` |
| 8 | `getByTestId` | Last resort, nothing else works | `screen.getByTestId('chart-canvas')` |

**Variants**: `getBy` (throws), `queryBy` (returns null), `findBy` (async), plus `All` versions of each.

---

## Common Assertions

```tsx
// Presence
expect(element).toBeInTheDocument();
expect(screen.queryByText(/gone/i)).not.toBeInTheDocument();

// Visibility
expect(element).toBeVisible();
expect(element).not.toBeVisible(); // hidden, display:none, etc.

// Text content
expect(element).toHaveTextContent(/welcome/i);
expect(element).toHaveTextContent('Exact text');

// Form state
expect(input).toHaveValue('alice@test.com');
expect(input).toHaveDisplayValue('alice@test.com');
expect(checkbox).toBeChecked();
expect(button).toBeDisabled();
expect(button).toBeEnabled();
expect(input).toBeRequired();
expect(input).toBeInvalid();
expect(input).toHaveAttribute('type', 'email');

// Accessibility
expect(element).toHaveAccessibleName(/submit form/i);
expect(element).toHaveAccessibleDescription(/sends your data/i);
expect(element).toHaveRole('button');

// CSS
expect(element).toHaveClass('active');
expect(element).toHaveStyle({ display: 'flex' });

// Focus
expect(input).toHaveFocus();
```

---

## MSW Handler Templates

```tsx
import { http, HttpResponse, delay } from 'msw';

// GET — return list
http.get('/api/items', () => {
  return HttpResponse.json([{ id: '1', name: 'Item' }]);
});

// GET — with params
http.get('/api/items/:id', ({ params }) => {
  return HttpResponse.json({ id: params.id, name: 'Item' });
});

// GET — with query string
http.get('/api/search', ({ request }) => {
  const url = new URL(request.url);
  const q = url.searchParams.get('q');
  return HttpResponse.json({ results: [], query: q });
});

// POST — read body, return 201
http.post('/api/items', async ({ request }) => {
  const body = await request.json();
  return HttpResponse.json({ id: '2', ...body }, { status: 201 });
});

// PUT
http.put('/api/items/:id', async ({ params, request }) => {
  const body = await request.json();
  return HttpResponse.json({ id: params.id, ...body });
});

// DELETE — 204 no content
http.delete('/api/items/:id', () => {
  return new HttpResponse(null, { status: 204 });
});

// Error response
http.get('/api/items', () => {
  return HttpResponse.json({ message: 'Server Error' }, { status: 500 });
});

// Delayed response
http.get('/api/items', async () => {
  await delay(2000);
  return HttpResponse.json([]);
});

// Network error
http.get('/api/items', () => {
  return HttpResponse.error();
});
```

---

## Custom Render Wrapper Template

```tsx
// src/test/render.tsx
import { render, type RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, type MemoryRouterProps } from 'react-router-dom';
import type { ReactElement, ReactNode } from 'react';

interface Options extends Omit<RenderOptions, 'wrapper'> {
  routerProps?: MemoryRouterProps;
  queryClient?: QueryClient;
}

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: Infinity },
      mutations: { retry: false },
    },
  });
}

export function renderWithProviders(
  ui: ReactElement,
  { routerProps, queryClient = createTestQueryClient(), ...rest }: Options = {}
) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter {...routerProps}>{children}</MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { ...render(ui, { wrapper: Wrapper, ...rest }), queryClient };
}
```

---

## Test Structure Template

```tsx
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '@/test/render';
import { server } from '@/test/mocks/server';
import { http, HttpResponse } from 'msw';
import { MyComponent } from './MyComponent';

describe('MyComponent', () => {
  describe('when data loads successfully', () => {
    it('displays the data', async () => {
      renderWithProviders(<MyComponent />);
      expect(await screen.findByText(/expected text/i)).toBeInTheDocument();
    });
  });

  describe('when the API fails', () => {
    it('shows an error message', async () => {
      server.use(
        http.get('/api/endpoint', () =>
          HttpResponse.json({ message: 'fail' }, { status: 500 })
        )
      );
      renderWithProviders(<MyComponent />);
      expect(await screen.findByRole('alert')).toHaveTextContent(/error/i);
    });
  });

  describe('user interactions', () => {
    it('does something on click', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MyComponent />);
      await user.click(screen.getByRole('button', { name: /action/i }));
      await waitFor(() => {
        expect(screen.getByText(/result/i)).toBeInTheDocument();
      });
    });
  });
});
```

---

## userEvent Cheat Sheet

```tsx
const user = userEvent.setup();

// Click
await user.click(element);
await user.dblClick(element);
await user.tripleClick(element);    // Select full line of text

// Typing
await user.type(input, 'hello');    // Types one char at a time
await user.clear(input);            // Clears input value
await user.type(input, '{Enter}');  // Special keys in braces

// Keyboard (no target needed — uses focused element)
await user.keyboard('hello');
await user.keyboard('{Enter}');
await user.keyboard('{Shift>}A{/Shift}');  // Hold Shift, type A
await user.keyboard('{Control>}a{/Control}'); // Ctrl+A (select all)
await user.keyboard('{ArrowDown}');
await user.keyboard('{Escape}');
await user.keyboard('{Backspace}');

// Tab navigation
await user.tab();                   // Tab forward
await user.tab({ shift: true });    // Shift+Tab backward

// Select / dropdown
await user.selectOptions(select, 'value');
await user.selectOptions(select, ['val1', 'val2']); // Multi-select
await user.deselectOptions(select, 'value');

// Pointer
await user.hover(element);
await user.unhover(element);
await user.pointer('[MouseLeft]');  // Low-level pointer API

// Clipboard
await user.copy();
await user.cut();
await user.paste('pasted text');

// File upload
const file = new File(['content'], 'file.txt', { type: 'text/plain' });
await user.upload(fileInput, file);
await user.upload(fileInput, [file1, file2]); // Multiple files
```

---

## waitFor / findBy Patterns

```tsx
// findBy = getBy + waitFor (preferred for single element)
const heading = await screen.findByRole('heading', { name: /title/i });

// waitFor — for complex assertions or multiple checks
await waitFor(() => {
  expect(screen.getByText(/saved/i)).toBeInTheDocument();
  expect(screen.queryByText(/saving/i)).not.toBeInTheDocument();
});

// waitForElementToBeRemoved — loading spinners
await waitForElementToBeRemoved(() => screen.queryByText(/loading/i));

// Custom timeout
await screen.findByText(/slow data/i, {}, { timeout: 5000 });
await waitFor(() => expect(el).toBeVisible(), { timeout: 5000 });

// Custom interval (default is 50ms)
await waitFor(() => expect(el).toBeVisible(), { interval: 100 });
```

---

## vi.mock Patterns

```tsx
// Mock entire module
vi.mock('@/lib/analytics', () => ({
  track: vi.fn(),
  identify: vi.fn(),
}));

// Partial mock (keep real implementations)
vi.mock('@/utils/format', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/utils/format')>();
  return { ...actual, formatDate: vi.fn(() => '2025-01-01') };
});

// Mock with hoisted variable (for per-test control)
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return { ...actual, useNavigate: () => mockNavigate };
});

// Spy on named export
import * as mod from '@/hooks/useAuth';
vi.spyOn(mod, 'useAuth').mockReturnValue({ user: null, isAuthenticated: false });

// Mock default export
vi.mock('@/components/Chart', () => ({
  default: () => <div data-testid="mock-chart" />,
}));

// Reset between tests
afterEach(() => {
  vi.restoreAllMocks();  // Restores spies to original
  // or
  vi.resetAllMocks();    // Resets call history + implementations
  // or
  vi.clearAllMocks();    // Resets call history only
});
```
