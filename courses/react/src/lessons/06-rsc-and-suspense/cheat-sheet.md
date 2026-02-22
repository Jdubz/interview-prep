# Cheat Sheet: RSC & Suspense

## Server vs Client Component Rules

| Capability | Server Component | Client Component |
|---|---|---|
| `async/await` at component level | Yes | No |
| `useState`, `useEffect`, `useRef` | No | Yes |
| Event handlers (`onClick`, etc.) | No | Yes |
| Browser APIs (`window`, `localStorage`) | No | Yes (after hydration) |
| Direct DB/filesystem access | Yes | No |
| Import server-only modules | Yes | No |
| Import server components | Yes | No (accept as `children` only) |
| Ship JS to browser | No | Yes |
| Access `cookies()`, `headers()` | Yes | No |
| Pass server actions as props | Yes | Yes |

## "use client" / "use server" Quick Rules

```
"use client"
  - Must be the FIRST line of the file (before imports)
  - Marks a serialization boundary, NOT "runs only on client"
  - Client components still SSR (server-render for initial HTML)
  - All imports from this module are pulled into the client bundle
  - Only needed at boundary entry points, not every client file
  - Props from server parent must be serializable (no functions, no classes)

"use server"
  - Module-level: all exports become server actions
  - Inline: marks a single async function as a server action
  - Server actions can ONLY be async functions
  - Arguments and return values must be serializable
  - Can be passed to client components as props (the ONE exception to "no functions")
  - Bound arguments are encrypted (safe from client tampering)
```

## Next.js App Router File Conventions

| File | Purpose | Component Type |
|---|---|---|
| `layout.tsx` | Shared UI for segment and children. Persists across navigations. | Server (default) |
| `page.tsx` | Unique UI for a route. Makes the route publicly accessible. | Server (default) |
| `loading.tsx` | Loading UI (auto-wrapped in Suspense). | Server (default) |
| `error.tsx` | Error UI (auto-wrapped in Error Boundary). | **Must be Client** |
| `template.tsx` | Like layout but re-mounts on navigation. | Server (default) |
| `not-found.tsx` | 404 UI for the segment. | Server (default) |
| `global-error.tsx` | Root-level error boundary (catches layout errors). | **Must be Client** |
| `route.ts` | API endpoint (GET, POST, PUT, DELETE, PATCH). | N/A (handler) |
| `default.tsx` | Fallback for parallel routes when no match. | Server (default) |
| `middleware.ts` | Runs before every request (at `/` root only). | Edge Runtime |

**Nesting order (what wraps what):**

```
<Layout>
  <Template>
    <ErrorBoundary fallback={<Error />}>
      <Suspense fallback={<Loading />}>
        <Page />
      </Suspense>
    </ErrorBoundary>
  </Template>
</Layout>
```

## Data Fetching Patterns

| Pattern | Where | When to Use | Example |
|---|---|---|---|
| Async server component | Server | Initial page data, SEO content | `const data = await db.query()` |
| `fetch()` in server component | Server | External APIs, cacheable data | `await fetch(url, { next: { revalidate: 60 } })` |
| Server action | Server (called from client) | Mutations, form submissions | `'use server'; async function create(fd: FormData)` |
| `use()` with promise prop | Client | Consuming server-started fetches | `const data = use(promiseProp)` |
| `useEffect` + fetch | Client | Client-only data, polling, WebSocket | `useEffect(() => { fetch(...) }, [])` |
| React Query / SWR | Client | Complex client caching, optimistic updates | `const { data } = useQuery(...)` |

## Caching Quick Reference

### Fetch Cache (Data Cache)

```tsx
// Cache forever (default for static routes)
fetch(url)

// Revalidate every N seconds
fetch(url, { next: { revalidate: 60 } })

// No cache
fetch(url, { cache: 'no-store' })

// Tag for on-demand revalidation
fetch(url, { next: { tags: ['posts'] } })
```

### Route-Level Config

```tsx
export const dynamic = 'force-dynamic';  // No caching
export const dynamic = 'force-static';   // Build-time only
export const revalidate = 60;            // Revalidation interval (seconds)
export const dynamicParams = true;       // Allow params beyond generateStaticParams
```

### Revalidation

```tsx
revalidatePath('/posts');            // Specific path
revalidatePath('/posts', 'layout');  // Path + all children
revalidateTag('posts');              // All fetches tagged 'posts'
```

### Cache Layers Summary

```
Request Memoization  -> Per-request dedupe         -> Automatic for fetch, manual via cache()
Data Cache           -> Cross-request persistence   -> Controlled via fetch options / unstable_cache
Full Route Cache     -> Build-time HTML + RSC cache -> Static routes only, invalidated with data cache
Router Cache         -> Client-side nav cache       -> 0s dynamic, 5min static (Next.js 15)
```

## Common Patterns

### Auth Check (Server Component)

```tsx
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { verifySession } from '@/lib/auth';

export default async function ProtectedPage() {
  const cookieStore = await cookies();
  const session = await verifySession(cookieStore.get('token')?.value);

  if (!session) {
    redirect('/login');
  }

  return <Dashboard user={session.user} />;
}
```

### Form Submission (Server Action + useActionState)

```tsx
// action.ts
'use server';
export async function submitForm(prev: State, formData: FormData) {
  const data = Object.fromEntries(formData);
  // validate, save, revalidate
  return { success: true, error: null };
}

// Form.tsx
'use client';
import { useActionState } from 'react';
import { submitForm } from './action';

export function Form() {
  const [state, action, pending] = useActionState(submitForm, { success: false, error: null });
  return (
    <form action={action}>
      <input name="email" required />
      {state.error && <p>{state.error}</p>}
      <button disabled={pending}>{pending ? 'Submitting...' : 'Submit'}</button>
    </form>
  );
}
```

### Optimistic Update

```tsx
'use client';
import { useOptimistic } from 'react';
import { toggleLike } from './actions';

export function LikeButton({ liked, count }: { liked: boolean; count: number }) {
  const [optimistic, setOptimistic] = useOptimistic(
    { liked, count },
    (state, _: void) => ({
      liked: !state.liked,
      count: state.liked ? state.count - 1 : state.count + 1,
    })
  );

  return (
    <form action={async () => {
      setOptimistic(undefined);
      await toggleLike();
    }}>
      <button>{optimistic.liked ? 'Unlike' : 'Like'} ({optimistic.count})</button>
    </form>
  );
}
```
