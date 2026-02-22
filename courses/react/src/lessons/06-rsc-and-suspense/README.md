# Lesson 06: React Server Components & Suspense

## Table of Contents

1. [Server vs Client Components](#server-vs-client-components)
2. [The "use client" Directive](#the-use-client-directive)
3. [The "use server" Directive](#the-use-server-directive)
4. [RSC Data Fetching](#rsc-data-fetching)
5. [Suspense Boundaries](#suspense-boundaries)
6. [Error Boundaries with Suspense](#error-boundaries-with-suspense)
7. [Streaming SSR](#streaming-ssr)
8. [Next.js App Router](#nextjs-app-router)
9. [Caching in RSC](#caching-in-rsc)
10. [The use() Hook](#the-use-hook)
11. [When NOT to Use RSC](#when-not-to-use-rsc)
12. [Common Interview Questions](#common-interview-questions)

---

## Server vs Client Components

**Server components are the default. Client components are opt-in.** This inverts the pre-RSC model where every component was implicitly a client component.

```
Server Components:                          Client Components:
  - Execute ONLY on the server/build time     - Execute on server (SSR) AND browser
  - Zero JS shipped to browser                - JS shipped for hydration + updates
  - No hooks (useState, useEffect, etc.)      - Full hook access
  - No browser APIs                           - Browser APIs after hydration
  - CAN: async/await, direct DB access        - CANNOT: async components, server imports
  - CAN: import server-only modules           - CAN: event handlers, local state
```

Think of the component tree as having a **boundary**. Server components live above it; client components live below.

```
        ServerLayout              <-- Server Component
         /        \
  ServerNav     ServerMain        <-- Server Components
     |              |
"use client"   "use client"       <-- BOUNDARY
     |              |
ClientDropdown  ClientForm        <-- Client Components
```

A client component **cannot import** a server component, but it CAN accept one as `children` (the "donut pattern"):

```tsx
// ServerPage.tsx (server component)
import { ClientSidebar } from './ClientSidebar';
import { ServerUserProfile } from './ServerUserProfile';

export default function ServerPage() {
  return (
    <ClientSidebar>
      <ServerUserProfile /> {/* Rendered on server, passed as serialized JSX */}
    </ClientSidebar>
  );
}
```

```tsx
// ClientSidebar.tsx
'use client';
export function ClientSidebar({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(true);
  return (
    <aside className={isOpen ? 'open' : 'closed'}>
      <button onClick={() => setIsOpen(!isOpen)}>Toggle</button>
      {children}
    </aside>
  );
}
```

---

## The "use client" Directive

`"use client"` does NOT mean "runs only on client." It means **"this is a serialization boundary -- this module and its imports are included in the client bundle."** Client components still get SSR'd.

### How it cascades

The directive applies at the **module boundary**. You only need it at entry points where server code transitions to client code -- not in every client file.

```tsx
// ClientForm.tsx -- this is the boundary
'use client';
import { FormField } from './FormField';       // No directive needed
import { useValidation } from './useValidation'; // Also no directive needed

export function ClientForm() {
  const { validate } = useValidation();
  return <form><FormField onValidate={validate} /></form>;
}
```

### Serialization constraints

Props passed from server to client components must be serializable:

```tsx
// ALLOWED: primitives, plain objects/arrays, Date, Map, Set,
//          TypedArrays, JSX elements (ReactNode), Promises, server actions
// NOT ALLOWED: functions (except server actions), class instances, Symbols, closures
```

```tsx
// ServerParent.tsx
export default async function ServerParent() {
  const data = await db.analytics.findMany();
  return (
    <ClientChart
      data={data}                         // OK: serializable array
      title="Revenue"                     // OK: string
      onRefresh={refreshData}             // ERROR: function
      formatter={new Intl.NumberFormat()} // ERROR: class instance
    />
  );
}
```

---

## The "use server" Directive

Marks functions as **server actions** -- server-side functions callable from the client. The RPC mechanism of RSC.

```tsx
// Module-level: ALL exports become server actions
// app/actions.ts
'use server';
export async function createUser(formData: FormData) {
  await db.user.create({ data: { name: formData.get('name') as string } });
  revalidatePath('/users');
}

// Inline: single function within a server component
export default function UsersPage() {
  async function handleDelete(formData: FormData) {
    'use server';
    await db.user.delete({ where: { id: formData.get('id') as string } });
    revalidatePath('/users');
  }
  return (
    <form action={handleDelete}>
      <input type="hidden" name="id" value="123" />
      <button type="submit">Delete</button>
    </form>
  );
}
```

### Progressive enhancement

Server actions passed to `<form action>` work **without JavaScript** (standard POST). With JS, React intercepts for seamless SPA behavior.

```tsx
'use client';
import { useActionState } from 'react';
import { createPost } from '@/app/actions';

export function CreatePostForm() {
  const [state, formAction, isPending] = useActionState(createPost, {
    error: null, success: false,
  });
  return (
    <form action={formAction}>
      <input name="title" required />
      <textarea name="body" required />
      {state.error && <p className="error">{state.error}</p>}
      <button disabled={isPending}>{isPending ? 'Creating...' : 'Create'}</button>
    </form>
  );
}
```

### Imperative calls

```tsx
'use client';
import { deleteItem } from '@/app/actions';
import { useTransition } from 'react';

export function DeleteButton({ id }: { id: string }) {
  const [isPending, startTransition] = useTransition();
  return (
    <button
      onClick={() => startTransition(async () => { await deleteItem(id); })}
      disabled={isPending}
    >
      {isPending ? 'Deleting...' : 'Delete'}
    </button>
  );
}
```

---

## RSC Data Fetching

Data fetching moves into the component tree. No useEffect, no loading state management -- just `async`/`await` at the component level.

```tsx
// This IS the data fetching layer. Direct DB access. No API route.
export default async function UsersPage() {
  const users = await db.user.findMany({
    orderBy: { createdAt: 'desc' },
    include: { posts: { select: { id: true } } },
  });
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name} -- {user.posts.length} posts</li>
      ))}
    </ul>
  );
}
```

### Parallel fetching (avoid waterfalls)

```tsx
// BAD: sequential (~500ms)
const user = await getUser();           // 200ms
const analytics = await getAnalytics(); // 300ms

// GOOD: parallel (~300ms)
const [user, analytics] = await Promise.all([getUser(), getAnalytics()]);

// ALSO GOOD: independent Suspense boundaries (stream independently)
export default function Dashboard() {
  return (
    <>
      <Suspense fallback={<UserSkeleton />}><UserSection /></Suspense>
      <Suspense fallback={<AnalyticsSkeleton />}><AnalyticsSection /></Suspense>
    </>
  );
}
```

---

## Suspense Boundaries

When a server component `await`s, the nearest Suspense boundary shows its fallback until the component resolves.

### Nested Suspense

```tsx
export default function Page() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <UserProfile />  {/* resolves ~100ms */}
      <Suspense fallback={<FeedSkeleton />}>
        <ActivityFeed />  {/* resolves ~500ms */}
        <Suspense fallback={<CommentsSkeleton />}>
          <Comments />    {/* resolves ~800ms */}
        </Suspense>
      </Suspense>
    </Suspense>
  );
}
// 0ms: PageSkeleton | 100ms: UserProfile + FeedSkeleton
// 500ms: ActivityFeed + CommentsSkeleton | 800ms: Comments
```

All async children within the **same** Suspense boundary must resolve before the fallback is replaced. Granular boundaries enable independent streaming.

---

## Error Boundaries with Suspense

### Next.js error.tsx

```tsx
// app/dashboard/error.tsx -- MUST be a client component
'use client';
export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div>
      <h2>Dashboard failed to load</h2>
      <p>{error.message}</p>
      <button onClick={() => reset()}>Retry</button>
    </div>
  );
}
```

### Combined pattern

```tsx
<ErrorBoundary fallback={(error, reset) => (
  <div>
    <p>{error.message}</p>
    <button onClick={reset}>Retry</button>
  </div>
)}>
  <Suspense fallback={<Skeleton />}>
    <AsyncDataComponent />
  </Suspense>
</ErrorBoundary>
```

### Retry via router.refresh()

```tsx
'use client';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';

export function RetryButton() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  return (
    <button
      onClick={() => startTransition(() => router.refresh())}
      disabled={isPending}
    >
      {isPending ? 'Retrying...' : 'Retry'}
    </button>
  );
}
```

---

## Streaming SSR

Traditional SSR waits for ALL data before sending HTML. Streaming SSR uses **chunked transfer encoding** to send HTML progressively.

```
Traditional: [wait for ALL data] -> [render ALL HTML] -> [send complete response]
Streaming:   [send shell + fallbacks] -> [stream chunk 1] -> [stream chunk 2] -> ...
```

The server sends fallbacks first. As async components resolve, it streams `<script>` tags that swap fallback HTML with resolved content.

### Selective hydration

React hydrates components independently. If a user interacts with a not-yet-hydrated component, React **prioritizes** hydrating it.

```
Time 0ms:    Browser renders [Header][HeroSkeleton][FeedSkeleton][SidebarSkeleton]
Time 50ms:   Header hydrates (interactive)
Time 200ms:  Server streams HeroSection -> replaces skeleton -> hydrates
Time 400ms:  Server streams Sidebar -> replaces skeleton -> hydrates
Time 800ms:  Server streams ContentFeed (slowest) -> replaces skeleton -> hydrates
```

---

## Next.js App Router

### File conventions

```
app/
  layout.tsx       // Root layout (persists across navigations)
  page.tsx         // Home (/)
  loading.tsx      // Suspense fallback
  error.tsx        // Error boundary (must be client component)
  not-found.tsx    // 404
  global-error.tsx // Catches root layout errors

  dashboard/
    layout.tsx     // Nested layout
    page.tsx       // /dashboard
    loading.tsx    // Dashboard loading state
    error.tsx      // Dashboard error boundary
    template.tsx   // Like layout but RE-MOUNTS on navigation
    [teamId]/
      page.tsx     // /dashboard/:teamId

  (marketing)/     // Route group (no URL segment)
    about/page.tsx // /about

  @modal/          // Parallel route (named slot)
    login/page.tsx
```

### Layout vs Template

**Layout** persists across navigations (state preserved). **Template** re-mounts on every navigation (useful for animations, per-page analytics, resetting state).

### Auto-wrapping

Next.js wraps `page.tsx` automatically:

```tsx
<Layout>
  <ErrorBoundary fallback={<Error />}>
    <Suspense fallback={<Loading />}>
      <Page />
    </Suspense>
  </ErrorBoundary>
</Layout>
```

---

## Caching in RSC

### Fetch caching

```tsx
await fetch(url);                                  // Cached indefinitely (default)
await fetch(url, { next: { revalidate: 60 } });    // Revalidate every 60s
await fetch(url, { cache: 'no-store' });            // Never cache
await fetch(url, { next: { tags: ['posts'] } });    // Tag-based revalidation
```

### Non-fetch caching

```tsx
import { unstable_cache } from 'next/cache';

const getCachedUser = unstable_cache(
  async (id: string) => db.user.findUnique({ where: { id } }),
  ['user-by-id'],
  { revalidate: 3600, tags: ['users'] }
);
```

### Four caching layers

```
1. Request Memoization   -- Dedupes identical requests within ONE render pass
2. Data Cache            -- Persists across requests; controlled by revalidate/tags
3. Full Route Cache      -- Build-time HTML + RSC payload for static routes
4. Router Cache          -- Client-side RSC payload cache (0s dynamic, 5min static)
```

### On-demand revalidation

```tsx
import { revalidatePath, revalidateTag } from 'next/cache';

revalidatePath('/profile');          // Specific path
revalidatePath('/profile', 'layout'); // Path + all sub-pages
revalidateTag('posts');               // All fetches tagged 'posts'
```

---

## The use() Hook

`use()` unwraps promises and reads context in client components. It bridges server-created promises to client consumption.

```tsx
// Server: create the promise, pass it as a prop
export default function UsersPage() {
  const usersPromise = db.user.findMany(); // DON'T await
  return (
    <Suspense fallback={<Skeleton />}>
      <UserList usersPromise={usersPromise} />
    </Suspense>
  );
}

// Client: unwrap with use()
'use client';
import { use } from 'react';

export function UserList({ usersPromise }: { usersPromise: Promise<User[]> }) {
  const users = use(usersPromise); // Suspends until resolved
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}
```

**Key rules:** Can be called conditionally (unlike other hooks). Do NOT create promises inside client render (infinite loops). Must have Suspense boundary above. Also works for context: `const theme = use(ThemeContext)`.

---

## When NOT to Use RSC

- **Highly interactive UIs** -- drag-and-drop, kanban boards, rich text editors. Need mouse/touch handlers, RAF animations, complex local state.
- **Real-time features** -- WebSockets, SSE, polling. Require persistent client connections.
- **Browser-only APIs** -- Geolocation, camera, localStorage, IndexedDB, Canvas, WebGL, Service Workers.
- **60fps animations** -- Cannot round-trip to the server. Use client components with `useRef` + `requestAnimationFrame`.

The pattern: fetch data in a server component, pass it as props, then let client components handle interactivity.

---

## Common Interview Questions

### Q1: Server component vs SSR -- what is the difference?

SSR renders client components to HTML on the server for initial load, then ships JS for hydration. Server components ONLY run on the server -- no JS shipped, no hydration, no client re-render. Client components still get SSR'd. `"use client"` components render on server during SSR AND on client during hydration.

### Q2: Can a client component import a server component?

No. Once past the `"use client"` boundary, all imports go into the client bundle. But a client component CAN receive a server component as `children` or any `ReactNode` prop (donut pattern). The server component renders on the server and its output is serialized as RSC payload.

### Q3: Server actions vs API routes?

Server actions integrate with React's component model: progressive enhancement, transition system integration, `revalidatePath`/`revalidateTag` support. API routes are standalone HTTP endpoints for webhooks, third-party consumers, non-React clients.

### Q4: Multiple async children in one Suspense boundary?

ALL must resolve before the fallback is replaced. For independent streaming, wrap each in its own Suspense boundary.

### Q5: Explain Next.js caching layers.

(1) Request Memoization: dedupes within one render. (2) Data Cache: persists across requests, controlled by `revalidate`/`tags`. (3) Full Route Cache: build-time HTML + RSC payload for static routes. (4) Router Cache: client-side RSC payload cache. Invalidation flows downward.

### Q6: use() vs await?

`await` in server components. `use()` in client components to unwrap promises passed from server. Enables "start fetching early, consume where needed." Also allows conditional context reading unlike `useContext`.

### Q7: How does streaming SSR handle SEO?

Initial HTML shell includes Suspense fallbacks. Chunks stream as inline `<script>` tags that swap content. JS-executing crawlers (Googlebot) see final content. For critical SEO content, avoid deep Suspense wrapping or ensure fast resolution.

### Q8: What is the RSC payload?

The Flight payload is a line-delimited streaming format representing the component tree. Unlike HTML, it preserves component boundaries, props, and client component references. During navigation, React fetches the RSC payload (not HTML) and reconciles with the existing tree for SPA-like transitions.
