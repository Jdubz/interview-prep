# Lesson 04: Advanced React Patterns

## Overview

This lesson covers the compositional and architectural patterns that separate
senior React engineers from those who just use React. These patterns appear
frequently in interviews at companies building design systems, component
libraries, and complex UIs. You are expected to not just know these patterns but
to articulate **when and why** to reach for each one.

---

## 1. Compound Components

Compound components model implicit parent-child relationships where a parent
component shares state with its children through Context, without requiring
the consumer to wire props manually.

**Why interviewers ask:** It tests whether you understand React Context, component
composition, and API design. It is the foundation of every serious component
library (Radix, Reach UI, Headless UI, MUI).

### Implementation

```tsx
import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
  type Dispatch,
  type SetStateAction,
} from "react";

// --- Internal shared state ---
interface SelectContextValue {
  value: string | null;
  onChange: (value: string) => void;
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
}

const SelectContext = createContext<SelectContextValue | null>(null);

function useSelectContext() {
  const ctx = useContext(SelectContext);
  if (!ctx) {
    throw new Error(
      "Select compound components must be rendered within <Select>"
    );
  }
  return ctx;
}

// --- Public API ---
interface SelectProps {
  value?: string | null;
  defaultValue?: string | null;
  onChange?: (value: string) => void;
  children: ReactNode;
}

function Select({ value, defaultValue = null, onChange, children }: SelectProps) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const [open, setOpen] = useState(false);

  // Support both controlled and uncontrolled usage
  const resolvedValue = value !== undefined ? value : internalValue;

  const handleChange = useCallback(
    (next: string) => {
      if (value === undefined) setInternalValue(next);
      onChange?.(next);
      setOpen(false);
    },
    [value, onChange]
  );

  return (
    <SelectContext.Provider
      value={{ value: resolvedValue, onChange: handleChange, open, setOpen }}
    >
      <div className="select-root">{children}</div>
    </SelectContext.Provider>
  );
}

function Trigger({ children }: { children: ReactNode }) {
  const { value, open, setOpen } = useSelectContext();
  return (
    <button
      aria-expanded={open}
      onClick={() => setOpen((prev) => !prev)}
    >
      {value ?? children}
    </button>
  );
}

function Option({ value, children }: { value: string; children: ReactNode }) {
  const { value: selected, onChange } = useSelectContext();
  return (
    <div
      role="option"
      aria-selected={selected === value}
      onClick={() => onChange(value)}
    >
      {children}
    </div>
  );
}

// Attach sub-components for dot-notation API
Select.Trigger = Trigger;
Select.Option = Option;
```

### Usage

```tsx
<Select onChange={(v) => console.log(v)}>
  <Select.Trigger>Pick a fruit</Select.Trigger>
  <Select.Option value="apple">Apple</Select.Option>
  <Select.Option value="banana">Banana</Select.Option>
</Select>
```

### Interview talking points

- Context is the communication channel; `children` is the composition mechanism.
- The `useSelectContext` guard enforces correct usage at runtime.
- This pattern cleanly supports controlled and uncontrolled modes (see section 5).
- For static sub-component typing with dot-notation, you can use a namespace
  object or assign to the function directly (as above). Both work; the namespace
  approach is slightly cleaner for tree-shaking.

---

## 2. Render Props

A render prop is a function prop that a component calls to delegate rendering.
Despite hooks replacing most render prop use cases, the pattern remains relevant
for **headless logic sharing** where you want to inject UI into a component's
render cycle.

### When render props still win over hooks

| Scenario | Hooks | Render Props |
|---|---|---|
| Sharing stateful logic | Preferred | Works |
| Injecting UI into a third-party component's render cycle | Cannot | Required |
| Conditional rendering based on internal state (e.g., virtualized list) | Awkward | Natural |
| Library consumers who want zero abstraction | N/A | Ideal |

### Implementation

```tsx
import { useState, useEffect, type ReactNode } from "react";

interface MousePosition {
  x: number;
  y: number;
}

interface MouseTrackerProps {
  children: (position: MousePosition) => ReactNode;
}

function MouseTracker({ children }: MouseTrackerProps) {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, []);

  return <>{children(position)}</>;
}

// Usage
<MouseTracker>
  {({ x, y }) => (
    <div>
      Cursor: {x}, {y}
    </div>
  )}
</MouseTracker>;
```

### Real-world example: Downshift

Downshift (by Kent C. Dodds) is the canonical render-prop library. Even after
adding hooks (`useCombobox`, `useSelect`), the render-prop API persists because
it allows full control over what gets rendered inside the dropdown without
requiring the consumer to manage any internal state.

---

## 3. Higher-Order Components (HOCs)

An HOC is a function that takes a component and returns a new component with
additional behavior. This is a **legacy pattern** -- hooks have replaced nearly
every legitimate HOC use case. Interviews still ask about HOCs because:

1. Large codebases (especially pre-hooks) are full of them.
2. Understanding HOCs proves you understand closures, component identity, and ref forwarding.
3. Some cross-cutting concerns (e.g., auth gating at the route level) are still
   occasionally expressed as HOCs.

### Implementation

```tsx
import { type ComponentType, type ComponentProps, forwardRef } from "react";

// --- withAuth: gate a component behind authentication ---
interface WithAuthProps {
  isAuthenticated: boolean;
}

function withAuth<T extends ComponentType<any>>(WrappedComponent: T) {
  type Props = Omit<ComponentProps<T>, keyof WithAuthProps> & WithAuthProps;

  const AuthGated = forwardRef<any, Props>(
    ({ isAuthenticated, ...rest }, ref) => {
      if (!isAuthenticated) {
        return <div>Please log in.</div>;
      }
      return <WrappedComponent ref={ref} {...(rest as any)} />;
    }
  );

  AuthGated.displayName = `withAuth(${
    WrappedComponent.displayName || WrappedComponent.name || "Component"
  })`;

  return AuthGated;
}

// --- withLogging: log renders ---
function withLogging<P extends object>(WrappedComponent: ComponentType<P>) {
  function Logged(props: P) {
    console.log(`[Render] ${WrappedComponent.displayName || WrappedComponent.name}`, props);
    return <WrappedComponent {...props} />;
  }

  Logged.displayName = `withLogging(${
    WrappedComponent.displayName || WrappedComponent.name || "Component"
  })`;

  return Logged;
}
```

### Why HOCs are problematic

- **Prop collisions:** Multiple HOCs can inject props with the same name.
- **Wrapper hell:** DevTools become unreadable with deeply nested HOC wrappers.
- **Static typing pain:** Inferring the resulting prop types across composed HOCs
  is extremely difficult in TypeScript.
- **Ref forwarding:** Every HOC must remember to forward refs or they silently break.

---

## 4. Headless Components / Hooks

The headless pattern separates **logic** (state, keyboard interactions, ARIA
attributes) from **UI** (markup, styles). The consumer provides all rendering;
the library provides all behavior.

### Hook-based headless component

```tsx
import { useState, useCallback, useRef, useId, type KeyboardEvent } from "react";

interface UseToggleOptions {
  defaultPressed?: boolean;
  onChange?: (pressed: boolean) => void;
}

interface UseToggleReturn {
  pressed: boolean;
  buttonProps: {
    id: string;
    role: "switch";
    "aria-checked": boolean;
    onClick: () => void;
    onKeyDown: (e: KeyboardEvent) => void;
  };
}

function useToggle({
  defaultPressed = false,
  onChange,
}: UseToggleOptions = {}): UseToggleReturn {
  const [pressed, setPressed] = useState(defaultPressed);
  const id = useId();

  const toggle = useCallback(() => {
    setPressed((prev) => {
      const next = !prev;
      onChange?.(next);
      return next;
    });
  }, [onChange]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    },
    [toggle]
  );

  return {
    pressed,
    buttonProps: {
      id,
      role: "switch",
      "aria-checked": pressed,
      onClick: toggle,
      onKeyDown,
    },
  };
}

// --- Usage: consumer owns all rendering ---
function FancyToggle() {
  const { pressed, buttonProps } = useToggle({
    onChange: (v) => console.log("Toggled:", v),
  });

  return (
    <button
      {...buttonProps}
      className={pressed ? "bg-green-500" : "bg-gray-300"}
    >
      {pressed ? "ON" : "OFF"}
    </button>
  );
}
```

### Libraries to know

| Library | Approach | Notes |
|---|---|---|
| Radix Primitives | Unstyled components + CSS | Compound component API, full ARIA |
| Headless UI (Tailwind) | Unstyled components | Designed for Tailwind, render props + hooks |
| Downshift | Hooks + render props | Combobox/select, pioneered the pattern |
| TanStack Table | Headless hooks | Zero UI, full table logic |
| React Aria (Adobe) | Hooks only | Most complete ARIA implementation |

---

## 5. Controlled vs. Uncontrolled Components

A **controlled** component derives its state from props. An **uncontrolled**
component manages its own internal state. The distinction applies to any
stateful component, not just form inputs.

### The dual-mode pattern

Production components should support both modes. This is how every serious
component library works.

```tsx
import { useState, useCallback } from "react";

/**
 * A hook that supports both controlled and uncontrolled state.
 * If the consumer passes a value, the component is controlled.
 * Otherwise, it manages its own internal state.
 */
function useControllableState<T>({
  value: controlledValue,
  defaultValue,
  onChange,
}: {
  value?: T;
  defaultValue: T;
  onChange?: (value: T) => void;
}) {
  const [internalValue, setInternalValue] = useState(defaultValue);

  const isControlled = controlledValue !== undefined;
  const value = isControlled ? controlledValue : internalValue;

  const setValue = useCallback(
    (next: T | ((prev: T) => T)) => {
      const resolvedNext =
        typeof next === "function" ? (next as (prev: T) => T)(value) : next;

      if (!isControlled) {
        setInternalValue(resolvedNext);
      }
      onChange?.(resolvedNext);
    },
    [isControlled, value, onChange]
  );

  return [value, setValue] as const;
}

// --- Usage ---
interface InputProps {
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
}

function Input({ value, defaultValue = "", onChange }: InputProps) {
  const [inputValue, setInputValue] = useControllableState({
    value,
    defaultValue,
    onChange,
  });

  return (
    <input
      value={inputValue}
      onChange={(e) => setInputValue(e.target.value)}
    />
  );
}
```

### Interview talking points

- **Never switch between controlled and uncontrolled** during a component's
  lifetime. React will warn. The `useControllableState` hook handles this
  cleanly by checking `controlledValue !== undefined` on every render.
- **Uncontrolled is simpler** for forms where you only need the value on submit
  (`useRef` + `ref`).
- **Controlled is required** when you need to validate, transform, or
  synchronize values in real time.

---

## 6. Polymorphic Components

A polymorphic component lets the consumer change the rendered element via an
`as` prop. This is essential for design systems where a `<Button>` might need to
render as an `<a>`, a `<Link>`, or a `<div>`.

### TypeScript-safe implementation

```tsx
import {
  forwardRef,
  type ElementType,
  type ComponentPropsWithRef,
  type ReactNode,
} from "react";

// --- Type utilities ---
type PolymorphicRef<C extends ElementType> =
  ComponentPropsWithRef<C>["ref"];

type PolymorphicProps<
  C extends ElementType,
  OwnProps = {},
> = OwnProps &
  Omit<ComponentPropsWithRef<C>, keyof OwnProps | "as"> & {
    as?: C;
  };

// --- Component ---
type ButtonOwnProps = {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  children: ReactNode;
};

type ButtonProps<C extends ElementType = "button"> = PolymorphicProps<
  C,
  ButtonOwnProps
>;

type ButtonComponent = <C extends ElementType = "button">(
  props: ButtonProps<C>
) => ReactNode;

const Button: ButtonComponent = forwardRef(
  <C extends ElementType = "button">(
    { as, variant = "primary", size = "md", children, ...rest }: ButtonProps<C>,
    ref: PolymorphicRef<C>
  ) => {
    const Component = as || "button";
    return (
      <Component ref={ref} {...rest}>
        {children}
      </Component>
    );
  }
) as ButtonComponent;
```

### Usage with full type safety

```tsx
// Renders <button>, gets ButtonHTMLAttributes
<Button variant="primary" onClick={() => {}}>Click</Button>

// Renders <a>, gets AnchorHTMLAttributes (href is valid)
<Button as="a" href="/about" variant="secondary">About</Button>

// Renders a custom Link component, gets its props
<Button as={Link} to="/dashboard" variant="ghost">Dashboard</Button>

// Type error: href is not valid on <button>
<Button href="/nope">Nope</Button>
```

### The `forwardRef` + generics problem

`forwardRef` does not natively support generic components -- its type signature
fixes the props at the call site. The `as ButtonComponent` cast above is the
standard workaround. React 19's `ref` as a regular prop may eliminate this
friction entirely.

---

## 7. Slot Pattern

The slot pattern uses `children` and named props to give consumers control over
specific rendering regions of a component.

### Named slots via props

```tsx
import { type ReactNode } from "react";

interface CardProps {
  header?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
}

function Card({ header, footer, children, actions }: CardProps) {
  return (
    <div className="card">
      {header && <div className="card-header">{header}</div>}
      <div className="card-body">{children}</div>
      {actions && <div className="card-actions">{actions}</div>}
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

// Usage
<Card
  header={<h2>Title</h2>}
  footer={<small>Last updated: today</small>}
  actions={
    <>
      <button>Save</button>
      <button>Cancel</button>
    </>
  }
>
  <p>Card body content</p>
</Card>;
```

### React.Children utilities (and why to avoid them)

```tsx
import { Children, isValidElement, type ReactNode, type ReactElement } from "react";

// Filtering children by type -- brittle, avoid in new code
function getChildrenByType(children: ReactNode, type: React.FC<any>) {
  return Children.toArray(children).filter(
    (child): child is ReactElement => isValidElement(child) && child.type === type
  );
}
```

`React.Children` utilities (`map`, `forEach`, `toArray`, `count`) are considered
legacy. They break when children are wrapped in fragments or returned from other
components. Prefer explicit named-slot props or compound components with Context.

---

## 8. Provider Pattern

The provider pattern uses React Context to inject dependencies (state, dispatch
functions, services) into a subtree.

### The problem: provider hell

```tsx
// This is real and it is terrible
function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <I18nProvider>
          <RouterProvider>
            <QueryClientProvider>
              <NotificationProvider>
                <FeatureFlagProvider>
                  <AppContent />
                </FeatureFlagProvider>
              </NotificationProvider>
            </QueryClientProvider>
          </RouterProvider>
        </I18nProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
```

### Solution: provider composition utility

```tsx
import { type ReactNode, type ComponentType } from "react";

type ProviderWithProps = [ComponentType<{ children: ReactNode }>, Record<string, unknown>?];

function ComposeProviders({
  providers,
  children,
}: {
  providers: ProviderWithProps[];
  children: ReactNode;
}) {
  return providers.reduceRight<ReactNode>(
    (acc, [Provider, props]) => <Provider {...props}>{acc}</Provider>,
    children
  );
}

// Usage
function App() {
  return (
    <ComposeProviders
      providers={[
        [ThemeProvider, { theme: darkTheme }],
        [AuthProvider],
        [I18nProvider, { locale: "en" }],
        [RouterProvider],
        [QueryClientProvider, { client: queryClient }],
        [NotificationProvider],
        [FeatureFlagProvider],
      ]}
    >
      <AppContent />
    </ComposeProviders>
  );
}
```

### Interview talking points

- Provider ordering matters. A provider can only consume contexts above it.
- Split contexts: separate read-heavy (state) from write-heavy (dispatch)
  contexts to avoid unnecessary re-renders.
- Consider colocation: not everything needs to be at the app root. A `FormContext`
  should wrap only the form.

---

## 9. Error Boundaries

Error boundaries catch JavaScript errors in their child component tree during
rendering, in lifecycle methods, and in constructors. They **do not** catch
errors in event handlers, async code, or server-side rendering.

### Why still class-based

There is no hook equivalent for `componentDidCatch` or
`getDerivedStateFromError`. These are the only two class lifecycle methods
without hook counterparts. React has not introduced a hook-based error boundary
API (as of React 19).

### Implementation

```tsx
import {
  Component,
  type ReactNode,
  type ErrorInfo,
} from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.props.onError?.(error, errorInfo);
  }

  private reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    const { fallback, children } = this.props;

    if (error) {
      if (typeof fallback === "function") {
        return fallback(error, this.reset);
      }
      return fallback;
    }

    return children;
  }
}
```

### Usage with recovery

```tsx
<ErrorBoundary
  fallback={(error, reset) => (
    <div role="alert">
      <h2>Something went wrong</h2>
      <pre>{error.message}</pre>
      <button onClick={reset}>Try again</button>
    </div>
  )}
  onError={(error, info) => {
    // Send to error tracking service
    Sentry.captureException(error, { extra: info });
  }}
>
  <SomeFragileComponent />
</ErrorBoundary>
```

### Patterns for production

- **Granular boundaries:** Wrap individual features, not the entire app.
  A crashing sidebar should not take down the main content.
- **Error boundary + Suspense:** Place `ErrorBoundary` above `Suspense` to
  catch both rendering errors and failed lazy loads.
- **Key-based reset:** Change the `key` prop on the ErrorBoundary to force
  a full remount of the subtree instead of using imperative `reset()`.

```tsx
function ResettableFeature() {
  const [boundaryKey, setBoundaryKey] = useState(0);

  return (
    <ErrorBoundary
      key={boundaryKey}
      fallback={
        <button onClick={() => setBoundaryKey((k) => k + 1)}>
          Retry
        </button>
      }
    >
      <FeatureComponent />
    </ErrorBoundary>
  );
}
```

---

## 10. Discriminated Union Props

Use TypeScript discriminated unions to make component prop combinations
**mutually exclusive** and **exhaustively type-checked**.

### The problem

```tsx
// Bad: nothing prevents passing both href and onClick, or neither
interface ButtonProps {
  href?: string;
  onClick?: () => void;
  external?: boolean; // only relevant if href is set
}
```

### The solution: discriminated unions

```tsx
type LinkButtonProps = {
  as: "link";
  href: string;
  external?: boolean;
  onClick?: never; // explicitly disallowed
};

type ActionButtonProps = {
  as: "button";
  onClick: () => void;
  href?: never;
  external?: never;
};

type SubmitButtonProps = {
  as: "submit";
  href?: never;
  onClick?: never;
  external?: never;
};

type ButtonProps = (LinkButtonProps | ActionButtonProps | SubmitButtonProps) & {
  children: ReactNode;
  variant?: "primary" | "secondary";
  disabled?: boolean;
};

function Button(props: ButtonProps) {
  switch (props.as) {
    case "link":
      return (
        <a
          href={props.href}
          target={props.external ? "_blank" : undefined}
          rel={props.external ? "noopener noreferrer" : undefined}
        >
          {props.children}
        </a>
      );
    case "button":
      return (
        <button onClick={props.onClick} disabled={props.disabled}>
          {props.children}
        </button>
      );
    case "submit":
      return (
        <button type="submit" disabled={props.disabled}>
          {props.children}
        </button>
      );
  }
}
```

### Usage

```tsx
// Valid
<Button as="link" href="/about">About</Button>
<Button as="link" href="/ext" external>External</Button>
<Button as="button" onClick={() => {}}>Click</Button>
<Button as="submit">Submit</Button>

// Type errors (caught at compile time)
<Button as="link" onClick={() => {}}>Nope</Button>      // onClick is never
<Button as="button" href="/bad">Nope</Button>            // href is never
<Button as="link" external>Nope</Button>                 // href is required
```

### Advanced: exhaustiveness checking

```tsx
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}

function renderButton(props: ButtonProps) {
  switch (props.as) {
    case "link":
      return /* ... */;
    case "button":
      return /* ... */;
    case "submit":
      return /* ... */;
    default:
      // If a new variant is added but not handled, this is a compile error
      return assertNever(props.as);
  }
}
```

---

## Key Takeaways for Interviews

1. **Compound components** are the gold standard for related component groups.
   Know how to build one from scratch with Context.
2. **Render props** are not dead. They are the right tool when hooks cannot
   inject into a render cycle.
3. **HOCs** are legacy. Know their problems (prop collision, ref forwarding,
   type inference) and why hooks replaced them.
4. **Headless hooks** are the modern approach to logic sharing. Name-drop Radix,
   React Aria, or TanStack.
5. **Controlled vs. uncontrolled** is fundamental. The `useControllableState`
   hook is a pattern worth memorizing.
6. **Polymorphic `as` prop** with correct TypeScript is a design system must-have.
   Know the `forwardRef` generics limitation.
7. **Provider composition** solves provider hell. Split read/write contexts.
8. **Error boundaries** are still class-based. Know why, and know the
   key-based reset trick.
9. **Discriminated unions** are the TypeScript-native way to model component
   variants. Use `never` to ban invalid prop combinations.
