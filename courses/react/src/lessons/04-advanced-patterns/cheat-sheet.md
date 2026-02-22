# Cheat Sheet: Advanced React Patterns

## Pattern Decision Matrix

| Pattern | Use Case | Complexity | TS Support | Status |
|---|---|---|---|---|
| Compound Components | Related component groups (Select, Tabs, Accordion) | Medium | Excellent | Active |
| Render Props | Inject UI into a component's render cycle | Low | Good | Niche |
| HOCs | Cross-cutting concerns (auth, logging) | High | Poor | Legacy |
| Headless Hooks | Reusable logic with zero UI opinions | Medium | Excellent | Preferred |
| Controlled/Uncontrolled | Stateful inputs, toggles, any interactive widget | Low | Good | Fundamental |
| Polymorphic `as` Prop | Design system primitives (Button, Text, Box) | High | Excellent | Active |
| Slot Pattern | Components with named render regions | Low | Good | Active |
| Provider Pattern | Dependency injection, theming, auth context | Low | Good | Active |
| Error Boundaries | Fault isolation, fallback UI | Low | Good | Required |
| Discriminated Unions | Type-safe component variants | Medium | Excellent | Preferred |

---

## Compound Component Template

```tsx
import { createContext, useContext, useState, type ReactNode } from "react";

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (id: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabsContext() {
  const ctx = useContext(TabsContext);
  if (!ctx) throw new Error("Tabs components must be used within <Tabs>");
  return ctx;
}

function Tabs({ defaultTab, children }: { defaultTab: string; children: ReactNode }) {
  const [activeTab, setActiveTab] = useState(defaultTab);
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div role="tablist">{children}</div>
    </TabsContext.Provider>
  );
}

function Tab({ id, children }: { id: string; children: ReactNode }) {
  const { activeTab, setActiveTab } = useTabsContext();
  return (
    <button role="tab" aria-selected={activeTab === id} onClick={() => setActiveTab(id)}>
      {children}
    </button>
  );
}

function Panel({ id, children }: { id: string; children: ReactNode }) {
  const { activeTab } = useTabsContext();
  if (activeTab !== id) return null;
  return <div role="tabpanel">{children}</div>;
}

Tabs.Tab = Tab;
Tabs.Panel = Panel;
```

---

## Polymorphic Component Template

```tsx
import { forwardRef, type ElementType, type ComponentPropsWithRef, type ReactNode } from "react";

type PolyProps<C extends ElementType, Own = {}> =
  Own & Omit<ComponentPropsWithRef<C>, keyof Own | "as"> & { as?: C };

type TextOwnProps = { size?: "sm" | "md" | "lg"; weight?: "normal" | "bold" };

type TextProps<C extends ElementType = "span"> = PolyProps<C, TextOwnProps>;

type TextComponent = <C extends ElementType = "span">(props: TextProps<C>) => ReactNode;

const Text: TextComponent = forwardRef(
  <C extends ElementType = "span">(
    { as, size = "md", weight = "normal", ...rest }: TextProps<C>,
    ref: ComponentPropsWithRef<C>["ref"]
  ) => {
    const Comp = as || "span";
    return <Comp ref={ref} data-size={size} data-weight={weight} {...rest} />;
  }
) as TextComponent;
```

---

## Error Boundary Template

```tsx
import { Component, type ReactNode, type ErrorInfo } from "react";

interface Props {
  children: ReactNode;
  fallback: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface State {
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info);
  }

  private reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return typeof this.props.fallback === "function"
        ? this.props.fallback(this.state.error, this.reset)
        : this.props.fallback;
    }
    return this.props.children;
  }
}
```

---

## Discriminated Union Props Template

```tsx
type ModalProps =
  | { variant: "confirm"; onConfirm: () => void; onCancel: () => void; destructive?: boolean }
  | { variant: "alert"; onAcknowledge: () => void }
  | { variant: "prompt"; onSubmit: (value: string) => void; defaultValue?: string };

type CommonModalProps = {
  title: string;
  open: boolean;
  onClose: () => void;
};

type FullModalProps = ModalProps & CommonModalProps;

function Modal(props: FullModalProps) {
  if (!props.open) return null;

  switch (props.variant) {
    case "confirm":
      return (/* confirm UI: onConfirm + onCancel */);
    case "alert":
      return (/* alert UI: onAcknowledge */);
    case "prompt":
      return (/* prompt UI: onSubmit */);
  }
}
```

---

## Common Patterns One-Pager

**Controllable state hook** -- support both controlled and uncontrolled:
```tsx
function useControllableState<T>({ value, defaultValue, onChange }: {
  value?: T; defaultValue: T; onChange?: (v: T) => void;
}) {
  const [internal, setInternal] = useState(defaultValue);
  const isControlled = value !== undefined;
  const resolved = isControlled ? value : internal;
  const setValue = (next: T) => { if (!isControlled) setInternal(next); onChange?.(next); };
  return [resolved, setValue] as const;
}
```

**Provider composition** -- flatten nested providers:
```tsx
function ComposeProviders({ providers, children }: {
  providers: [React.ComponentType<{ children: ReactNode }>, Record<string, unknown>?][];
  children: ReactNode;
}) {
  return providers.reduceRight<ReactNode>(
    (acc, [P, props]) => <P {...props}>{acc}</P>, children
  );
}
```

**Exhaustive switch** -- compile-time guard against unhandled variants:
```tsx
function assertNever(x: never): never { throw new Error(`Unhandled: ${x}`); }
```

**Key-based error recovery** -- remount subtree on error:
```tsx
const [key, setKey] = useState(0);
<ErrorBoundary key={key} fallback={<button onClick={() => setKey(k => k + 1)}>Retry</button>}>
  <Feature />
</ErrorBoundary>
```

**Generic list component** -- type-safe data rendering:
```tsx
function List<T>({ items, renderItem, keyFn }: {
  items: T[]; renderItem: (item: T) => ReactNode; keyFn: (item: T) => string;
}) {
  return <ul>{items.map(item => <li key={keyFn(item)}>{renderItem(item)}</li>)}</ul>;
}
```
