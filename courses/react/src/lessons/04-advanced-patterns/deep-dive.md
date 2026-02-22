# Deep Dive: Advanced Patterns

Optional extended material for candidates targeting L6+ roles or companies with
heavy design system / component library work (Vercel, Shopify, Adobe, Atlassian).

---

## 1. Inversion of Control

Inversion of control (IoC) means shifting decision-making from the library to
the consumer. Instead of the component deciding *how* to render, *what* to
filter, or *when* to update, the consumer provides those decisions via props,
render functions, or hooks.

### Levels of IoC in React

| Level | Mechanism | Control given to consumer |
|---|---|---|
| 0 | Fixed component | None |
| 1 | Prop configuration | Toggle features (e.g., `showIcon`) |
| 2 | Render slot props | Inject UI into specific regions |
| 3 | Compound components | Restructure the component tree |
| 4 | Headless hooks | Full rendering control |

### Example: progressive IoC for a list component

```tsx
// Level 1: prop configuration (low IoC)
<List items={items} renderAs="grid" showCheckboxes />

// Level 2: render slot (medium IoC)
<List items={items} renderItem={(item) => <CustomCard item={item} />} />

// Level 3: compound components (high IoC)
<List>
  <List.Header>
    <SearchInput />
  </List.Header>
  <List.Body>
    {items.map((item) => (
      <List.Item key={item.id}>
        <CustomCard item={item} />
      </List.Item>
    ))}
  </List.Body>
  <List.Footer>
    <Pagination />
  </List.Footer>
</List>

// Level 4: headless hook (maximum IoC)
function MyList() {
  const { getListProps, getItemProps, selectedIds } = useList({ items });

  return (
    <ul {...getListProps()}>
      {items.map((item) => (
        <li key={item.id} {...getItemProps(item)}>
          {item.name} {selectedIds.has(item.id) && "(selected)"}
        </li>
      ))}
    </ul>
  );
}
```

**Design principle:** Start at the lowest IoC level that satisfies your use
cases. Over-abstracting up front leads to APIs that are powerful but
incomprehensible. Libraries like Downshift and TanStack Table deliberately ship
multiple API levels for this reason.

---

## 2. Generic Component Typing

Generic components let you build type-safe, reusable structures (tables, lists,
selects) that adapt their types to the data they receive.

### The `forwardRef` generics problem

`React.forwardRef` erases generic type parameters because it returns a
`ForwardRefExoticComponent` with fixed props. There are several workarounds.

#### Workaround 1: Type assertion

```tsx
import { forwardRef, type Ref } from "react";

interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => ReactNode;
  keyExtractor: (item: T) => string;
}

// Inner implementation is generic
function ListInner<T>(
  { items, renderItem, keyExtractor }: ListProps<T>,
  ref: Ref<HTMLUListElement>
) {
  return (
    <ul ref={ref}>
      {items.map((item) => (
        <li key={keyExtractor(item)}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}

// Cast to recover the generic
export const List = forwardRef(ListInner) as <T>(
  props: ListProps<T> & { ref?: Ref<HTMLUListElement> }
) => ReactNode;
```

#### Workaround 2: Wrapper component (no forwardRef)

```tsx
// React 19: ref is a regular prop, no forwardRef needed
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => ReactNode;
  keyExtractor: (item: T) => string;
  ref?: Ref<HTMLUListElement>;
}

function List<T>({ items, renderItem, keyExtractor, ref }: ListProps<T>) {
  return (
    <ul ref={ref}>
      {items.map((item) => (
        <li key={keyExtractor(item)}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}
```

### Generic table component

```tsx
interface Column<T> {
  key: keyof T & string;
  header: string;
  render?: (value: T[keyof T], row: T) => ReactNode;
  width?: number;
}

interface TableProps<T extends Record<string, unknown>> {
  data: T[];
  columns: Column<T>[];
  rowKey: keyof T & string;
  onRowClick?: (row: T) => void;
}

function Table<T extends Record<string, unknown>>({
  data,
  columns,
  rowKey,
  onRowClick,
}: TableProps<T>) {
  return (
    <table>
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col.key} style={{ width: col.width }}>{col.header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row) => (
          <tr key={String(row[rowKey])} onClick={() => onRowClick?.(row)}>
            {columns.map((col) => (
              <td key={col.key}>
                {col.render
                  ? col.render(row[col.key], row)
                  : String(row[col.key] ?? "")}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Usage: T is inferred from `data`
interface User {
  id: number;
  name: string;
  email: string;
  role: "admin" | "user";
}

<Table<User>
  data={users}
  columns={[
    { key: "name", header: "Name" },
    { key: "email", header: "Email" },
    {
      key: "role",
      header: "Role",
      render: (value) => <Badge>{value}</Badge>, // value is User["role"]
    },
  ]}
  rowKey="id"
  onRowClick={(user) => navigate(`/users/${user.id}`)}
/>;
```

---

## 3. Builder Pattern for Complex Component APIs

When a component has many configuration options that interact (e.g., a form
builder, chart, or data grid), a builder API can provide guided construction
with full type inference at each step.

```tsx
// Builder for a type-safe form schema
class FormBuilder<TFields extends Record<string, unknown> = {}> {
  private fields: Map<string, FieldConfig> = new Map();

  addField<K extends string, V>(
    name: K,
    config: FieldConfig<V>
  ): FormBuilder<TFields & Record<K, V>> {
    this.fields.set(name, config);
    return this as unknown as FormBuilder<TFields & Record<K, V>>;
  }

  build(): FormSchema<TFields> {
    return {
      fields: Object.fromEntries(this.fields) as any,
      validate: (values: TFields) => this.runValidation(values),
    };
  }

  private runValidation(values: TFields): ValidationResult<TFields> {
    // ... validation logic
  }
}

interface FieldConfig<V = unknown> {
  type: "text" | "number" | "select" | "checkbox";
  label: string;
  required?: boolean;
  validate?: (value: V) => string | null;
}

// Usage: types accumulate as fields are added
const schema = new FormBuilder()
  .addField("name", {
    type: "text",
    label: "Name",
    required: true,
    validate: (v: string) => (v.length < 2 ? "Too short" : null),
  })
  .addField("age", {
    type: "number",
    label: "Age",
    validate: (v: number) => (v < 0 ? "Invalid" : null),
  })
  .addField("role", {
    type: "select",
    label: "Role",
  })
  .build();

// schema is FormSchema<{ name: string; age: number; role: unknown }>
```

This pattern is used by libraries like Zod (schema builder), tRPC (procedure
builder), and Prisma (query builder). It is less common in React components
directly but appears in configuration objects passed to complex components.

---

## 4. Component Composition vs. Configuration

The fundamental API design decision: do consumers **configure** behavior
through props or **compose** behavior through children?

### Configuration approach

```tsx
// Highly configurable, low composability
<DataGrid
  data={users}
  columns={columnDefs}
  sortable
  filterable
  paginated
  pageSize={25}
  onSort={handleSort}
  onFilter={handleFilter}
  emptyState="No users found"
  loading={isLoading}
  rowActions={[
    { label: "Edit", onClick: handleEdit },
    { label: "Delete", onClick: handleDelete },
  ]}
/>
```

### Composition approach

```tsx
// Highly composable, requires more consumer code
<DataGrid data={users}>
  <DataGrid.Toolbar>
    <DataGrid.Search />
    <DataGrid.FilterMenu />
  </DataGrid.Toolbar>
  <DataGrid.Table>
    <DataGrid.Column field="name" sortable />
    <DataGrid.Column field="email" />
    <DataGrid.Column field="role" render={(role) => <Badge>{role}</Badge>} />
  </DataGrid.Table>
  <DataGrid.Pagination pageSize={25} />
  <DataGrid.EmptyState>
    <EmptyIllustration />
    <p>No users found</p>
  </DataGrid.EmptyState>
</DataGrid>
```

### Trade-offs

| Dimension | Configuration | Composition |
|---|---|---|
| API surface | Large (many props) | Small (few props per component) |
| Flexibility | Low (predefined options) | High (arbitrary structure) |
| Learning curve | Low (one component) | Higher (many components) |
| TypeScript complexity | Simpler | More complex (context, generics) |
| Tree-shaking | Poor (monolithic) | Good (import only what you use) |
| Testability | Harder (many code paths) | Easier (isolated units) |

**Heuristic:** Use configuration for 80% use cases, offer composition for the
20% that need customization. Ship both if the component is complex enough.

---

## 5. Design System Patterns

### Variant systems with CVA (class-variance-authority)

CVA provides a type-safe way to define component variants that map to class
names. It has become the de facto standard in Tailwind-based design systems.

```tsx
import { cva, type VariantProps } from "class-variance-authority";

const buttonVariants = cva(
  // Base classes (always applied)
  "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2",
  {
    variants: {
      variant: {
        primary: "bg-blue-600 text-white hover:bg-blue-700",
        secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200",
        destructive: "bg-red-600 text-white hover:bg-red-700",
        ghost: "hover:bg-gray-100",
      },
      size: {
        sm: "h-8 px-3 text-sm",
        md: "h-10 px-4 text-base",
        lg: "h-12 px-6 text-lg",
      },
    },
    compoundVariants: [
      {
        variant: "ghost",
        size: "sm",
        className: "px-2", // override padding for small ghost buttons
      },
    ],
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
);

// Extract variant props as a TypeScript type
type ButtonVariantProps = VariantProps<typeof buttonVariants>;

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    ButtonVariantProps {
  children: ReactNode;
}

function Button({ variant, size, className, children, ...rest }: ButtonProps) {
  return (
    <button className={buttonVariants({ variant, size, className })} {...rest}>
      {children}
    </button>
  );
}
```

### Token-based theming with CSS custom properties

```tsx
// tokens.ts
const tokens = {
  color: {
    primary: "var(--color-primary)",
    secondary: "var(--color-secondary)",
    surface: "var(--color-surface)",
    text: "var(--color-text)",
  },
  spacing: {
    xs: "var(--spacing-xs)",
    sm: "var(--spacing-sm)",
    md: "var(--spacing-md)",
    lg: "var(--spacing-lg)",
  },
  radius: {
    sm: "var(--radius-sm)",
    md: "var(--radius-md)",
    lg: "var(--radius-lg)",
    full: "var(--radius-full)",
  },
} as const;

// Theme provider sets CSS variables on a wrapping element
interface Theme {
  [key: `--${string}`]: string;
}

const lightTheme: Theme = {
  "--color-primary": "#2563eb",
  "--color-secondary": "#64748b",
  "--color-surface": "#ffffff",
  "--color-text": "#0f172a",
  "--spacing-xs": "4px",
  "--spacing-sm": "8px",
  "--spacing-md": "16px",
  "--spacing-lg": "32px",
  "--radius-sm": "4px",
  "--radius-md": "8px",
  "--radius-lg": "16px",
  "--radius-full": "9999px",
};

function ThemeProvider({
  theme,
  children,
}: {
  theme: Theme;
  children: ReactNode;
}) {
  return (
    <div style={theme as React.CSSProperties}>
      {children}
    </div>
  );
}
```

---

## 6. Module Federation and Micro-Frontends

Module federation (Webpack 5 / Rspack) lets independently deployed applications
share React components at runtime. This is the infrastructure behind
micro-frontend architectures.

### Key concepts

- **Host:** The shell application that loads remote modules.
- **Remote:** An independently deployed app that exposes modules.
- **Shared:** Dependencies (React, React DOM) that are shared to avoid
  duplication and ensure singleton instances.

### Critical constraints for shared React components

1. **Single React instance:** Multiple React instances cause hooks to break
   ("Invalid hook call"). Use the `shared` configuration to enforce a singleton.
2. **Version alignment:** All federated apps must use compatible React versions.
   Use `requiredVersion` and `singleton: true`.
3. **Type safety across boundaries:** Types are not shared at runtime. Use
   a shared package (published to npm or a monorepo) that exports only
   TypeScript types and interfaces.
4. **Error isolation:** Wrap federated components in error boundaries. A crash
   in a remote should not take down the host.

```tsx
// Consuming a federated component
import { lazy, Suspense } from "react";

// Dynamic import from a remote -- resolved at runtime via module federation
const RemoteCheckout = lazy(() => import("checkout/CheckoutWidget"));

function App() {
  return (
    <ErrorBoundary fallback={<div>Checkout unavailable</div>}>
      <Suspense fallback={<Skeleton />}>
        <RemoteCheckout cartId={cartId} />
      </Suspense>
    </ErrorBoundary>
  );
}
```

### Interview perspective

Micro-frontends are a **deployment and organizational** pattern, not a React
pattern. Interviewers ask about them to test whether you understand the
trade-offs: independent deployment and team autonomy vs. increased complexity,
bundle duplication, and coordination overhead. The correct senior answer is
usually "we considered it and decided a monorepo was simpler for our scale."

---

## 7. Advanced TypeScript Patterns for React

### Template literal types for event handler props

```tsx
// Generate event handler prop names from a list of events
type EventName = "click" | "hover" | "focus" | "blur";

type EventHandlerProps = {
  [K in EventName as `on${Capitalize<K>}`]?: (event: Event) => void;
};

// Result:
// {
//   onClick?: (event: Event) => void;
//   onHover?: (event: Event) => void;
//   onFocus?: (event: Event) => void;
//   onBlur?: (event: Event) => void;
// }
```

### Mapped types for component variant generation

```tsx
// Generate variant-specific components from a variant map
type Variant = "info" | "warning" | "error" | "success";

type AlertComponents = {
  [V in Variant as `${Capitalize<V>}Alert`]: React.FC<{ message: string }>;
};

// Result:
// {
//   InfoAlert: React.FC<{ message: string }>;
//   WarningAlert: React.FC<{ message: string }>;
//   ErrorAlert: React.FC<{ message: string }>;
//   SuccessAlert: React.FC<{ message: string }>;
// }

// Implementation
function createAlertComponents(): AlertComponents {
  const variants: Variant[] = ["info", "warning", "error", "success"];

  return Object.fromEntries(
    variants.map((v) => [
      `${v.charAt(0).toUpperCase()}${v.slice(1)}Alert`,
      ({ message }: { message: string }) => (
        <div role="alert" className={`alert-${v}`}>
          {message}
        </div>
      ),
    ])
  ) as AlertComponents;
}
```

### Conditional props with `Extract` and `Exclude`

```tsx
// Only allow `placeholder` when `type` is a text-like input
type TextInputType = "text" | "email" | "password" | "search" | "url";
type NonTextInputType = "checkbox" | "radio" | "range" | "file";

type InputProps =
  | {
      type: TextInputType;
      placeholder?: string;
      value: string;
      onChange: (value: string) => void;
    }
  | {
      type: Extract<NonTextInputType, "checkbox" | "radio">;
      placeholder?: never;
      checked: boolean;
      onChange: (checked: boolean) => void;
    }
  | {
      type: Extract<NonTextInputType, "range">;
      placeholder?: never;
      value: number;
      min: number;
      max: number;
      onChange: (value: number) => void;
    }
  | {
      type: Extract<NonTextInputType, "file">;
      placeholder?: never;
      accept?: string;
      onChange: (files: FileList) => void;
    };
```

### Extracting component prop types from third-party libraries

```tsx
import { type ComponentProps, type ComponentRef } from "react";

// Get props from any component
type ButtonProps = ComponentProps<typeof Button>;

// Get the ref type from any component
type ButtonRef = ComponentRef<typeof Button>;

// Get props from an HTML element
type DivProps = ComponentProps<"div">;

// Combine with Omit for wrapping
type CustomInputProps = Omit<ComponentProps<"input">, "onChange"> & {
  onChange: (value: string) => void; // simplified onChange
};
```

---

## Summary

These deep-dive topics are not required for most interviews, but demonstrating
familiarity with them signals that you have built and maintained real component
libraries at scale. The unifying theme is **type safety and composability**: the
best React patterns are the ones where TypeScript catches misuse at compile time
and consumers can adapt behavior without forking the source.
