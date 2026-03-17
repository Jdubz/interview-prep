# 01 — Stripe API Fundamentals & Core Objects

Before diving into payments or billing, you need to understand how Stripe's API is designed and the primitives everything else builds on. Stripe is an API-first company — the API *is* the product.

---

## API Design Philosophy

Stripe's API is often cited as the gold standard for REST API design. Key principles:

- **Resource-oriented URLs** — every object has a predictable path: `/v1/customers`, `/v1/payment_intents`, `/v1/subscriptions/{id}`
- **Form-encoded request bodies** — POST requests use `application/x-www-form-urlencoded`, not JSON (though responses are JSON)
- **Consistent CRUD patterns** — Create, Retrieve, Update, List, Delete where applicable
- **Dot-notation for nested params** — `payment_method_data[type]=card` rather than nested JSON

### Authentication

Two types of API keys, determined by mode:

| Key prefix | Mode | Purpose |
|------------|------|---------|
| `sk_test_` | Test | Safe to use freely, no real money moves |
| `sk_live_` | Live | Production, real charges |
| `pk_test_` / `pk_live_` | Publishable | Client-side only (Stripe.js, mobile SDKs), can only confirm PaymentIntents and tokenize |

Secret keys authenticate server-side requests via Bearer token in the `Authorization` header. Publishable keys are safe to expose in client code — they can't read or modify most objects.

**Restricted keys** allow scoping permissions to specific resources (e.g., read-only access to charges). Used for microservices or third-party integrations.

### API Versioning

- Stripe releases new API versions regularly (e.g., `2024-12-18.acacia`)
- Your account is pinned to the version at signup
- Override per-request with `Stripe-Version` header
- Webhook events use the version of the endpoint that created them, not the version pinned to your account
- Breaking changes are gated behind version upgrades; additive changes (new fields, new event types) ship without version bumps

---

## Core Objects

### Customer

The anchor object for a person or business in Stripe. Nearly everything connects back to a Customer.

```
Customer
├── id: "cus_xxx"
├── email, name, phone, address
├── metadata: {}              ← arbitrary key-value pairs you control
├── default_source            ← legacy (Sources/Tokens API)
├── invoice_settings
│   └── default_payment_method  ← the PaymentMethod used for invoices/subscriptions
├── balance                   ← credit balance applied to future invoices
└── tax_ids[]                 ← VAT numbers, etc.
```

Key points:
- A Customer can have multiple PaymentMethods attached
- The `invoice_settings.default_payment_method` is what Billing uses — this is distinct from the legacy `default_source`
- Customer `balance` is a credit/debit ledger — positive balance means the customer has credit that will be applied to the next invoice
- Customers are shared across Payments, Billing, and Connect (with nuances per Connect charge type)

### PaymentMethod

Represents a specific payment instrument. Replaced the legacy Token and Source objects.

```
PaymentMethod
├── id: "pm_xxx"
├── type: "card" | "us_bank_account" | "sepa_debit" | "link" | ...
├── card
│   ├── brand: "visa", last4: "4242", exp_month, exp_year
│   └── checks: { cvc_check, address_line1_check, address_postal_code_check }
├── billing_details: { name, email, address }
└── customer: "cus_xxx" | null   ← attached to a customer, or unattached
```

Key points:
- PaymentMethods are **attached** to Customers via `POST /v1/payment_methods/{id}/attach`
- A detached PaymentMethod can only be used once; attached ones can be reused
- Different `type` values expose different sub-objects (`card`, `us_bank_account`, etc.)
- The PaymentMethod API is the modern approach — avoid Sources/Tokens in new integrations

### SetupIntent

Used to collect and save a payment method for **future** use without charging immediately.

```
SetupIntent
├── id: "seti_xxx"
├── status: "requires_payment_method" | "requires_confirmation" | "requires_action" | "processing" | "succeeded" | "canceled"
├── payment_method: "pm_xxx"
├── customer: "cus_xxx"
├── usage: "off_session" | "on_session"
└── client_secret: "seti_xxx_secret_xxx"
```

Key points:
- Runs 3D Secure / authentication if required, without charging
- After `succeeded`, the PaymentMethod is attached to the Customer and ready for future charges
- `usage: "off_session"` tells Stripe to request exemptions and optimize for recurring charges
- Essential for saving cards for subscriptions, usage-based billing, or "charge later" flows

---

## Cross-Cutting Concepts

### Idempotency

Every mutating (`POST`) request should include an `Idempotency-Key` header.

- Stripe saves the status code and response body for the first request with a given key
- Subsequent requests with the same key return the cached response, preventing double-charges
- Keys expire after **24 hours**
- Keys can be up to **255 characters** — typically a UUID v4
- If parameters differ on retry, Stripe rejects the request with an error
- `GET` and `DELETE` are inherently idempotent — don't send keys for these
- Validation errors (400) before processing begins are *not* cached — you can fix the request and retry with the same key

**Why this matters in interviews:** Idempotency is a core Stripe design principle. Being able to explain how it prevents double-charges in unreliable network conditions is fundamental.

### Metadata

Every Stripe object supports a `metadata` hash — up to 50 keys, each key up to 40 chars, each value up to 500 chars.

- Use for: order IDs, internal user IDs, campaign tracking, feature flags
- Searchable in the Dashboard
- Carried through to related objects (e.g., PaymentIntent metadata copies to Charges at creation)
- **Never store sensitive data** (PII, credentials) in metadata

### The `expand` Parameter

By default, related objects return as just an ID string. Use `expand[]` to inline the full object:

```
GET /v1/payment_intents/pi_xxx?expand[]=payment_method&expand[]=customer
```

- Reduces roundtrips — fetch a PaymentIntent with its Customer and PaymentMethod in one call
- Works up to 4 levels deep
- Available on most endpoints, including list operations (`expand[]=data.payment_method`)

### Pagination

List endpoints use cursor-based pagination:

- `limit` — number of objects per page (max 100, default 10)
- `starting_after` — cursor: return objects after this ID
- `ending_before` — cursor: return objects before this ID
- `has_more` — boolean indicating more results exist

Stripe uses cursor-based (not offset-based) pagination for consistency during concurrent writes.

---

## Webhooks

Webhooks are how Stripe tells your application about events asynchronously. This is critical infrastructure, not optional.

### Why webhooks matter

Many payment flows are asynchronous:
- 3D Secure requires customer action after your API call returns
- Bank debits (ACH, SEPA) take days to confirm
- Disputes, refunds, and payouts happen outside your control

Polling the API is expensive and unreliable. Webhooks push events to you in near real-time.

### Event structure

```json
{
  "id": "evt_xxx",
  "type": "payment_intent.succeeded",
  "data": {
    "object": { ... }    // snapshot of the object at event time
  },
  "api_version": "2024-12-18.acacia",
  "created": 1234567890,
  "livemode": true
}
```

### Signature verification

Every webhook includes a `Stripe-Signature` header with:
1. A **timestamp** (`t=`)
2. A **HMAC-SHA256 signature** (`v1=`) computed from `{timestamp}.{raw_body}` using your endpoint's signing secret

Verification steps:
1. Extract `t` and `v1` from the header
2. Compute HMAC-SHA256 of `{t}.{raw_body}` with your signing secret
3. Compare using constant-time comparison (prevents timing attacks)
4. Check timestamp is within tolerance (default 5 minutes) to prevent replay attacks

**Always use Stripe's SDK verification helpers** — don't roll your own.

### Best practices

- **Return 2xx immediately**, then process asynchronously (queue the event)
- **Handle out-of-order delivery** — Stripe doesn't guarantee order
- **Deduplicate** — log processed `evt_` IDs and skip duplicates
- **Subscribe only to events you need** — reduces load
- Stripe retries failed deliveries for up to **3 days** with exponential backoff (live mode)
- Up to **16 webhook endpoints** per account
- Test locally with `stripe listen --forward-to localhost:4242/webhook`

### Key event types to know

| Event | When it fires |
|-------|---------------|
| `payment_intent.succeeded` | Payment completed successfully |
| `payment_intent.payment_failed` | Payment attempt failed |
| `payment_intent.requires_action` | Customer needs to authenticate (3DS) |
| `customer.subscription.created` | New subscription started |
| `customer.subscription.updated` | Subscription changed (status, price, etc.) |
| `customer.subscription.deleted` | Subscription canceled |
| `invoice.paid` | Invoice successfully paid |
| `invoice.payment_failed` | Invoice payment attempt failed |
| `charge.dispute.created` | Customer disputed a charge |
| `account.updated` | Connect: connected account info changed |

---

## Error Handling

Stripe errors have a consistent structure:

```json
{
  "error": {
    "type": "card_error",
    "code": "card_declined",
    "decline_code": "insufficient_funds",
    "message": "Your card has insufficient funds.",
    "param": "payment_method"
  }
}
```

Error types:

| Type | HTTP Status | Meaning |
|------|-------------|---------|
| `card_error` | 402 | Card was declined |
| `invalid_request_error` | 400 | Missing/invalid parameters |
| `authentication_error` | 401 | Bad API key |
| `rate_limit_error` | 429 | Too many requests |
| `api_error` | 500/502/503 | Stripe-side issue (rare) |
| `idempotency_error` | 400 | Idempotency key reused with different params |

**Decline codes** (on `card_error`) tell you *why* a card was declined: `insufficient_funds`, `lost_card`, `stolen_card`, `expired_card`, `incorrect_cvc`, `processing_error`, `do_not_honor`, etc.

---

## Check Yourself

1. **What's the difference between a secret key and a publishable key? Why does this distinction exist?**

   Secret keys (`sk_`) authenticate server-side and can read/write all objects. Publishable keys (`pk_`) are safe for client-side code and can only tokenize payment details and confirm PaymentIntents. The split prevents exposing full API access in browser/mobile code.

2. **A customer says they were double-charged. What Stripe mechanism should have prevented this, and how does it work?**

   Idempotency keys. If the server retried a `POST /v1/payment_intents` with the same `Idempotency-Key`, Stripe returns the cached response from the first request. Keys are stored for 24 hours and prevent duplicate object creation even through network failures.

3. **You need to save a customer's card for future subscription charges without billing them today. What object do you create?**

   A SetupIntent. It handles authentication (3D Secure) without charging, and on success attaches the PaymentMethod to the Customer for future off-session use.

4. **Your webhook handler takes 45 seconds to process an event. What happens?**

   Stripe times out (30 second threshold) and marks the delivery as failed. It will retry with exponential backoff. The fix: return 2xx immediately and process the event asynchronously via a queue.

5. **A Stripe API response includes `"customer": "cus_abc123"` as just a string. How do you get the full customer object without a second API call?**

   Use `expand[]`: `GET /v1/payment_intents/pi_xxx?expand[]=customer`. This inlines the full Customer object in the response.

6. **Why does Stripe use cursor-based pagination instead of offset-based?**

   Cursor-based pagination is stable during concurrent writes. With offset-based pagination, new objects inserted during paging would cause items to shift and either be duplicated or skipped. Cursors anchor to a specific object ID.
