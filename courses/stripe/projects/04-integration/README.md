# Integration Exercise Simulation: Merchant Payment Service

**Timer: 50 minutes.** The actual round is 60 minutes (2:30-3:30 PM with Adam Fuller). This is the LAST interview of the day -- you will be fatigued. Practice this one last in your prep sessions to simulate real conditions.

---

## What This Round Tests

The Integration Exercise is a practical coding round. In the real interview, you clone an existing repo that has a partially-built service, read the API documentation for an external service, and extend the codebase by making HTTP requests to a running server. The work is ETL-shaped: read data from APIs, transform it, store results, handle errors.

This simulation mirrors that format. The "existing codebase" is `server.ts`. The "external API" is the `PaymentProvider` class (standing in for a real HTTP service). Your implementation goes in `starter.ts`.

### What the interviewer evaluates

| Dimension | What they watch for |
|---|---|
| **Ability to help yourself** | Do you read the existing code and docs before asking? Do you explore the codebase to understand conventions? Do you use the API reference effectively? |
| **Abstractions and writing code** | Is your code clean and well-structured? Do you follow the patterns already in the codebase? Do you handle edge cases? |
| **Correctness / Testing / Debugging** | Does your code work? Do you write tests? When something breaks, can you diagnose it systematically? |
| **Interaction and collaboration** | Do you think out loud? Do you ask clarifying questions when genuinely stuck (not before reading the docs)? Do you communicate trade-offs? |

---

## Setup

Read this ENTIRE spec before writing any code. The existing codebase is in `server.ts` -- read that too.

Your implementation goes in `starter.ts`.

```
npx tsx starter.ts
```

**Do not modify `server.ts`.** It exports everything you need.

---

## Existing Codebase (server.ts)

Spend the first few minutes reading this. Understand the patterns before you write anything.

| Component | Purpose |
|---|---|
| `Database` | Map-based store with products (pre-seeded) and orders |
| `Router` | Express-like router with `get/post/put/delete`, middleware via `use()` |
| `PaymentProvider` | Simulates an external payment API -- PaymentIntents, refunds, webhooks |
| `registerExistingRoutes()` | Already registers product + order routes |
| `makeRequest()` | Test helper -- `makeRequest(router, "POST", "/path", body)` |

### Existing Routes (already working)

```
GET  /products          -> { data: Product[], count }
GET  /products/:id      -> Product | 404
POST /orders            -> body: { customer_id, items: [{ product_id, quantity }] }
GET  /orders/:id        -> Order | 404
GET  /orders?customer_id=X -> { data: Order[], count }
```

### Order Status Flow

```
pending -> payment_processing -> paid -> partially_refunded -> refunded
                              \-> payment_failed
```

---

## PaymentProvider API Reference

All methods are synchronous. Errors throw `PaymentError` with `.code`, `.message`, `.statusCode`.

### createPaymentIntent(amount, currency, metadata?, captureMethod?, idempotencyKey?)

Creates a PaymentIntent. Returns object with `id`, `amount`, `currency`, `status: "requires_payment_method"`, `client_secret`.

Errors: `invalid_amount` (400), `amount_too_large` (400).

### confirmPaymentIntent(id, paymentMethod)

Confirms a PI. Test methods: `pm_card_visa` (succeeds), `pm_card_declined` (402), `pm_card_3ds_required` (requires_action).

Errors: `not_found` (404), `invalid_state` (400), `card_declined` (402).

### capturePaymentIntent(id, amountToCapture?)

Captures an authorized PI (manual capture only).

Errors: `not_found` (404), `invalid_state` (400), `invalid_capture_amount` (400).

### getPaymentIntent(id)

Retrieves a PI by ID. Errors: `not_found` (404).

### createRefund(paymentIntentId, amount?, reason?)

Refunds a succeeded PI. Returns `Refund` object. Defaults to full remaining amount.

Errors: `not_found` (404), `invalid_state` (400), `invalid_refund_amount` (400), `refund_exceeds_payment` (400).

### constructWebhookEvent(rawBody, signature)

Verifies webhook signature, returns parsed event. Errors: `signature_verification_failed` (400).

### createTestEvent(type, data) -- Test Helper

Creates a test webhook event with valid signature. Returns `{ rawBody, signature, event }`.

---

## Requirements

### Part 1: Checkout Flow (15 min)

**`POST /orders/:id/pay`**
- Look up order (404 if not found), must be `"pending"` (400 otherwise)
- Create PaymentIntent with order's `total`, `currency`, `metadata: { order_id }`
- Pass through `capture_method` and `idempotency_key` from body if present
- Store `payment_intent_id` on order, set status to `"payment_processing"`
- Return 200 + `{ client_secret, payment_intent_id }`
- Catch `PaymentError` -> return its `statusCode` + `{ error: message }`

**`POST /orders/:id/confirm`**
- Look up order (404), must be `"payment_processing"` (400)
- `payment_method` required in body (400 if missing)
- Call `confirmPaymentIntent(payment_intent_id, payment_method)`
- `"succeeded"` -> order `"paid"`, return `{ status: "succeeded", payment_intent }`
- `"requires_action"` -> keep `"payment_processing"`, return `{ status: "requires_action", payment_intent }`
- `"requires_capture"` -> keep `"payment_processing"`, return `{ status: "requires_capture", payment_intent }`
- `card_declined` error -> order `"payment_failed"`, return 402

### Part 2: Capture & Refunds (15 min)

**`POST /orders/:id/capture`**
- Look up order (404), must be `"payment_processing"` with `payment_intent_id` (400)
- Verify PI is in `"requires_capture"` (400 if not)
- Call `capturePaymentIntent()`, pass optional `amount` from body
- Set order to `"paid"`, return 200 + `{ payment_intent }`

**`POST /orders/:id/refund`**
- Look up order (404), must be `"paid"` or `"partially_refunded"` (400)
- Optional `amount` (positive integer) and `reason` from body
- Call `createRefund(payment_intent_id, amount, reason)`
- Record on `order.refunds` array
- If total refunded >= `order.total` -> `"refunded"`, else `"partially_refunded"`
- Return 200 + `{ refund, order }`

### Part 3: Webhook Handler (12 min)

**`POST /webhooks`**
- Read `stripe-signature` header (400 if missing)
- Verify with `constructWebhookEvent(req.rawBody, signature)` -- 400 on failure
- Deduplicate by `event.id` (return 200 if already processed)
- `payment_intent.succeeded` -> find order via metadata, set `"paid"`
- `payment_intent.payment_failed` -> find order, set `"payment_failed"`
- `charge.refunded` -> acknowledge (no order update)
- Return 200 + `{ received: true }`

### Part 4: Your Tests (8 min)

Write at least 5 tests in `runCandidateTests()`:
1. Happy path: create order, pay, confirm, verify `"paid"`
2. Declined card: confirm with `pm_card_declined`, verify `"payment_failed"`
3. Refund: pay an order, refund it, verify status
4. Webhook: send valid event, verify dedup
5. Idempotency: pay with same key twice, verify same PI returned

---

## How to Approach This (Read Before Starting)

**First 3-5 minutes: Read, don't write.** Open `server.ts` and understand the Router, Database, and PaymentProvider. Look at how the existing routes are structured. Your code should follow the same patterns.

**Use the API reference.** In the real interview, you get docs for the external service. Practice using the reference above instead of guessing at method signatures. The test payment methods (`pm_card_visa`, `pm_card_declined`, `pm_card_3ds_required`) are critical -- know what each one does.

**Work sequentially.** Part 1 -> Part 2 -> Part 3 -> Part 4. Each part builds on the previous. If you get stuck on something, leave a reasonable stub and move on. Partial credit is real.

**Write tests as you go, not just at the end.** The interviewers care about correctness and debugging ability. If you verify each route works before moving to the next, you will catch issues early and look more confident.

**Talk out loud.** Narrate your approach: "I'm going to read the order, validate the status, then call the payment provider." When you encounter an error, say what you think the issue is before diving in. This is collaboration signal.

**Handle errors like you mean it.** Every PaymentProvider call can throw `PaymentError`. The `.code`, `.message`, and `.statusCode` fields map directly to your HTTP response. This is a core Stripe pattern -- show that you understand it.

---

## Quick Reference

| Test Payment Method | Behavior |
|---|---|
| `pm_card_visa` | Succeeds |
| `pm_card_declined` | Throws `card_declined` (402) |
| `pm_card_3ds_required` | Returns `requires_action` |

| Error Class | Fields |
|---|---|
| `PaymentError` | `.code`, `.message`, `.statusCode` |

| Helper | Usage |
|---|---|
| `makeRequest(router, method, path, body?, headers?)` | Returns `{ status, body }` |
| `paymentProvider.createTestEvent(type, data)` | Returns `{ rawBody, signature, event }` for webhook testing |
