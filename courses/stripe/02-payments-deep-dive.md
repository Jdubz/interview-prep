# 02 — Payments Deep Dive

Payments is Stripe's core product. As a TSE, this is where the majority of developer questions land. You need to know the PaymentIntent lifecycle cold, understand how different payment methods behave, and be able to debug common integration mistakes.

---

## The PaymentIntent Lifecycle

A PaymentIntent tracks a payment from creation through completion. It's the modern replacement for the legacy Charges API.

### Statuses

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
requires_payment_method ──► requires_confirmation ──► requires_action
         ▲                          │                      │
         │                          │                      │
         │ (decline)                ▼                      ▼
         └──────────────────── processing ──────────► succeeded
                                    │
                                    ▼
                              requires_capture ──────► succeeded

         Any pre-processing state ──────────────────► canceled
```

| Status | Meaning |
|--------|---------|
| `requires_payment_method` | No payment method attached yet, or previous attempt was declined |
| `requires_confirmation` | Has a payment method, waiting for you to confirm |
| `requires_action` | Customer must complete additional authentication (3D Secure, redirect) |
| `processing` | Payment is being processed (async methods like ACH can stay here for days) |
| `requires_capture` | Auth succeeded, funds are held — you must capture within 7 days |
| `succeeded` | Payment is complete |
| `canceled` | Explicitly canceled — cannot be reused |

**Critical transitions to know:**
- A **decline** returns the PI to `requires_payment_method` — the same PI can be retried with a new payment method
- `requires_action` means the client must handle 3D Secure or a redirect — your server can't complete this alone
- `requires_capture` only happens with `capture_method: "manual"` (authorization hold pattern)

### Creating and confirming

Two main flows:

**Server-side confirmation:**
```typescript
// 1. Create with payment method and confirm in one step
const pi = await stripe.paymentIntents.create({
  amount: 2000,            // $20.00 in cents
  currency: 'usd',
  payment_method: 'pm_xxx',
  customer: 'cus_xxx',
  confirm: true,           // confirm immediately
  return_url: 'https://example.com/return',  // for redirect-based methods
}, {
  idempotencyKey: 'order_12345',
});
```

**Client-side confirmation (most common for card payments):**
```typescript
// Server: create the PaymentIntent
const pi = await stripe.paymentIntents.create({
  amount: 2000,
  currency: 'usd',
  automatic_payment_methods: { enabled: true },
});
// Send pi.client_secret to the frontend

// Client (Stripe.js):
const { error } = await stripe.confirmPayment({
  elements,
  confirmParams: {
    return_url: 'https://example.com/return',
  },
});
```

**Why client-side confirmation matters:** 3D Secure modals, bank redirects, and other customer-action flows are handled natively by Stripe.js. If you confirm server-side, you need to handle `requires_action` yourself.

### The `client_secret`

- Generated when the PaymentIntent is created
- Passed to the frontend to complete payment
- **Must not be logged, stored in URLs, or exposed beyond the customer's session**
- Grants limited ability to confirm and read that specific PI — it's scoped, not a full API key

---

## Payment Methods

### `automatic_payment_methods`

The modern approach: set `automatic_payment_methods: { enabled: true }` on the PaymentIntent and let Stripe determine which methods are available based on the customer's location, currency, and your Dashboard settings.

This replaces the old approach of explicitly listing `payment_method_types: ['card', 'ideal', ...]`.

### Key payment method types

| Type | Behavior | Confirmation | Settlement |
|------|----------|-------------|------------|
| `card` | Synchronous | Instant | Instant (or manual capture) |
| `us_bank_account` | Asynchronous | Days (ACH) | 4-5 business days |
| `sepa_debit` | Asynchronous | Days | 5-14 business days |
| `ideal` / `bancontact` / `sofort` | Redirect | Redirect to bank | Near-instant |
| `link` | Saved payment info | Fast checkout | Depends on underlying method |
| `klarna` / `afterpay_clearpay` | Buy now pay later | Redirect | Next business day |

**Why this matters for TSE:** Developers often assume all payment methods behave like cards (instant confirmation). A common support case is "my webhook never fires `payment_intent.succeeded`" — because ACH takes days.

### Saving payment methods for later

Two patterns:

1. **SetupIntent** — save without charging (described in lesson 01)
2. **`setup_future_usage` on a PaymentIntent** — charge now AND save for later

```typescript
const pi = await stripe.paymentIntents.create({
  amount: 2000,
  currency: 'usd',
  customer: 'cus_xxx',
  setup_future_usage: 'off_session',  // save for recurring/off-session use
});
```

`off_session` vs `on_session`:
- `off_session` — you'll charge later without the customer present (subscriptions, metered billing). Stripe requests SCA exemptions and may trigger 3DS upfront to avoid future declines.
- `on_session` — you'll charge later but the customer will be in-session (e.g., reorder button). Less friction upfront.

---

## Checkout Sessions

Stripe Checkout is a pre-built, hosted payment page. Developers love it for speed; TSEs field fewer bugs with Checkout than custom integrations.

```typescript
const session = await stripe.checkout.sessions.create({
  mode: 'payment',          // or 'subscription' or 'setup'
  line_items: [{
    price: 'price_xxx',     // a Price object ID
    quantity: 1,
  }],
  success_url: 'https://example.com/success?session_id={CHECKOUT_SESSION_ID}',
  cancel_url: 'https://example.com/cancel',
});
// Redirect customer to session.url
```

Key points:
- Three modes: `payment` (one-time), `subscription` (recurring), `setup` (save card)
- Handles payment method selection, 3DS, and localization automatically
- Creates the PaymentIntent/Subscription/SetupIntent under the hood
- `{CHECKOUT_SESSION_ID}` is a template — Stripe replaces it with the actual ID on redirect
- **Always verify payment status server-side** — don't trust the redirect alone. Use the `checkout.session.completed` webhook.

---

## Refunds

```typescript
const refund = await stripe.refunds.create({
  payment_intent: 'pi_xxx',    // or charge: 'ch_xxx'
  amount: 1000,                // partial refund; omit for full refund
  reason: 'requested_by_customer',  // optional metadata
});
```

Key points:
- Refund object has its own statuses: `succeeded`, `pending`, `failed`, `canceled`
- Partial refunds are supported — you can refund multiple times up to the original amount
- Refund timing depends on the payment method (card refunds: 5-10 business days to appear on statement)
- Refunding updates the PaymentIntent status only if it's a full refund on certain flows
- The original Charge object tracks `amount_refunded` and `refunded: true/false`

---

## Disputes (Chargebacks)

When a customer disputes a charge with their bank:

1. Stripe creates a `Dispute` object and fires `charge.dispute.created`
2. The disputed amount is immediately deducted from your balance (plus a dispute fee)
3. You can submit evidence via the API or Dashboard
4. The bank decides — if you win, funds and fee are returned

```typescript
// Submit evidence
await stripe.disputes.update('dp_xxx', {
  evidence: {
    customer_email_address: 'customer@example.com',
    product_description: 'Premium subscription',
    uncategorized_text: 'Customer used the service for 30 days before disputing...',
  },
});
```

**Dispute reasons:** `duplicate`, `fraudulent`, `subscription_canceled`, `product_unacceptable`, `product_not_received`, `unrecognized`, `credit_not_processed`, `general`

---

## Authorization & Capture (Manual Capture)

For "place a hold" scenarios (hotels, rental cars, restaurants with tips):

```typescript
// 1. Authorize (hold funds)
const pi = await stripe.paymentIntents.create({
  amount: 15000,
  currency: 'usd',
  payment_method: 'pm_xxx',
  customer: 'cus_xxx',
  capture_method: 'manual',   // hold, don't charge
  confirm: true,
});
// Status: requires_capture

// 2. Capture (charge the held funds, possibly a different amount)
const captured = await stripe.paymentIntents.capture('pi_xxx', {
  amount_to_capture: 17500,   // can capture up to 120% of original for certain MCCs
});
// Status: succeeded
```

- Authorization hold expires after **7 days** (card-dependent)
- If you don't capture, the hold releases and the PI is effectively abandoned
- You can capture less than the authorized amount (partial capture)

---

## Common Integration Mistakes (TSE Knowledge)

These are the issues you'd help developers debug:

1. **Not handling `requires_action`** — developer confirms server-side, gets `requires_action` for 3DS, has no client-side code to handle the authentication modal. Fix: use Stripe.js `confirmPayment` or handle the `next_action` redirect.

2. **Relying on redirect instead of webhooks** — checking `success_url` redirect to confirm payment. The redirect can fail (user closes tab, network issue). Fix: always use `payment_intent.succeeded` webhook as source of truth.

3. **Double-charging from retries** — server creates a new PaymentIntent on each retry instead of reusing the same one. Fix: use idempotency keys, or store the PI ID and retrieve it.

4. **Wrong `client_secret` handling** — logging it, putting it in a URL, or sharing across sessions. Fix: pass it directly to the frontend in the same request/render, don't persist it.

5. **Assuming instant confirmation for all methods** — ACH, SEPA, and BECS take days. Status stays at `processing`. Fix: inform the user their payment is pending and fulfill on `payment_intent.succeeded` webhook.

6. **Not setting `return_url`** — required for redirect-based payment methods (iDEAL, Bancontact, etc.) and 3DS. Without it, the customer can't return to your site after authenticating.

7. **Currency in wrong format** — `amount: 20` means $0.20, not $20.00. Stripe uses the smallest currency unit (cents for USD). Zero-decimal currencies like JPY use the full amount.

---

## Check Yourself

1. **A PaymentIntent is in `requires_action` status. What does the developer need to do?**

   The customer needs to complete additional authentication (usually 3D Secure). The developer should use Stripe.js on the client side to handle the `next_action` — either a 3DS modal or a redirect to the bank. This cannot be completed server-side.

2. **What's the difference between `automatic_payment_methods: { enabled: true }` and `payment_method_types: ['card']`?**

   `automatic_payment_methods` lets Stripe dynamically select available methods based on the customer's location, currency, and your Dashboard settings — it's the modern approach and shows more relevant options. `payment_method_types` is an explicit list you manage yourself and requires code changes to add new methods.

3. **A developer says "I created a PaymentIntent and the charge went through, but my webhook handler for `payment_intent.succeeded` never fired." What do you check?**

   Check: (a) is the webhook endpoint registered in the Dashboard for that event type? (b) is the endpoint URL correct and reachable? (c) is the endpoint returning 2xx? Check failed delivery attempts in Dashboard > Developers > Webhooks. (d) is the webhook using the right API version? (e) for test mode, is the endpoint registered for test mode events?

4. **When should you use `capture_method: 'manual'` vs the default?**

   Manual capture for scenarios where the final amount isn't known at authorization time (tips, hotel incidentals, rental damage deposits) or where you need to verify inventory/availability before actually charging. Default (automatic) for standard purchases where you charge the full amount immediately.

5. **A developer is building a subscription service and wants to save the card during the first payment. What parameter do they add to the PaymentIntent?**

   `setup_future_usage: 'off_session'` — this tells Stripe to authenticate the card for future recurring use and attach the PaymentMethod to the Customer.

6. **Explain the difference between a Refund and a Dispute in terms a non-technical PM would understand.**

   A refund is when the merchant voluntarily returns money — like a store return. A dispute (chargeback) is when the customer goes to their bank and says "I didn't authorize this" or "I never received this." The bank forces the money back and charges the merchant a fee. The merchant can fight it with evidence, but the bank decides.
