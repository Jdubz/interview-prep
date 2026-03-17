# 03 — Billing & Subscriptions

Stripe Billing builds on the Payments primitives. As a TSE, you'll field questions about subscription lifecycles, failed invoice payments, proration confusion, and metered billing setups. This lesson covers the real objects and how they connect.

---

## The Object Hierarchy

```
Product (what you sell)
└── Price (how much it costs, how often)
    └── Subscription (ongoing agreement with a customer)
        └── SubscriptionItem (line item linking a Price to the Subscription)
            └── Invoice (periodic bill generated automatically)
                └── InvoiceItem / InvoiceLineItem (individual charges on the bill)
                    └── PaymentIntent (the actual payment attempt)
```

### Product

```typescript
const product = await stripe.products.create({
  name: 'Pro Plan',
  description: 'Full access to all features',
  metadata: { tier: 'pro' },
});
```

- Represents what you sell — a SaaS plan, a physical good, a service
- Has no pricing info — that's on the Price object
- One Product can have many Prices (monthly/annual, different currencies, tiers)
- `active: true/false` controls visibility

### Price

```typescript
// Recurring price (subscription)
const monthlyPrice = await stripe.prices.create({
  product: 'prod_xxx',
  unit_amount: 2000,          // $20.00
  currency: 'usd',
  recurring: {
    interval: 'month',        // 'day', 'week', 'month', 'year'
    interval_count: 1,        // every 1 month
    usage_type: 'licensed',   // or 'metered'
  },
});

// One-time price
const setupFee = await stripe.prices.create({
  product: 'prod_xxx',
  unit_amount: 5000,
  currency: 'usd',
  // no recurring = one-time
});
```

Key fields:
- `unit_amount` — price in smallest currency unit (cents)
- `recurring.interval` — billing frequency
- `recurring.usage_type` — `licensed` (flat per-seat) or `metered` (pay for what you use)
- `billing_scheme` — `per_unit` (simple) or `tiered` (graduated/volume pricing)
- `tiers_mode` — `graduated` (each tier priced separately) or `volume` (all units priced at the tier they fall into)
- Prices are **immutable** once created — to change pricing, create a new Price and migrate subscriptions

---

## Subscription Lifecycle

### Statuses (8 states)

| Status | What it means | Provision access? |
|--------|---------------|-------------------|
| `trialing` | Free trial active, no payment collected yet | Yes |
| `active` | Payment succeeded, subscription is current | Yes |
| `incomplete` | Initial payment failed or requires authentication; 23-hour window to fix | Check your policy |
| `incomplete_expired` | 23-hour window passed without successful payment | No |
| `past_due` | Renewal payment failed; Stripe is retrying per your retry schedule | Check your policy |
| `unpaid` | All retries exhausted; subscription frozen but not canceled | No |
| `canceled` | Terminal state — no new invoices, can't be reactivated | No |
| `paused` | Collection paused (e.g., trial ended without payment method) | Your choice |

### Creating a subscription

```typescript
const subscription = await stripe.subscriptions.create({
  customer: 'cus_xxx',
  items: [
    { price: 'price_monthly_pro' },
  ],
  default_payment_method: 'pm_xxx',  // or use customer's default
  trial_period_days: 14,             // optional trial
  payment_behavior: 'default_incomplete',  // see below
  expand: ['latest_invoice.payment_intent'],
});
```

**What happens on creation:**
1. Stripe creates the Subscription object
2. Stripe generates an Invoice for the first period
3. Stripe creates a PaymentIntent from the Invoice and attempts to collect payment
4. If payment succeeds → `active`. If it fails → `incomplete` (23-hour window).

### `payment_behavior` options

| Value | Behavior |
|-------|----------|
| `default_incomplete` | Attempt payment; if it fails or needs 3DS, subscription stays `incomplete` and you handle it client-side. **Recommended for SPA integrations.** |
| `allow_incomplete` | Attempt payment; if it fails, subscription goes `incomplete` but you're signaling you'll handle recovery later |
| `error_if_incomplete` | If payment fails immediately, the API call returns an error and no subscription is created |

### Trial periods

```typescript
// Fixed trial
{ trial_period_days: 14 }

// Exact trial end
{ trial_end: Math.floor(Date.now() / 1000) + (14 * 86400) }

// Trial without payment method (paused)
{ trial_settings: { end_behavior: { missing_payment_method: 'pause' } } }
```

- During trial: `status: 'trialing'`, no invoices generated
- At trial end: Stripe generates the first invoice and attempts payment
- You can require a payment method upfront (SetupIntent during trial) or collect it at trial end

---

## Invoices

Invoices are the billing event that triggers payment collection. Stripe generates them automatically for subscriptions.

### Invoice lifecycle

```
draft ──► open ──► paid
                ──► void
                ──► uncollectible
```

| Status | Meaning |
|--------|---------|
| `draft` | Not yet finalized; you can still add/modify line items |
| `open` | Finalized; payment is being attempted or awaiting payment |
| `paid` | Successfully paid |
| `void` | Explicitly voided — no payment will be collected |
| `uncollectible` | Marked as uncollectible (bad debt) |

### Key behaviors

- **Auto-advance**: When `auto_advance: true` (default), Stripe finalizes draft invoices and attempts payment automatically
- **First invoice timing**: For `charge_automatically` subscriptions, the first invoice finalizes and charges immediately — no draft window
- **Editing window**: For `send_invoice` subscriptions, you have ~1 hour to modify the draft before it's finalized
- **Invoice items**: Additional one-time charges can be added to the next invoice via `InvoiceItem`

### Collection methods

| Method | Behavior |
|--------|----------|
| `charge_automatically` | Stripe charges the customer's default payment method. Default. |
| `send_invoice` | Stripe emails the invoice with a hosted payment page. Customer pays on their own. `days_until_due` sets the deadline. |

### Upcoming invoice preview

```typescript
// Preview what the next invoice will look like
const upcoming = await stripe.invoices.retrieveUpcoming({
  customer: 'cus_xxx',
  subscription: 'sub_xxx',
  subscription_items: [
    { id: 'si_xxx', price: 'price_new_plan' },  // preview a plan change
  ],
});
```

This is read-only — it shows what the next invoice *would* look like without making changes. Essential for showing customers the impact of a plan change before they confirm.

---

## Proration

Proration adjusts charges when a subscription changes mid-cycle. This is one of the most common sources of developer confusion.

### When proration happens

- Changing a subscription's price (upgrade/downgrade)
- Adding or removing subscription items (seats)
- Changing quantity
- Canceling mid-period (not at period end)

### How it works

When you update a subscription:

1. Stripe calculates **credit** for unused time on the old price
2. Stripe calculates **charge** for remaining time on the new price
3. These appear as separate line items on the next invoice (or immediately, depending on `proration_behavior`)

Example: Customer on $30/month plan upgrades to $50/month halfway through the cycle:
- Credit: $15 (unused half-month of $30 plan)
- Charge: $25 (remaining half-month of $50 plan)
- Net: $10 extra on the next invoice

### `proration_behavior` parameter

```typescript
await stripe.subscriptions.update('sub_xxx', {
  items: [{ id: 'si_xxx', price: 'price_new' }],
  proration_behavior: 'create_prorations',  // default
});
```

| Value | Behavior |
|-------|----------|
| `create_prorations` | Create proration line items, bill on next invoice |
| `always_invoice` | Create prorations AND immediately generate/charge an invoice |
| `none` | No proration — customer pays full new price at next billing cycle |

### Previewing prorations

```typescript
const preview = await stripe.invoices.retrieveUpcoming({
  customer: 'cus_xxx',
  subscription: 'sub_xxx',
  subscription_items: [
    { id: 'si_xxx', price: 'price_new' },
  ],
  subscription_proration_date: Math.floor(Date.now() / 1000),
});
// preview.lines.data shows the proration line items
```

**Important:** Pass the same `proration_date` to both the preview and the actual update. Stripe prorates to the second, so the amounts can differ if there's a delay between preview and update.

---

## Metered Billing

For usage-based pricing (API calls, compute hours, data storage):

```typescript
// 1. Create a metered price
const price = await stripe.prices.create({
  product: 'prod_xxx',
  currency: 'usd',
  recurring: {
    interval: 'month',
    usage_type: 'metered',
  },
  unit_amount: 10,  // $0.10 per unit
});

// 2. Create subscription (no immediate charge)
const sub = await stripe.subscriptions.create({
  customer: 'cus_xxx',
  items: [{ price: price.id }],
});

// 3. Report usage throughout the period
await stripe.subscriptionItems.createUsageRecord('si_xxx', {
  quantity: 150,
  timestamp: Math.floor(Date.now() / 1000),
  action: 'increment',  // or 'set' to override
});
```

- No charge at subscription creation — first invoice is $0
- Usage records accumulate throughout the billing period
- At period end, Stripe generates an invoice with the total usage
- `action: 'increment'` adds to the running total; `action: 'set'` replaces it
- Use `set` for absolute readings (current storage used); `increment` for events (API calls)

---

## Revenue Recovery (Smart Retries & Dunning)

When a renewal payment fails:

1. Subscription moves to `past_due`
2. Stripe retries the payment on a **Smart Retry** schedule (ML-based timing for maximum recovery)
3. You can configure a custom retry schedule in Dashboard > Settings > Billing > Revenue Recovery
4. After all retries exhaust, subscription moves to `unpaid` or `canceled` (your choice in settings)

**Dunning emails**: Stripe can automatically email customers when payments fail, including a link to update their payment method (Customer Portal or hosted invoice page).

### Customer Portal (Billing Portal)

Pre-built page where customers can manage their own subscriptions:

```typescript
const session = await stripe.billingPortal.sessions.create({
  customer: 'cus_xxx',
  return_url: 'https://example.com/account',
});
// Redirect customer to session.url
```

Customers can:
- Update payment methods
- View invoice history
- Change plans (upgrade/downgrade)
- Cancel subscriptions

Configure allowed actions in Dashboard > Settings > Billing > Customer Portal.

---

## Common Billing Issues (TSE Knowledge)

1. **"Customer was charged twice on plan change"** — Developer used `always_invoice` proration but didn't realize it generates an immediate invoice. The "double charge" is actually the prorated adjustment + the next period. Fix: explain proration line items and preview before changing.

2. **"Subscription is stuck in `incomplete`"** — Initial payment required 3D Secure authentication but the frontend doesn't handle `requires_action`. Fix: expand `latest_invoice.payment_intent` and use the PaymentIntent's `client_secret` on the frontend.

3. **"Trial ended but customer wasn't charged"** — Customer had no payment method, subscription moved to `paused` instead of `active`. Fix: collect payment method during trial with a SetupIntent, or use `trial_settings.end_behavior`.

4. **"Invoice shows $0"** — Metered billing with no usage records reported, or proration credits exactly offset the charge. Check usage records and proration line items.

5. **"How do I give a customer credit?"** — Use `Customer.balance` (adding a negative invoice item) or create a credit note against a paid invoice. Don't manually adjust amounts.

---

## Check Yourself

1. **What's the difference between a Product and a Price? Why are they separate objects?**

   A Product is what you sell (the plan or service). A Price is a specific cost configuration for that Product (amount, currency, billing interval). They're separate because one Product can have many Prices — monthly vs annual, different currencies, different tiers. Prices are immutable so you can safely change pricing by creating new Prices without affecting existing subscriptions.

2. **A subscription is in `past_due` status. Walk through what happened and what will happen next.**

   A renewal invoice was generated but the payment attempt failed (card declined, expired, etc.). Stripe is automatically retrying the payment on a smart schedule. The subscription stays `past_due` until either: (a) a retry succeeds → `active`, (b) all retries are exhausted → `unpaid` or `canceled` based on your Billing settings, or (c) the customer updates their payment method and pays the open invoice.

3. **A developer wants to show a customer "If you upgrade to Pro, your next charge will be $X." What API call do they use?**

   `stripe.invoices.retrieveUpcoming()` with the proposed subscription changes. Pass `subscription_items` with the new price and `subscription_proration_date` set to the current timestamp. The response shows exact line items including proration credits and charges.

4. **Explain the difference between `licensed` and `metered` usage types in a Price.**

   `licensed` means the customer pays a fixed amount per billing period regardless of usage — like $20/seat/month. Stripe charges at subscription creation and each renewal. `metered` means the customer pays based on reported usage — like $0.10 per API call. Stripe only charges at the end of each period based on accumulated usage records.

5. **When would you recommend `send_invoice` over `charge_automatically`?**

   For B2B / enterprise customers who need to route invoices through their internal procurement/AP process before paying — typically larger contracts with `days_until_due: 30`. For consumer SaaS, `charge_automatically` is almost always the right choice.
