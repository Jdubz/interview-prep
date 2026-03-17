# 04 — Connect & Platform Payments

Stripe Connect enables multi-party payments — platforms and marketplaces where money flows between multiple parties. As a TSE, you'll help developers choose the right integration architecture and debug fund-flow issues. Connect is the most architecturally complex Stripe product.

---

## When Connect Applies

Any time money needs to go to someone other than the platform:
- **Marketplaces** — Etsy-like: buyer pays, platform takes a cut, seller receives the rest
- **SaaS platforms** — Shopify-like: platform provides the commerce tool, connected accounts are the merchants
- **On-demand services** — Uber-like: customer pays, money splits between driver, restaurant, platform
- **Crowdfunding** — funds route to project creators

---

## Account Types

The first design decision: what type of connected accounts will your users have?

| | Standard | Express | Custom |
|--|----------|---------|--------|
| **Onboarding** | Redirects to Stripe; user creates their own full Stripe account | Stripe-hosted or embedded; lighter onboarding flow | You build everything yourself via API |
| **Dashboard** | Full Stripe Dashboard | Stripe Express Dashboard (limited view: balance, earnings, payouts) | No Stripe dashboard — you build it |
| **Branding** | Stripe's branding on statements | Co-branded (your platform + Stripe) | Fully your branding |
| **API control** | Account owner manages via their own Dashboard | Platform manages most things via API | Platform manages everything via API |
| **Support** | Stripe supports the account holder directly | Stripe provides limited support | You provide all support |
| **Best for** | Users who already have/want their own Stripe account | Most marketplaces and platforms (recommended default) | Financial platforms needing full white-label control |
| **Effort** | Lowest | Low-medium | Highest |

**Stripe's recommendation:** Start with Express unless you have a specific reason to need Standard or Custom. Most platforms use Express.

### Creating accounts

```typescript
// Express account
const account = await stripe.accounts.create({
  type: 'express',
  country: 'US',
  email: 'seller@example.com',
  capabilities: {
    card_payments: { requested: true },
    transfers: { requested: true },
  },
});

// Generate onboarding link
const accountLink = await stripe.accountLinks.create({
  account: account.id,
  refresh_url: 'https://example.com/reauth',
  return_url: 'https://example.com/onboarding-complete',
  type: 'account_onboarding',
});
// Redirect user to accountLink.url
```

### Capabilities

Connected accounts must have specific **capabilities** enabled before they can do things:

| Capability | Allows |
|-----------|--------|
| `card_payments` | Accept card payments |
| `transfers` | Receive transfers from the platform |
| `us_bank_account_ach_payments` | Accept ACH payments |
| `tax_reporting_us_1099_k` | 1099-K tax reporting |

Capabilities require identity verification. Stripe progressively collects requirements — the `account.updated` webhook fires when status changes. Check `requirements.currently_due` and `requirements.eventually_due` to know what's needed.

---

## Charge Models

The second design decision: how do funds flow?

### Direct Charges

The payment is created **on the connected account's Stripe account**.

```typescript
const paymentIntent = await stripe.paymentIntents.create({
  amount: 10000,
  currency: 'usd',
  application_fee_amount: 1000,  // platform takes $10
}, {
  stripeAccount: 'acct_xxx',     // charge on this connected account
});
```

**Fund flow:**
```
Customer → Connected Account ($90) + Platform ($10 fee)
```

- Customer relationship belongs to the **connected account**
- Connected account's info appears on bank statements
- The connected account handles disputes
- Platform collects revenue via `application_fee_amount`
- Best for: SaaS platforms where sellers are independent businesses (they have their own customer relationships)

### Destination Charges

The payment is created **on the platform** and funds are routed to a connected account.

```typescript
const paymentIntent = await stripe.paymentIntents.create({
  amount: 10000,
  currency: 'usd',
  transfer_data: {
    destination: 'acct_xxx',
    amount: 9000,               // connected account gets $90
  },
  // Platform keeps the difference ($10)
});
```

**Fund flow:**
```
Customer → Platform ($10 kept) → Connected Account ($90 transferred)
```

- Customer relationship belongs to the **platform**
- Platform's info appears on bank statements
- Platform handles disputes
- Simple fee structure: you control exactly how much goes to each party
- Best for: branded marketplaces where the customer interacts with the platform, not the seller

### Separate Charges and Transfers

The payment is created on the platform, and you transfer funds to connected accounts separately.

```typescript
// 1. Charge on platform
const paymentIntent = await stripe.paymentIntents.create({
  amount: 10000,
  currency: 'usd',
});

// 2. After payment succeeds, transfer to multiple accounts
await stripe.transfers.create({
  amount: 6000,
  currency: 'usd',
  destination: 'acct_restaurant',
  transfer_group: 'order_123',     // links related transfers
});

await stripe.transfers.create({
  amount: 2000,
  currency: 'usd',
  destination: 'acct_driver',
  transfer_group: 'order_123',
});
// Platform keeps $20
```

**Fund flow:**
```
Customer → Platform → Restaurant ($60) + Driver ($20) + Platform ($20 kept)
```

- Most flexible but most complex
- Charge and transfers are decoupled — you can transfer later, to multiple accounts, or adjust amounts
- Requires you to monitor your platform's balance (insufficient funds = failed transfers)
- Best for: complex multi-party splits (food delivery, travel booking with multiple suppliers)

### Decision Matrix

| Consideration | Direct | Destination | Separate Charges + Transfers |
|--------------|--------|-------------|-------------------------------|
| Who owns the customer? | Connected account | Platform | Platform |
| Who appears on statement? | Connected account | Platform | Platform |
| Who handles disputes? | Connected account | Platform | Platform |
| Multi-party splits? | No | No (1 destination) | Yes |
| Complexity | Low | Low | High |
| When to use | Sellers are independent businesses | Branded marketplace | Multi-party or delayed transfers |

### The `on_behalf_of` parameter

Applied to destination charges or separate charges to make the connected account the "business of record":

```typescript
const paymentIntent = await stripe.paymentIntents.create({
  amount: 10000,
  currency: 'usd',
  on_behalf_of: 'acct_xxx',        // settlement in this account's country
  transfer_data: {
    destination: 'acct_xxx',
  },
});
```

- Settles in the connected account's country/currency
- Uses the connected account's statement descriptor
- Optimizes for the connected account's payout schedule
- Reduces declines and currency conversion fees

---

## Application Fees

How the platform makes money:

**On direct charges:**
```typescript
{ application_fee_amount: 1000 }  // $10 fee to platform
```

**On destination charges:**
```typescript
{
  transfer_data: {
    destination: 'acct_xxx',
    amount: 9000,  // send $90, keep $10 implicitly
  },
}
// OR
{
  application_fee_amount: 1000,  // explicit: $10 fee to platform
  transfer_data: { destination: 'acct_xxx' },
}
```

Application fees create a separate `ApplicationFee` object that appears in the platform's Dashboard and reporting.

---

## Payouts

Payouts move money from a Stripe balance to an external bank account or debit card.

```typescript
// Manual payout (if automatic payouts are disabled)
const payout = await stripe.payouts.create({
  amount: 5000,
  currency: 'usd',
}, {
  stripeAccount: 'acct_xxx',   // for connected account
});
```

### Payout schedule

- **Automatic** (default): Stripe pays out on a schedule (daily, weekly, monthly) based on the account's country and settings
- **Manual**: Platform controls when payouts happen via API
- Payout timing: typically 2 business days for US card payments (varies by country and payment method)

### Payout statuses

| Status | Meaning |
|--------|---------|
| `pending` | Created, waiting for processing |
| `in_transit` | Funds are moving to the bank |
| `paid` | Funds deposited in the bank account |
| `failed` | Payout failed (invalid bank details, etc.) |
| `canceled` | Canceled before processing |

---

## Negative Balances & Risk

With Connect, the platform may be liable for connected accounts' negative balances (from refunds, disputes on accounts that have already been paid out).

**Two approaches:**

1. **Platform liability** (default for most setups): Your platform balance is debited if a connected account can't cover a refund/dispute. Requires you to build fraud monitoring.
2. **Stripe liability**: Stripe absorbs losses from connected account negatives. Available for certain account configurations. Stripe recommends this as the default when possible.

---

## Common Connect Issues (TSE Knowledge)

1. **"Transfer failed: insufficient funds"** — Platform balance doesn't have enough to cover a separate-charges-and-transfers flow. Fix: ensure platform balance covers transfers before creating them, or use destination charges which handle this automatically.

2. **"Connected account can't accept payments"** — `card_payments` capability not yet active. Check `account.requirements.currently_due` — identity verification may be incomplete. The `account.updated` webhook tells you when capabilities change.

3. **"Customer was charged but money didn't appear in connected account"** — For destination charges, funds transfer immediately but may be in `pending` balance (not yet available for payout). For separate charges, the developer may have forgotten to create the transfer.

4. **"Refund failed on connected account"** — Connected account's balance is insufficient to cover the refund. For direct charges, refund comes from the connected account's balance. For destination charges, you can reverse the transfer or refund from the platform.

5. **Onboarding link expired** — Account links are **single-use** and expire. Generate a new one by calling `stripe.accountLinks.create()` again with `type: 'account_onboarding'`.

---

## Check Yourself

1. **A developer is building a food delivery app. The customer pays $30 — $20 goes to the restaurant, $5 to the driver, and $5 to the platform. Which charge model should they use and why?**

   Separate charges and transfers. This is the only model that supports splitting a single payment across multiple connected accounts. They'd create a PaymentIntent for $30 on the platform, then create two transfers: $20 to the restaurant account and $5 to the driver account, keeping $5.

2. **What's the difference between `application_fee_amount` on a direct charge vs setting a specific `transfer_data.amount` on a destination charge?**

   With `application_fee_amount` on a direct charge, the payment lands in the connected account's balance and the fee is extracted to the platform — the connected account "owns" the payment. With `transfer_data.amount` on a destination charge, the payment lands in the platform's balance and you decide how much to send to the connected account — the platform "owns" the payment. Both achieve a fee, but ownership, statement descriptors, and dispute responsibility differ.

3. **An Express connected account just signed up but `capabilities.card_payments` shows `inactive`. What's happening?**

   Stripe is still collecting and verifying identity information. Check `account.requirements.currently_due` for what's needed. The account may need to provide additional documents, banking info, or identity verification. Subscribe to `account.updated` webhooks to know when capabilities become active.

4. **When would you use `on_behalf_of` and what does it change?**

   Use it when the connected account should be the business of record — their statement descriptor, their country's settlement, their fee structure. It's common in destination charge setups where you want the connected account to appear as the merchant to the cardholder, reducing declines and avoiding unnecessary currency conversions. It doesn't change who owns the customer object — the platform still owns it.

5. **A marketplace platform is seeing losses from connected accounts that get paid out, then the customer disputes. How should they address this?**

   Short-term: Hold payouts longer to build a buffer (increase payout delay). Consider requiring reserves on connected accounts. Long-term: use Stripe's risk-transfer options where Stripe absorbs negative balance losses. Also implement Radar rules and monitor connected accounts' dispute rates. For destination charges, the platform handles disputes directly and can control refund/payout timing.
