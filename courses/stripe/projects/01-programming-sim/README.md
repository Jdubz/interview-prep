# Programming Exercise Simulation

## Payment Fee Calculator

**Timer: 40 minutes.** Your actual round is 45 min. Train tight.

Read this entire spec before writing any code. Talk out loud — explain your approach, trade-offs, and decisions.

```
npx tsx starter.ts
```

Do **NOT** open `solution.ts` until you have completed your attempt.

---

## Scenario

You are building a fee calculation engine for Stripe's merchant payment processing. Merchants pay processing fees on every transaction, with rates that vary by transaction type. High-volume merchants earn discounted rates.

All monetary values are in **cents** (integers only — no floating point).

---

## Data Types

```typescript
type Transaction = {
  id: string;         // unique transaction id
  merchant: string;   // merchant id
  amount: number;     // transaction amount in cents (e.g. 10000 = $100.00)
  currency: string;   // 3-letter uppercase currency code, e.g. "USD"
  type: string;       // transaction type, e.g. "card_present"
};

type FeeRule = {
  type: string;         // matches Transaction.type
  flatFee: number;      // flat fee in cents (e.g. 30 = $0.30)
  percentFee: number;   // fee rate in basis points (e.g. 290 = 2.90%)
};

type VolumeTier = {
  upTo: number;           // volume ceiling in cents (use Infinity for the last tier)
  discountPercent: number; // discount applied to total fees (e.g. 10 = 10%)
};
```

---

## Part 1 — Fee Calculation (12 min)

Implement:

- **`calculateFees(transactions, feeSchedule): Map<string, MerchantSummary>`**

For each transaction, calculate: `fee = flatFee + Math.ceil(amount * percentFee / 10000)`

Return a `Map<merchantId, MerchantSummary>` where:

```typescript
type MerchantSummary = {
  totalVolume: number;       // sum of transaction amounts
  totalFees: number;         // sum of calculated fees
  transactionCount: number;  // number of transactions
};
```

### Example

```
Fee schedule:
  card_present:     flatFee = 10, percentFee = 200  (2.00%)
  card_not_present: flatFee = 30, percentFee = 290  (2.90%)

Transactions:
  { id: "t1", merchant: "m1", amount: 10000, currency: "USD", type: "card_present" }
  { id: "t2", merchant: "m1", amount: 25000, currency: "USD", type: "card_not_present" }

Fee for t1: 10 + ceil(10000 * 200 / 10000) = 10 + 200 = 210
Fee for t2: 30 + ceil(25000 * 290 / 10000) = 30 + 725 = 755

Result: Map { "m1" => { totalVolume: 35000, totalFees: 965, transactionCount: 2 } }
```

---

## Part 2 — Tiered Volume Discounts (13 min)

Implement:

- **`applyVolumeDiscounts(summaries, tiers): Map<string, DiscountedSummary>`**

Each merchant's total fees are discounted based on their `totalVolume`. The discount tier is determined by volume, not per-transaction.

```typescript
type DiscountedSummary = MerchantSummary & {
  discountPercent: number;  // the discount tier that applied
  discountedFees: number;   // fees after discount
};
```

Discount formula: `discountedFees = totalFees - Math.floor(totalFees * discountPercent / 100)`

Tiers are sorted by `upTo` ascending. A merchant falls into the tier where `totalVolume <= upTo`.

### Example

```
Tiers:
  { upTo: 50000,   discountPercent: 0 }
  { upTo: 200000,  discountPercent: 10 }
  { upTo: Infinity, discountPercent: 20 }

Merchant with totalVolume = 58000, totalFees = 1750:
  Volume 58000 > 50000, 58000 <= 200000 => 10% discount
  discountedFees = 1750 - Math.floor(1750 * 10 / 100) = 1750 - 175 = 1575
```

---

## Part 3 — Transaction Validation & Rejection (10 min)

Implement:

- **`processTransactions(transactions, feeSchedule, tiers): ProcessingResult`**

Before calculating fees, validate each transaction:

1. `amount` must be > 0 and <= 999999
2. `currency` must be exactly 3 uppercase letters (A-Z)
3. `type` must exist in the fee schedule

Invalid transactions are **rejected** — they do not count toward volume or fees.

```typescript
type RejectedTransaction = {
  id: string;
  reason: string;  // e.g. "invalid_amount", "invalid_currency", "invalid_type"
};

type ProcessingResult = {
  processed: Map<string, DiscountedSummary>;
  rejected: RejectedTransaction[];
};
```

Rejection reasons (use these exact strings):
- `"invalid_amount"` — amount <= 0 or amount > 999999
- `"invalid_currency"` — not exactly 3 uppercase A-Z letters
- `"invalid_type"` — type not found in fee schedule

If a transaction has multiple issues, return the **first** reason in the order above.

---

## Part 4 — Settlement Report (5 min, stretch)

Implement:

- **`generateSettlement(transactions, feeSchedule): SettlementEntry[]`**

Group **valid** transactions by `(merchant, currency)`. For each group:

```typescript
type SettlementEntry = {
  merchant: string;
  currency: string;
  grossVolume: number;   // sum of amounts
  totalFees: number;     // sum of fees
  netAmount: number;     // grossVolume - totalFees
};
```

Return sorted by `merchant` ascending, then `currency` ascending.

**Note:** This function only filters out transactions with invalid types (type not in fee schedule). It does NOT apply the full validation from Part 3 — only type checking is needed to look up fee rates.

### Example

```
Transactions for merchant "m1":
  { amount: 10000, currency: "USD", type: "card_present" }   => fee 210
  { amount: 15000, currency: "EUR", type: "international" }  => fee 615
  { amount: 25000, currency: "USD", type: "card_not_present" } => fee 755

Settlement:
  { merchant: "m1", currency: "EUR", grossVolume: 15000, totalFees: 615, netAmount: 14385 }
  { merchant: "m1", currency: "USD", grossVolume: 35000, totalFees: 965, netAmount: 34035 }
```

---

## Pacing

| Part | Target | Cumulative |
|---|---|---|
| Read spec + design | 3 min | 3 min |
| Part 1 | 12 min | 15 min |
| Part 2 | 13 min | 28 min |
| Part 3 | 10 min | 38 min |
| Part 4 | 5 min | 43 min |

Parts 1-2 clean is a strong result. If stuck 3+ min past target, move on.
