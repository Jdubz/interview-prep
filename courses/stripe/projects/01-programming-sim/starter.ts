/*
Stripe Programming Exercise Simulation
───────────────────────────────────────
Payment Fee Calculator

Time limit: 40 minutes. Set a timer now.

Read the README.md for the full problem spec.
Run with: npx tsx starter.ts
Do NOT open solution.ts until you are done.
*/

// ─── Types ───────────────────────────────────────────────────────

export type Transaction = {
  id: string;
  merchant: string;
  amount: number;
  currency: string;
  type: string;
};

export type FeeRule = {
  type: string;
  flatFee: number;
  percentFee: number;
};

export type VolumeTier = {
  upTo: number;
  discountPercent: number;
};

export type MerchantSummary = {
  totalVolume: number;
  totalFees: number;
  transactionCount: number;
};

export type DiscountedSummary = MerchantSummary & {
  discountPercent: number;
  discountedFees: number;
};

export type RejectedTransaction = {
  id: string;
  reason: string;
};

export type ProcessingResult = {
  processed: Map<string, DiscountedSummary>;
  rejected: RejectedTransaction[];
};

export type SettlementEntry = {
  merchant: string;
  currency: string;
  grossVolume: number;
  totalFees: number;
  netAmount: number;
};

// ─── Part 1 — Fee Calculation ────────────────────────────────────

export function calculateFees(
  transactions: Transaction[],
  feeSchedule: FeeRule[],
): Map<string, MerchantSummary> {
  throw new Error("TODO: implement calculateFees");
}

// ─── Part 2 — Tiered Volume Discounts ────────────────────────────

export function applyVolumeDiscounts(
  summaries: Map<string, MerchantSummary>,
  tiers: VolumeTier[],
): Map<string, DiscountedSummary> {
  throw new Error("TODO: implement applyVolumeDiscounts");
}

// ─── Part 3 — Transaction Validation & Rejection ─────────────────

export function processTransactions(
  transactions: Transaction[],
  feeSchedule: FeeRule[],
  tiers: VolumeTier[],
): ProcessingResult {
  throw new Error("TODO: implement processTransactions");
}

// ─── Part 4 — Settlement Report ──────────────────────────────────

export function generateSettlement(
  transactions: Transaction[],
  feeSchedule: FeeRule[],
): SettlementEntry[] {
  throw new Error("TODO: implement generateSettlement");
}

// ─── Self-Checks (do not edit below this line) ──────────────────

let _passed = 0;
let _failed = 0;

function check(label: string, actual: unknown, expected: unknown): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (Object.is(actual, expected) || a === e) {
    _passed++;
    console.log(`  \u2713 ${label}`);
  } else {
    _failed++;
    console.log(`  \u2717 ${label}`);
    console.log(`    expected: ${e}`);
    console.log(`         got: ${a}`);
  }
}

function checkMap(label: string, actual: Map<string, unknown>, expected: Record<string, unknown>): void {
  const actualObj: Record<string, unknown> = {};
  for (const [k, v] of actual) actualObj[k] = v;
  const a = JSON.stringify(actualObj, Object.keys(actualObj).sort());
  const e = JSON.stringify(expected, Object.keys(expected).sort());
  if (a === e) {
    _passed++;
    console.log(`  \u2713 ${label}`);
  } else {
    _failed++;
    console.log(`  \u2717 ${label}`);
    console.log(`    expected: ${e}`);
    console.log(`         got: ${a}`);
  }
}

function level(name: string, fn: () => void): void {
  console.log(name);
  try {
    fn();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.startsWith("TODO:")) {
      console.log(`  \u25CB ${msg}`);
    } else {
      _failed++;
      console.log(`  \u2717 ${msg}`);
    }
  }
}

// ─── Test Data ───────────────────────────────────────────────────

const FEE_SCHEDULE: FeeRule[] = [
  { type: "card_present",     flatFee: 10, percentFee: 200 },  // 2.00%
  { type: "card_not_present", flatFee: 30, percentFee: 290 },  // 2.90%
  { type: "international",    flatFee: 30, percentFee: 390 },  // 3.90%
];

const VOLUME_TIERS: VolumeTier[] = [
  { upTo: 50000,    discountPercent: 0 },
  { upTo: 200000,   discountPercent: 10 },
  { upTo: Infinity,  discountPercent: 20 },
];

// Valid transactions (used in Parts 1, 2, 4)
const TRANSACTIONS: Transaction[] = [
  { id: "t01", merchant: "m1", amount: 10000,  currency: "USD", type: "card_present" },
  { id: "t02", merchant: "m1", amount: 25000,  currency: "USD", type: "card_not_present" },
  { id: "t03", merchant: "m1", amount: 15000,  currency: "EUR", type: "international" },
  { id: "t04", merchant: "m1", amount: 8000,   currency: "USD", type: "card_present" },
  { id: "t05", merchant: "m2", amount: 3000,   currency: "USD", type: "card_not_present" },
  { id: "t06", merchant: "m2", amount: 7500,   currency: "USD", type: "card_present" },
  { id: "t07", merchant: "m2", amount: 4200,   currency: "GBP", type: "international" },
  { id: "t08", merchant: "m3", amount: 120000, currency: "USD", type: "card_not_present" },
  { id: "t09", merchant: "m3", amount: 95000,  currency: "USD", type: "card_present" },
  { id: "t10", merchant: "m3", amount: 45000,  currency: "EUR", type: "international" },
];

// Transactions with some invalid entries (used in Part 3)
const TRANSACTIONS_WITH_INVALID: Transaction[] = [
  ...TRANSACTIONS,
  { id: "t11", merchant: "m1", amount: -500,    currency: "USD",  type: "card_present" },
  { id: "t12", merchant: "m2", amount: 1000000, currency: "USD",  type: "card_not_present" },
  { id: "t13", merchant: "m1", amount: 5000,    currency: "usd",  type: "card_present" },
  { id: "t14", merchant: "m2", amount: 3000,    currency: "USD",  type: "wire_transfer" },
  { id: "t15", merchant: "m1", amount: 2000,    currency: "USDD", type: "card_present" },
];

// ─── Checks ──────────────────────────────────────────────────────

function runSelfChecks(): void {
  // ── Part 1 ──────────────────────────────────────────────────────
  //
  // Fee calculations:
  //   t01: m1, 10000, card_present     -> 10 + ceil(10000*200/10000) = 10 + 200 = 210
  //   t02: m1, 25000, card_not_present -> 30 + ceil(25000*290/10000) = 30 + 725 = 755
  //   t03: m1, 15000, international    -> 30 + ceil(15000*390/10000) = 30 + 585 = 615
  //   t04: m1,  8000, card_present     -> 10 + ceil(8000*200/10000)  = 10 + 160 = 170
  //   t05: m2,  3000, card_not_present -> 30 + ceil(3000*290/10000)  = 30 + 87  = 117
  //   t06: m2,  7500, card_present     -> 10 + ceil(7500*200/10000)  = 10 + 150 = 160
  //   t07: m2,  4200, international    -> 30 + ceil(4200*390/10000)  = 30 + 164 = 194
  //   t08: m3,120000, card_not_present -> 30 + ceil(120000*290/10000)= 30 + 3480= 3510
  //   t09: m3, 95000, card_present     -> 10 + ceil(95000*200/10000) = 10 + 1900= 1910
  //   t10: m3, 45000, international    -> 30 + ceil(45000*390/10000) = 30 + 1755= 1785
  //
  //   m1: volume=58000,  fees=1750, count=4
  //   m2: volume=14700,  fees=471,  count=3
  //   m3: volume=260000, fees=7205, count=3

  level("Part 1 \u2014 Fee Calculation", () => {
    const result = calculateFees(TRANSACTIONS, FEE_SCHEDULE);

    checkMap("merchant summaries", result, {
      m1: { totalVolume: 58000,  totalFees: 1750, transactionCount: 4 },
      m2: { totalVolume: 14700,  totalFees: 471,  transactionCount: 3 },
      m3: { totalVolume: 260000, totalFees: 7205, transactionCount: 3 },
    });

    // Spot-check individual merchant
    const m1 = result.get("m1")!;
    check("m1 totalVolume", m1.totalVolume, 58000);
    check("m1 totalFees", m1.totalFees, 1750);
    check("m1 transactionCount", m1.transactionCount, 4);

    const m3 = result.get("m3")!;
    check("m3 totalFees", m3.totalFees, 7205);
  });

  // ── Part 2 ──────────────────────────────────────────────────────
  //
  // Volume discount tiers:
  //   m1: volume=58000  -> 58000 <= 200000 => 10% discount
  //       discountedFees = 1750 - floor(1750*10/100) = 1750 - 175 = 1575
  //   m2: volume=14700  -> 14700 <= 50000  => 0% discount
  //       discountedFees = 471
  //   m3: volume=260000 -> 260000 > 200000 => 20% discount
  //       discountedFees = 7205 - floor(7205*20/100) = 7205 - 1441 = 5764

  level("Part 2 \u2014 Tiered Volume Discounts", () => {
    const summaries = calculateFees(TRANSACTIONS, FEE_SCHEDULE);
    const discounted = applyVolumeDiscounts(summaries, VOLUME_TIERS);

    const m1 = discounted.get("m1")!;
    check("m1 discountPercent", m1.discountPercent, 10);
    check("m1 discountedFees", m1.discountedFees, 1575);

    const m2 = discounted.get("m2")!;
    check("m2 discountPercent", m2.discountPercent, 0);
    check("m2 discountedFees", m2.discountedFees, 471);

    const m3 = discounted.get("m3")!;
    check("m3 discountPercent", m3.discountPercent, 20);
    check("m3 discountedFees", m3.discountedFees, 5764);
  });

  // ── Part 3 ──────────────────────────────────────────────────────
  //
  // Invalid transactions:
  //   t11: amount=-500     -> "invalid_amount"
  //   t12: amount=1000000  -> "invalid_amount"
  //   t13: currency="usd"  -> "invalid_currency"
  //   t14: type="wire_transfer" -> "invalid_type"
  //   t15: currency="USDD" -> "invalid_currency"

  level("Part 3 \u2014 Transaction Validation & Rejection", () => {
    const result = processTransactions(TRANSACTIONS_WITH_INVALID, FEE_SCHEDULE, VOLUME_TIERS);

    check("rejected count", result.rejected.length, 5);

    check("t11 rejected", result.rejected[0], { id: "t11", reason: "invalid_amount" });
    check("t12 rejected", result.rejected[1], { id: "t12", reason: "invalid_amount" });
    check("t13 rejected", result.rejected[2], { id: "t13", reason: "invalid_currency" });
    check("t14 rejected", result.rejected[3], { id: "t14", reason: "invalid_type" });
    check("t15 rejected", result.rejected[4], { id: "t15", reason: "invalid_currency" });

    // Valid transactions still processed correctly (same as Part 2 results)
    const m1 = result.processed.get("m1")!;
    check("m1 still correct after filtering", m1.discountedFees, 1575);
  });

  // ── Part 4 ──────────────────────────────────────────────────────
  //
  // Settlement by (merchant, currency):
  //   (m1, EUR): gross=15000,  fees=615,  net=14385
  //   (m1, USD): gross=43000,  fees=1135, net=41865
  //   (m2, GBP): gross=4200,   fees=194,  net=4006
  //   (m2, USD): gross=10500,  fees=277,  net=10223
  //   (m3, EUR): gross=45000,  fees=1785, net=43215
  //   (m3, USD): gross=215000, fees=5420, net=209580

  level("Part 4 \u2014 Settlement Report", () => {
    const settlement = generateSettlement(TRANSACTIONS, FEE_SCHEDULE);

    check("settlement length", settlement.length, 6);

    check("entry 0", settlement[0], { merchant: "m1", currency: "EUR", grossVolume: 15000,  totalFees: 615,  netAmount: 14385 });
    check("entry 1", settlement[1], { merchant: "m1", currency: "USD", grossVolume: 43000,  totalFees: 1135, netAmount: 41865 });
    check("entry 2", settlement[2], { merchant: "m2", currency: "GBP", grossVolume: 4200,   totalFees: 194,  netAmount: 4006 });
    check("entry 3", settlement[3], { merchant: "m2", currency: "USD", grossVolume: 10500,  totalFees: 277,  netAmount: 10223 });
    check("entry 4", settlement[4], { merchant: "m3", currency: "EUR", grossVolume: 45000,  totalFees: 1785, netAmount: 43215 });
    check("entry 5", settlement[5], { merchant: "m3", currency: "USD", grossVolume: 215000, totalFees: 5420, netAmount: 209580 });
  });
}

function main(): void {
  console.log("\nPayment Fee Calculator\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
