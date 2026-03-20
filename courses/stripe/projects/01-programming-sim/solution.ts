/*
+================================================================+
|                                                                  |
|   REFERENCE SOLUTION -- Do not open until you've completed       |
|   your attempt. Seriously. Close this file now.                  |
|                                                                  |
+================================================================+
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
  const feeMap = new Map<string, FeeRule>();
  for (const rule of feeSchedule) {
    feeMap.set(rule.type, rule);
  }

  const summaries = new Map<string, MerchantSummary>();

  for (const tx of transactions) {
    const rule = feeMap.get(tx.type);
    if (!rule) continue; // skip unknown types in Part 1

    const fee = rule.flatFee + Math.ceil(tx.amount * rule.percentFee / 10000);

    const existing = summaries.get(tx.merchant);
    if (existing) {
      existing.totalVolume += tx.amount;
      existing.totalFees += fee;
      existing.transactionCount += 1;
    } else {
      summaries.set(tx.merchant, {
        totalVolume: tx.amount,
        totalFees: fee,
        transactionCount: 1,
      });
    }
  }

  return summaries;
}

// ─── Part 2 — Tiered Volume Discounts ────────────────────────────

export function applyVolumeDiscounts(
  summaries: Map<string, MerchantSummary>,
  tiers: VolumeTier[],
): Map<string, DiscountedSummary> {
  const sortedTiers = [...tiers].sort((a, b) => a.upTo - b.upTo);
  const result = new Map<string, DiscountedSummary>();

  for (const [merchantId, summary] of summaries) {
    let discountPercent = 0;
    for (const tier of sortedTiers) {
      if (summary.totalVolume <= tier.upTo) {
        discountPercent = tier.discountPercent;
        break;
      }
    }

    const discountAmount = Math.floor(summary.totalFees * discountPercent / 100);
    const discountedFees = summary.totalFees - discountAmount;

    result.set(merchantId, {
      ...summary,
      discountPercent,
      discountedFees,
    });
  }

  return result;
}

// ─── Part 3 — Transaction Validation & Rejection ─────────────────

function validateTransaction(tx: Transaction, feeSchedule: FeeRule[]): string | null {
  // Check amount
  if (tx.amount <= 0 || tx.amount > 999999) {
    return "invalid_amount";
  }

  // Check currency: exactly 3 uppercase A-Z
  if (!/^[A-Z]{3}$/.test(tx.currency)) {
    return "invalid_currency";
  }

  // Check type exists in fee schedule
  const validTypes = new Set(feeSchedule.map(r => r.type));
  if (!validTypes.has(tx.type)) {
    return "invalid_type";
  }

  return null;
}

export function processTransactions(
  transactions: Transaction[],
  feeSchedule: FeeRule[],
  tiers: VolumeTier[],
): ProcessingResult {
  const valid: Transaction[] = [];
  const rejected: RejectedTransaction[] = [];

  for (const tx of transactions) {
    const reason = validateTransaction(tx, feeSchedule);
    if (reason) {
      rejected.push({ id: tx.id, reason });
    } else {
      valid.push(tx);
    }
  }

  const summaries = calculateFees(valid, feeSchedule);
  const processed = applyVolumeDiscounts(summaries, tiers);

  return { processed, rejected };
}

// ─── Part 4 — Settlement Report ──────────────────────────────────

export function generateSettlement(
  transactions: Transaction[],
  feeSchedule: FeeRule[],
): SettlementEntry[] {
  const feeMap = new Map<string, FeeRule>();
  for (const rule of feeSchedule) {
    feeMap.set(rule.type, rule);
  }

  // Group by (merchant, currency)
  const groups = new Map<string, SettlementEntry>();

  for (const tx of transactions) {
    const rule = feeMap.get(tx.type);
    if (!rule) continue; // skip unknown types

    const fee = rule.flatFee + Math.ceil(tx.amount * rule.percentFee / 10000);
    const key = `${tx.merchant}|${tx.currency}`;

    const existing = groups.get(key);
    if (existing) {
      existing.grossVolume += tx.amount;
      existing.totalFees += fee;
      existing.netAmount = existing.grossVolume - existing.totalFees;
    } else {
      groups.set(key, {
        merchant: tx.merchant,
        currency: tx.currency,
        grossVolume: tx.amount,
        totalFees: fee,
        netAmount: tx.amount - fee,
      });
    }
  }

  const entries = [...groups.values()];
  entries.sort((a, b) => {
    if (a.merchant !== b.merchant) return a.merchant < b.merchant ? -1 : 1;
    return a.currency < b.currency ? -1 : a.currency > b.currency ? 1 : 0;
  });

  return entries;
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
  { type: "card_present",     flatFee: 10, percentFee: 200 },
  { type: "card_not_present", flatFee: 30, percentFee: 290 },
  { type: "international",    flatFee: 30, percentFee: 390 },
];

const VOLUME_TIERS: VolumeTier[] = [
  { upTo: 50000,    discountPercent: 0 },
  { upTo: 200000,   discountPercent: 10 },
  { upTo: Infinity,  discountPercent: 20 },
];

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
  level("Part 1 \u2014 Fee Calculation", () => {
    const result = calculateFees(TRANSACTIONS, FEE_SCHEDULE);

    checkMap("merchant summaries", result, {
      m1: { totalVolume: 58000,  totalFees: 1750, transactionCount: 4 },
      m2: { totalVolume: 14700,  totalFees: 471,  transactionCount: 3 },
      m3: { totalVolume: 260000, totalFees: 7205, transactionCount: 3 },
    });

    const m1 = result.get("m1")!;
    check("m1 totalVolume", m1.totalVolume, 58000);
    check("m1 totalFees", m1.totalFees, 1750);
    check("m1 transactionCount", m1.transactionCount, 4);

    const m3 = result.get("m3")!;
    check("m3 totalFees", m3.totalFees, 7205);
  });

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

  level("Part 3 \u2014 Transaction Validation & Rejection", () => {
    const result = processTransactions(TRANSACTIONS_WITH_INVALID, FEE_SCHEDULE, VOLUME_TIERS);

    check("rejected count", result.rejected.length, 5);

    check("t11 rejected", result.rejected[0], { id: "t11", reason: "invalid_amount" });
    check("t12 rejected", result.rejected[1], { id: "t12", reason: "invalid_amount" });
    check("t13 rejected", result.rejected[2], { id: "t13", reason: "invalid_currency" });
    check("t14 rejected", result.rejected[3], { id: "t14", reason: "invalid_type" });
    check("t15 rejected", result.rejected[4], { id: "t15", reason: "invalid_currency" });

    const m1 = result.processed.get("m1")!;
    check("m1 still correct after filtering", m1.discountedFees, 1575);
  });

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
  console.log("\nPayment Fee Calculator (Solution)\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
