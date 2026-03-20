/*
Drill 07 — Tiered Pricing / Shipping Cost Calculator

Build a pricing engine that computes shipping costs. This is the
single most commonly reported Stripe Programming Exercise problem.
Focus on clean design, speed of implementation, and handling edge cases.

Target time: 30 minutes for all 4 levels.

────────────────────────────────────────
Level 1 — Flat Per-Unit Pricing (5 min)

  type FlatConfig = {
    [destination: string]: {
      [product: string]: { unitPrice: number }
    }
  }

  type Order = { destination: string; product: string; quantity: number }

  calculateFlat(config: FlatConfig, order: Order): number
    Returns unitPrice * quantity.
    If the (destination, product) pair is not in config, return -1.

────────────────────────────────────────
Level 2 — Tiered Per-Unit Pricing (10 min)

  type Tier = { upTo: number; unitPrice: number }

  type TieredConfig = {
    [destination: string]: {
      [product: string]: { tiers: Tier[] }
    }
  }

  calculateTiered(config: TieredConfig, order: Order): number
    Each tier's upTo is inclusive. Apply each tier only to items
    in that range (graduated/marginal pricing).

    Tiers are sorted ascending by upTo. The last tier may use Infinity.

    Example: tiers [{upTo: 5, unitPrice: 10}, {upTo: 10, unitPrice: 8}, {upTo: Infinity, unitPrice: 5}]
      quantity=7:  first 5 at $10 = $50, next 2 at $8 = $16 → total $66
      quantity=12: first 5 at $10 = $50, next 5 at $8 = $40, next 2 at $5 = $10 → total $100

    If the (destination, product) pair is not in config, return -1.

────────────────────────────────────────
Level 3 — Flat Base + Tiered Overflow (10 min)

  type BaseFlat = { amount: number; coversUpTo: number }

  type BasePlusTieredConfig = {
    [destination: string]: {
      [product: string]: { baseFlat: BaseFlat; tiers: Tier[] }
    }
  }

  calculateBasePlusTiered(config: BasePlusTieredConfig, order: Order): number
    For quantities <= coversUpTo: charge baseFlat.amount.
    For quantities > coversUpTo: charge baseFlat.amount + tiered pricing
    on (quantity - coversUpTo) using the tiers array.

    If the (destination, product) pair is not in config, return -1.

────────────────────────────────────────
Level 4 — Multi-Product Orders with Discounts (10 min)

  type LineItem = { product: string; quantity: number }

  type MultiOrder = { destination: string; items: LineItem[] }

  type ProductPricing =
    | { type: "flat"; unitPrice: number }
    | { type: "tiered"; tiers: Tier[] }
    | { type: "basePlusTiered"; baseFlat: BaseFlat; tiers: Tier[] }

  type MultiConfig = {
    [destination: string]: {
      [product: string]: ProductPricing
    }
  }

  type Discount = { threshold: number; percentage: number }

  type OrderResult = {
    items: { product: string; subtotal: number }[];
    subtotal: number;
    discount: number;
    total: number;
  }

  calculateMulti(config: MultiConfig, order: MultiOrder, discount?: Discount): OrderResult
    Process each line item using its pricing type (flat, tiered, or basePlusTiered).
    If any product is missing from config, its subtotal is -1 and it is excluded
    from the order subtotal/discount/total (treat as an error line).

    If discount is provided and the subtotal (of valid items) > threshold,
    apply the percentage discount (0-100) to get the discount amount.
    total = subtotal - discount.

    Return itemized breakdown and totals.
*/

// ─── Types ───────────────────────────────────────────────────────

type Tier = { upTo: number; unitPrice: number };
type BaseFlat = { amount: number; coversUpTo: number };

type FlatConfig = {
  [destination: string]: {
    [product: string]: { unitPrice: number };
  };
};

type TieredConfig = {
  [destination: string]: {
    [product: string]: { tiers: Tier[] };
  };
};

type BasePlusTieredConfig = {
  [destination: string]: {
    [product: string]: { baseFlat: BaseFlat; tiers: Tier[] };
  };
};

type LineItem = { product: string; quantity: number };
type Order = { destination: string; product: string; quantity: number };
type MultiOrder = { destination: string; items: LineItem[] };

type ProductPricing =
  | { type: "flat"; unitPrice: number }
  | { type: "tiered"; tiers: Tier[] }
  | { type: "basePlusTiered"; baseFlat: BaseFlat; tiers: Tier[] };

type MultiConfig = {
  [destination: string]: {
    [product: string]: ProductPricing;
  };
};

type Discount = { threshold: number; percentage: number };

type OrderResult = {
  items: { product: string; subtotal: number }[];
  subtotal: number;
  discount: number;
  total: number;
};

// ─── Implementation ──────────────────────────────────────────────

// Level 1
function calculateFlat(config: FlatConfig, order: Order): number {
  throw new Error("TODO: implement calculateFlat");
}

// Level 2
function calculateTiered(config: TieredConfig, order: Order): number {
  throw new Error("TODO: implement calculateTiered");
}

// Level 3
function calculateBasePlusTiered(config: BasePlusTieredConfig, order: Order): number {
  throw new Error("TODO: implement calculateBasePlusTiered");
}

// Level 4
function calculateMulti(config: MultiConfig, order: MultiOrder, discount?: Discount): OrderResult {
  throw new Error("TODO: implement calculateMulti");
}

// ─── Self-Checks (do not edit below this line) ──────────────────

let _passed = 0;
let _failed = 0;

function check(label: string, actual: unknown, expected: unknown): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (Object.is(actual, expected) || a === e) {
    _passed++;
    console.log(`  ✓ ${label}`);
  } else {
    _failed++;
    console.log(`  ✗ ${label}`);
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
      console.log(`  ○ ${msg}`);
    } else {
      _failed++;
      console.log(`  ✗ ${msg}`);
    }
  }
}

function runSelfChecks(): void {
  const flatConfig: FlatConfig = {
    US: { widget: { unitPrice: 5 }, gadget: { unitPrice: 12 } },
    CA: { widget: { unitPrice: 7 } },
  };

  const tieredConfig: TieredConfig = {
    US: {
      widget: {
        tiers: [
          { upTo: 5, unitPrice: 10 },
          { upTo: 10, unitPrice: 8 },
          { upTo: Infinity, unitPrice: 5 },
        ],
      },
    },
    CA: {
      widget: {
        tiers: [
          { upTo: 3, unitPrice: 15 },
          { upTo: Infinity, unitPrice: 10 },
        ],
      },
    },
  };

  const baseTieredConfig: BasePlusTieredConfig = {
    US: {
      widget: {
        baseFlat: { amount: 20, coversUpTo: 3 },
        tiers: [
          { upTo: 5, unitPrice: 8 },
          { upTo: Infinity, unitPrice: 5 },
        ],
      },
    },
  };

  const multiConfig: MultiConfig = {
    US: {
      widget: { type: "flat", unitPrice: 5 },
      gadget: {
        type: "tiered",
        tiers: [
          { upTo: 5, unitPrice: 10 },
          { upTo: Infinity, unitPrice: 6 },
        ],
      },
      gizmo: {
        type: "basePlusTiered",
        baseFlat: { amount: 15, coversUpTo: 2 },
        tiers: [
          { upTo: 3, unitPrice: 7 },
          { upTo: Infinity, unitPrice: 4 },
        ],
      },
    },
  };

  level("Level 1 — Flat Per-Unit Pricing", () => {
    check("flat basic", calculateFlat(flatConfig, { destination: "US", product: "widget", quantity: 4 }), 20);
    check("flat different product", calculateFlat(flatConfig, { destination: "US", product: "gadget", quantity: 3 }), 36);
    check("flat missing product", calculateFlat(flatConfig, { destination: "US", product: "unknown", quantity: 1 }), -1);
    check("flat missing destination", calculateFlat(flatConfig, { destination: "MX", product: "widget", quantity: 1 }), -1);
  });

  level("Level 2 — Tiered Per-Unit Pricing", () => {
    check("tiered within first tier", calculateTiered(tieredConfig, { destination: "US", product: "widget", quantity: 3 }), 30);
    check("tiered spanning two tiers", calculateTiered(tieredConfig, { destination: "US", product: "widget", quantity: 7 }), 66);
    // 5 at $10 = $50, 5 at $8 = $40, 2 at $5 = $10, total = $100
    check("tiered spanning all tiers", calculateTiered(tieredConfig, { destination: "US", product: "widget", quantity: 12 }), 100);
    check("tiered missing config", calculateTiered(tieredConfig, { destination: "US", product: "unknown", quantity: 1 }), -1);
  });

  level("Level 3 — Flat Base + Tiered Overflow", () => {
    check("base covers all", calculateBasePlusTiered(baseTieredConfig, { destination: "US", product: "widget", quantity: 2 }), 20);
    check("base exactly at limit", calculateBasePlusTiered(baseTieredConfig, { destination: "US", product: "widget", quantity: 3 }), 20);
    // overflow = 7-3 = 4, tiers: [{upTo:5, unitPrice:8}, ...], 4 at $8 = $32, total = 20+32 = 52
    check("base + overflow", calculateBasePlusTiered(baseTieredConfig, { destination: "US", product: "widget", quantity: 7 }), 52);
  });

  level("Level 4 — Multi-Product Orders with Discounts", () => {
    const order: MultiOrder = {
      destination: "US",
      items: [
        { product: "widget", quantity: 4 },  // flat: 4*5 = 20
        { product: "gadget", quantity: 7 },   // tiered: 5*10 + 2*6 = 62
        { product: "gizmo", quantity: 5 },    // base+tiered: 15 + (3*7) = 15+21 = 36
      ],
    };
    const result = calculateMulti(multiConfig, order);
    check("multi itemized widget", result.items[0], { product: "widget", subtotal: 20 });
    check("multi itemized gadget", result.items[1], { product: "gadget", subtotal: 62 });
    check("multi itemized gizmo", result.items[2], { product: "gizmo", subtotal: 36 });
    check("multi subtotal", result.subtotal, 118);
    check("multi no discount", result.discount, 0);
    check("multi total", result.total, 118);

    const discountedResult = calculateMulti(multiConfig, order, { threshold: 100, percentage: 10 });
    check("multi discount applied", discountedResult.discount, 11.8);
    check("multi discounted total", discountedResult.total, 106.2);
  });
}

function main(): void {
  console.log("\nTiered Pricing / Shipping Cost Calculator\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
