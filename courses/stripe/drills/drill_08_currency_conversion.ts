/*
Drill 08 — Currency Conversion

Build a currency converter that supports graph traversal for
multi-hop conversions. This is a well-known Stripe interview
classic that tests graph modeling and traversal algorithms.

Target time: 30 minutes for all 4 levels.

────────────────────────────────────────
Level 1 — Direct Conversion (5 min)

  constructor(ratesStr: string)
    Parse exchange rates from a string like
    "USD:EUR:0.85,USD:JPY:110,EUR:GBP:0.88".
    Build a bidirectional rate map: if USD→EUR = 0.85,
    then EUR→USD = 1/0.85.

  convert(from: string, to: string, amount: number): number
    Return the converted amount using a direct rate.
    If from === to, return amount unchanged.
    Return -1 if no direct rate exists.

────────────────────────────────────────
Level 2 — Multi-Hop Conversion via DFS (15 min)

  convertDFS(from: string, to: string, amount: number):
    { path: string[]; rate: number; amount: number } | null

    Find any valid conversion path using depth-first search.
    Track visited nodes to avoid cycles.
    Multiply rates along the path.
    Return { path, rate, amount } or null if no path exists.
    Example: USD→EUR→GBP, rate = 0.85 * 0.88 = 0.748,
    amount = 100 * 0.748 = 74.8

────────────────────────────────────────
Level 3 — Shortest Path via BFS (10 min)

  convertBFS(from: string, to: string, amount: number):
    { path: string[]; rate: number; amount: number } | null

    Find the path with fewest intermediate conversions using BFS.
    Track parent pointers and accumulate rates along the path.
    Return the shortest path result (same shape as Level 2).

────────────────────────────────────────
Level 4 — Best Rate Path (10 min)

  convertBestRate(from: string, to: string, amount: number):
    { path: string[]; rate: number; amount: number } | null

    Find the path that gives the best (highest) conversion rate.
    Explore all paths (modified DFS) to find the optimal one.
    Return the best-rate result (same shape as Level 2).
*/

export class CurrencyConverter {
  constructor(ratesStr: string) {
    throw new Error("TODO: implement constructor — parse rates string and build bidirectional rate map");
  }

  // Level 1
  convert(from: string, to: string, amount: number): number {
    throw new Error("TODO: implement convert — direct conversion, return -1 if no direct rate");
  }

  // Level 2
  convertDFS(from: string, to: string, amount: number): { path: string[]; rate: number; amount: number } | null {
    throw new Error("TODO: implement convertDFS — find any path via DFS");
  }

  // Level 3
  convertBFS(from: string, to: string, amount: number): { path: string[]; rate: number; amount: number } | null {
    throw new Error("TODO: implement convertBFS — find shortest path via BFS");
  }

  // Level 4
  convertBestRate(from: string, to: string, amount: number): { path: string[]; rate: number; amount: number } | null {
    throw new Error("TODO: implement convertBestRate — find path with best (highest) rate");
  }
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
  const RATES = "USD:EUR:0.85,USD:JPY:110,EUR:GBP:0.88,GBP:CHF:1.18";

  level("Level 1 — Direct Conversion", () => {
    const cc = new CurrencyConverter(RATES);
    check("same currency", cc.convert("USD", "USD", 100), 100);
    check("USD to EUR", cc.convert("USD", "EUR", 100), 85);
    check("EUR to USD (reverse)", cc.convert("EUR", "USD", 85), 100);
    check("no direct rate", cc.convert("USD", "GBP", 100), -1);
  });

  level("Level 2 — Multi-Hop DFS", () => {
    const cc = new CurrencyConverter(RATES);
    const result = cc.convertDFS("USD", "GBP", 100);
    check("DFS path exists", result !== null, true);
    check("DFS path is USD→EUR→GBP", result!.path, ["USD", "EUR", "GBP"]);
    check("DFS rate", result!.rate, 0.85 * 0.88);
    check("DFS amount", result!.amount, 100 * 0.85 * 0.88);
    const same = cc.convertDFS("USD", "USD", 50);
    check("DFS same currency", same, { path: ["USD"], rate: 1, amount: 50 });
    const none = cc.convertDFS("USD", "XYZ", 100);
    check("DFS no path", none, null);
  });

  level("Level 3 — Shortest Path BFS", () => {
    // Two paths exist from USD to CHF:
    //   Short: USD→EUR→CHF (2 hops) via direct EUR:CHF rate
    //   Long:  USD→EUR→GBP→CHF (3 hops)
    // BFS should find the shorter one.
    const rates = RATES + ",EUR:CHF:1.08";
    const cc = new CurrencyConverter(rates);
    const result = cc.convertBFS("USD", "CHF", 100);
    check("BFS path exists", result !== null, true);
    check("BFS shortest path", result!.path, ["USD", "EUR", "CHF"]);
    const expectedRate = 0.85 * 1.08;
    check("BFS rate", result!.rate, expectedRate);
    check("BFS amount", result!.amount, 100 * expectedRate);
    check("BFS no path", cc.convertBFS("USD", "XYZ", 100), null);
  });

  level("Level 4 — Best Rate Path", () => {
    // Two paths from USD to GBP:
    //   Direct: USD→GBP = 0.72
    //   Via EUR: USD→EUR→GBP = 0.85 * 0.88 = 0.748
    // Best rate is via EUR (0.748 > 0.72)
    const rates = "USD:EUR:0.85,USD:GBP:0.72,EUR:GBP:0.88";
    const cc = new CurrencyConverter(rates);
    const result = cc.convertBestRate("USD", "GBP", 100);
    check("best rate path exists", result !== null, true);
    check("best rate picks higher", result!.path, ["USD", "EUR", "GBP"]);
    check("best rate", result!.rate, 0.85 * 0.88);
    check("best rate amount", result!.amount, 100 * 0.85 * 0.88);
    check("best rate no path", cc.convertBestRate("USD", "XYZ", 100), null);
  });
}

function main(): void {
  console.log("\nCurrency Conversion\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
