/*
Drill 09 — String Parsing

String parsing patterns that appear constantly in Stripe interviews.
Accept-Language header parsing, invoice memo extraction, delimiter-based
parsing with edge cases. Focus on clean splitting, normalization, and
matching logic.

Target time: 30 minutes for all 4 levels.

────────────────────────────────────────
Level 1 — Accept-Language Basic Matching (8 min)

  parseAcceptLanguage(header: string, supported: string[]): string[]

    Parse an Accept-Language header like "en-US, fr-CA, fr-FR".
    Return the subset of supported languages that appear in the header,
    in preference order (left-to-right from the header).

    - Trim whitespace around each tag
    - Case-insensitive matching ("en-us" matches "en-US")
    - Only return exact tag matches at this level
    - If no supported languages match, return []

────────────────────────────────────────
Level 2 — Language Variants and Wildcards (8 min)

  parseAcceptLanguageWithVariants(header: string, supported: string[]): string[]

    Extends Level 1 with two new features:

    - Non-region tags: "en" matches any supported language starting with
      "en-" (e.g., "en-US", "en-GB"). Exact matches still take priority
      over prefix matches.
    - Wildcard: "*" matches all remaining supported languages not yet
      in the result.
    - Explicitly listed languages always take precedence over wildcard.
    - Return deduplicated results in preference order.

────────────────────────────────────────
Level 3 — Quality Factors (q-values) (8 min)

  parseAcceptLanguageQuality(header: string, supported: string[]): string[]

    Parse quality weights: "en-US;q=0.8, fr;q=0.9, *;q=0.1"

    - Default q=1.0 if no ;q= is specified
    - q=0 means "specifically not wanted" — exclude these tags entirely
    - Sort matched languages by q-value descending
    - For ties in q-value, preserve original header order
    - When a prefix tag matches multiple supported languages (e.g., "fr"
      matches "fr-CA" and "fr-FR"), they appear in the order they are
      listed in the supported array, all at the same q-value
    - Supports prefix matching ("en" → "en-US") and wildcard ("*")
    - Return sorted, deduplicated matching languages

────────────────────────────────────────
Level 4 — Invoice Memo Parser (8 min)

  reconcilePayments(payments: string[], invoices: string[]): Reconciliation[]

    Parse payment strings:  "PAY001,1500,Paying off: INV-2024-001"
      Format: paymentId,amount,memo
      The memo contains an invoice ID after "Paying off: " or
      "Payment for: " (case-insensitive).

    Parse invoice strings:  "INV-2024-001,2024-01-15,2000"
      Format: invoiceId,date,total

    Match payments to invoices by extracting the invoice ID from the memo.

    Return an array of Reconciliation objects:
      { paymentId, invoiceId, amount, invoiceDate, invoiceTotal, remaining }

    Where remaining = invoiceTotal - (sum of all payments to that invoice).

    Handle:
      - Partial payments (remaining > 0)
      - Multiple payments to the same invoice
      - Payments referencing non-existent invoices (invoiceDate = null,
        invoiceTotal = 0, remaining = -amount)
      - Return results in the order payments appear
*/

interface Reconciliation {
  paymentId: string;
  invoiceId: string;
  amount: number;
  invoiceDate: string | null;
  invoiceTotal: number;
  remaining: number;
}

// Level 1
function parseAcceptLanguage(header: string, supported: string[]): string[] {
  throw new Error("TODO: implement parseAcceptLanguage");
}

// Level 2
function parseAcceptLanguageWithVariants(header: string, supported: string[]): string[] {
  throw new Error("TODO: implement parseAcceptLanguageWithVariants");
}

// Level 3
function parseAcceptLanguageQuality(header: string, supported: string[]): string[] {
  throw new Error("TODO: implement parseAcceptLanguageQuality");
}

// Level 4
function reconcilePayments(payments: string[], invoices: string[]): Reconciliation[] {
  throw new Error("TODO: implement reconcilePayments");
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
  level("Level 1 — Accept-Language Basic Matching", () => {
    const supported = ["en-US", "fr-CA", "fr-FR", "de-DE"];

    check("basic match",
      parseAcceptLanguage("en-US, fr-CA, fr-FR", supported),
      ["en-US", "fr-CA", "fr-FR"]);

    check("whitespace trimming",
      parseAcceptLanguage("  en-US ,  fr-CA  ", supported),
      ["en-US", "fr-CA"]);

    check("case insensitive",
      parseAcceptLanguage("EN-US, FR-ca", supported),
      ["en-US", "fr-CA"]);

    check("no matches",
      parseAcceptLanguage("ja-JP, ko-KR", supported),
      []);

    check("partial overlap preserves order",
      parseAcceptLanguage("fr-FR, ja-JP, en-US", supported),
      ["fr-FR", "en-US"]);
  });

  level("Level 2 — Language Variants and Wildcards", () => {
    const supported = ["en-US", "en-GB", "fr-CA", "fr-FR", "de-DE"];

    check("prefix match",
      parseAcceptLanguageWithVariants("en", supported),
      ["en-US", "en-GB"]);

    check("exact before prefix",
      parseAcceptLanguageWithVariants("en-GB, en", supported),
      ["en-GB", "en-US"]);

    check("wildcard fills remaining",
      parseAcceptLanguageWithVariants("fr-CA, *", supported),
      ["fr-CA", "en-US", "en-GB", "fr-FR", "de-DE"]);

    check("prefix + wildcard dedup",
      parseAcceptLanguageWithVariants("en, *", supported),
      ["en-US", "en-GB", "fr-CA", "fr-FR", "de-DE"]);
  });

  level("Level 3 — Quality Factors", () => {
    const supported = ["en-US", "en-GB", "fr-CA", "fr-FR", "de-DE"];

    check("q-value sorting",
      parseAcceptLanguageQuality("en-US;q=0.8, fr-CA;q=0.9", supported),
      ["fr-CA", "en-US"]);

    check("default q=1.0",
      parseAcceptLanguageQuality("de-DE, en-US;q=0.5", supported),
      ["de-DE", "en-US"]);

    check("q=0 excluded",
      parseAcceptLanguageQuality("en-US, fr-CA;q=0, de-DE", supported),
      ["en-US", "de-DE"]);

    check("wildcard with q + prefix",
      parseAcceptLanguageQuality("fr;q=0.9, *;q=0.1", supported),
      ["fr-CA", "fr-FR", "en-US", "en-GB", "de-DE"]);
  });

  level("Level 4 — Invoice Memo Parser", () => {
    const invoices = [
      "INV-2024-001,2024-01-15,2000",
      "INV-2024-002,2024-02-20,3000",
    ];

    check("single full payment",
      reconcilePayments(
        ["PAY001,2000,Paying off: INV-2024-001"],
        invoices
      ),
      [{
        paymentId: "PAY001", invoiceId: "INV-2024-001",
        amount: 2000, invoiceDate: "2024-01-15",
        invoiceTotal: 2000, remaining: 0
      }]);

    check("partial payment",
      reconcilePayments(
        ["PAY002,500,Payment for: INV-2024-002"],
        invoices
      ),
      [{
        paymentId: "PAY002", invoiceId: "INV-2024-002",
        amount: 500, invoiceDate: "2024-02-20",
        invoiceTotal: 3000, remaining: 2500
      }]);

    check("multiple payments to same invoice",
      reconcilePayments(
        [
          "PAY003,1000,Paying off: INV-2024-001",
          "PAY004,800,Payment for: INV-2024-001",
        ],
        invoices
      ),
      [
        {
          paymentId: "PAY003", invoiceId: "INV-2024-001",
          amount: 1000, invoiceDate: "2024-01-15",
          invoiceTotal: 2000, remaining: 1000
        },
        {
          paymentId: "PAY004", invoiceId: "INV-2024-001",
          amount: 800, invoiceDate: "2024-01-15",
          invoiceTotal: 2000, remaining: 200
        },
      ]);

    check("payment to non-existent invoice",
      reconcilePayments(
        ["PAY005,750,Paying off: INV-9999-999"],
        invoices
      ),
      [{
        paymentId: "PAY005", invoiceId: "INV-9999-999",
        amount: 750, invoiceDate: null,
        invoiceTotal: 0, remaining: -750
      }]);

    // Case-insensitive memo prefix
    check("case-insensitive memo",
      reconcilePayments(
        ["PAY006,500,paying off: INV-2024-002"],
        invoices
      ),
      [{
        paymentId: "PAY006", invoiceId: "INV-2024-002",
        amount: 500, invoiceDate: "2024-02-01",
        invoiceTotal: 3000, remaining: 2500
      }]);
  });
}

function main(): void {
  console.log("\nString Parsing\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
