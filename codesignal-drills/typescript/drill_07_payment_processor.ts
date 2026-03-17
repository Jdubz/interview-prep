/*
Drill 07 — Payment Processor

Implement a PaymentProcessor class with account management,
payment processing, scheduled payments, and atomic payment groups.

This drill tests multi-entity state management, state machines,
time-based processing, and transactional semantics.

────────────────────────────────────────
Level 1 — Accounts

  createAccount(accountId: string, initialBalance: number): boolean
    Create an account with the given balance.
    Returns false if accountId already exists.

  getBalance(accountId: string): number | null
    Returns the account balance, or null if not found.

  deposit(accountId: string, amount: number): number | null
    Add amount to the account balance.
    Returns the new balance, or null if account not found.

  withdraw(accountId: string, amount: number): number | null
    Subtract amount from the account balance.
    Returns the new balance, or null if account not found
    or insufficient funds.

────────────────────────────────────────
Level 2 — Payments

  createPayment(paymentId: string, fromId: string, toId: string, amount: number): boolean
    Create a payment in PENDING status.
    Returns false if paymentId already exists, either account
    not found, fromId equals toId, or amount <= 0.

  processPayment(paymentId: string): boolean
    Process a PENDING payment: debit sender, credit receiver.
    Marks status COMPLETED on success.
    Marks status FAILED if insufficient funds.
    Returns false if payment not found or not in PENDING status.

  getPaymentStatus(paymentId: string): string | null
    Returns the payment status string, or null if not found.
    Statuses: "PENDING" | "COMPLETED" | "FAILED" | "CANCELLED" | "SCHEDULED"

  getAccountPayments(accountId: string): string[]
    Returns all payment ids involving this account
    (as sender or receiver), in creation order.
    Returns [] if account not found or has no payments.

────────────────────────────────────────
Level 3 — Scheduling & Cancellation

  schedulePayment(paymentId: string, fromId: string, toId: string,
                   amount: number, executeAt: number): boolean
    Like createPayment but with SCHEDULED status and a timestamp.
    Same validation rules as createPayment.

  processScheduled(currentTime: number): string[]
    Process all SCHEDULED payments where executeAt <= currentTime.
    Process in order of executeAt ascending, then paymentId ascending.
    Payments with insufficient funds move to FAILED.
    Returns ids of successfully completed payments only.

  cancelPayment(paymentId: string): boolean
    Cancel a payment based on its current status:
    - PENDING or SCHEDULED → move to CANCELLED (no balance change).
    - COMPLETED → reverse the transfer and move to CANCELLED.
    - FAILED or CANCELLED → return false (cannot cancel).

────────────────────────────────────────
Level 4 — Payment Groups & Chains

  createPaymentGroup(groupId: string, paymentIds: string[]): boolean
    Group existing PENDING payments for atomic processing.
    Returns false if groupId already exists, paymentIds is empty,
    any payment doesn't exist, any payment is not PENDING,
    or any payment is already in a group.

  processGroup(groupId: string): boolean
    All-or-nothing atomic processing.
    Simulate all transfers in the order the payments were added
    to the group, tracking running balances.
    If every payment would succeed (no overdraft), commit all
    transfers and mark all as COMPLETED. Returns true.
    If any payment would overdraft, mark ALL as FAILED and
    make no balance changes. Returns false.
    Also returns false if group not found or already processed.

  chainPayment(paymentId: string, afterPaymentId: string): boolean
    Declare that paymentId cannot be processed until
    afterPaymentId is COMPLETED.
    A payment can have at most one chain dependency.
    Returns false if either payment doesn't exist,
    paymentId already has a chain dependency,
    or adding this link would create a cycle.

  getProcessablePayments(): string[]
    Returns PENDING payments that are not in any payment group
    and whose chain dependency (if any) is COMPLETED.
    Sorted by paymentId ascending.
*/

export class PaymentProcessor {
  constructor() {
  }

  createAccount(accountId: string, initialBalance: number): boolean {
    throw new Error("TODO: implement createAccount");
  }

  getBalance(accountId: string): number | null {
    throw new Error("TODO: implement getBalance");
  }

  deposit(accountId: string, amount: number): number | null {
    throw new Error("TODO: implement deposit");
  }

  withdraw(accountId: string, amount: number): number | null {
    throw new Error("TODO: implement withdraw");
  }

  createPayment(paymentId: string, fromId: string, toId: string, amount: number): boolean {
    throw new Error("TODO: implement createPayment");
  }

  processPayment(paymentId: string): boolean {
    throw new Error("TODO: implement processPayment");
  }

  getPaymentStatus(paymentId: string): string | null {
    throw new Error("TODO: implement getPaymentStatus");
  }

  getAccountPayments(accountId: string): string[] {
    throw new Error("TODO: implement getAccountPayments");
  }

  schedulePayment(paymentId: string, fromId: string, toId: string, amount: number, executeAt: number): boolean {
    throw new Error("TODO: implement schedulePayment");
  }

  processScheduled(currentTime: number): string[] {
    throw new Error("TODO: implement processScheduled");
  }

  cancelPayment(paymentId: string): boolean {
    throw new Error("TODO: implement cancelPayment");
  }

  createPaymentGroup(groupId: string, paymentIds: string[]): boolean {
    throw new Error("TODO: implement createPaymentGroup");
  }

  processGroup(groupId: string): boolean {
    throw new Error("TODO: implement processGroup");
  }

  chainPayment(paymentId: string, afterPaymentId: string): boolean {
    throw new Error("TODO: implement chainPayment");
  }

  getProcessablePayments(): string[] {
    throw new Error("TODO: implement getProcessablePayments");
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
  level("Level 1 — Accounts", () => {
    const p = new PaymentProcessor();
    check("create", p.createAccount("acct1", 1000), true);
    check("create dup", p.createAccount("acct1", 500), false);
    check("balance", p.getBalance("acct1"), 1000);
    check("missing", p.getBalance("nope"), null);
    p.createAccount("acct2", 500);
    check("deposit", p.deposit("acct1", 500), 1500);
    check("deposit missing", p.deposit("nope", 100), null);
    check("withdraw", p.withdraw("acct1", 200), 1300);
    check("withdraw too much", p.withdraw("acct1", 2000), null);
    check("withdraw missing", p.withdraw("nope", 100), null);
  });

  level("Level 2 — Payments", () => {
    const p = new PaymentProcessor();
    p.createAccount("alice", 1000);
    p.createAccount("bob", 500);
    check("create pay", p.createPayment("p1", "alice", "bob", 200), true);
    check("dup pay", p.createPayment("p1", "alice", "bob", 100), false);
    check("bad from", p.createPayment("p2", "nope", "bob", 100), false);
    check("bad to", p.createPayment("p3", "alice", "nope", 100), false);
    check("self pay", p.createPayment("p4", "alice", "alice", 100), false);
    check("zero amount", p.createPayment("p5", "alice", "bob", 0), false);
    check("status", p.getPaymentStatus("p1"), "PENDING");
    check("status missing", p.getPaymentStatus("nope"), null);
    check("process", p.processPayment("p1"), true);
    check("status after", p.getPaymentStatus("p1"), "COMPLETED");
    check("alice bal", p.getBalance("alice"), 800);
    check("bob bal", p.getBalance("bob"), 700);
    check("process again", p.processPayment("p1"), false);
    p.createPayment("p6", "alice", "bob", 5000);
    check("process insuff", p.processPayment("p6"), false);
    check("p6 status", p.getPaymentStatus("p6"), "FAILED");
    check("alice same", p.getBalance("alice"), 800);
    check("acct pays", p.getAccountPayments("alice"), ["p1", "p6"]);
    check("bob pays", p.getAccountPayments("bob"), ["p1", "p6"]);
    check("empty pays", p.getAccountPayments("nope"), []);
  });

  level("Level 3 — Scheduling & Cancellation", () => {
    const p = new PaymentProcessor();
    p.createAccount("alice", 1000);
    p.createAccount("bob", 500);
    p.createAccount("carol", 300);
    check("schedule", p.schedulePayment("s1", "alice", "bob", 100, 10), true);
    check("schedule 2", p.schedulePayment("s2", "alice", "carol", 200, 5), true);
    check("schedule dup", p.schedulePayment("s1", "bob", "carol", 50, 15), false);
    check("schedule bad", p.schedulePayment("s3", "nope", "bob", 50, 15), false);
    check("status", p.getPaymentStatus("s1"), "SCHEDULED");
    // processScheduled at time 7: only s2 (executeAt=5) qualifies
    check("process t7", p.processScheduled(7), ["s2"]);
    check("s2 status", p.getPaymentStatus("s2"), "COMPLETED");
    check("alice after", p.getBalance("alice"), 800);
    check("carol after", p.getBalance("carol"), 500);
    check("s1 still", p.getPaymentStatus("s1"), "SCHEDULED");
    // processScheduled at time 15: s1 (executeAt=10) qualifies
    check("process t15", p.processScheduled(15), ["s1"]);
    check("alice after2", p.getBalance("alice"), 700);
    check("bob after", p.getBalance("bob"), 600);
    // cancel a completed payment — reverses the transfer
    check("cancel done", p.cancelPayment("s2"), true);
    check("s2 cancelled", p.getPaymentStatus("s2"), "CANCELLED");
    check("alice reversed", p.getBalance("alice"), 900);
    check("carol reversed", p.getBalance("carol"), 300);
    // cancel a pending payment — no balance change
    p.createPayment("px", "alice", "bob", 50);
    check("cancel pending", p.cancelPayment("px"), true);
    check("px cancelled", p.getPaymentStatus("px"), "CANCELLED");
    check("alice same", p.getBalance("alice"), 900);
    // can't cancel already cancelled or failed
    check("cancel again", p.cancelPayment("px"), false);
    p.createPayment("pf", "carol", "alice", 9999);
    p.processPayment("pf");
    check("pf failed", p.getPaymentStatus("pf"), "FAILED");
    check("cancel failed", p.cancelPayment("pf"), false);
    // scheduled payment with insufficient funds → FAILED
    p.schedulePayment("sf", "carol", "alice", 9999, 20);
    check("process t25", p.processScheduled(25), []);
    check("sf failed", p.getPaymentStatus("sf"), "FAILED");
  });

  level("Level 4 — Groups & Chains", () => {
    const p = new PaymentProcessor();
    p.createAccount("alice", 1000);
    p.createAccount("bob", 500);
    p.createAccount("carol", 200);
    p.createPayment("g1", "alice", "bob", 300);
    p.createPayment("g2", "alice", "carol", 400);
    // group: alice debited 300+400=700 total, has 1000 → OK
    check("create grp", p.createPaymentGroup("grp1", ["g1", "g2"]), true);
    check("dup grp", p.createPaymentGroup("grp1", ["g1"]), false);
    check("process grp", p.processGroup("grp1"), true);
    check("alice after grp", p.getBalance("alice"), 300);
    check("bob after grp", p.getBalance("bob"), 800);
    check("carol after grp", p.getBalance("carol"), 600);
    check("g1 done", p.getPaymentStatus("g1"), "COMPLETED");
    check("g2 done", p.getPaymentStatus("g2"), "COMPLETED");
    // already processed group
    check("reprocess grp", p.processGroup("grp1"), false);
    // group that overdrafts — both debit alice (300 remaining)
    p.createPayment("g3", "alice", "bob", 200);
    p.createPayment("g4", "alice", "carol", 200);
    check("create grp2", p.createPaymentGroup("grp2", ["g3", "g4"]), true);
    // alice has 300, g3 takes 200 → 100, g4 takes 200 → -100 overdraft
    check("process fail grp", p.processGroup("grp2"), false);
    check("alice unchanged", p.getBalance("alice"), 300);
    check("g3 failed", p.getPaymentStatus("g3"), "FAILED");
    check("g4 failed", p.getPaymentStatus("g4"), "FAILED");
    // empty group
    check("empty grp", p.createPaymentGroup("grp3", []), false);
    // payment already in a group
    p.createPayment("g5", "bob", "carol", 50);
    p.createPayment("g6", "bob", "carol", 50);
    p.createPaymentGroup("grp4", ["g5"]);
    check("already grouped", p.createPaymentGroup("grp5", ["g5", "g6"]), false);
    // chain dependencies
    p.createPayment("c1", "bob", "carol", 50);
    p.createPayment("c2", "carol", "alice", 30);
    check("chain", p.chainPayment("c2", "c1"), true);
    check("chain dup", p.chainPayment("c2", "c1"), false);
    // c2 depends on c1 (PENDING), so only c1 is processable
    // (g6 is PENDING and not grouped, c1 is PENDING and not grouped)
    check("processable", p.getProcessablePayments(), ["c1", "g6"]);
    p.processPayment("c1");
    // c1 COMPLETED → c2 now processable
    check("processable 2", p.getProcessablePayments(), ["c2", "g6"]);
    // cycle detection
    p.createPayment("cy1", "alice", "bob", 10);
    p.createPayment("cy2", "bob", "carol", 10);
    check("chain cy", p.chainPayment("cy2", "cy1"), true);
    check("cycle", p.chainPayment("cy1", "cy2"), false);
    check("chain missing", p.chainPayment("nope", "cy1"), false);
    // cy1 has no dep → processable, cy2 depends on cy1 (PENDING) → not
    check("processable 3", p.getProcessablePayments(), ["c2", "cy1", "g6"]);
  });
}

function main(): void {
  console.log("\nPayment Processor\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
