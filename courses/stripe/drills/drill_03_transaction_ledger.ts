/*
Drill 03 — Transaction Ledger

Build a double-entry bookkeeping ledger. Process transactions,
enforce rules, and generate reports. Commonly reported Stripe
Programming Exercise pattern.

Target time: 35 minutes for all 4 levels.
Levels 1–2 are implemented. Levels 3–4 are TODO.

────────────────────────────────────────
Level 1 — Accounts & Transfers (8 min) ✅

  createAccount(id: string, balance: number): boolean
    Create an account. Returns false if id already exists.

  transfer(fromId: string, toId: string, amount: number, timestamp?: number): boolean
    Move amount from one account to another.
    Optional timestamp for deterministic testing (defaults to Date.now()).
    Returns false if:
      - either account doesn't exist
      - from and to are the same account
      - amount <= 0
      - insufficient funds (balance < amount)

  getBalance(id: string): number | null
    Returns balance or null if account not found.

  getTransactionCount(): number
    Returns total number of successful transfers.

────────────────────────────────────────
Level 2 — Transaction History & Queries (10 min) ✅

  getHistory(accountId: string): Transaction[]
    Returns all transactions involving this account, in chronological order.
    Returns [] if account not found.

    Transaction = { id: string; from: string; to: string;
                    amount: number; timestamp: number }

  getBalanceAt(accountId: string, timestamp: number): number | null
    Returns the account balance as of the given timestamp
    (inclusive — includes transactions AT that timestamp).
    Returns the initial balance if no transactions exist at or
    before the given timestamp.
    Returns null if account doesn't exist.

  getTopAccounts(n: number): Array<{ id: string; balance: number }>
    Returns the top n accounts by balance, descending.
    Break ties by id ascending.

────────────────────────────────────────
Level 3 — Rules Engine (10 min)

  addRule(rule: TransferRule): void
    Add a rule that runs on every transfer attempt.

    TransferRule = {
      name: string;
      check: (tx: { from: string; to: string; amount: number;
                     fromBalance: number; toBalance: number }) => boolean;
    }

    check() returns true if the transfer should be ALLOWED.
    All rules must pass for a transfer to succeed.
    Rules run AFTER basic validation passes (account exists, amount > 0,
    sufficient funds). If basic validation fails, the transfer is rejected
    without consulting rules.

  transfer() now also returns false if any rule rejects.

  getBlockedTransfers(): Array<{ from: string; to: string;
                                  amount: number; blockedBy: string }>
    Returns all rule-blocked transfers with the name of the first
    blocking rule. In chronological order.
    Note: transfers that fail basic validation (missing account, etc.)
    are NOT included — only rule-blocked ones.

────────────────────────────────────────
Level 4 — Batch Processing & Rollback (7 min)

  processBatch(transfers: Array<{ from: string; to: string; amount: number }>): {
    succeeded: number;
    failed: number;
    results: boolean[];
  }
    Process multiple transfers in order. Each is independent —
    a failure doesn't stop the batch. Returns per-transfer results.

  checkpoint(): string
    Save a snapshot of all account balances AND transaction history.
    Returns a checkpoint id (any unique string).

  rollback(checkpointId: string): boolean
    Restore all account balances and transaction history to the
    checkpoint state. Any transactions that occurred after the
    checkpoint are discarded.
    The checkpoint is consumed — rolling back to the same
    checkpoint a second time returns false.
    Returns false if checkpoint not found.
*/
type Account = {
  id: string;
  balance: number;
  initialBalance: number;
  history: TransactionWithBalance[];
}

export type Transaction = {
  id: string;
  from: string;
  to: string;
  amount: number;
  timestamp: number;
};

type TransactionWithBalance = Transaction & { balance: number }

export type TransferRule = {
  name: string;
  check: (tx: {
    from: string;
    to: string;
    amount: number;
    fromBalance: number;
    toBalance: number;
  }) => boolean;
};

// REVIEW: This helper lives outside the class. Works fine, but
// makes the class harder to reason about in isolation. Consider
// making it a private method — especially since Level 3 will
// need to modify transfer(), and having all the logic inside
// the class keeps the flow in one place.
const modifyAccount = (account: Account, transaction: Transaction): void => {
  const balance = transaction.from === account.id ? account.balance - transaction.amount : account.balance + transaction.amount;
  account.balance = balance;
  account.history.push({ ...transaction, balance });
}

export class TransactionLedger {
  accounts: Map<string, Account>;
  constructor() {
    this.accounts = new Map();
  }

  // Level 1
  createAccount(id: string, balance: number): boolean {
    if (this.accounts.has(id)) return false;
    const newAccount: Account = {
      id,
      balance,
      initialBalance: balance,
      history: [],
    }
    this.accounts.set(id, newAccount);
    return true;
  }

  transfer(fromId: string, toId: string, amount: number, timestamp?: number): boolean {
    if (fromId === toId) return false;
    if (amount <= 0) return false;
    const fromAccount = this.accounts.get(fromId);
    const toAccount = this.accounts.get(toId);
    if (!fromAccount || !toAccount) return false;
    if (fromAccount.balance < amount) return false;

    const transaction: Transaction = {
      id: crypto.randomUUID(),
      from: fromId,
      to: toId,
      amount: amount,
      timestamp: timestamp ?? new Date().getTime(),
    }
    modifyAccount(fromAccount, transaction);
    modifyAccount(toAccount, transaction);
    return true;
  }

  getBalance(id: string): number | null {
    const account = this.accounts.get(id);
    if (!account) return null;
    return account.balance;
  }

  getTransactionCount(): number {
    const transactionIds = new Set();
    this.accounts.forEach((acc: Account) => {
      acc.history.forEach((transaction: Transaction) => transactionIds.add(transaction.id))
    });
    return transactionIds.size;
  }
  // REVIEW: This iterates every account and deduplicates with a
  // Set because each transfer appears in two histories. Simpler
  // alternative: add a `private txCount = 0` to the class and
  // increment it in transfer(). Then this method is just
  // `return this.txCount`. O(1) instead of O(n).

  // Level 2
  getHistory(accountId: string): Transaction[] {
    const account = this.accounts.get(accountId);
    if (!account) return [];
    return account.history;
  }
  // REVIEW: Returns the internal array. A caller doing
  // `getHistory("alice").push(fake)` would corrupt your ledger.
  // Fix: `return [...account.history]`. One-character difference,
  // interviewers notice it, shows you think about API boundaries.

  getBalanceAt(accountId: string, timestamp: number): number | null {
    const account = this.accounts.get(accountId);
    if (!account) return null;
    const lastTransaction = account.history.reduce((lt: TransactionWithBalance | null, ct: TransactionWithBalance) => {
      if (ct.timestamp <= timestamp) return ct;
      return lt;
    }, null);
    return lastTransaction ? lastTransaction.balance : account.initialBalance;
  }
  // REVIEW: Clean. Storing balance on each TransactionWithBalance
  // was a good call — it makes this a simple scan instead of
  // replaying all debits/credits from initial. The reduce finds
  // the last tx at or before the timestamp and reads its stored
  // balance. Only works because history is always chronological,
  // which your transfer() guarantees.

  getTopAccounts(n: number): Array<{ id: string; balance: number }> {
    const accounts = Array.from(this.accounts.values());
    accounts.sort((a,b) => {
      if (a.balance === b.balance) return a.id.localeCompare(b.id);
      return b.balance - a.balance
    });
    const topAccounts = accounts.slice(0, n);
    return topAccounts.map(a => ({ id: a.id, balance: a.balance }));
  }
  // REVIEW: Clean. sort + slice + map is the right pattern.
  // Minor: you can chain it —
  //   return [...this.accounts.values()]
  //     .sort((a,b) => b.balance - a.balance || a.id.localeCompare(b.id))
  //     .slice(0, n)
  //     .map(a => ({ id: a.id, balance: a.balance }));
  // Same logic, fewer intermediate variables.

  // Level 3
  addRule(rule: TransferRule): void {
    throw new Error("TODO: implement addRule");
  }

  getBlockedTransfers(): Array<{ from: string; to: string; amount: number; blockedBy: string }> {
    throw new Error("TODO: implement getBlockedTransfers");
  }

  // Level 4
  processBatch(transfers: Array<{ from: string; to: string; amount: number }>): {
    succeeded: number;
    failed: number;
    results: boolean[];
  } {
    throw new Error("TODO: implement processBatch");
  }

  checkpoint(): string {
    throw new Error("TODO: implement checkpoint");
  }

  rollback(checkpointId: string): boolean {
    throw new Error("TODO: implement rollback");
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
  level("Level 1 — Accounts & Transfers", () => {
    const l = new TransactionLedger();
    check("create", l.createAccount("alice", 1000), true);
    check("create dup", l.createAccount("alice", 500), false);
    check("create bob", l.createAccount("bob", 500), true);
    check("balance", l.getBalance("alice"), 1000);
    check("missing", l.getBalance("nope"), null);
    check("transfer", l.transfer("alice", "bob", 200), true);
    check("alice after", l.getBalance("alice"), 800);
    check("bob after", l.getBalance("bob"), 700);
    check("count", l.getTransactionCount(), 1);
    check("overdraft", l.transfer("alice", "bob", 5000), false);
    check("count unchanged", l.getTransactionCount(), 1);
    check("self transfer", l.transfer("alice", "alice", 100), false);
    check("zero amount", l.transfer("alice", "bob", 0), false);
    check("negative", l.transfer("alice", "bob", -10), false);
    check("bad from", l.transfer("nope", "bob", 100), false);
  });

  level("Level 2 — History & Queries", () => {
    const l = new TransactionLedger();
    l.createAccount("alice", 1000);
    l.createAccount("bob", 500);
    l.createAccount("carol", 200);
    l.transfer("alice", "bob", 100, 1000);
    l.transfer("bob", "carol", 50, 2000);
    l.transfer("alice", "carol", 200, 3000);

    const hist = l.getHistory("alice");
    check("history length", hist.length, 2);
    check("history[0] amount", hist[0].amount, 100);
    check("history[1] to", hist[1].to, "carol");

    const bobHist = l.getHistory("bob");
    check("bob history length", bobHist.length, 2);

    // Balance at timestamp
    // After tx 0 (t=1000): alice=900, bob=600, carol=200
    // After tx 1 (t=2000): alice=900, bob=550, carol=250
    // After tx 2 (t=3000): alice=700, bob=550, carol=450
    check("alice at t=1000", l.getBalanceAt("alice", 1000), 900);
    check("bob at t=1000", l.getBalanceAt("bob", 1000), 600);
    check("alice at t=2500 (between txs)", l.getBalanceAt("alice", 2500), 900);
    check("balanceAt before any tx", l.getBalanceAt("alice", 500), 1000);
    check("balanceAt missing account", l.getBalanceAt("nope", 0), null);

    check("top 2", l.getTopAccounts(2), [
      { id: "alice", balance: 700 },
      { id: "bob", balance: 550 },
    ]);
    check("top all", l.getTopAccounts(10).length, 3);
    check("history missing account", l.getHistory("nope"), []);
  });

  level("Level 3 — Rules Engine", () => {
    const l = new TransactionLedger();
    l.createAccount("alice", 1000);
    l.createAccount("bob", 500);
    l.createAccount("carol", 100);

    // Rule: no single transfer > 500
    l.addRule({
      name: "max_transfer",
      check: (tx) => tx.amount <= 500,
    });

    // Rule: receiver can't have balance > 2000
    l.addRule({
      name: "max_balance",
      check: (tx) => tx.toBalance + tx.amount <= 2000,
    });

    check("allowed", l.transfer("alice", "bob", 200), true);
    check("blocked by max_transfer", l.transfer("alice", "bob", 600), false);
    check("alice unchanged after block", l.getBalance("alice"), 800);
    check("bob unchanged after block", l.getBalance("bob"), 700);

    const blocked = l.getBlockedTransfers();
    check("blocked count", blocked.length, 1);
    check("blocked by", blocked[0].blockedBy, "max_transfer");
    check("blocked amount", blocked[0].amount, 600);

    // Test max_balance rule
    l.createAccount("rich", 1900);
    check("blocked by max_balance", l.transfer("alice", "rich", 200), false);
    check("blocked count 2", l.getBlockedTransfers().length, 2);
    check("second blocked by", l.getBlockedTransfers()[1].blockedBy, "max_balance");

    // Basic validation failures don't appear in blocked list
    l.transfer("nope", "bob", 100); // missing account — not rule-blocked
    check("basic failures not in blocked list", l.getBlockedTransfers().length, 2);
  });

  level("Level 4 — Batch & Rollback", () => {
    const l = new TransactionLedger();
    l.createAccount("alice", 1000);
    l.createAccount("bob", 500);
    l.createAccount("carol", 200);

    const cpId = l.checkpoint();

    const result = l.processBatch([
      { from: "alice", to: "bob", amount: 100 },
      { from: "alice", to: "carol", amount: 50 },
      { from: "bob", to: "carol", amount: 9999 }, // will fail — insufficient
      { from: "alice", to: "carol", amount: 100 },
    ]);
    check("batch succeeded", result.succeeded, 3);
    check("batch failed", result.failed, 1);
    check("batch results", result.results, [true, true, false, true]);
    check("alice after batch", l.getBalance("alice"), 750);
    check("carol after batch", l.getBalance("carol"), 350);

    // Rollback restores balances AND history
    check("rollback", l.rollback(cpId), true);
    check("alice restored", l.getBalance("alice"), 1000);
    check("bob restored", l.getBalance("bob"), 500);
    check("carol restored", l.getBalance("carol"), 200);
    check("history cleared", l.getHistory("alice").length, 0);
    check("rollback bad id", l.rollback("nope"), false);
    check("rollback consumed", l.rollback(cpId), false);

    // Checkpoint with rules active
    l.addRule({ name: "small_only", check: (tx) => tx.amount <= 300 });
    l.transfer("alice", "bob", 200);
    check("alice after rule-allowed transfer", l.getBalance("alice"), 800);
    const cp2 = l.checkpoint();
    l.transfer("alice", "bob", 100);
    check("alice before rollback2", l.getBalance("alice"), 700);
    l.rollback(cp2);
    check("alice after rollback2", l.getBalance("alice"), 800);
    check("history after rollback2", l.getHistory("alice").length, 1);
  });
}

function main(): void {
  console.log("\nTransaction Ledger\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
