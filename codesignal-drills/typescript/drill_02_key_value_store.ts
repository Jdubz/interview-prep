/*
Drill 02 — Key-Value Store

Implement a KeyValueStore class with CRUD, prefix scanning,
nested transactions, and value history.

────────────────────────────────────────
Level 1 — Basic Operations

  set(key: string, value: string): void
    Set a key to a value. Overwrites if key already exists.

  get(key: string): string | null
    Returns the value for key, or null if not found.

  delete(key: string): boolean
    Removes the key. Returns true if deleted, false if not found.

  count(value: string): number
    Returns the number of keys that currently have this value.

────────────────────────────────────────
Level 2 — Scanning

  keys(): string[]
    Returns all keys, sorted alphabetically.

  prefix(p: string): string[]
    Returns all keys starting with p, sorted alphabetically.

────────────────────────────────────────
Level 3 — Transactions

  begin(): void
    Start a new transaction. Transactions can be nested.

  commit(): boolean
    Commit the current transaction.
    Returns false if there is no active transaction.

  rollback(): boolean
    Rollback the current transaction, discarding all changes
    made since the matching begin().
    Returns false if there is no active transaction.

  Notes:
  - Nested transactions work like a stack.
  - Rolling back an inner transaction discards only that
    transaction's changes.
  - Committing an inner transaction merges its changes into
    the outer transaction (or into the main store if outermost).
  - get, count, keys, prefix all reflect uncommitted changes
    within the current transaction.

────────────────────────────────────────
Level 4 — History

  getHistory(key: string): string[]
    Returns all values ever successfully set for this key, in order.
    Does not include deletions. Returns [] if never set.
    Rolled-back sets do not appear in history.

  undoSet(key: string): boolean
    Reverts key to its previous value (or deletes it if only set once).
    Returns false if no history exists for this key.
    Removes the undone value from history.
*/

export class KeyValueStore {
  constructor() {
    // TODO: initialize your data structures
  }

  set(key: string, value: string): void {
    throw new Error("TODO: implement set");
  }

  get(key: string): string | null {
    throw new Error("TODO: implement get");
  }

  delete(key: string): boolean {
    throw new Error("TODO: implement delete");
  }

  count(value: string): number {
    throw new Error("TODO: implement count");
  }

  keys(): string[] {
    throw new Error("TODO: implement keys");
  }

  prefix(p: string): string[] {
    throw new Error("TODO: implement prefix");
  }

  begin(): void {
    throw new Error("TODO: implement begin");
  }

  commit(): boolean {
    throw new Error("TODO: implement commit");
  }

  rollback(): boolean {
    throw new Error("TODO: implement rollback");
  }

  getHistory(key: string): string[] {
    throw new Error("TODO: implement getHistory");
  }

  undoSet(key: string): boolean {
    throw new Error("TODO: implement undoSet");
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
  level("Level 1 — Basic Operations", () => {
    const s1 = new KeyValueStore();
    s1.set("a", "apple");
    s1.set("b", "banana");
    s1.set("c", "apple");
    check("get", s1.get("a"), "apple");
    check("get missing", s1.get("z"), null);
    check("count apple", s1.count("apple"), 2);
    check("count banana", s1.count("banana"), 1);
    check("count missing", s1.count("cherry"), 0);
    check("delete", s1.delete("c"), true);
    check("count after delete", s1.count("apple"), 1);
    check("delete missing", s1.delete("z"), false);
    s1.set("a", "avocado");
    check("overwrite", s1.get("a"), "avocado");
    check("count old val", s1.count("apple"), 0);
  });

  level("Level 2 — Scanning", () => {
    const s2 = new KeyValueStore();
    s2.set("app", "1");
    s2.set("api", "2");
    s2.set("beta", "3");
    check("keys", s2.keys(), ["api", "app", "beta"]);
    check("prefix ap", s2.prefix("ap"), ["api", "app"]);
    check("prefix z", s2.prefix("z"), []);
  });

  level("Level 3 — Transactions", () => {
    const s3 = new KeyValueStore();
    s3.set("x", "1");
    s3.begin();
    s3.set("x", "2");
    check("read in txn", s3.get("x"), "2");
    check("rollback", s3.rollback(), true);
    check("after rollback", s3.get("x"), "1");

    s3.begin();
    s3.set("y", "10");
    check("commit", s3.commit(), true);
    check("after commit", s3.get("y"), "10");

    // nested
    s3.begin();
    s3.set("z", "outer");
    s3.begin();
    s3.set("z", "inner");
    check("nested read", s3.get("z"), "inner");
    check("inner rollback", s3.rollback(), true);
    check("after inner rollback", s3.get("z"), "outer");
    check("outer commit", s3.commit(), true);
    check("after outer commit", s3.get("z"), "outer");

    check("no txn rollback", s3.rollback(), false);
    check("no txn commit", s3.commit(), false);
  });

  level("Level 4 — History", () => {
    const s4 = new KeyValueStore();
    s4.set("k", "first");
    s4.set("k", "second");
    s4.set("k", "third");
    check("history", s4.getHistory("k"), ["first", "second", "third"]);
    check("undo", s4.undoSet("k"), true);
    check("after undo", s4.get("k"), "second");
    check("history after undo", s4.getHistory("k"), ["first", "second"]);
    check("undo again", s4.undoSet("k"), true);
    check("after undo 2", s4.get("k"), "first");
    check("undo last", s4.undoSet("k"), true);
    check("after full undo", s4.get("k"), null);
    check("no more undo", s4.undoSet("k"), false);
    check("empty history", s4.getHistory("k"), []);
    check("never set", s4.getHistory("unknown"), []);
  });
}

function main(): void {
  console.log("\nKey-Value Store\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
