/*
Drill 06 — Inventory Tracker

Implement an InventoryTracker class with stock management,
categories, low-stock alerts, and change history.

────────────────────────────────────────
Level 1 — Stock Management

  addProduct(sku: string, quantity: number): boolean
    Add a product with initial quantity.
    Returns false if sku already exists.

  getQuantity(sku: string): number
    Returns the quantity for a sku, or -1 if not found.

  restock(sku: string, amount: number): boolean
    Add amount to the product's quantity.
    Returns false if sku not found.

  sell(sku: string, amount: number): boolean
    Subtract amount from the product's quantity.
    Returns false if sku not found or insufficient stock.

  removeProduct(sku: string): boolean
    Removes the product. Returns true if removed, false if not found.

────────────────────────────────────────
Level 2 — Categories & Search

  setCategory(sku: string, category: string): boolean
    Assign a category to a product.
    Returns false if sku not found.

  getByCategory(category: string): string[]
    Returns all skus in this category, sorted alphabetically.

  listProducts(): string[]
    Returns all skus, sorted alphabetically.

────────────────────────────────────────
Level 3 — Low Stock Alerts

  setThreshold(sku: string, threshold: number): boolean
    Set a low-stock threshold for a product.
    Returns false if sku not found.

  getLowStock(): string[]
    Returns skus whose quantity is at or below their threshold.
    Only includes products that have a threshold set.
    Sorted by quantity ascending, then sku ascending for ties.

  bulkRestock(updates: [string, number][]): number
    Apply multiple restocks. Each entry is [sku, amount].
    Returns the count of successful restocks.

────────────────────────────────────────
Level 4 — Change History

  getHistory(sku: string): number[]
    Returns all quantity values for this sku, in order.
    Includes the initial quantity from addProduct and every
    change from restock or sell.
    Returns [] if sku was never added.

  undoLastChange(sku: string): boolean
    Reverts to the previous quantity (or removes the product
    if only the initial add remains).
    Returns false if no history exists for this sku.
    Removes the undone value from history.
*/

export class InventoryTracker {
  constructor() {
  }

  addProduct(sku: string, quantity: number): boolean {
    throw new Error("TODO: implement addProduct");
  }

  getQuantity(sku: string): number {
    throw new Error("TODO: implement getQuantity");
  }

  restock(sku: string, amount: number): boolean {
    throw new Error("TODO: implement restock");
  }

  sell(sku: string, amount: number): boolean {
    throw new Error("TODO: implement sell");
  }

  removeProduct(sku: string): boolean {
    throw new Error("TODO: implement removeProduct");
  }

  setCategory(sku: string, category: string): boolean {
    throw new Error("TODO: implement setCategory");
  }

  getByCategory(category: string): string[] {
    throw new Error("TODO: implement getByCategory");
  }

  listProducts(): string[] {
    throw new Error("TODO: implement listProducts");
  }

  setThreshold(sku: string, threshold: number): boolean {
    throw new Error("TODO: implement setThreshold");
  }

  getLowStock(): string[] {
    throw new Error("TODO: implement getLowStock");
  }

  bulkRestock(updates: [string, number][]): number {
    throw new Error("TODO: implement bulkRestock");
  }

  getHistory(sku: string): number[] {
    throw new Error("TODO: implement getHistory");
  }

  undoLastChange(sku: string): boolean {
    throw new Error("TODO: implement undoLastChange");
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
  level("Level 1 — Stock Management", () => {
    const s1 = new InventoryTracker();
    check("add", s1.addProduct("SKU1", 100), true);
    check("add dup", s1.addProduct("SKU1", 50), false);
    check("get", s1.getQuantity("SKU1"), 100);
    check("get missing", s1.getQuantity("NOPE"), -1);
    check("restock", s1.restock("SKU1", 50), true);
    check("after restock", s1.getQuantity("SKU1"), 150);
    check("restock missing", s1.restock("NOPE", 10), false);
    check("sell", s1.sell("SKU1", 30), true);
    check("sell too many", s1.sell("SKU1", 200), false);
    check("after sell", s1.getQuantity("SKU1"), 120);
    check("remove", s1.removeProduct("SKU1"), true);
    check("remove missing", s1.removeProduct("SKU1"), false);
  });

  level("Level 2 — Categories & Search", () => {
    const s2 = new InventoryTracker();
    s2.addProduct("widget", 50);
    s2.addProduct("gadget", 30);
    s2.addProduct("gizmo", 20);
    check("set cat", s2.setCategory("widget", "electronics"), true);
    s2.setCategory("gadget", "electronics");
    check("set cat 2", s2.setCategory("gizmo", "tools"), true);
    check("cat missing", s2.setCategory("nope", "x"), false);
    check("by cat", s2.getByCategory("electronics"), ["gadget", "widget"]);
    check("cat empty", s2.getByCategory("food"), []);
    check("list", s2.listProducts(), ["gadget", "gizmo", "widget"]);
  });

  level("Level 3 — Low Stock Alerts", () => {
    const s3 = new InventoryTracker();
    s3.addProduct("a", 5);
    s3.addProduct("b", 20);
    s3.addProduct("c", 3);
    check("threshold", s3.setThreshold("a", 10), true);
    s3.setThreshold("c", 5);
    check("threshold missing", s3.setThreshold("nope", 5), false);
    check("low stock", s3.getLowStock(), ["c", "a"]);
    check("bulk", s3.bulkRestock([["a", 20], ["c", 10], ["nope", 5]]), 2);
    check("after bulk", s3.getQuantity("a"), 25);
    check("low stock after", s3.getLowStock(), []);
  });

  level("Level 4 — Change History", () => {
    const s4 = new InventoryTracker();
    s4.addProduct("item", 100);
    s4.restock("item", 50);
    s4.sell("item", 30);
    check("history", s4.getHistory("item"), [100, 150, 120]);
    check("undo", s4.undoLastChange("item"), true);
    check("after undo", s4.getQuantity("item"), 150);
    check("history after", s4.getHistory("item"), [100, 150]);
    check("undo again", s4.undoLastChange("item"), true);
    check("after undo 2", s4.getQuantity("item"), 100);
    check("undo last", s4.undoLastChange("item"), true);
    check("fully undone", s4.getQuantity("item"), -1);
    check("no more", s4.undoLastChange("item"), false);
    check("empty history", s4.getHistory("item"), []);
    check("never added", s4.getHistory("nope"), []);
  });
}

function main(): void {
  console.log("\nInventory Tracker\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
