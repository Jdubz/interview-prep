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

type Product = {
  sku: string;
  quantity: number;
  history: number[];
  category: Set<string>;
  threshold: number;
} 

export class InventoryTracker {
  inventory: Map<string, Product>;

  constructor() {
    this.inventory = new Map();
  }

  addProduct(sku: string, quantity: number): boolean {
    if (this.inventory.has(sku)) return false;
    this.inventory.set(sku, {
      sku,
      quantity,
      history: [quantity],
      category: new Set(),
      threshold: -1,
    });
    return true;
  }
  // REVIEW: Clean. Good use of early return for duplicate check.

  getQuantity(sku: string): number {
    const product = this.inventory.get(sku);
    if (!product) return -1;
    return product.quantity;
  }
  // REVIEW: Clean, no issues.

  restock(sku: string, amount: number): boolean {
    const product = this.inventory.get(sku);
    if (!product) return false;
    const newQuantity = product.quantity + amount;
    product.history.push(newQuantity);
    product.quantity = newQuantity;
    return true;
  }
  // REVIEW: Correct. History records the resulting quantity (not
  // the delta), which matches what getHistory/undoLastChange expect.

  sell(sku: string, amount: number): boolean {
    const product = this.inventory.get(sku);
    if (!product) return false;
    if (product.quantity < amount) return false;
    const newQuantity = product.quantity - amount;
    product.history.push(newQuantity);
    product.quantity = newQuantity;
    return true;
  }
  // REVIEW: Clean. Correctly guards against insufficient stock.

  removeProduct(sku: string): boolean {
    if (!this.inventory.has(sku)) return false;
    this.inventory.delete(sku);
    return true;
  }
  // REVIEW: Clean, no issues.

  setCategory(sku: string, category: string): boolean {
    const product = this.inventory.get(sku);
    if (!product) return false;
    product.category.add(category);
    return true;
  }
  // REVIEW: Works, but the spec says "assign a category" (singular),
  // implying one category per product. Using a Set allows multiple
  // categories. The tests don't catch this, but `category: string`
  // and reassignment (`product.category = category`) would match
  // the spec more precisely.

  getByCategory(category: string): string[] {
    const products = Array.from(this.inventory.values());
    const categoryProducts = products.filter(p => p.category.has(category));
    categoryProducts.sort((a, b) => a.sku.localeCompare(b.sku));
    return categoryProducts.map(p => p.sku);
  }
  // REVIEW: Correct. Could chain filter/sort/map for brevity.

  listProducts(): string[] {
    const products = Array.from(this.inventory.values());
    products.sort((a, b) => a.sku.localeCompare(b.sku));
    return products.map(p => p.sku);
  }
  // REVIEW: Works. Could simplify:
  //   return Array.from(this.inventory.keys()).sort();

  setThreshold(sku: string, threshold: number): boolean {
    const product = this.inventory.get(sku);
    if (!product) return false;
    product.threshold = threshold;
    return true;
  }
  // REVIEW: Clean, no issues.

  getLowStock(): string[] {
    const products = Array.from(this.inventory.values());
    const lowInventory = products.filter(p => {
      return p.threshold !== -1 && p.quantity <= p.threshold;
    });
    lowInventory.sort((a,b) => {
      return a.quantity !== b.quantity ? a.quantity - b.quantity : a.sku.localeCompare(b.sku);
    });
    return lowInventory.map(p => p.sku);
  }
  // REVIEW: Works. Using `-1` as sentinel for "no threshold" is
  // fragile — a threshold of 0 is valid, but -1 could be confused.
  // Consider `threshold: number | null` with a `!== null` check
  // instead. (The current approach works because -1 is never a
  // meaningful threshold value.)

  bulkRestock(updates: [string, number][]): number {
    let updated = 0;
    updates.forEach(([sku, restock]) => {
      if (this.restock(sku, restock)) updated++;
    });
    return updated;
  }
  // REVIEW: Nice — good reuse of `this.restock()` so history
  // tracking is handled in one place.

  getHistory(sku: string): number[] {
    const product = this.inventory.get(sku);
    if (!product) return [];
    return product.history;
  }
  // REVIEW: Same note as drill_05 — returns a reference to the
  // internal array, so callers could mutate it. Fine for a drill.

  undoLastChange(sku: string): boolean {
    const product = this.inventory.get(sku);
    if (!product) return false;
    product.history.pop();
    if (!product.history.length) {
      this.inventory.delete(sku);
    } else {
      product.quantity = product.history[product.history.length - 1];
    }
    return true;
  }
  // REVIEW: Nicely done — correctly removes the product when history
  // is exhausted and returns true (learned from the drill_05 bug).
  // Clean branching with `!product.history.length`.
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
