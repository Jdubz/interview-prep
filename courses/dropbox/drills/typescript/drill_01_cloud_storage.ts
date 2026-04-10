/**
 * Drill 01 — Cloud Storage System
 * ================================
 * Directly reported Dropbox CodeSignal OA problem.
 * Build a progressive in-memory cloud storage system.
 *
 * Level 1 — Basic File Operations (10 min)
 * -----------------------------------------
 *   addFile(name: string, size: number): boolean
 *       Add a file with the given name and size.
 *       Return false if a file with that name already exists.
 *
 *   getFile(name: string): number
 *       Return the size of the file, or -1 if not found.
 *
 *   deleteFile(name: string): boolean
 *       Remove the file. Return false if not found.
 *
 *   listFiles(): string[]
 *       Return all filenames sorted alphabetically.
 *
 * Level 2 — User Ownership (10 min)
 * -----------------------------------
 *   addFile now accepts a third parameter: owner: string
 *       addFile(name: string, size: number, owner: string): boolean
 *
 *   getFilesByOwner(owner: string): string[]
 *       Return files owned by user, sorted by size descending
 *       then name ascending.
 *
 *   changeOwner(name: string, newOwner: string): boolean
 *       Change the owner of a file. Return false if file not found.
 *
 * Level 3 — Search (10 min)
 * ---------------------------
 *   search(prefix: string): string[]
 *       Return files whose names start with prefix, sorted by
 *       size descending then name ascending, max 10 results.
 *
 *   searchBySuffix(suffix: string): string[]
 *       Return files whose names end with suffix, same sort
 *       order, max 10 results.
 *
 * Level 4 — Capacity & Rollback (15 min)
 * ----------------------------------------
 *   Constructor accepts optional capacity: number | null
 *       (max total bytes stored; null = unlimited).
 *
 *   addFile returns false if adding would exceed capacity.
 *
 *   getUsedSpace(): number
 *       Return total bytes currently stored.
 *
 *   rollback(): boolean
 *       Undo the last successful addFile / deleteFile /
 *       changeOwner operation. Return false if nothing to undo.
 *       Multiple rollbacks unwind the history stack.
 *
 * Examples
 * --------
 *   const cs = new CloudStorage(100);
 *   cs.addFile("readme.txt", 30, "alice");   // true
 *   cs.addFile("notes.txt", 50, "bob");      // true
 *   cs.getUsedSpace();                       // 80
 *   cs.addFile("big.bin", 30, "alice");      // false (80+30 > 100)
 *   cs.rollback();                           // true  (undoes notes.txt add)
 *   cs.getUsedSpace();                       // 30
 */

class CloudStorage {
  constructor(capacity: number | null = null) {
    // TODO: initialize your data structures
  }

  // -- Level 1 --------------------------------------------------

  addFile(name: string, size: number, owner: string = ""): boolean {
    /** Add a file. Return false if name exists or would exceed capacity. */
    throw new Error("TODO: addFile");
  }

  getFile(name: string): number {
    /** Return file size, or -1 if not found. */
    throw new Error("TODO: getFile");
  }

  deleteFile(name: string): boolean {
    /** Remove a file. Return false if not found. */
    throw new Error("TODO: deleteFile");
  }

  listFiles(): string[] {
    /** Return all filenames sorted alphabetically. */
    throw new Error("TODO: listFiles");
  }

  // -- Level 2 --------------------------------------------------

  getFilesByOwner(owner: string): string[] {
    /** Files owned by user, sorted by size desc then name asc. */
    throw new Error("TODO: getFilesByOwner");
  }

  changeOwner(name: string, newOwner: string): boolean {
    /** Change file owner. Return false if file not found. */
    throw new Error("TODO: changeOwner");
  }

  // -- Level 3 --------------------------------------------------

  search(prefix: string): string[] {
    /** Files starting with prefix, sorted size desc / name asc, max 10. */
    throw new Error("TODO: search");
  }

  searchBySuffix(suffix: string): string[] {
    /** Files ending with suffix, sorted size desc / name asc, max 10. */
    throw new Error("TODO: searchBySuffix");
  }

  // -- Level 4 --------------------------------------------------

  getUsedSpace(): number {
    /** Return total bytes currently stored. */
    throw new Error("TODO: getUsedSpace");
  }

  rollback(): boolean {
    /** Undo last successful add/delete/changeOwner. False if nothing to undo. */
    throw new Error("TODO: rollback");
  }
}

// --- Self-Checks (do not edit below this line) ------------------

let _passed = 0;
let _failed = 0;

function _check(label: string, actual: unknown, expected: unknown): void {
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    _passed++;
    console.log(`  \u2713 ${label}`);
  } else {
    _failed++;
    console.log(`  \u2717 ${label}`);
    console.log(`    expected: ${JSON.stringify(expected)}`);
    console.log(`         got: ${JSON.stringify(actual)}`);
  }
}

function _level(name: string, fn: () => void): void {
  console.log(name);
  try {
    fn();
  } catch (e: any) {
    if (e.message?.startsWith("TODO")) {
      console.log(`  \u25CB ${e.message}`);
    } else {
      _failed++;
      console.log(`  \u2717 ${e.message}`);
    }
  }
}

function _runSelfChecks(): void {
  function level1(): void {
    const cs = new CloudStorage();
    // add and retrieve
    _check("add new file returns true", cs.addFile("a.txt", 10), true);
    _check("get existing file size", cs.getFile("a.txt"), 10);
    // duplicate add
    _check("add duplicate returns false", cs.addFile("a.txt", 20), false);
    // get non-existent
    _check("get missing file returns -1", cs.getFile("missing"), -1);
    // delete
    _check("delete existing returns true", cs.deleteFile("a.txt"), true);
    _check("delete missing returns false", cs.deleteFile("a.txt"), false);
    // list files sorted
    cs.addFile("banana.txt", 5);
    cs.addFile("apple.txt", 3);
    cs.addFile("cherry.txt", 7);
    _check("list files sorted", cs.listFiles(), [
      "apple.txt",
      "banana.txt",
      "cherry.txt",
    ]);
    // empty storage
    const cs2 = new CloudStorage();
    _check("list files empty", cs2.listFiles(), []);
  }

  function level2(): void {
    const cs = new CloudStorage();
    cs.addFile("big.dat", 100, "alice");
    cs.addFile("small.dat", 10, "alice");
    cs.addFile("med.dat", 50, "bob");
    cs.addFile("also_big.dat", 100, "alice");
    // files by owner sorted: size desc, name asc
    _check("alice files sorted", cs.getFilesByOwner("alice"), [
      "also_big.dat",
      "big.dat",
      "small.dat",
    ]);
    _check("bob files sorted", cs.getFilesByOwner("bob"), ["med.dat"]);
    _check("unknown owner empty", cs.getFilesByOwner("charlie"), []);
    // change owner
    _check("change owner success", cs.changeOwner("med.dat", "alice"), true);
    _check(
      "change owner missing file",
      cs.changeOwner("nope", "alice"),
      false,
    );
    _check(
      "alice now has med.dat",
      cs.getFilesByOwner("alice").includes("med.dat"),
      true,
    );
    _check("bob now empty", cs.getFilesByOwner("bob"), []);
  }

  function level3(): void {
    const cs = new CloudStorage();
    cs.addFile("report_2024.pdf", 500, "alice");
    cs.addFile("report_2023.pdf", 300, "alice");
    cs.addFile("readme.md", 50, "bob");
    cs.addFile("recipe.txt", 200, "bob");
    cs.addFile("data.csv", 1000, "charlie");
    // prefix search
    _check("prefix 'rep' matches", cs.search("rep"), [
      "report_2024.pdf",
      "report_2023.pdf",
    ]);
    _check("prefix 're' includes readme and recipe", cs.search("re"), [
      "report_2024.pdf",
      "report_2023.pdf",
      "recipe.txt",
      "readme.md",
    ]);
    _check("prefix no match", cs.search("zzz"), []);
    // suffix search
    _check("suffix '.pdf'", cs.searchBySuffix(".pdf"), [
      "report_2024.pdf",
      "report_2023.pdf",
    ]);
    _check("suffix '.txt'", cs.searchBySuffix(".txt"), ["recipe.txt"]);
    _check("suffix no match", cs.searchBySuffix(".zip"), []);
    // max 10 results
    const cs2 = new CloudStorage();
    for (let i = 0; i < 15; i++) {
      const name = `file_${String(i).padStart(2, "0")}.txt`;
      cs2.addFile(name, i + 1, "user");
    }
    _check("search max 10 results", cs2.search("file").length, 10);
    _check(
      "searchBySuffix max 10 results",
      cs2.searchBySuffix(".txt").length,
      10,
    );
  }

  function level4(): void {
    // capacity
    const cs = new CloudStorage(100);
    _check("add within capacity", cs.addFile("a.txt", 60, "alice"), true);
    _check("used space after add", cs.getUsedSpace(), 60);
    _check("add exceeds capacity", cs.addFile("b.txt", 50, "bob"), false);
    _check("add exactly at capacity", cs.addFile("c.txt", 40, "bob"), true);
    _check("used space at capacity", cs.getUsedSpace(), 100);
    // rollback add
    _check("rollback last add", cs.rollback(), true);
    _check("used space after rollback", cs.getUsedSpace(), 60);
    _check("file gone after rollback", cs.getFile("c.txt"), -1);
    // rollback delete
    cs.deleteFile("a.txt");
    _check("used space after delete", cs.getUsedSpace(), 0);
    _check("rollback delete restores file", cs.rollback(), true);
    _check("file restored", cs.getFile("a.txt"), 60);
    _check("used space restored", cs.getUsedSpace(), 60);
    // rollback change_owner
    cs.changeOwner("a.txt", "bob");
    _check("rollback change_owner", cs.rollback(), true);
    _check(
      "owner reverted (file still owned by alice)",
      cs.getFilesByOwner("alice").includes("a.txt"),
      true,
    );
    // rollback empty history
    const cs2 = new CloudStorage();
    _check("rollback empty returns false", cs2.rollback(), false);
    // unlimited capacity
    const cs3 = new CloudStorage();
    _check("unlimited add", cs3.addFile("huge.bin", 999999, "user"), true);
    _check("unlimited used space", cs3.getUsedSpace(), 999999);
  }

  _level("Level 1 \u2014 Basic File Operations", level1);
  _level("Level 2 \u2014 User Ownership", level2);
  _level("Level 3 \u2014 Search", level3);
  _level("Level 4 \u2014 Capacity & Rollback", level4);
}

console.log("\nDrill 01 \u2014 Cloud Storage System\n");
_runSelfChecks();
const _total = _passed + _failed;
console.log(`\n${_passed}/${_total} passed`);
if (_failed === 0 && _total > 0) {
  console.log("All tests passed.");
}
