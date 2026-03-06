/*
Drill 01 — File Storage

Implement a FileStorage class that manages files in memory.
Each level adds new methods. Keep all previous methods working.

────────────────────────────────────────
Level 1 — Basic Operations

  addFile(name: string, size: number): boolean
    Add a file. Returns true if added, false if name already exists.

  getFileSize(name: string): number
    Returns the file's size, or -1 if not found.

  deleteFile(name: string): boolean
    Removes the file. Returns true if deleted, false if not found.

────────────────────────────────────────
Level 2 — Copy & Search

  copyFile(source: string, dest: string): boolean
    Copy source to dest (overwrites dest if it exists).
    Returns false if source does not exist.

  search(prefix: string): string[]
    Returns up to 10 file names that start with prefix.
    Sort by size descending, then name ascending for ties.

────────────────────────────────────────
Level 3 — Capacity

  Constructor accepts an optional capacity (max total bytes).
  No capacity means unlimited storage.

  addFile returns false if adding would exceed capacity.
  copyFile returns false if copying would exceed capacity
    (only counts the net change — overwriting a file replaces its size).

  getUsedSpace(): number — total bytes stored
  getRemainingSpace(): number — bytes remaining (Infinity if unlimited)

────────────────────────────────────────
Level 4 — Undo

  undo(): boolean
    Reverts the last successful add, delete, or copy.
    Returns false if there is nothing to undo.
    Multiple undo() calls walk back through history.
*/

export class FileStorage {
  capacity: number | null;
  fileStore: Map<string, number>;

  constructor(capacity?: number) {
    this.capacity = capacity || null
    this.fileStore = new Map();
  }

  addFile(name: string, size: number): boolean {
    const existingFile = this.fileStore.get(name);
    if (existingFile) return false;
    this.fileStore.set(name, size);
    return true;
  }

  getFileSize(name: string): number {
    const existingFile = this.fileStore.get(name);
    if (existingFile) return existingFile;
    return -1;
  }

  deleteFile(name: string): boolean {
    const existingFile = this.fileStore.get(name);
    if (existingFile) {
      this.fileStore.delete(name);
      return true;
    }
    return false;
  }

  copyFile(source: string, dest: string): boolean {
    throw new Error("TODO: implement copyFile");
  }

  search(prefix: string): string[] {
    throw new Error("TODO: implement search");
  }

  getUsedSpace(): number {
    throw new Error("TODO: implement getUsedSpace");
  }

  getRemainingSpace(): number {
    throw new Error("TODO: implement getRemainingSpace");
  }

  undo(): boolean {
    throw new Error("TODO: implement undo");
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
    const s1 = new FileStorage();
    check("add new", s1.addFile("a.txt", 100), true);
    check("add dup", s1.addFile("a.txt", 999), false);
    check("get exists", s1.getFileSize("a.txt"), 100);
    check("get missing", s1.getFileSize("z.txt"), -1);
    check("delete exists", s1.deleteFile("a.txt"), true);
    check("delete missing", s1.deleteFile("a.txt"), false);
    check("get after delete", s1.getFileSize("a.txt"), -1);
  });

  level("Level 2 — Copy & Search", () => {
    const s2 = new FileStorage();
    s2.addFile("app.js", 300);
    s2.addFile("api.js", 300);
    s2.addFile("assets.zip", 100);
    s2.addFile("readme.md", 50);
    check("copy ok", s2.copyFile("app.js", "app_backup.js"), true);
    check("copy missing", s2.copyFile("nope.txt", "x.txt"), false);
    check("search", s2.search("a"),
      ["api.js", "app.js", "app_backup.js", "assets.zip"]);
    check("search no match", s2.search("zzz"), []);
  });

  level("Level 3 — Capacity", () => {
    const s3 = new FileStorage(500);
    check("add within", s3.addFile("a.txt", 200), true);
    check("add within 2", s3.addFile("b.txt", 200), true);
    check("add exceeds", s3.addFile("c.txt", 200), false);
    check("used", s3.getUsedSpace(), 400);
    check("remaining", s3.getRemainingSpace(), 100);
    const s3b = new FileStorage();
    s3b.addFile("x.txt", 1000);
    check("no limit remaining", s3b.getRemainingSpace(), Infinity);
  });

  level("Level 4 — Undo", () => {
    const s4 = new FileStorage();
    s4.addFile("file.txt", 100);
    s4.deleteFile("file.txt");
    check("after delete", s4.getFileSize("file.txt"), -1);
    check("undo delete", s4.undo(), true);
    check("restored", s4.getFileSize("file.txt"), 100);
    check("undo add", s4.undo(), true);
    check("fully undone", s4.getFileSize("file.txt"), -1);
    check("nothing left", s4.undo(), false);

    const s4b = new FileStorage();
    s4b.addFile("src.txt", 200);
    s4b.copyFile("src.txt", "dst.txt");
    check("copy created", s4b.getFileSize("dst.txt"), 200);
    check("undo copy", s4b.undo(), true);
    check("copy undone", s4b.getFileSize("dst.txt"), -1);
    check("src intact", s4b.getFileSize("src.txt"), 200);
  });
}

function main(): void {
  console.log("\nFile Storage\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
