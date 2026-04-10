"""
Drill 01 — Cloud Storage System
================================
Directly reported Dropbox CodeSignal OA problem.
Build a progressive in-memory cloud storage system.

Level 1 — Basic File Operations (10 min)
-----------------------------------------
  add_file(name: str, size: int) -> bool
      Add a file with the given name and size.
      Return False if a file with that name already exists.

  get_file(name: str) -> int
      Return the size of the file, or -1 if not found.

  delete_file(name: str) -> bool
      Remove the file. Return False if not found.

  list_files() -> list[str]
      Return all filenames sorted alphabetically.

Level 2 — User Ownership (10 min)
-----------------------------------
  add_file now accepts a third parameter: owner: str
      add_file(name: str, size: int, owner: str) -> bool

  get_files_by_owner(owner: str) -> list[str]
      Return files owned by user, sorted by size descending
      then name ascending.

  change_owner(name: str, new_owner: str) -> bool
      Change the owner of a file. Return False if file not found.

Level 3 — Search (10 min)
---------------------------
  search(prefix: str) -> list[str]
      Return files whose names start with prefix, sorted by
      size descending then name ascending, max 10 results.

  search_by_suffix(suffix: str) -> list[str]
      Return files whose names end with suffix, same sort
      order, max 10 results.

Level 4 — Capacity & Rollback (15 min)
----------------------------------------
  Constructor accepts optional capacity: int
      (max total bytes stored; None = unlimited).

  add_file returns False if adding would exceed capacity.

  get_used_space() -> int
      Return total bytes currently stored.

  rollback() -> bool
      Undo the last successful add_file / delete_file /
      change_owner operation. Return False if nothing to undo.
      Multiple rollbacks unwind the history stack.

Examples
--------
  cs = CloudStorage(capacity=100)
  cs.add_file("readme.txt", 30, "alice")   # True
  cs.add_file("notes.txt", 50, "bob")      # True
  cs.get_used_space()                       # 80
  cs.add_file("big.bin", 30, "alice")       # False (80+30 > 100)
  cs.rollback()                             # True  (undoes notes.txt add)
  cs.get_used_space()                       # 30
"""


class CloudStorage:
    def __init__(self, capacity: int | None = None):
        # TODO: initialize your data structures
        pass

    # ── Level 1 ──────────────────────────────────────────────

    def add_file(self, name: str, size: int, owner: str = "") -> bool:
        """Add a file. Return False if name exists or would exceed capacity."""
        raise NotImplementedError("Level 1: add_file")

    def get_file(self, name: str) -> int:
        """Return file size, or -1 if not found."""
        raise NotImplementedError("Level 1: get_file")

    def delete_file(self, name: str) -> bool:
        """Remove a file. Return False if not found."""
        raise NotImplementedError("Level 1: delete_file")

    def list_files(self) -> list[str]:
        """Return all filenames sorted alphabetically."""
        raise NotImplementedError("Level 1: list_files")

    # ── Level 2 ──────────────────────────────────────────────

    def get_files_by_owner(self, owner: str) -> list[str]:
        """Files owned by user, sorted by size desc then name asc."""
        raise NotImplementedError("Level 2: get_files_by_owner")

    def change_owner(self, name: str, new_owner: str) -> bool:
        """Change file owner. Return False if file not found."""
        raise NotImplementedError("Level 2: change_owner")

    # ── Level 3 ──────────────────────────────────────────────

    def search(self, prefix: str) -> list[str]:
        """Files starting with prefix, sorted size desc / name asc, max 10."""
        raise NotImplementedError("Level 3: search")

    def search_by_suffix(self, suffix: str) -> list[str]:
        """Files ending with suffix, sorted size desc / name asc, max 10."""
        raise NotImplementedError("Level 3: search_by_suffix")

    # ── Level 4 ──────────────────────────────────────────────

    def get_used_space(self) -> int:
        """Return total bytes currently stored."""
        raise NotImplementedError("Level 4: get_used_space")

    def rollback(self) -> bool:
        """Undo last successful add/delete/change_owner. False if nothing to undo."""
        raise NotImplementedError("Level 4: rollback")


# ─── Self-Checks (do not edit below this line) ──────────────────

_passed = 0
_failed = 0

def _check(label: str, actual: object, expected: object) -> None:
    global _passed, _failed
    if actual == expected:
        _passed += 1
        print(f"  ✓ {label}")
    else:
        _failed += 1
        print(f"  ✗ {label}")
        print(f"    expected: {expected!r}")
        print(f"         got: {actual!r}")

def _level(name: str, fn) -> None:
    global _failed
    print(name)
    try:
        fn()
    except NotImplementedError as e:
        print(f"  ○ {e}")
    except Exception as e:
        _failed += 1
        print(f"  ✗ {e}")

def _run_self_checks() -> None:

    def level_1():
        cs = CloudStorage()
        # add and retrieve
        _check("add new file returns True", cs.add_file("a.txt", 10), True)
        _check("get existing file size", cs.get_file("a.txt"), 10)
        # duplicate add
        _check("add duplicate returns False", cs.add_file("a.txt", 20), False)
        # get non-existent
        _check("get missing file returns -1", cs.get_file("missing"), -1)
        # delete
        _check("delete existing returns True", cs.delete_file("a.txt"), True)
        _check("delete missing returns False", cs.delete_file("a.txt"), False)
        # list files sorted
        cs.add_file("banana.txt", 5)
        cs.add_file("apple.txt", 3)
        cs.add_file("cherry.txt", 7)
        _check("list files sorted", cs.list_files(), ["apple.txt", "banana.txt", "cherry.txt"])
        # empty storage
        cs2 = CloudStorage()
        _check("list files empty", cs2.list_files(), [])

    def level_2():
        cs = CloudStorage()
        cs.add_file("big.dat", 100, "alice")
        cs.add_file("small.dat", 10, "alice")
        cs.add_file("med.dat", 50, "bob")
        cs.add_file("also_big.dat", 100, "alice")
        # files by owner sorted: size desc, name asc
        _check("alice files sorted", cs.get_files_by_owner("alice"),
               ["also_big.dat", "big.dat", "small.dat"])
        _check("bob files sorted", cs.get_files_by_owner("bob"), ["med.dat"])
        _check("unknown owner empty", cs.get_files_by_owner("charlie"), [])
        # change owner
        _check("change owner success", cs.change_owner("med.dat", "alice"), True)
        _check("change owner missing file", cs.change_owner("nope", "alice"), False)
        _check("alice now has med.dat", "med.dat" in cs.get_files_by_owner("alice"), True)
        _check("bob now empty", cs.get_files_by_owner("bob"), [])

    def level_3():
        cs = CloudStorage()
        cs.add_file("report_2024.pdf", 500, "alice")
        cs.add_file("report_2023.pdf", 300, "alice")
        cs.add_file("readme.md", 50, "bob")
        cs.add_file("recipe.txt", 200, "bob")
        cs.add_file("data.csv", 1000, "charlie")
        # prefix search
        _check("prefix 'rep' matches", cs.search("rep"),
               ["report_2024.pdf", "report_2023.pdf"])
        _check("prefix 're' includes readme and recipe",
               cs.search("re"),
               ["report_2024.pdf", "report_2023.pdf", "recipe.txt", "readme.md"])
        _check("prefix no match", cs.search("zzz"), [])
        # suffix search
        _check("suffix '.pdf'", cs.search_by_suffix(".pdf"),
               ["report_2024.pdf", "report_2023.pdf"])
        _check("suffix '.txt'", cs.search_by_suffix(".txt"), ["recipe.txt"])
        _check("suffix no match", cs.search_by_suffix(".zip"), [])
        # max 10 results
        cs2 = CloudStorage()
        for i in range(15):
            cs2.add_file(f"file_{i:02d}.txt", i + 1, "user")
        _check("search max 10 results", len(cs2.search("file")), 10)
        _check("search_by_suffix max 10 results", len(cs2.search_by_suffix(".txt")), 10)

    def level_4():
        # capacity
        cs = CloudStorage(capacity=100)
        _check("add within capacity", cs.add_file("a.txt", 60, "alice"), True)
        _check("used space after add", cs.get_used_space(), 60)
        _check("add exceeds capacity", cs.add_file("b.txt", 50, "bob"), False)
        _check("add exactly at capacity", cs.add_file("c.txt", 40, "bob"), True)
        _check("used space at capacity", cs.get_used_space(), 100)
        # rollback add
        _check("rollback last add", cs.rollback(), True)
        _check("used space after rollback", cs.get_used_space(), 60)
        _check("file gone after rollback", cs.get_file("c.txt"), -1)
        # rollback delete
        cs.delete_file("a.txt")
        _check("used space after delete", cs.get_used_space(), 0)
        _check("rollback delete restores file", cs.rollback(), True)
        _check("file restored", cs.get_file("a.txt"), 60)
        _check("used space restored", cs.get_used_space(), 60)
        # rollback change_owner
        cs.change_owner("a.txt", "bob")
        _check("rollback change_owner", cs.rollback(), True)
        _check("owner reverted (file still owned by alice)",
               "a.txt" in cs.get_files_by_owner("alice"), True)
        # rollback empty history
        cs2 = CloudStorage()
        _check("rollback empty returns False", cs2.rollback(), False)
        # unlimited capacity
        cs3 = CloudStorage()
        _check("unlimited add", cs3.add_file("huge.bin", 999999, "user"), True)
        _check("unlimited used space", cs3.get_used_space(), 999999)

    _level("Level 1 — Basic File Operations", level_1)
    _level("Level 2 — User Ownership", level_2)
    _level("Level 3 — Search", level_3)
    _level("Level 4 — Capacity & Rollback", level_4)

def main() -> None:
    print("\nDrill 01 — Cloud Storage System\n")
    _run_self_checks()
    total = _passed + _failed
    print(f"\n{_passed}/{total} passed")
    if _failed == 0 and total > 0:
        print("All tests passed.")

if __name__ == "__main__":
    main()
