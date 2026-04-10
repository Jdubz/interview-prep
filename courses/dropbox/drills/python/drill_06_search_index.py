"""
Drill 06 — Search Index
========================
Dash universal search — build an inverted index that supports keyword
search across documents from multiple sources.

Level 1 — Basic Indexing (10 min)
----------------------------------
  add_document(doc_id: str, content: str) -> None
      Index a document. Content is a space-separated string of
      words. If doc_id already exists, replace its content.

  search(keyword: str) -> list[str]
      Return doc_ids containing the keyword (case-insensitive).
      Sorted alphabetically.

  remove_document(doc_id: str) -> bool
      Remove from index. Return False if not found.

Level 2 — Multi-Keyword Search (10 min)
-----------------------------------------
  search_all(keywords: list[str]) -> list[str]
      Return doc_ids containing ALL keywords (AND).
      Case-insensitive. Sorted alphabetically.

  search_any(keywords: list[str]) -> list[str]
      Return doc_ids containing ANY keyword (OR).
      Case-insensitive. Sorted alphabetically.

  get_word_count(doc_id: str) -> int
      Number of unique words in the document. -1 if not found.

Level 3 — Ranked Results (10 min)
-----------------------------------
  search_ranked(keyword: str) -> list[str]
      Return doc_ids sorted by relevance: number of times the
      keyword appears in the document (descending), then doc_id
      alphabetically for ties. Case-insensitive.

  add_document now accepts an optional source: str parameter
      (e.g. "gmail", "slack", "gdrive"). Default is "unknown".
      add_document(doc_id: str, content: str, source: str = "unknown")

  search_by_source(keyword: str, source: str) -> list[str]
      Search filtered to a specific source. Sorted alphabetically.

Level 4 — Prefix Search / Autocomplete (15 min)
-------------------------------------------------
  autocomplete(prefix: str, limit: int = 5) -> list[str]
      Return unique words in the index that start with prefix
      (case-insensitive). Sorted alphabetically. Limited to
      `limit` results.

  get_index_stats() -> dict
      Return {"documents": int, "unique_words": int,
              "sources": list[str]}
      where sources is sorted alphabetically.

Examples
--------
  idx = SearchIndex()
  idx.add_document("d1", "hello world hello", source="gmail")
  idx.search("hello")           # -> ["d1"]
  idx.search_ranked("hello")    # -> ["d1"]  (2 occurrences)
  idx.autocomplete("hel")       # -> ["hello"]
"""


class SearchIndex:
    def __init__(self):
        # TODO: initialize your data structures
        pass

    # ── Level 1 ──────────────────────────────────────────────

    def add_document(self, doc_id: str, content: str, source: str = "unknown") -> None:
        """Index a document. Replace if doc_id already exists."""
        raise NotImplementedError("Level 1: add_document")

    def search(self, keyword: str) -> list[str]:
        """Return doc_ids containing keyword (case-insensitive), sorted."""
        raise NotImplementedError("Level 1: search")

    def remove_document(self, doc_id: str) -> bool:
        """Remove from index. Return False if not found."""
        raise NotImplementedError("Level 1: remove_document")

    # ── Level 2 ──────────────────────────────────────────────

    def search_all(self, keywords: list[str]) -> list[str]:
        """Return doc_ids containing ALL keywords (AND). Sorted."""
        raise NotImplementedError("Level 2: search_all")

    def search_any(self, keywords: list[str]) -> list[str]:
        """Return doc_ids containing ANY keyword (OR). Sorted."""
        raise NotImplementedError("Level 2: search_any")

    def get_word_count(self, doc_id: str) -> int:
        """Number of unique words in the document. -1 if not found."""
        raise NotImplementedError("Level 2: get_word_count")

    # ── Level 3 ──────────────────────────────────────────────

    def search_ranked(self, keyword: str) -> list[str]:
        """Doc_ids sorted by keyword frequency desc, then id asc."""
        raise NotImplementedError("Level 3: search_ranked")

    def search_by_source(self, keyword: str, source: str) -> list[str]:
        """Search filtered to a specific source. Sorted alphabetically."""
        raise NotImplementedError("Level 3: search_by_source")

    # ── Level 4 ──────────────────────────────────────────────

    def autocomplete(self, prefix: str, limit: int = 5) -> list[str]:
        """Words starting with prefix (case-insensitive), sorted, limited."""
        raise NotImplementedError("Level 4: autocomplete")

    def get_index_stats(self) -> dict:
        """Return {documents: int, unique_words: int, sources: list[str]}."""
        raise NotImplementedError("Level 4: get_index_stats")


# ─── Self-Checks (do not edit below this line) ──────────────────

_passed = 0
_failed = 0

def _check(label: str, actual: object, expected: object) -> None:
    global _passed, _failed
    if actual == expected:
        _passed += 1
        print(f"  \u2713 {label}")
    else:
        _failed += 1
        print(f"  \u2717 {label}")
        print(f"    expected: {expected!r}")
        print(f"         got: {actual!r}")

def _level(name: str, fn) -> None:
    global _failed
    print(name)
    try:
        fn()
    except NotImplementedError as e:
        print(f"  \u25cb {e}")
    except Exception as e:
        _failed += 1
        print(f"  \u2717 {e}")

def _run_self_checks() -> None:

    def level_1():
        idx = SearchIndex()
        # Basic add and search
        idx.add_document("doc1", "the quick brown fox")
        idx.add_document("doc2", "the lazy brown dog")
        idx.add_document("doc3", "quick fox jumps high")
        _check("search 'brown'",
               idx.search("brown"), ["doc1", "doc2"])
        _check("search 'quick'",
               idx.search("quick"), ["doc1", "doc3"])

        # Case-insensitive search
        idx.add_document("doc4", "Hello World HELLO")
        _check("case-insensitive search 'hello'",
               idx.search("hello"), ["doc4"])
        _check("case-insensitive search 'HELLO'",
               idx.search("HELLO"), ["doc4"])

        # Search miss
        _check("search miss",
               idx.search("zebra"), [])

        # Replace existing document
        idx.add_document("doc1", "completely new content")
        _check("replaced doc no longer matches 'fox'",
               idx.search("fox"), ["doc3"])
        _check("replaced doc matches 'completely'",
               idx.search("completely"), ["doc1"])

        # Remove
        _check("remove existing returns True",
               idx.remove_document("doc2"), True)
        _check("removed doc gone from search",
               idx.search("lazy"), [])
        _check("remove missing returns False",
               idx.remove_document("doc999"), False)

    def level_2():
        idx = SearchIndex()
        idx.add_document("d1", "python java rust")
        idx.add_document("d2", "python rust go")
        idx.add_document("d3", "java go elixir")
        idx.add_document("d4", "python java go rust elixir")

        # search_all — AND
        _check("search_all ['python', 'rust']",
               idx.search_all(["python", "rust"]),
               ["d1", "d2", "d4"])
        _check("search_all ['java', 'go', 'elixir']",
               idx.search_all(["java", "go", "elixir"]),
               ["d3", "d4"])
        _check("search_all no match",
               idx.search_all(["python", "elixir", "haskell"]),
               [])

        # search_any — OR
        _check("search_any ['elixir']",
               idx.search_any(["elixir"]),
               ["d3", "d4"])
        _check("search_any ['python', 'elixir']",
               idx.search_any(["python", "elixir"]),
               ["d1", "d2", "d3", "d4"])
        _check("search_any no match",
               idx.search_any(["haskell", "cobol"]),
               [])

        # get_word_count
        _check("word count d1", idx.get_word_count("d1"), 3)
        _check("word count d4", idx.get_word_count("d4"), 5)
        _check("word count missing", idx.get_word_count("d99"), -1)

        # Duplicate words — unique count
        idx.add_document("d5", "go go go python python")
        _check("word count with dupes", idx.get_word_count("d5"), 2)

    def level_3():
        idx = SearchIndex()
        # Frequency ranking: "data" appears 3x in d1, 1x in d2, 2x in d3
        idx.add_document("d1", "data data data science", source="gmail")
        idx.add_document("d2", "data engineering", source="slack")
        idx.add_document("d3", "big data data pipeline", source="gmail")
        idx.add_document("d4", "machine learning model", source="gdrive")

        _check("search_ranked by frequency",
               idx.search_ranked("data"),
               ["d1", "d3", "d2"])  # 3, 2, 1 occurrences

        # Tie-breaking by doc_id alphabetically
        idx.add_document("a_doc", "data results", source="slack")
        idx.add_document("z_doc", "data results", source="slack")
        ranked = idx.search_ranked("data")
        # a_doc and z_doc both have 1 occurrence, same as d2
        _check("search_ranked tie-breaking",
               ranked,
               ["d1", "d3", "a_doc", "d2", "z_doc"])

        _check("search_ranked no match",
               idx.search_ranked("quantum"), [])

        # search_by_source
        _check("search_by_source gmail 'data'",
               idx.search_by_source("data", "gmail"),
               ["d1", "d3"])
        _check("search_by_source slack 'data'",
               idx.search_by_source("data", "slack"),
               ["a_doc", "z_doc"])
        _check("search_by_source gdrive 'data'",
               idx.search_by_source("data", "gdrive"),
               [])

        # Case-insensitive ranked
        idx.add_document("d5", "Data DATA data", source="gmail")
        _check("ranked case-insensitive — d5 has 3 occurrences",
               idx.search_ranked("DATA")[0:2],
               ["d1", "d5"])  # both have 3

    def level_4():
        idx = SearchIndex()
        idx.add_document("d1", "dropbox desktop application sync", source="gdrive")
        idx.add_document("d2", "dropbox drive document sharing", source="gmail")
        idx.add_document("d3", "application programming interface api", source="slack")

        # Autocomplete basic
        _check("autocomplete 'drop'",
               idx.autocomplete("drop"),
               ["dropbox"])
        _check("autocomplete 'dr'",
               idx.autocomplete("dr"),
               ["drive", "dropbox"])
        _check("autocomplete 'app'",
               idx.autocomplete("app"),
               ["application"])

        # Autocomplete case-insensitive
        _check("autocomplete 'D' case-insensitive",
               idx.autocomplete("D"),
               ["desktop", "document", "drive", "dropbox"])

        # Autocomplete with limit
        idx.add_document("d4", "alpha beta gamma delta epsilon zeta")
        _check("autocomplete limit 3",
               idx.autocomplete("", 3),
               ["alpha", "api", "application"])

        # Autocomplete no match
        _check("autocomplete no match",
               idx.autocomplete("xyz"),
               [])

        # get_index_stats
        stats = idx.get_index_stats()
        _check("stats documents count",
               stats["documents"], 4)
        _check("stats unique_words > 0",
               stats["unique_words"] > 0, True)
        _check("stats sources sorted",
               stats["sources"],
               ["gdrive", "gmail", "slack"])

        # Remove a doc and check stats update
        idx.remove_document("d3")
        stats2 = idx.get_index_stats()
        _check("stats after remove — documents count",
               stats2["documents"], 3)

    _level("Level 1 \u2014 Basic Indexing", level_1)
    _level("Level 2 \u2014 Multi-Keyword Search", level_2)
    _level("Level 3 \u2014 Ranked Results", level_3)
    _level("Level 4 \u2014 Prefix Search / Autocomplete", level_4)

def main() -> None:
    print("\nDrill 06 \u2014 Search Index\n")
    _run_self_checks()
    total = _passed + _failed
    print(f"\n{_passed}/{total} passed")
    if _failed == 0 and total > 0:
        print("All tests passed.")

if __name__ == "__main__":
    main()
