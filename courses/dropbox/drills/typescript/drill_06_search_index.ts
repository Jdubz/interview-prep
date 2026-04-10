/**
 * Drill 06 — Search Index
 * ========================
 * Dash universal search — build an inverted index that supports keyword
 * search across documents from multiple sources.
 *
 * Level 1 — Basic Indexing (10 min)
 * ----------------------------------
 *   addDocument(docId: string, content: string, source?: string): void
 *       Index a document. Content is a space-separated string of
 *       words. If docId already exists, replace its content.
 *
 *   search(keyword: string): string[]
 *       Return docIds containing the keyword (case-insensitive).
 *       Sorted alphabetically.
 *
 *   removeDocument(docId: string): boolean
 *       Remove from index. Return false if not found.
 *
 * Level 2 — Multi-Keyword Search (10 min)
 * -----------------------------------------
 *   searchAll(keywords: string[]): string[]
 *       Return docIds containing ALL keywords (AND).
 *       Case-insensitive. Sorted alphabetically.
 *
 *   searchAny(keywords: string[]): string[]
 *       Return docIds containing ANY keyword (OR).
 *       Case-insensitive. Sorted alphabetically.
 *
 *   getWordCount(docId: string): number
 *       Number of unique words in the document. -1 if not found.
 *
 * Level 3 — Ranked Results (10 min)
 * -----------------------------------
 *   searchRanked(keyword: string): string[]
 *       Return docIds sorted by relevance: number of times the
 *       keyword appears in the document (descending), then docId
 *       alphabetically for ties. Case-insensitive.
 *
 *   addDocument now accepts an optional source parameter
 *       (e.g. "gmail", "slack", "gdrive"). Default is "unknown".
 *
 *   searchBySource(keyword: string, source: string): string[]
 *       Search filtered to a specific source. Sorted alphabetically.
 *
 * Level 4 — Prefix Search / Autocomplete (15 min)
 * -------------------------------------------------
 *   autocomplete(prefix: string, limit?: number): string[]
 *       Return unique words in the index that start with prefix
 *       (case-insensitive). Sorted alphabetically. Limited to
 *       `limit` results (default 5).
 *
 *   getIndexStats(): { documents: number; unique_words: number; sources: string[] }
 *       where sources is sorted alphabetically.
 *
 * Examples
 * --------
 *   const idx = new SearchIndex();
 *   idx.addDocument("d1", "hello world hello", "gmail");
 *   idx.search("hello");           // -> ["d1"]
 *   idx.searchRanked("hello");     // -> ["d1"]  (2 occurrences)
 *   idx.autocomplete("hel");       // -> ["hello"]
 */


class SearchIndex {
  constructor() {
    // TODO: initialize your data structures
  }

  // -- Level 1 --------------------------------------------------

  /** Index a document. Replace if docId already exists. */
  addDocument(docId: string, content: string, source: string = "unknown"): void {
    throw new Error("TODO: addDocument");
  }

  /** Return docIds containing keyword (case-insensitive), sorted. */
  search(keyword: string): string[] {
    throw new Error("TODO: search");
  }

  /** Remove from index. Return false if not found. */
  removeDocument(docId: string): boolean {
    throw new Error("TODO: removeDocument");
  }

  // -- Level 2 --------------------------------------------------

  /** Return docIds containing ALL keywords (AND). Sorted. */
  searchAll(keywords: string[]): string[] {
    throw new Error("TODO: searchAll");
  }

  /** Return docIds containing ANY keyword (OR). Sorted. */
  searchAny(keywords: string[]): string[] {
    throw new Error("TODO: searchAny");
  }

  /** Number of unique words in the document. -1 if not found. */
  getWordCount(docId: string): number {
    throw new Error("TODO: getWordCount");
  }

  // -- Level 3 --------------------------------------------------

  /** DocIds sorted by keyword frequency desc, then id asc. */
  searchRanked(keyword: string): string[] {
    throw new Error("TODO: searchRanked");
  }

  /** Search filtered to a specific source. Sorted alphabetically. */
  searchBySource(keyword: string, source: string): string[] {
    throw new Error("TODO: searchBySource");
  }

  // -- Level 4 --------------------------------------------------

  /** Words starting with prefix (case-insensitive), sorted, limited. */
  autocomplete(prefix: string, limit: number = 5): string[] {
    throw new Error("TODO: autocomplete");
  }

  /** Return {documents, unique_words, sources}. */
  getIndexStats(): { documents: number; unique_words: number; sources: string[] } {
    throw new Error("TODO: getIndexStats");
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
      console.log(`  \u25cb ${e.message}`);
    } else {
      _failed++;
      console.log(`  \u2717 ${e.message}`);
    }
  }
}

function _runSelfChecks(): void {

  function level1(): void {
    const idx = new SearchIndex();
    // Basic add and search
    idx.addDocument("doc1", "the quick brown fox");
    idx.addDocument("doc2", "the lazy brown dog");
    idx.addDocument("doc3", "quick fox jumps high");
    _check("search 'brown'",
      idx.search("brown"), ["doc1", "doc2"]);
    _check("search 'quick'",
      idx.search("quick"), ["doc1", "doc3"]);

    // Case-insensitive search
    idx.addDocument("doc4", "Hello World HELLO");
    _check("case-insensitive search 'hello'",
      idx.search("hello"), ["doc4"]);
    _check("case-insensitive search 'HELLO'",
      idx.search("HELLO"), ["doc4"]);

    // Search miss
    _check("search miss",
      idx.search("zebra"), []);

    // Replace existing document
    idx.addDocument("doc1", "completely new content");
    _check("replaced doc no longer matches 'fox'",
      idx.search("fox"), ["doc3"]);
    _check("replaced doc matches 'completely'",
      idx.search("completely"), ["doc1"]);

    // Remove
    _check("remove existing returns true",
      idx.removeDocument("doc2"), true);
    _check("removed doc gone from search",
      idx.search("lazy"), []);
    _check("remove missing returns false",
      idx.removeDocument("doc999"), false);
  }

  function level2(): void {
    const idx = new SearchIndex();
    idx.addDocument("d1", "python java rust");
    idx.addDocument("d2", "python rust go");
    idx.addDocument("d3", "java go elixir");
    idx.addDocument("d4", "python java go rust elixir");

    // searchAll — AND
    _check("searchAll ['python', 'rust']",
      idx.searchAll(["python", "rust"]),
      ["d1", "d2", "d4"]);
    _check("searchAll ['java', 'go', 'elixir']",
      idx.searchAll(["java", "go", "elixir"]),
      ["d3", "d4"]);
    _check("searchAll no match",
      idx.searchAll(["python", "elixir", "haskell"]),
      []);

    // searchAny — OR
    _check("searchAny ['elixir']",
      idx.searchAny(["elixir"]),
      ["d3", "d4"]);
    _check("searchAny ['python', 'elixir']",
      idx.searchAny(["python", "elixir"]),
      ["d1", "d2", "d3", "d4"]);
    _check("searchAny no match",
      idx.searchAny(["haskell", "cobol"]),
      []);

    // getWordCount
    _check("word count d1", idx.getWordCount("d1"), 3);
    _check("word count d4", idx.getWordCount("d4"), 5);
    _check("word count missing", idx.getWordCount("d99"), -1);

    // Duplicate words — unique count
    idx.addDocument("d5", "go go go python python");
    _check("word count with dupes", idx.getWordCount("d5"), 2);
  }

  function level3(): void {
    const idx = new SearchIndex();
    // Frequency ranking: "data" appears 3x in d1, 1x in d2, 2x in d3
    idx.addDocument("d1", "data data data science", "gmail");
    idx.addDocument("d2", "data engineering", "slack");
    idx.addDocument("d3", "big data data pipeline", "gmail");
    idx.addDocument("d4", "machine learning model", "gdrive");

    _check("searchRanked by frequency",
      idx.searchRanked("data"),
      ["d1", "d3", "d2"]); // 3, 2, 1 occurrences

    // Tie-breaking by docId alphabetically
    idx.addDocument("a_doc", "data results", "slack");
    idx.addDocument("z_doc", "data results", "slack");
    const ranked = idx.searchRanked("data");
    // a_doc and z_doc both have 1 occurrence, same as d2
    _check("searchRanked tie-breaking",
      ranked,
      ["d1", "d3", "a_doc", "d2", "z_doc"]);

    _check("searchRanked no match",
      idx.searchRanked("quantum"), []);

    // searchBySource
    _check("searchBySource gmail 'data'",
      idx.searchBySource("data", "gmail"),
      ["d1", "d3"]);
    _check("searchBySource slack 'data'",
      idx.searchBySource("data", "slack"),
      ["a_doc", "z_doc"]);
    _check("searchBySource gdrive 'data'",
      idx.searchBySource("data", "gdrive"),
      []);

    // Case-insensitive ranked
    idx.addDocument("d5", "Data DATA data", "gmail");
    _check("ranked case-insensitive -- d5 has 3 occurrences",
      idx.searchRanked("DATA").slice(0, 2),
      ["d1", "d5"]); // both have 3
  }

  function level4(): void {
    const idx = new SearchIndex();
    idx.addDocument("d1", "dropbox desktop application sync", "gdrive");
    idx.addDocument("d2", "dropbox drive document sharing", "gmail");
    idx.addDocument("d3", "application programming interface api", "slack");

    // Autocomplete basic
    _check("autocomplete 'drop'",
      idx.autocomplete("drop"),
      ["dropbox"]);
    _check("autocomplete 'dr'",
      idx.autocomplete("dr"),
      ["drive", "dropbox"]);
    _check("autocomplete 'app'",
      idx.autocomplete("app"),
      ["application"]);

    // Autocomplete case-insensitive
    _check("autocomplete 'D' case-insensitive",
      idx.autocomplete("D"),
      ["desktop", "document", "drive", "dropbox"]);

    // Autocomplete with limit
    idx.addDocument("d4", "alpha beta gamma delta epsilon zeta");
    _check("autocomplete limit 3",
      idx.autocomplete("", 3),
      ["alpha", "api", "application"]);

    // Autocomplete no match
    _check("autocomplete no match",
      idx.autocomplete("xyz"),
      []);

    // getIndexStats
    const stats = idx.getIndexStats();
    _check("stats documents count",
      stats["documents"], 4);
    _check("stats unique_words > 0",
      stats["unique_words"] > 0, true);
    _check("stats sources sorted",
      stats["sources"],
      ["gdrive", "gmail", "slack"]);

    // Remove a doc and check stats update
    idx.removeDocument("d3");
    const stats2 = idx.getIndexStats();
    _check("stats after remove -- documents count",
      stats2["documents"], 3);
  }

  _level("Level 1 \u2014 Basic Indexing", level1);
  _level("Level 2 \u2014 Multi-Keyword Search", level2);
  _level("Level 3 \u2014 Ranked Results", level3);
  _level("Level 4 \u2014 Prefix Search / Autocomplete", level4);
}

console.log("\nDrill 06 \u2014 Search Index\n");
_runSelfChecks();
const _total = _passed + _failed;
console.log(`\n${_passed}/${_total} passed`);
if (_failed === 0 && _total > 0) {
  console.log("All tests passed.");
}
