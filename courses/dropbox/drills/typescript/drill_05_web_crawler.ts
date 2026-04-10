/**
 * Drill 05 — Web Crawler
 * =======================
 * Classic Dropbox interview problem.
 * Build a web crawler that explores pages from a simulated web.
 *
 * A FakeWeb class is provided that simulates fetching pages — you
 * implement the Crawler class.
 *
 * Level 1 — Basic BFS Crawl (10 min)
 * ------------------------------------
 *   crawl(startUrl: string): string[]
 *       BFS from startUrl. Return all reachable URLs (including
 *       start). Sorted alphabetically. Don't visit the same URL twice.
 *
 * Level 2 — Domain Filtering (10 min)
 * -------------------------------------
 *   Constructor accepts allowedDomains: string[] | null
 *       If provided, only crawl URLs whose domain matches.
 *       Domain is the part between "://" and the next "/".
 *       e.g. "https://dropbox.com/page" -> "dropbox.com"
 *       If allowedDomains is null, crawl all domains.
 *
 *   getCrawledDomains(): string[]
 *       Return sorted list of unique domains that were actually crawled.
 *
 * Level 3 — Depth Limit (10 min)
 * --------------------------------
 *   Constructor accepts maxDepth: number | null
 *       Maximum link depth from start (startUrl is depth 0).
 *       null means no limit.
 *
 *   crawlWithDepth(startUrl: string): Record<string, number>
 *       Return object mapping URL -> depth at which it was first
 *       discovered. Pages beyond maxDepth should NOT be fetched
 *       (saving fetch calls).
 *
 * Level 4 — Rate Limiting (15 min)
 * ----------------------------------
 *   Constructor accepts maxFetches: number | null
 *       Maximum number of fetch calls allowed across all crawls.
 *       null means unlimited. If the limit is reached during a crawl,
 *       stop crawling and return what you have so far.
 *
 *   getFetchCount(): number
 *       Total fetches made across all crawls.
 *
 *   getUnfetched(): string[]
 *       URLs that were discovered but not yet fetched (due to rate
 *       limit or not yet crawled). Sorted alphabetically.
 *
 * Examples
 * --------
 *   const web = new FakeWeb({
 *     "http://a.com/": ["http://a.com/about", "http://b.com/"],
 *     "http://a.com/about": ["http://a.com/"],
 *     "http://b.com/": [],
 *   });
 *   const c = new Crawler(web);
 *   c.crawl("http://a.com/");
 *   // -> ["http://a.com/", "http://a.com/about", "http://b.com/"]
 */


/** Simulates a web of linked pages. Do NOT modify this class. */
class FakeWeb {
  private pages: Record<string, string[]>;
  fetchCount: number;

  constructor(pages: Record<string, string[]>) {
    this.pages = pages;
    this.fetchCount = 0;
  }

  /** Returns list of links on the page, or null if page doesn't exist. */
  fetch(url: string): string[] | null {
    this.fetchCount++;
    return this.pages[url] ?? null;
  }
}


class Crawler {
  constructor(
    web: FakeWeb,
    options?: {
      allowedDomains?: string[] | null;
      maxDepth?: number | null;
      maxFetches?: number | null;
    },
  ) {
    // TODO: initialize your data structures
  }

  // -- Level 1 --------------------------------------------------

  /** BFS crawl. Return all reachable URLs sorted alphabetically. */
  crawl(startUrl: string): string[] {
    throw new Error("TODO: crawl");
  }

  // -- Level 2 --------------------------------------------------

  /** Return sorted list of unique domains that were crawled. */
  getCrawledDomains(): string[] {
    throw new Error("TODO: getCrawledDomains");
  }

  // -- Level 3 --------------------------------------------------

  /** Return object mapping URL -> depth of first discovery. */
  crawlWithDepth(startUrl: string): Record<string, number> {
    throw new Error("TODO: crawlWithDepth");
  }

  // -- Level 4 --------------------------------------------------

  /** Total fetches made across all crawls. */
  getFetchCount(): number {
    throw new Error("TODO: getFetchCount");
  }

  /** URLs discovered but not yet fetched. Sorted alphabetically. */
  getUnfetched(): string[] {
    throw new Error("TODO: getUnfetched");
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
    // Simple linear chain
    const web1 = new FakeWeb({
      "http://a.com/": ["http://a.com/p1"],
      "http://a.com/p1": ["http://a.com/p2"],
      "http://a.com/p2": [],
    });
    const c1 = new Crawler(web1);
    _check("linear chain crawl",
      c1.crawl("http://a.com/"),
      ["http://a.com/", "http://a.com/p1", "http://a.com/p2"]);

    // Graph with a cycle
    const web2 = new FakeWeb({
      "http://x.com/": ["http://x.com/a", "http://x.com/b"],
      "http://x.com/a": ["http://x.com/b", "http://x.com/"],
      "http://x.com/b": ["http://x.com/a"],
    });
    const c2 = new Crawler(web2);
    _check("cycle handled correctly",
      c2.crawl("http://x.com/"),
      ["http://x.com/", "http://x.com/a", "http://x.com/b"]);

    // Start page links to non-existent page
    const web3 = new FakeWeb({
      "http://y.com/": ["http://y.com/missing"],
    });
    const c3 = new Crawler(web3);
    _check("non-existent page still listed as discovered",
      c3.crawl("http://y.com/"),
      ["http://y.com/", "http://y.com/missing"]);

    // Disconnected — only reachable from start
    const web4 = new FakeWeb({
      "http://d.com/": ["http://d.com/a"],
      "http://d.com/a": [],
      "http://d.com/island": ["http://d.com/a"],
    });
    const c4 = new Crawler(web4);
    _check("disconnected node not reached",
      c4.crawl("http://d.com/"),
      ["http://d.com/", "http://d.com/a"]);

    // Single node, no links
    const web5 = new FakeWeb({ "http://solo.com/": [] });
    const c5 = new Crawler(web5);
    _check("single node crawl",
      c5.crawl("http://solo.com/"),
      ["http://solo.com/"]);

    // Cross-domain links
    const web6 = new FakeWeb({
      "http://a.com/": ["http://b.com/", "http://c.com/"],
      "http://b.com/": ["http://c.com/"],
      "http://c.com/": [],
    });
    const c6 = new Crawler(web6);
    _check("cross-domain BFS",
      c6.crawl("http://a.com/"),
      ["http://a.com/", "http://b.com/", "http://c.com/"]);

    // Start URL doesn't exist
    const web7 = new FakeWeb({});
    const c7 = new Crawler(web7);
    _check("start URL doesn't exist",
      c7.crawl("http://nowhere.com/"),
      ["http://nowhere.com/"]);
  }

  function level2(): void {
    const web = new FakeWeb({
      "https://dropbox.com/": ["https://dropbox.com/docs", "https://google.com/"],
      "https://dropbox.com/docs": ["https://dropbox.com/", "https://slack.com/"],
      "https://google.com/": ["https://google.com/search"],
      "https://google.com/search": [],
      "https://slack.com/": [],
    });

    // Filter to dropbox.com only
    const c1 = new Crawler(web, { allowedDomains: ["dropbox.com"] });
    _check("domain filter -- dropbox only",
      c1.crawl("https://dropbox.com/"),
      ["https://dropbox.com/", "https://dropbox.com/docs"]);
    _check("crawled domains -- dropbox only",
      c1.getCrawledDomains(),
      ["dropbox.com"]);

    // Filter to two domains
    const c2 = new Crawler(web, { allowedDomains: ["dropbox.com", "google.com"] });
    _check("domain filter -- two domains",
      c2.crawl("https://dropbox.com/"),
      ["https://dropbox.com/", "https://dropbox.com/docs",
       "https://google.com/", "https://google.com/search"]);
    _check("crawled domains -- two domains",
      c2.getCrawledDomains(),
      ["dropbox.com", "google.com"]);

    // No domain filter
    const c3 = new Crawler(web, { allowedDomains: null });
    _check("no domain filter -- all reachable",
      c3.crawl("https://dropbox.com/"),
      ["https://dropbox.com/", "https://dropbox.com/docs",
       "https://google.com/", "https://google.com/search",
       "https://slack.com/"]);
    _check("crawled domains -- all three",
      c3.getCrawledDomains(),
      ["dropbox.com", "google.com", "slack.com"]);

    // Allowed domain not reachable
    const c4 = new Crawler(web, { allowedDomains: ["slack.com"] });
    _check("start domain not in allowed -- only start listed",
      c4.crawl("https://dropbox.com/"),
      ["https://dropbox.com/"]);

    // Empty allowed domains — nothing should be crawled (not even start)
    const c5 = new Crawler(web, { allowedDomains: [] });
    _check("empty allowed_domains -- only start URL",
      c5.crawl("https://dropbox.com/"),
      ["https://dropbox.com/"]);
  }

  function level3(): void {
    // Deep chain: 0 -> 1 -> 2 -> 3 -> 4
    const web1 = new FakeWeb({
      "http://a.com/0": ["http://a.com/1"],
      "http://a.com/1": ["http://a.com/2"],
      "http://a.com/2": ["http://a.com/3"],
      "http://a.com/3": ["http://a.com/4"],
      "http://a.com/4": [],
    });
    const c1 = new Crawler(web1, { maxDepth: 2 });
    const result1 = c1.crawlWithDepth("http://a.com/0");
    _check("depth limit 2 -- depths correct",
      result1,
      { "http://a.com/0": 0, "http://a.com/1": 1, "http://a.com/2": 2 });
    _check("depth limit 2 -- fetch count",
      web1.fetchCount, 3); // only 0, 1, 2 fetched

    // No depth limit
    const web2 = new FakeWeb({
      "http://b.com/0": ["http://b.com/1"],
      "http://b.com/1": ["http://b.com/2"],
      "http://b.com/2": [],
    });
    const c2 = new Crawler(web2, { maxDepth: null });
    _check("no depth limit -- all discovered",
      c2.crawlWithDepth("http://b.com/0"),
      { "http://b.com/0": 0, "http://b.com/1": 1, "http://b.com/2": 2 });

    // Depth 0 — only start
    const web3 = new FakeWeb({
      "http://c.com/": ["http://c.com/a", "http://c.com/b"],
      "http://c.com/a": [],
      "http://c.com/b": [],
    });
    const c3 = new Crawler(web3, { maxDepth: 0 });
    const result3 = c3.crawlWithDepth("http://c.com/");
    _check("depth 0 -- only start",
      result3,
      { "http://c.com/": 0 });
    _check("depth 0 -- only 1 fetch",
      web3.fetchCount, 1);

    // Diamond graph: A->B, A->C, B->D, C->D — D should be depth 2
    const web4 = new FakeWeb({
      "http://d.com/a": ["http://d.com/b", "http://d.com/c"],
      "http://d.com/b": ["http://d.com/d"],
      "http://d.com/c": ["http://d.com/d"],
      "http://d.com/d": [],
    });
    const c4 = new Crawler(web4, { maxDepth: 5 });
    const result4 = c4.crawlWithDepth("http://d.com/a");
    _check("diamond -- D discovered at depth 2",
      result4["http://d.com/d"], 2);
    _check("diamond -- all nodes found",
      Object.keys(result4).sort(),
      ["http://d.com/a", "http://d.com/b", "http://d.com/c", "http://d.com/d"]);

    // Cycle with depth limit
    const web5 = new FakeWeb({
      "http://e.com/0": ["http://e.com/1"],
      "http://e.com/1": ["http://e.com/0", "http://e.com/2"],
      "http://e.com/2": [],
    });
    const c5 = new Crawler(web5, { maxDepth: 1 });
    _check("cycle + depth limit 1",
      c5.crawlWithDepth("http://e.com/0"),
      { "http://e.com/0": 0, "http://e.com/1": 1 });
  }

  function level4(): void {
    const web1 = new FakeWeb({
      "http://a.com/": ["http://a.com/1", "http://a.com/2", "http://a.com/3"],
      "http://a.com/1": ["http://a.com/4"],
      "http://a.com/2": ["http://a.com/5"],
      "http://a.com/3": ["http://a.com/6"],
      "http://a.com/4": [],
      "http://a.com/5": [],
      "http://a.com/6": [],
    });

    // Rate limit: only 3 fetches allowed
    const c1 = new Crawler(web1, { maxFetches: 3 });
    const result1 = c1.crawl("http://a.com/");
    _check("rate limit -- fetch count is 3",
      c1.getFetchCount(), 3);
    _check("rate limit -- start URL in results",
      result1.includes("http://a.com/"), true);
    _check("rate limit -- result length <= 7 (partial crawl)",
      result1.length <= 7, true);

    // Unfetched URLs should exist
    const unfetched1 = c1.getUnfetched();
    _check("rate limit -- has unfetched URLs",
      unfetched1.length > 0, true);
    _check("rate limit -- unfetched are sorted",
      JSON.stringify(unfetched1) === JSON.stringify([...unfetched1].sort()), true);

    // Unlimited fetches
    const web2 = new FakeWeb({
      "http://b.com/": ["http://b.com/x"],
      "http://b.com/x": [],
    });
    const c2 = new Crawler(web2, { maxFetches: null });
    c2.crawl("http://b.com/");
    _check("unlimited -- fetch count correct",
      c2.getFetchCount(), 2);
    _check("unlimited -- no unfetched",
      c2.getUnfetched(), []);

    // Rate limit of 1 — only start page fetched
    const web3 = new FakeWeb({
      "http://c.com/": ["http://c.com/a", "http://c.com/b"],
      "http://c.com/a": [],
      "http://c.com/b": [],
    });
    const c3 = new Crawler(web3, { maxFetches: 1 });
    const result3 = c3.crawl("http://c.com/");
    _check("rate limit 1 -- discovered links listed",
      [...result3].sort(),
      ["http://c.com/", "http://c.com/a", "http://c.com/b"]);
    _check("rate limit 1 -- unfetched has 2 URLs",
      c3.getUnfetched(),
      ["http://c.com/a", "http://c.com/b"]);
  }

  _level("Level 1 \u2014 Basic BFS Crawl", level1);
  _level("Level 2 \u2014 Domain Filtering", level2);
  _level("Level 3 \u2014 Depth Limit", level3);
  _level("Level 4 \u2014 Rate Limiting", level4);
}

console.log("\nDrill 05 \u2014 Web Crawler\n");
_runSelfChecks();
const _total = _passed + _failed;
console.log(`\n${_passed}/${_total} passed`);
if (_failed === 0 && _total > 0) {
  console.log("All tests passed.");
}
