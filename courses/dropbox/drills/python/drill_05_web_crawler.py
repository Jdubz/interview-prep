"""
Drill 05 — Web Crawler
=======================
Classic Dropbox interview problem.
Build a web crawler that explores pages from a simulated web.

A FakeWeb class is provided that simulates fetching pages — you
implement the Crawler class.

Level 1 — Basic BFS Crawl (10 min)
------------------------------------
  crawl(start_url: str) -> list[str]
      BFS from start_url. Return all reachable URLs (including
      start). Sorted alphabetically. Don't visit the same URL twice.

Level 2 — Domain Filtering (10 min)
-------------------------------------
  Constructor accepts allowed_domains: list[str] | None
      If provided, only crawl URLs whose domain matches.
      Domain is the part between "://" and the next "/".
      e.g. "https://dropbox.com/page" -> "dropbox.com"
      If allowed_domains is None, crawl all domains.

  get_crawled_domains() -> list[str]
      Return sorted list of unique domains that were actually crawled.

Level 3 — Depth Limit (10 min)
--------------------------------
  Constructor accepts max_depth: int | None
      Maximum link depth from start (start_url is depth 0).
      None means no limit.

  crawl_with_depth(start_url: str) -> dict[str, int]
      Return dict mapping URL -> depth at which it was first
      discovered. Pages beyond max_depth should NOT be fetched
      (saving fetch calls).

Level 4 — Rate Limiting (15 min)
----------------------------------
  Constructor accepts max_fetches: int | None
      Maximum number of fetch calls allowed across all crawls.
      None means unlimited. If the limit is reached during a crawl,
      stop crawling and return what you have so far.

  get_fetch_count() -> int
      Total fetches made across all crawls.

  get_unfetched() -> list[str]
      URLs that were discovered but not yet fetched (due to rate
      limit or not yet crawled). Sorted alphabetically.

Examples
--------
  web = FakeWeb({"http://a.com/": ["http://a.com/about", "http://b.com/"],
                 "http://a.com/about": ["http://a.com/"],
                 "http://b.com/": []})
  c = Crawler(web)
  c.crawl("http://a.com/")
  # -> ["http://a.com/", "http://a.com/about", "http://b.com/"]
"""


class FakeWeb:
    """Simulates a web of linked pages. Do NOT modify this class."""

    def __init__(self, pages: dict[str, list[str]]):
        """pages maps URL -> list of linked URLs."""
        self.pages = pages
        self.fetch_count = 0

    def fetch(self, url: str) -> list[str] | None:
        """Returns list of links on the page, or None if page doesn't exist."""
        self.fetch_count += 1
        return self.pages.get(url)


class Crawler:
    def __init__(
        self,
        web: FakeWeb,
        allowed_domains: list[str] | None = None,
        max_depth: int | None = None,
        max_fetches: int | None = None,
    ):
        # TODO: initialize your data structures
        pass

    # ── Level 1 ──────────────────────────────────────────────

    def crawl(self, start_url: str) -> list[str]:
        """BFS crawl. Return all reachable URLs sorted alphabetically."""
        raise NotImplementedError("Level 1: crawl")

    # ── Level 2 ──────────────────────────────────────────────

    def get_crawled_domains(self) -> list[str]:
        """Return sorted list of unique domains that were crawled."""
        raise NotImplementedError("Level 2: get_crawled_domains")

    # ── Level 3 ──────────────────────────────────────────────

    def crawl_with_depth(self, start_url: str) -> dict[str, int]:
        """Return dict mapping URL -> depth of first discovery."""
        raise NotImplementedError("Level 3: crawl_with_depth")

    # ── Level 4 ──────────────────────────────────────────────

    def get_fetch_count(self) -> int:
        """Total fetches made across all crawls."""
        raise NotImplementedError("Level 4: get_fetch_count")

    def get_unfetched(self) -> list[str]:
        """URLs discovered but not yet fetched. Sorted alphabetically."""
        raise NotImplementedError("Level 4: get_unfetched")


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
        # Simple linear chain
        web1 = FakeWeb({
            "http://a.com/": ["http://a.com/p1"],
            "http://a.com/p1": ["http://a.com/p2"],
            "http://a.com/p2": [],
        })
        c1 = Crawler(web1)
        _check("linear chain crawl",
               c1.crawl("http://a.com/"),
               ["http://a.com/", "http://a.com/p1", "http://a.com/p2"])

        # Graph with a cycle
        web2 = FakeWeb({
            "http://x.com/": ["http://x.com/a", "http://x.com/b"],
            "http://x.com/a": ["http://x.com/b", "http://x.com/"],
            "http://x.com/b": ["http://x.com/a"],
        })
        c2 = Crawler(web2)
        _check("cycle handled correctly",
               c2.crawl("http://x.com/"),
               ["http://x.com/", "http://x.com/a", "http://x.com/b"])

        # Start page links to non-existent page
        web3 = FakeWeb({
            "http://y.com/": ["http://y.com/missing"],
        })
        c3 = Crawler(web3)
        _check("non-existent page still listed as discovered",
               c3.crawl("http://y.com/"),
               ["http://y.com/", "http://y.com/missing"])

        # Disconnected — only reachable from start
        web4 = FakeWeb({
            "http://d.com/": ["http://d.com/a"],
            "http://d.com/a": [],
            "http://d.com/island": ["http://d.com/a"],
        })
        c4 = Crawler(web4)
        _check("disconnected node not reached",
               c4.crawl("http://d.com/"),
               ["http://d.com/", "http://d.com/a"])

        # Single node, no links
        web5 = FakeWeb({"http://solo.com/": []})
        c5 = Crawler(web5)
        _check("single node crawl",
               c5.crawl("http://solo.com/"),
               ["http://solo.com/"])

        # Cross-domain links
        web6 = FakeWeb({
            "http://a.com/": ["http://b.com/", "http://c.com/"],
            "http://b.com/": ["http://c.com/"],
            "http://c.com/": [],
        })
        c6 = Crawler(web6)
        _check("cross-domain BFS",
               c6.crawl("http://a.com/"),
               ["http://a.com/", "http://b.com/", "http://c.com/"])

        # Start URL doesn't exist
        web7 = FakeWeb({})
        c7 = Crawler(web7)
        _check("start URL doesn't exist",
               c7.crawl("http://nowhere.com/"),
               ["http://nowhere.com/"])

    def level_2():
        web = FakeWeb({
            "https://dropbox.com/": ["https://dropbox.com/docs", "https://google.com/"],
            "https://dropbox.com/docs": ["https://dropbox.com/", "https://slack.com/"],
            "https://google.com/": ["https://google.com/search"],
            "https://google.com/search": [],
            "https://slack.com/": [],
        })

        # Filter to dropbox.com only
        c1 = Crawler(web, allowed_domains=["dropbox.com"])
        _check("domain filter — dropbox only",
               c1.crawl("https://dropbox.com/"),
               ["https://dropbox.com/", "https://dropbox.com/docs"])
        _check("crawled domains — dropbox only",
               c1.get_crawled_domains(),
               ["dropbox.com"])

        # Filter to two domains
        c2 = Crawler(web, allowed_domains=["dropbox.com", "google.com"])
        _check("domain filter — two domains",
               c2.crawl("https://dropbox.com/"),
               ["https://dropbox.com/", "https://dropbox.com/docs",
                "https://google.com/", "https://google.com/search"])
        _check("crawled domains — two domains",
               c2.get_crawled_domains(),
               ["dropbox.com", "google.com"])

        # No domain filter
        c3 = Crawler(web, allowed_domains=None)
        _check("no domain filter — all reachable",
               c3.crawl("https://dropbox.com/"),
               ["https://dropbox.com/", "https://dropbox.com/docs",
                "https://google.com/", "https://google.com/search",
                "https://slack.com/"])
        _check("crawled domains — all three",
               c3.get_crawled_domains(),
               ["dropbox.com", "google.com", "slack.com"])

        # Allowed domain not reachable
        c4 = Crawler(web, allowed_domains=["slack.com"])
        _check("start domain not in allowed — only start listed",
               c4.crawl("https://dropbox.com/"),
               ["https://dropbox.com/"])

        # Empty allowed domains — nothing should be crawled (not even start)
        c5 = Crawler(web, allowed_domains=[])
        _check("empty allowed_domains — only start URL",
               c5.crawl("https://dropbox.com/"),
               ["https://dropbox.com/"])

    def level_3():
        # Deep chain: 0 -> 1 -> 2 -> 3 -> 4
        web1 = FakeWeb({
            "http://a.com/0": ["http://a.com/1"],
            "http://a.com/1": ["http://a.com/2"],
            "http://a.com/2": ["http://a.com/3"],
            "http://a.com/3": ["http://a.com/4"],
            "http://a.com/4": [],
        })
        c1 = Crawler(web1, max_depth=2)
        result1 = c1.crawl_with_depth("http://a.com/0")
        _check("depth limit 2 — depths correct",
               result1,
               {"http://a.com/0": 0, "http://a.com/1": 1, "http://a.com/2": 2})
        _check("depth limit 2 — fetch count",
               web1.fetch_count, 3)  # only 0, 1, 2 fetched

        # No depth limit
        web2 = FakeWeb({
            "http://b.com/0": ["http://b.com/1"],
            "http://b.com/1": ["http://b.com/2"],
            "http://b.com/2": [],
        })
        c2 = Crawler(web2, max_depth=None)
        _check("no depth limit — all discovered",
               c2.crawl_with_depth("http://b.com/0"),
               {"http://b.com/0": 0, "http://b.com/1": 1, "http://b.com/2": 2})

        # Depth 0 — only start
        web3 = FakeWeb({
            "http://c.com/": ["http://c.com/a", "http://c.com/b"],
            "http://c.com/a": [],
            "http://c.com/b": [],
        })
        c3 = Crawler(web3, max_depth=0)
        result3 = c3.crawl_with_depth("http://c.com/")
        _check("depth 0 — only start",
               result3,
               {"http://c.com/": 0})
        _check("depth 0 — only 1 fetch",
               web3.fetch_count, 1)

        # Diamond graph: A->B, A->C, B->D, C->D — D should be depth 2
        web4 = FakeWeb({
            "http://d.com/a": ["http://d.com/b", "http://d.com/c"],
            "http://d.com/b": ["http://d.com/d"],
            "http://d.com/c": ["http://d.com/d"],
            "http://d.com/d": [],
        })
        c4 = Crawler(web4, max_depth=5)
        result4 = c4.crawl_with_depth("http://d.com/a")
        _check("diamond — D discovered at depth 2",
               result4["http://d.com/d"], 2)
        _check("diamond — all nodes found",
               sorted(result4.keys()),
               ["http://d.com/a", "http://d.com/b", "http://d.com/c", "http://d.com/d"])

        # Cycle with depth limit
        web5 = FakeWeb({
            "http://e.com/0": ["http://e.com/1"],
            "http://e.com/1": ["http://e.com/0", "http://e.com/2"],
            "http://e.com/2": [],
        })
        c5 = Crawler(web5, max_depth=1)
        _check("cycle + depth limit 1",
               c5.crawl_with_depth("http://e.com/0"),
               {"http://e.com/0": 0, "http://e.com/1": 1})

    def level_4():
        web1 = FakeWeb({
            "http://a.com/": ["http://a.com/1", "http://a.com/2", "http://a.com/3"],
            "http://a.com/1": ["http://a.com/4"],
            "http://a.com/2": ["http://a.com/5"],
            "http://a.com/3": ["http://a.com/6"],
            "http://a.com/4": [],
            "http://a.com/5": [],
            "http://a.com/6": [],
        })

        # Rate limit: only 3 fetches allowed
        c1 = Crawler(web1, max_fetches=3)
        result1 = c1.crawl("http://a.com/")
        _check("rate limit — fetch count is 3",
               c1.get_fetch_count(), 3)
        _check("rate limit — start URL in results",
               "http://a.com/" in result1, True)
        _check("rate limit — result length <= 7 (partial crawl)",
               len(result1) <= 7, True)

        # Unfetched URLs should exist
        unfetched1 = c1.get_unfetched()
        _check("rate limit — has unfetched URLs",
               len(unfetched1) > 0, True)
        _check("rate limit — unfetched are sorted",
               unfetched1 == sorted(unfetched1), True)

        # Unlimited fetches
        web2 = FakeWeb({
            "http://b.com/": ["http://b.com/x"],
            "http://b.com/x": [],
        })
        c2 = Crawler(web2, max_fetches=None)
        c2.crawl("http://b.com/")
        _check("unlimited — fetch count correct",
               c2.get_fetch_count(), 2)
        _check("unlimited — no unfetched",
               c2.get_unfetched(), [])

        # Rate limit of 1 — only start page fetched
        web3 = FakeWeb({
            "http://c.com/": ["http://c.com/a", "http://c.com/b"],
            "http://c.com/a": [],
            "http://c.com/b": [],
        })
        c3 = Crawler(web3, max_fetches=1)
        result3 = c3.crawl("http://c.com/")
        _check("rate limit 1 — discovered links listed",
               sorted(result3),
               ["http://c.com/", "http://c.com/a", "http://c.com/b"])
        _check("rate limit 1 — unfetched has 2 URLs",
               c3.get_unfetched(),
               ["http://c.com/a", "http://c.com/b"])

    _level("Level 1 \u2014 Basic BFS Crawl", level_1)
    _level("Level 2 \u2014 Domain Filtering", level_2)
    _level("Level 3 \u2014 Depth Limit", level_3)
    _level("Level 4 \u2014 Rate Limiting", level_4)

def main() -> None:
    print("\nDrill 05 \u2014 Web Crawler\n")
    _run_self_checks()
    total = _passed + _failed
    print(f"\n{_passed}/{total} passed")
    if _failed == 0 and total > 0:
        print("All tests passed.")

if __name__ == "__main__":
    main()
