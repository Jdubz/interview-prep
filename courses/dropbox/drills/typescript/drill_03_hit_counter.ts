/**
 * Drill 03 — Hit Counter
 *
 * Implement a HitCounter class that counts hits within a sliding time window.
 * Classic Dropbox interview problem — used for rate limiting and analytics.
 *
 * ----------------------------------------
 * Level 1 — Basic Counter (10 min)
 *
 *   hit(timestamp: number): void
 *     Record a hit at the given timestamp (in seconds).
 *     Timestamps are monotonically non-decreasing.
 *
 *   getHits(timestamp: number): number
 *     Return the number of hits in the past 300 seconds.
 *     Window is (timestamp - 300, timestamp], i.e. inclusive of
 *     timestamp, exclusive of timestamp - 300.
 *
 * ----------------------------------------
 * Level 2 — Per-Endpoint Tracking (10 min)
 *
 *   hit(timestamp: number, endpoint?: string): void
 *     Record a hit at the given timestamp for the given endpoint.
 *     Defaults to "/".
 *
 *   getHits(timestamp: number, endpoint?: string | null): number
 *     If endpoint is provided (not null/undefined), return hits for
 *     just that endpoint within the window. If null/undefined, return
 *     total across all endpoints.
 *
 *   getEndpoints(): string[]
 *     Return all endpoints that have been hit, sorted alphabetically.
 *
 * ----------------------------------------
 * Level 3 — Configurable Window (10 min)
 *
 *   Constructor accepts window (default 300) — the time window
 *   in seconds. All methods use this configurable window.
 *
 *   getHitRate(timestamp: number, endpoint?: string | null): number
 *     Return hits per second in the current window (hits / window).
 *     Return 0.0 if no hits in the window.
 *
 * ----------------------------------------
 * Level 4 — Top K (15 min)
 *
 *   topEndpoints(timestamp: number, k: number): [string, number][]
 *     Return the top k endpoints by hit count in the current window.
 *     Sorted by count descending, then endpoint ascending for ties.
 *     Return list of [endpoint, count] tuples.
 *
 *   clearBefore(timestamp: number): number
 *     Remove all hits strictly before the given timestamp.
 *     Return the count of hits removed.
 */

class HitCounter {
  constructor(window: number = 300) {
    // TODO: initialize your data structures
  }

  hit(timestamp: number, endpoint: string = "/"): void {
    throw new Error("TODO: hit");
  }

  getHits(timestamp: number, endpoint?: string | null): number {
    throw new Error("TODO: getHits");
  }

  getEndpoints(): string[] {
    throw new Error("TODO: getEndpoints");
  }

  getHitRate(timestamp: number, endpoint?: string | null): number {
    throw new Error("TODO: getHitRate");
  }

  topEndpoints(timestamp: number, k: number): [string, number][] {
    throw new Error("TODO: topEndpoints");
  }

  clearBefore(timestamp: number): number {
    throw new Error("TODO: clearBefore");
  }
}

// ─── Self-Checks (do not edit below this line) ──────────────────

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
    const c = new HitCounter();

    // no hits yet
    _check("no hits", c.getHits(1), 0);

    // single hit
    c.hit(1);
    _check("one hit", c.getHits(1), 1);

    // multiple hits at same timestamp
    c.hit(1);
    c.hit(1);
    _check("three hits at t=1", c.getHits(1), 3);

    // hits within window
    c.hit(100);
    c.hit(200);
    _check("all hits in window", c.getHits(300), 5);

    // hit exactly at window boundary — t=1 is NOT in (0, 300]
    // wait, (300-300, 300] = (0, 300] so t=1 IS included
    _check("boundary inclusive", c.getHits(300), 5);

    // hits falling outside the window
    _check("expired hits", c.getHits(301), 2); // t=1 hits expire, t=100 and t=200 remain

    // all expired
    _check("all expired", c.getHits(600), 0);

    // new hit after expiry
    c.hit(600);
    _check("fresh hit after gap", c.getHits(600), 1);
  }

  _level("Level 1 \u2014 Basic Counter", level1);

  function level2(): void {
    const c = new HitCounter();

    // default endpoint
    c.hit(1);
    _check("default endpoint total", c.getHits(1), 1);
    _check("default endpoint specific", c.getHits(1, "/"), 1);

    // multiple endpoints
    c.hit(2, "/api");
    c.hit(3, "/api");
    c.hit(4, "/home");
    _check("total all endpoints", c.getHits(10), 4);
    _check("/api count", c.getHits(10, "/api"), 2);
    _check("/home count", c.getHits(10, "/home"), 1);
    _check("/ count", c.getHits(10, "/"), 1);

    // unknown endpoint
    _check("unknown endpoint", c.getHits(10, "/missing"), 0);

    // getEndpoints sorted
    _check("sorted endpoints", c.getEndpoints(), ["/", "/api", "/home"]);
  }

  _level("Level 2 \u2014 Per-Endpoint Tracking", level2);

  function level3(): void {
    // custom window
    const c = new HitCounter(10);
    c.hit(1);
    c.hit(5);
    c.hit(10);
    _check("all in 10s window", c.getHits(10), 3);
    _check("one expired", c.getHits(11), 2); // t=1 is outside (1, 11]

    // hit rate
    _check("hit rate", c.getHitRate(10), 3 / 10);

    // hit rate with no hits
    const c2 = new HitCounter(60);
    _check("hit rate no hits", c2.getHitRate(100), 0.0);

    // hit rate per endpoint
    const c3 = new HitCounter(100);
    c3.hit(10, "/a");
    c3.hit(20, "/a");
    c3.hit(30, "/b");
    _check("rate for /a", c3.getHitRate(50, "/a"), 2 / 100);
    _check("rate for /b", c3.getHitRate(50, "/b"), 1 / 100);
    _check("rate total", c3.getHitRate(50), 3 / 100);

    // default window still works
    const c4 = new HitCounter();
    c4.hit(1);
    _check("default 300s window", c4.getHits(300), 1);
  }

  _level("Level 3 \u2014 Configurable Window", level3);

  function level4(): void {
    const c = new HitCounter(100);

    c.hit(10, "/a");
    c.hit(20, "/a");
    c.hit(30, "/a");
    c.hit(40, "/b");
    c.hit(50, "/b");
    c.hit(60, "/c");

    // top endpoints
    _check("top 2", c.topEndpoints(60, 2), [["/a", 3], ["/b", 2]]);
    _check("top all", c.topEndpoints(60, 5), [["/a", 3], ["/b", 2], ["/c", 1]]);
    _check("top 0", c.topEndpoints(60, 0), []);

    // ties — sorted by endpoint asc
    const c2 = new HitCounter(100);
    c2.hit(10, "/x");
    c2.hit(20, "/y");
    c2.hit(30, "/z");
    _check("tie breaking", c2.topEndpoints(30, 3), [["/x", 1], ["/y", 1], ["/z", 1]]);

    // clearBefore
    const c3 = new HitCounter(300);
    c3.hit(10);
    c3.hit(20);
    c3.hit(30);
    c3.hit(100);
    const removed = c3.clearBefore(25);
    _check("clear_before count", removed, 2);
    _check("hits after clear", c3.getHits(100), 2);

    // clearBefore with nothing to clear
    _check("clear nothing", c3.clearBefore(5), 0);

    // topEndpoints respects window
    const c4 = new HitCounter(10);
    c4.hit(1, "/old");
    c4.hit(50, "/new");
    _check("top respects window", c4.topEndpoints(50, 5), [["/new", 1]]);
  }

  _level("Level 4 \u2014 Top K", level4);
}

function main(): void {
  console.log("\nHit Counter\n");
  _runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) {
    console.log("All tests passed.");
  }
}

main();
