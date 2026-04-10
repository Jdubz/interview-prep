/**
 * Drill 02 — File Chunker
 * ========================
 * File transfer techniques — classic Dropbox domain.
 * Build a chunked upload system with deduplication.
 *
 * Level 1 — Basic Chunking (10 min)
 * -----------------------------------
 *   chunk(data: string, chunkSize: number): string[]
 *       Split data into chunks of chunkSize characters.
 *       Last chunk may be smaller.
 *
 *   reassemble(chunks: string[]): string
 *       Join chunks back into the original data string.
 *
 *   getChunkCount(data: string, chunkSize: number): number
 *       Return the number of chunks needed to store data.
 *
 * Level 2 — Upload Session (10 min)
 * -----------------------------------
 *   initUpload(filename: string, totalChunks: number): string
 *       Start an upload session. Return a unique sessionId.
 *
 *   uploadChunk(sessionId: string, chunkIndex: number, data: string): boolean
 *       Upload a chunk at the given index. Return false if
 *       sessionId is invalid or chunkIndex is out of range
 *       [0, totalChunks). Idempotent: re-uploading the same
 *       chunk index is allowed (overwrites).
 *
 *   getProgress(sessionId: string): number
 *       Return upload progress from 0.0 to 1.0.
 *       Return -1.0 if sessionId is invalid.
 *
 * Level 3 — Completion & Integrity (10 min)
 * -------------------------------------------
 *   completeUpload(sessionId: string): string | null
 *       If all chunks are uploaded, reassemble and return the
 *       full data string. Return null if any chunks are missing
 *       or sessionId is invalid. Cleans up the session after
 *       successful completion.
 *
 *   getMissingChunks(sessionId: string): number[]
 *       Return sorted list of chunk indices not yet uploaded.
 *       Return [] if sessionId is invalid.
 *
 *   Chunks can be uploaded in any order (not just sequential).
 *
 * Level 4 — Deduplication (15 min)
 * ---------------------------------
 *   computeHash(data: string): string
 *       Return a hash of the data string. Use a simple string
 *       hash that produces a hex string.
 *
 *   When uploading a chunk, if a chunk with the same hash
 *   already exists in the global content store, do not store
 *   a duplicate copy. The chunk still counts as uploaded for
 *   that session (it references the existing stored data).
 *
 *   getStorageSaved(): number
 *       Return the total number of characters saved by dedup
 *       (sum of lengths of all skipped duplicate chunks).
 *
 * Examples
 * --------
 *   const fc = new FileChunker();
 *   fc.chunk("HelloWorld!", 4);           // ["Hell", "oWor", "ld!"]
 *   fc.reassemble(["Hell", "oWor", "ld!"]); // "HelloWorld!"
 *   const sid = fc.initUpload("test.txt", 3);
 *   fc.uploadChunk(sid, 0, "Hell");       // true
 *   fc.uploadChunk(sid, 2, "ld!");        // true (out of order)
 *   fc.getProgress(sid);                  // ~0.6667
 *   fc.getMissingChunks(sid);             // [1]
 *   fc.uploadChunk(sid, 1, "oWor");       // true
 *   fc.completeUpload(sid);               // "HelloWorld!"
 */

class FileChunker {
  constructor() {
    // TODO: initialize your data structures
  }

  // -- Level 1 --------------------------------------------------

  chunk(data: string, chunkSize: number): string[] {
    /** Split data into chunks of chunkSize. Last chunk may be smaller. */
    throw new Error("TODO: chunk");
  }

  reassemble(chunks: string[]): string {
    /** Join chunks back into original data. */
    throw new Error("TODO: reassemble");
  }

  getChunkCount(data: string, chunkSize: number): number {
    /** Number of chunks needed. */
    throw new Error("TODO: getChunkCount");
  }

  // -- Level 2 --------------------------------------------------

  initUpload(filename: string, totalChunks: number): string {
    /** Start upload session, return sessionId. */
    throw new Error("TODO: initUpload");
  }

  uploadChunk(sessionId: string, chunkIndex: number, data: string): boolean {
    /** Upload a chunk. False if invalid session or index out of range. */
    throw new Error("TODO: uploadChunk");
  }

  getProgress(sessionId: string): number {
    /** Progress 0.0-1.0, or -1.0 if invalid session. */
    throw new Error("TODO: getProgress");
  }

  // -- Level 3 --------------------------------------------------

  completeUpload(sessionId: string): string | null {
    /** Reassemble if complete, else null. Cleans up session. */
    throw new Error("TODO: completeUpload");
  }

  getMissingChunks(sessionId: string): number[] {
    /** Sorted list of missing chunk indices. [] if invalid session. */
    throw new Error("TODO: getMissingChunks");
  }

  // -- Level 4 --------------------------------------------------

  computeHash(data: string): string {
    /** Simple string hash, returned as hex. */
    throw new Error("TODO: computeHash");
  }

  getStorageSaved(): number {
    /** Total characters saved by deduplication. */
    throw new Error("TODO: getStorageSaved");
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
    const fc = new FileChunker();
    // basic chunking
    _check("chunk even split", fc.chunk("abcdef", 3), ["abc", "def"]);
    _check("chunk uneven split", fc.chunk("HelloWorld!", 4), [
      "Hell",
      "oWor",
      "ld!",
    ]);
    _check("chunk size 1", fc.chunk("abc", 1), ["a", "b", "c"]);
    _check("chunk empty string", fc.chunk("", 5), []);
    _check("chunk size larger than data", fc.chunk("hi", 10), ["hi"]);
    // reassemble
    _check(
      "reassemble chunks",
      fc.reassemble(["Hell", "oWor", "ld!"]),
      "HelloWorld!",
    );
    _check("reassemble empty list", fc.reassemble([]), "");
    // chunk count
    _check("chunk count even", fc.getChunkCount("abcdef", 3), 2);
    _check("chunk count uneven", fc.getChunkCount("abcdefg", 3), 3);
  }

  function level2(): void {
    const fc = new FileChunker();
    const sid = fc.initUpload("test.txt", 3);
    _check("initUpload returns string", typeof sid === "string", true);
    // upload valid chunk
    _check("upload chunk 0", fc.uploadChunk(sid, 0, "aaa"), true);
    _check("progress after 1/3", fc.getProgress(sid), 1.0 / 3.0);
    // upload out of range
    _check("upload chunk -1 invalid", fc.uploadChunk(sid, -1, "x"), false);
    _check("upload chunk 3 invalid", fc.uploadChunk(sid, 3, "x"), false);
    // invalid session
    _check("upload bad session", fc.uploadChunk("fake", 0, "x"), false);
    _check("progress bad session", fc.getProgress("fake"), -1.0);
    // idempotent re-upload
    _check("re-upload chunk 0", fc.uploadChunk(sid, 0, "bbb"), true);
    _check(
      "progress still 1/3 after re-upload",
      fc.getProgress(sid),
      1.0 / 3.0,
    );
  }

  function level3(): void {
    const fc = new FileChunker();
    const sid = fc.initUpload("doc.txt", 3);
    // missing chunks before any upload
    _check("all chunks missing", fc.getMissingChunks(sid), [0, 1, 2]);
    // upload out of order
    fc.uploadChunk(sid, 2, "ld!");
    fc.uploadChunk(sid, 0, "Hell");
    _check("missing chunk 1", fc.getMissingChunks(sid), [1]);
    // complete with missing chunk
    _check(
      "complete with missing returns null",
      fc.completeUpload(sid),
      null,
    );
    // upload final chunk
    fc.uploadChunk(sid, 1, "oWor");
    _check("no missing chunks", fc.getMissingChunks(sid), []);
    _check("complete returns data", fc.completeUpload(sid), "HelloWorld!");
    // session cleaned up after complete
    _check("missing chunks after cleanup", fc.getMissingChunks(sid), []);
    _check("complete again returns null", fc.completeUpload(sid), null);
    // invalid session
    _check("missing chunks invalid session", fc.getMissingChunks("fake"), []);
  }

  function level4(): void {
    const fc = new FileChunker();
    // computeHash deterministic
    const h1 = fc.computeHash("hello");
    const h2 = fc.computeHash("hello");
    const h3 = fc.computeHash("world");
    _check("hash deterministic", h1, h2);
    _check("hash differs for different data", h1 !== h3, true);
    _check(
      "hash is hex string",
      h1.split("").every((c) => "0123456789abcdef".includes(c)),
      true,
    );
    // dedup within same session
    const sid1 = fc.initUpload("a.txt", 3);
    fc.uploadChunk(sid1, 0, "AAA");
    fc.uploadChunk(sid1, 1, "AAA"); // duplicate content
    fc.uploadChunk(sid1, 2, "BBB");
    _check("dedup saved within session", fc.getStorageSaved(), 3);
    // dedup across sessions
    const sid2 = fc.initUpload("b.txt", 2);
    fc.uploadChunk(sid2, 0, "AAA"); // already stored globally
    fc.uploadChunk(sid2, 1, "CCC"); // new content
    _check("dedup saved across sessions", fc.getStorageSaved(), 6);
    // chunks still count as uploaded despite dedup
    _check("progress with deduped chunks", fc.getProgress(sid2), 1.0);
    _check(
      "complete with deduped chunks",
      fc.completeUpload(sid2),
      "AAACCC",
    );
  }

  _level("Level 1 \u2014 Basic Chunking", level1);
  _level("Level 2 \u2014 Upload Session", level2);
  _level("Level 3 \u2014 Completion & Integrity", level3);
  _level("Level 4 \u2014 Deduplication", level4);
}

console.log("\nDrill 02 \u2014 File Chunker\n");
_runSelfChecks();
const _total = _passed + _failed;
console.log(`\n${_passed}/${_total} passed`);
if (_failed === 0 && _total > 0) {
  console.log("All tests passed.");
}
