/**
 * Data Structures Exercises
 *
 * Classic interview data structure problems implemented in TypeScript.
 * Trie, deep clone, caches, and merge algorithms — the problems that
 * show up on whiteboard rounds and take-home assignments.
 *
 * Run:  npx tsx exercises/04-data-structures.ts
 */


// ============================================================================
// EXERCISE 1: Group By
// ============================================================================
//
// RELATED READING:
//   - ../10-interview-prep/01-interview-fundamentals.md (coding patterns)
//   - ../00-ts-node-fundamentals.md (generics, Map)
//
// Implement a generic groupBy function that groups items by a key derived
// from a key function. Returns a Map<K, T[]>.
//
// Requirements:
//   - groupBy(items, keyFn) returns a Map grouping items by keyFn result
//   - Each key maps to an array of items that produced that key
//   - Preserves insertion order within each group
//   - Works with any key type that can be a Map key
//   - Empty input returns an empty Map
//
// Hints:
//   - Use a Map<K, T[]> to accumulate groups
//   - For each item, compute the key, get-or-create the array, push the item
//
//   Pattern:
//     function groupBy<T, K>(items: T[], keyFn: (item: T) => K): Map<K, T[]> {
//       const groups = new Map<K, T[]>();
//       for (const item of items) {
//         const key = keyFn(item);
//         if (!groups.has(key)) groups.set(key, []);
//         groups.get(key)!.push(item);
//       }
//       return groups;
//     }
//
//   Key concepts:
//   - Map preserves insertion order (unlike plain objects with numeric keys)
//   - Any value can be a Map key (objects, functions, etc.)
//   - The ! assertion is safe because we just set the key
//
// Expected behavior:
//   groupBy(["hello", "hi", "hey", "world", "wow"], w => w[0])
//   // Map { "h" => ["hello", "hi", "hey"], "w" => ["world", "wow"] }

function groupBy<T, K>(items: T[], keyFn: (item: T) => K): Map<K, T[]> {
  const groups = new Map<K, T[]>();
  for (const item of items) {
    const key = keyFn(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(item);
  }
  return groups;
}


// ============================================================================
// EXERCISE 2: Deep Clone
// ============================================================================
//
// RELATED READING:
//   - ../10-interview-prep/01-interview-fundamentals.md (object manipulation)
//   - ../00-ts-node-fundamentals.md (TypeScript types)
//
// Implement a deep clone function that creates a complete copy of an object,
// including nested objects, arrays, and special types.
//
// Requirements:
//   - Handles plain objects and arrays (recursive)
//   - Handles Date (clone via new Date(original.getTime()))
//   - Handles RegExp (clone via new RegExp(original.source, original.flags))
//   - Handles Map (clone entries recursively)
//   - Handles Set (clone values recursively)
//   - Primitives and null are returned as-is
//   - No need to handle circular references
//
// Hints:
//   - Use instanceof checks to identify special types
//   - For plain objects, iterate Object.entries and recurse on values
//   - For arrays, map over elements and recurse
//   - Order of checks matters: Array before Object (arrays are objects)
//
//   Pattern:
//     function deepClone<T>(obj: T): T {
//       if (obj === null || typeof obj !== "object") return obj;
//       if (obj instanceof Date) return new Date(obj.getTime()) as T;
//       if (obj instanceof RegExp) return new RegExp(obj.source, obj.flags) as T;
//       if (obj instanceof Map) { /* clone entries */ }
//       if (obj instanceof Set) { /* clone values */ }
//       if (Array.isArray(obj)) return obj.map(deepClone) as T;
//       // Plain object
//       const result = {} as Record<string, unknown>;
//       for (const [key, value] of Object.entries(obj)) {
//         result[key] = deepClone(value);
//       }
//       return result as T;
//     }
//
//   Key concepts:
//   - typeof null === "object" (JavaScript quirk, check null first)
//   - instanceof narrows the type for special objects
//   - Object.entries gives [key, value] pairs for plain objects
//
// Expected behavior:
//   const original = { a: [1, { b: 2 }], c: new Date(), d: new Map([["x", 1]]) };
//   const cloned = deepClone(original);
//   cloned.a[1].b = 99;
//   original.a[1].b; // still 2 (not affected by mutation)

function deepClone<T>(_obj: T): T {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 3: Trie (Prefix Tree)
// ============================================================================
//
// RELATED READING:
//   - ../10-interview-prep/01-interview-fundamentals.md (tree structures)
//   - ../01-typescript-advanced/02-advanced-type-patterns.md
//
// Implement a Trie data structure for efficient prefix-based string operations.
// Commonly used in autocomplete, spell checkers, and IP routing tables.
//
// Requirements:
//   - insert(word) adds a word to the trie
//   - search(word) returns true if the exact word exists
//   - startsWith(prefix) returns true if any word starts with prefix
//   - autoComplete(prefix, limit) returns up to `limit` words with that prefix
//   - Case-sensitive (no normalization needed)
//
// Hints:
//   - Each node has a Map<string, TrieNode> of children (one per character)
//   - A boolean flag marks the end of a word
//   - insert: walk the tree, creating nodes for new characters
//   - search: walk the tree, check if final node is end-of-word
//   - autoComplete: find the prefix node, then DFS to collect all words
//
//   Pattern:
//     interface TrieNode {
//       children: Map<string, TrieNode>;
//       isEnd: boolean;
//     }
//
//     function createNode(): TrieNode {
//       return { children: new Map(), isEnd: false };
//     }
//
//     // insert: for each char, get-or-create child node, mark last as isEnd
//     // search: walk chars, return false if missing child, check isEnd
//     // autoComplete: find prefix node, DFS with character accumulation
//
//   Key concepts:
//   - Map<string, TrieNode> gives O(1) child lookup per character
//   - DFS with path accumulation collects all words under a prefix
//   - Space-efficient for shared prefixes (e.g., "cat", "car", "card")
//
// Expected behavior:
//   const trie = new Trie();
//   trie.insert("cat"); trie.insert("car"); trie.insert("card"); trie.insert("dog");
//   trie.search("car");      // true
//   trie.search("ca");       // false
//   trie.startsWith("ca");   // true
//   trie.autoComplete("ca", 10); // ["cat", "car", "card"]

class Trie {
  insert(_word: string): void {
    throw new Error("Not implemented");
  }

  search(_word: string): boolean {
    throw new Error("Not implemented");
  }

  startsWith(_prefix: string): boolean {
    throw new Error("Not implemented");
  }

  autoComplete(_prefix: string, _limit: number): string[] {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 4: Deep Flatten
// ============================================================================
//
// RELATED READING:
//   - ../10-interview-prep/01-interview-fundamentals.md (recursion)
//   - ../01-typescript-advanced/01-conditional-and-mapped-types.md (recursive types)
//
// Flatten an arbitrarily nested array into a single-level array. Implement
// both a recursive and an iterative approach.
//
// Requirements:
//   - deepFlatten([1, [2, [3, [4]]]]) → [1, 2, 3, 4]
//   - Handles arbitrary nesting depth
//   - Preserves element order
//   - Works with any element type
//   - Implement two versions: recursive and iterative (stack-based)
//
// Hints:
//   - Recursive: if element is array, recurse; else push to result
//   - Iterative: use a stack (array), push elements in reverse order
//   - Array.isArray() checks if a value is an array
//
//   Type definition:
//     type NestedArray<T> = (T | NestedArray<T>)[];
//
//   Recursive pattern:
//     function deepFlatten<T>(arr: NestedArray<T>): T[] {
//       const result: T[] = [];
//       for (const item of arr) {
//         if (Array.isArray(item)) result.push(...deepFlatten(item));
//         else result.push(item);
//       }
//       return result;
//     }
//
//   Iterative pattern (stack-based):
//     function deepFlattenIterative<T>(arr: NestedArray<T>): T[] {
//       const stack = [...arr];
//       const result: T[] = [];
//       while (stack.length) {
//         const item = stack.pop()!;
//         if (Array.isArray(item)) stack.push(...item);
//         else result.push(item);
//       }
//       return result.reverse(); // pop gives reverse order
//     }
//
// Expected behavior:
//   deepFlatten([1, [2, [3]], [4, [5, [6]]]]) // [1, 2, 3, 4, 5, 6]
//   deepFlattenIterative([1, [2, [3]], [4, [5, [6]]]]) // [1, 2, 3, 4, 5, 6]

type NestedArray<T> = (T | NestedArray<T>)[];

function deepFlatten<T>(_arr: NestedArray<T>): T[] {
  throw new Error("Not implemented");
}

function deepFlattenIterative<T>(_arr: NestedArray<T>): T[] {
  throw new Error("Not implemented");
}


// ============================================================================
// EXERCISE 5: LFU Cache
// ============================================================================
//
// RELATED READING:
//   - ../08-performance-scaling/01-caching-and-redis.md (caching strategies)
//   - ../10-interview-prep/01-interview-fundamentals.md (data structure design)
//
// Implement a Least Frequently Used (LFU) cache. When the cache is full,
// evict the item with the lowest access frequency. If tied, evict the
// least recently used among the least frequent.
//
// Requirements:
//   - LFUCache(capacity) creates a cache with the given capacity
//   - .get(key) returns the value (or undefined) and increments frequency
//   - .put(key, value) adds/updates an entry; evicts LFU item if at capacity
//   - On tie (same frequency), evict the least recently used
//   - O(1) for get and put operations
//
// Hints:
//   - Three maps: keyToValue, keyToFreq, freqToKeys (ordered set of keys)
//   - Track minFreq to know which frequency bucket to evict from
//   - freqToKeys uses Map<number, Set<string>> — Set preserves insertion order
//   - On access: remove key from old freq bucket, add to freq+1 bucket
//
//   Pattern:
//     class LFUCache<K, V> {
//       private vals = new Map<K, V>();
//       private freqs = new Map<K, number>();
//       private freqBuckets = new Map<number, Set<K>>();
//       private minFreq = 0;
//
//       get(key: K): V | undefined {
//         if (!this.vals.has(key)) return undefined;
//         this.incrementFreq(key);
//         return this.vals.get(key)!;
//       }
//
//       private incrementFreq(key: K) {
//         const freq = this.freqs.get(key)!;
//         this.freqBuckets.get(freq)!.delete(key);
//         if (this.freqBuckets.get(freq)!.size === 0 && freq === this.minFreq) {
//           this.minFreq++;
//         }
//         const newFreq = freq + 1;
//         this.freqs.set(key, newFreq);
//         if (!this.freqBuckets.has(newFreq)) this.freqBuckets.set(newFreq, new Set());
//         this.freqBuckets.get(newFreq)!.add(key);
//       }
//     }
//
//   Key concepts:
//   - Frequency buckets group keys by access count
//   - Set iteration order = insertion order (LRU within frequency)
//   - minFreq tracks the eviction target
//
// Expected behavior:
//   const cache = new LFUCache<string, number>(2);
//   cache.put("a", 1); cache.put("b", 2);
//   cache.get("a");      // 1 (freq: a=2, b=1)
//   cache.put("c", 3);   // evicts "b" (lowest frequency)
//   cache.get("b");      // undefined

class LFUCache<K, V> {
  constructor(_capacity: number) {
    throw new Error("Not implemented");
  }

  get(_key: K): V | undefined {
    throw new Error("Not implemented");
  }

  put(_key: K, _value: V): void {
    throw new Error("Not implemented");
  }
}


// ============================================================================
// EXERCISE 6: Merge K Sorted Arrays
// ============================================================================
//
// RELATED READING:
//   - ../10-interview-prep/01-interview-fundamentals.md (heap/priority queue)
//   - ../01-typescript-advanced/02-advanced-type-patterns.md
//
// Merge K sorted arrays into a single sorted array. Use a min-heap approach
// for optimal O(N log K) time complexity.
//
// Requirements:
//   - mergeSorted(arrays) merges K sorted arrays into one sorted array
//   - Optional comparator function (defaults to numeric ascending)
//   - O(N log K) time complexity where N = total elements, K = number of arrays
//   - Handles empty arrays and arrays of different lengths
//
// Hints:
//   - Use a min-heap (priority queue) of size K
//   - Initialize heap with the first element of each non-empty array
//   - Pop min, push result, then push next element from that array
//   - Track which array and index each heap element came from
//
//   Min-heap pattern (array-based):
//     class MinHeap<T> {
//       private data: T[] = [];
//       constructor(private compare: (a: T, b: T) => number) {}
//       push(val: T) { this.data.push(val); this.bubbleUp(this.data.length - 1); }
//       pop(): T | undefined {
//         if (this.data.length === 0) return undefined;
//         const top = this.data[0];
//         const last = this.data.pop()!;
//         if (this.data.length > 0) { this.data[0] = last; this.sinkDown(0); }
//         return top;
//       }
//       get size() { return this.data.length; }
//       // bubbleUp and sinkDown implement heap property
//     }
//
//   Key concepts:
//   - Heap gives O(log K) per insertion/extraction
//   - Tracking (arrayIndex, elementIndex) lets you pull the next element
//   - Total: N elements * O(log K) per element = O(N log K)
//
// Expected behavior:
//   mergeSorted([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
//   // [1, 2, 3, 4, 5, 6, 7, 8, 9]
//
//   mergeSorted([[1, 3], [], [2, 4, 5]]);
//   // [1, 2, 3, 4, 5]

type Comparator<T> = (a: T, b: T) => number;

function mergeSorted<T>(
  _arrays: T[][],
  _compare?: Comparator<T>,
): T[] {
  throw new Error("Not implemented");
}


// ============================================================================
// TESTS
// ============================================================================

function test_group_by(): void {
  console.log("\n=== EXERCISE 1: Group By ===");

  const words = ["hello", "hi", "hey", "world", "wow"];
  const grouped = groupBy(words, (w) => w[0]);

  console.assert(grouped.get("h")!.length === 3, "3 h-words");
  console.assert(grouped.get("w")!.length === 2, "2 w-words");

  // Group numbers by even/odd
  const nums = [1, 2, 3, 4, 5, 6];
  const byParity = groupBy(nums, (n) => (n % 2 === 0 ? "even" : "odd"));
  console.assert(
    JSON.stringify(byParity.get("even")) === JSON.stringify([2, 4, 6]),
    "Even numbers grouped",
  );
  console.assert(
    JSON.stringify(byParity.get("odd")) === JSON.stringify([1, 3, 5]),
    "Odd numbers grouped",
  );

  // Empty input
  const empty = groupBy([], (x: string) => x);
  console.assert(empty.size === 0, "Empty input returns empty Map");

  console.log("Edge cases passed");
  console.log("EXERCISE 1: PASSED");
}

function test_deep_clone(): void {
  console.log("\n=== EXERCISE 2: Deep Clone ===");

  // Plain objects and arrays
  const obj = { a: [1, { b: 2 }], c: "hello" };
  const cloned = deepClone(obj);
  (cloned.a[1] as { b: number }).b = 99;
  console.assert((obj.a[1] as { b: number }).b === 2, "Original not mutated");

  // Date
  const date = new Date(2024, 0, 1);
  const clonedDate = deepClone(date);
  console.assert(clonedDate.getTime() === date.getTime(), "Date cloned");
  console.assert(clonedDate !== date, "Date is a new instance");

  // RegExp
  const regex = /hello/gi;
  const clonedRegex = deepClone(regex);
  console.assert(clonedRegex.source === "hello", "RegExp source cloned");
  console.assert(clonedRegex.flags === "gi", "RegExp flags cloned");
  console.assert(clonedRegex !== regex, "RegExp is a new instance");

  // Map
  const map = new Map([["a", { x: 1 }]]);
  const clonedMap = deepClone(map);
  clonedMap.get("a")!.x = 99;
  console.assert(map.get("a")!.x === 1, "Map values deeply cloned");

  // Set
  const set = new Set([1, 2, 3]);
  const clonedSet = deepClone(set);
  console.assert(clonedSet.size === 3, "Set cloned");
  console.assert(clonedSet !== set, "Set is a new instance");

  // Primitives
  console.assert(deepClone(42) === 42, "Number passthrough");
  console.assert(deepClone("hello") === "hello", "String passthrough");
  console.assert(deepClone(null) === null, "Null passthrough");

  console.log("Edge cases passed");
  console.log("EXERCISE 2: PASSED");
}

function test_trie(): void {
  console.log("\n=== EXERCISE 3: Trie ===");

  const trie = new Trie();
  trie.insert("cat");
  trie.insert("car");
  trie.insert("card");
  trie.insert("dog");
  trie.insert("door");

  console.assert(trie.search("car") === true, "Search exact word");
  console.assert(trie.search("ca") === false, "Prefix is not a word");
  console.assert(trie.search("cart") === false, "Non-existent word");
  console.assert(trie.startsWith("ca") === true, "Prefix exists");
  console.assert(trie.startsWith("do") === true, "Another prefix");
  console.assert(trie.startsWith("xyz") === false, "Prefix doesn't exist");

  const completions = trie.autoComplete("ca", 10);
  console.assert(completions.length === 3, "3 completions for 'ca'");
  console.assert(completions.includes("cat"), "cat in completions");
  console.assert(completions.includes("car"), "car in completions");
  console.assert(completions.includes("card"), "card in completions");

  // Limit
  const limited = trie.autoComplete("ca", 2);
  console.assert(limited.length === 2, "Limit respected");

  // Empty trie
  const emptyTrie = new Trie();
  console.assert(emptyTrie.search("x") === false, "Empty trie search");
  console.assert(emptyTrie.autoComplete("x", 10).length === 0, "Empty trie autocomplete");

  console.log("Edge cases passed");
  console.log("EXERCISE 3: PASSED");
}

function test_deep_flatten(): void {
  console.log("\n=== EXERCISE 4: Deep Flatten ===");

  // Recursive version
  const result1 = deepFlatten([1, [2, [3]], [4, [5, [6]]]]);
  console.assert(
    JSON.stringify(result1) === JSON.stringify([1, 2, 3, 4, 5, 6]),
    "Recursive flatten",
  );

  // Iterative version
  const result2 = deepFlattenIterative([1, [2, [3]], [4, [5, [6]]]]);
  console.assert(
    JSON.stringify(result2) === JSON.stringify([1, 2, 3, 4, 5, 6]),
    "Iterative flatten",
  );

  // Already flat
  console.assert(
    JSON.stringify(deepFlatten([1, 2, 3])) === JSON.stringify([1, 2, 3]),
    "Already flat",
  );

  // Empty
  console.assert(deepFlatten([]).length === 0, "Empty array");

  // Deeply nested
  console.assert(
    JSON.stringify(deepFlatten([[[[[1]]]]])) === JSON.stringify([1]),
    "Very deep nesting",
  );

  // Strings
  const strResult = deepFlatten(["a", ["b", ["c"]]]);
  console.assert(
    JSON.stringify(strResult) === JSON.stringify(["a", "b", "c"]),
    "String flatten",
  );

  console.log("Edge cases passed");
  console.log("EXERCISE 4: PASSED");
}

function test_lfu_cache(): void {
  console.log("\n=== EXERCISE 5: LFU Cache ===");

  const cache = new LFUCache<string, number>(2);

  cache.put("a", 1);
  cache.put("b", 2);
  console.assert(cache.get("a") === 1, "Get a");

  // a has freq 2, b has freq 1 — b should be evicted
  cache.put("c", 3);
  console.assert(cache.get("b") === undefined, "b evicted (lowest freq)");
  console.assert(cache.get("a") === 1, "a still exists");
  console.assert(cache.get("c") === 3, "c exists");

  // Now a has freq 3, c has freq 2 — c should be evicted
  cache.put("d", 4);
  console.assert(cache.get("c") === undefined, "c evicted");
  console.assert(cache.get("a") === 1, "a still exists");
  console.assert(cache.get("d") === 4, "d exists");

  // Update existing key
  cache.put("a", 10);
  console.assert(cache.get("a") === 10, "a updated");

  console.log("Edge cases passed");
  console.log("EXERCISE 5: PASSED");
}

function test_merge_sorted(): void {
  console.log("\n=== EXERCISE 6: Merge K Sorted Arrays ===");

  const result = mergeSorted([[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
  console.assert(
    JSON.stringify(result) === JSON.stringify([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "3 sorted arrays merged",
  );

  // With empty arrays
  const result2 = mergeSorted([[1, 3], [], [2, 4, 5]]);
  console.assert(
    JSON.stringify(result2) === JSON.stringify([1, 2, 3, 4, 5]),
    "Handles empty arrays",
  );

  // Single array
  console.assert(
    JSON.stringify(mergeSorted([[1, 2, 3]])) === JSON.stringify([1, 2, 3]),
    "Single array",
  );

  // All empty
  console.assert(mergeSorted([[], [], []]).length === 0, "All empty arrays");

  // No arrays
  console.assert(mergeSorted([]).length === 0, "No arrays");

  // Custom comparator (descending)
  const desc = mergeSorted(
    [[9, 5, 1], [8, 4], [7, 3, 2]],
    (a, b) => a - b,
  );
  // Input arrays are sorted descending, but we still merge ascending
  // Actually, for this to work correctly, input should be sorted per comparator
  // Let's test with properly sorted ascending arrays
  const result3 = mergeSorted(
    [[1, 5, 9], [2, 4, 8], [3, 7]],
    (a, b) => a - b,
  );
  console.assert(
    JSON.stringify(result3) === JSON.stringify([1, 2, 3, 4, 5, 7, 8, 9]),
    "Custom comparator",
  );

  console.log("Edge cases passed");
  console.log("EXERCISE 6: PASSED");
}


if (require.main === module) {
  console.log("Data Structures Exercises");
  console.log("=".repeat(60));

  const tests: [string, () => void][] = [
    ["Exercise 1: Group By", test_group_by],
    ["Exercise 2: Deep Clone", test_deep_clone],
    ["Exercise 3: Trie", test_trie],
    ["Exercise 4: Deep Flatten", test_deep_flatten],
    ["Exercise 5: LFU Cache", test_lfu_cache],
    ["Exercise 6: Merge K Sorted Arrays", test_merge_sorted],
  ];

  let passed = 0;
  let failed = 0;

  for (const [name, testFn] of tests) {
    try {
      testFn();
      passed++;
    } catch (e) {
      if (e instanceof Error && e.message === "Not implemented") {
        console.log(`  ${name}: NOT IMPLEMENTED`);
      } else if (e instanceof Error) {
        console.log(`  ${name}: FAILED -- ${e.message}`);
      } else {
        console.log(`  ${name}: ERROR -- ${e}`);
      }
      failed++;
    }
  }

  console.log();
  console.log("=".repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed out of ${tests.length}`);
  console.log("=".repeat(60));
}
