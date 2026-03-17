/*
Drill 08 — Resource Scheduler

Implement a ResourceScheduler class with resource management,
bookings with capacity constraints, availability queries,
and a priority waitlist.

This drill tests interval overlap detection, capacity management,
event-sweep algorithms, and priority-based queue processing.

────────────────────────────────────────
Level 1 — Resources & Bookings

  addResource(resourceId: string, capacity: number): boolean
    Register a resource with a given capacity (max concurrent bookings).
    Returns false if resourceId already exists or capacity <= 0.

  createBooking(bookingId: string, resourceId: string,
                start: number, end: number): boolean
    Book 1 unit of a resource for the half-open interval [start, end).
    Returns false if bookingId already exists, resource not found,
    start >= end, or insufficient capacity at any point during [start, end).

  cancelBooking(bookingId: string): boolean
    Cancel an active booking, freeing its capacity.
    Returns false if not found or already cancelled.

  getBooking(bookingId: string):
      { resourceId: string, start: number, end: number, status: string } | null
    Returns booking details, or null if not found.
    Statuses: "ACTIVE" | "CANCELLED"

────────────────────────────────────────
Level 2 — Availability & Queries

  getAvailability(resourceId: string, start: number, end: number): number | null
    Returns the minimum available capacity during [start, end).
    Considers only ACTIVE bookings (including multi-unit from Level 3).
    Returns null if resource not found.

  getBookingsForResource(resourceId: string): string[]
    Returns ACTIVE booking ids for this resource, sorted by
    start ascending, then bookingId ascending for ties.
    Returns [] if resource not found.

  getConflicts(resourceId: string, start: number, end: number): string[]
    Returns ACTIVE bookings that overlap with [start, end),
    sorted by start ascending, then bookingId ascending.
    Two intervals [a, b) and [c, d) overlap iff a < d AND c < b.
    Returns [] if resource not found.

────────────────────────────────────────
Level 3 — Multi-unit & Rescheduling

  createBulkBooking(bookingId: string, resourceId: string,
                    start: number, end: number, units: number): boolean
    Book multiple units of capacity for [start, end).
    Same validation as createBooking but checks for `units`
    available capacity. Also returns false if units <= 0.

  moveBooking(bookingId: string, newStart: number, newEnd: number): boolean
    Reschedule an active booking to a new time slot.
    The booking's capacity is freed before checking the new slot,
    so moving within an overlapping window is valid if capacity allows.
    The booking retains its original unit count.
    Returns false if booking not found, not active,
    newStart >= newEnd, or insufficient capacity at new time.

  batchCreate(entries: [string, string, number, number][]): number
    Each entry is [bookingId, resourceId, start, end] for 1 unit.
    Process entries in order, creating bookings that pass validation
    and skipping those that don't. Earlier bookings in the batch
    affect capacity for later ones.
    Returns the count of successfully created bookings.

────────────────────────────────────────
Level 4 — Waitlist

  addToWaitlist(waitlistId: string, resourceId: string,
                start: number, end: number, priority: number): boolean
    Queue a booking request with a priority.
    Returns false if waitlistId already exists (as a waitlist entry
    or as an existing booking id) or resource not found.

  processWaitlist(resourceId: string): string[]
    Process waitlist entries for this resource in priority descending
    order (then waitlistId ascending for ties).
    For each entry, if 1 unit of capacity is available during
    [start, end), create a booking using waitlistId as the bookingId
    and remove the entry from the waitlist.
    Each successful booking affects capacity for subsequent entries.
    Returns ids of successfully booked entries.

  getWaitlist(resourceId: string): string[]
    Returns pending waitlist entry ids, sorted by priority descending,
    then waitlistId ascending.
    Returns [] if resource not found or waitlist is empty.
*/

export class ResourceScheduler {
  constructor() {
  }

  addResource(resourceId: string, capacity: number): boolean {
    throw new Error("TODO: implement addResource");
  }

  createBooking(bookingId: string, resourceId: string, start: number, end: number): boolean {
    throw new Error("TODO: implement createBooking");
  }

  cancelBooking(bookingId: string): boolean {
    throw new Error("TODO: implement cancelBooking");
  }

  getBooking(bookingId: string): { resourceId: string, start: number, end: number, status: string } | null {
    throw new Error("TODO: implement getBooking");
  }

  getAvailability(resourceId: string, start: number, end: number): number | null {
    throw new Error("TODO: implement getAvailability");
  }

  getBookingsForResource(resourceId: string): string[] {
    throw new Error("TODO: implement getBookingsForResource");
  }

  getConflicts(resourceId: string, start: number, end: number): string[] {
    throw new Error("TODO: implement getConflicts");
  }

  createBulkBooking(bookingId: string, resourceId: string, start: number, end: number, units: number): boolean {
    throw new Error("TODO: implement createBulkBooking");
  }

  moveBooking(bookingId: string, newStart: number, newEnd: number): boolean {
    throw new Error("TODO: implement moveBooking");
  }

  batchCreate(entries: [string, string, number, number][]): number {
    throw new Error("TODO: implement batchCreate");
  }

  addToWaitlist(waitlistId: string, resourceId: string, start: number, end: number, priority: number): boolean {
    throw new Error("TODO: implement addToWaitlist");
  }

  processWaitlist(resourceId: string): string[] {
    throw new Error("TODO: implement processWaitlist");
  }

  getWaitlist(resourceId: string): string[] {
    throw new Error("TODO: implement getWaitlist");
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
  level("Level 1 — Resources & Bookings", () => {
    const r = new ResourceScheduler();
    check("add", r.addResource("room1", 2), true);
    check("add dup", r.addResource("room1", 3), false);
    check("add bad cap", r.addResource("room2", 0), false);
    r.addResource("room2", 1);
    check("book", r.createBooking("b1", "room1", 0, 10), true);
    check("book dup", r.createBooking("b1", "room1", 20, 30), false);
    check("book bad res", r.createBooking("b2", "nope", 0, 10), false);
    check("book bad time", r.createBooking("b3", "room1", 10, 5), false);
    check("book equal time", r.createBooking("b3b", "room1", 5, 5), false);
    // room1 capacity=2, second overlapping booking is fine
    check("book overlap ok", r.createBooking("b4", "room1", 5, 15), true);
    // third overlapping would exceed capacity
    check("book overlap full", r.createBooking("b5", "room1", 7, 8), false);
    // room2 capacity=1
    check("book room2", r.createBooking("b6", "room2", 0, 10), true);
    check("book room2 full", r.createBooking("b7", "room2", 5, 15), false);
    // non-overlapping is fine
    check("book no overlap", r.createBooking("b8", "room2", 10, 20), true);
    // get booking
    check("get", r.getBooking("b1"),
      { resourceId: "room1", start: 0, end: 10, status: "ACTIVE" });
    check("get missing", r.getBooking("nope"), null);
    // cancel
    check("cancel", r.cancelBooking("b1"), true);
    check("cancel again", r.cancelBooking("b1"), false);
    check("get cancelled", r.getBooking("b1"),
      { resourceId: "room1", start: 0, end: 10, status: "CANCELLED" });
    // after cancelling b1, capacity freed — can now book in that slot
    check("book after cancel", r.createBooking("b9", "room1", 7, 8), true);
  });

  level("Level 2 — Availability & Queries", () => {
    const r = new ResourceScheduler();
    r.addResource("room", 3);
    r.createBooking("a1", "room", 0, 10);   // ████████░░░░░░░░
    r.createBooking("a2", "room", 5, 15);   // ░░░░░████████░░░
    r.createBooking("a3", "room", 8, 12);   // ░░░░░░░░████░░░░
    // [6,11): a1+a2 at [6,8), a1+a2+a3 at [8,10), a2+a3 at [10,11) → max=3, avail=0
    check("avail full", r.getAvailability("room", 6, 11), 0);
    // [0,5): only a1 → max=1, avail=2
    check("avail partial", r.getAvailability("room", 0, 5), 2);
    // [12,20): only a2 at [12,15) → max=1, avail=2
    check("avail later", r.getAvailability("room", 12, 20), 2);
    // [20,30): nothing → avail=3
    check("avail empty", r.getAvailability("room", 20, 30), 3);
    check("avail missing", r.getAvailability("nope", 0, 10), null);
    // bookings for resource (sorted by start, then id)
    check("all bookings", r.getBookingsForResource("room"), ["a1", "a2", "a3"]);
    check("bookings missing", r.getBookingsForResource("nope"), []);
    // conflicts with [7, 9)
    check("conflicts all", r.getConflicts("room", 7, 9), ["a1", "a2", "a3"]);
    // conflicts with [0, 6): a1 [0,10) overlaps, a2 [5,15) overlaps
    check("conflicts partial", r.getConflicts("room", 0, 6), ["a1", "a2"]);
    check("conflicts none", r.getConflicts("room", 20, 30), []);
    // cancel a2, verify availability changes
    r.cancelBooking("a2");
    check("avail after cancel", r.getAvailability("room", 8, 10), 1);
    check("bookings after cancel", r.getBookingsForResource("room"), ["a1", "a3"]);
  });

  level("Level 3 — Multi-unit & Rescheduling", () => {
    const r = new ResourceScheduler();
    r.addResource("server", 5);
    // bulk: 3 units at once
    check("bulk", r.createBulkBooking("bk1", "server", 0, 10, 3), true);
    check("avail after bulk", r.getAvailability("server", 0, 10), 2);
    // add 2 more units overlapping
    check("bulk 2", r.createBulkBooking("bk2", "server", 5, 15, 2), true);
    // at [5,10): 3+2=5 concurrent, avail=0
    check("avail overlap", r.getAvailability("server", 5, 10), 0);
    // can't add even 1 unit in that range
    check("bulk too many", r.createBulkBooking("bk3", "server", 7, 8, 1), false);
    check("bulk bad units", r.createBulkBooking("bk4", "server", 20, 30, 0), false);
    // [10,15): only bk2 (2 units), avail=3
    check("avail partial", r.getAvailability("server", 10, 15), 3);

    // move booking
    r.addResource("desk", 1);
    r.createBooking("m1", "desk", 0, 10);
    // desk full at [0,10). Move m1 to [10,20) — freed first, so OK
    check("move", r.moveBooking("m1", 10, 20), true);
    check("moved", r.getBooking("m1"),
      { resourceId: "desk", start: 10, end: 20, status: "ACTIVE" });
    // [0,10) now free
    check("book freed slot", r.createBooking("m2", "desk", 0, 10), true);
    // try to move m1 to overlap with m2 — desk cap=1, fails
    check("move conflict", r.moveBooking("m1", 5, 15), false);
    // m1 stays at original position after failed move
    check("after fail", r.getBooking("m1"),
      { resourceId: "desk", start: 10, end: 20, status: "ACTIVE" });
    check("move missing", r.moveBooking("nope", 0, 10), false);
    check("move bad time", r.moveBooking("m1", 20, 15), false);

    // batch create
    r.addResource("lab", 2);
    const count = r.batchCreate([
      ["bc1", "lab", 0, 10],
      ["bc2", "lab", 0, 10],
      ["bc3", "lab", 0, 10],  // fails: lab cap=2, already 2 booked
      ["bc4", "lab", 10, 20], // succeeds: different time
    ]);
    check("batch count", count, 3);
    check("batch bc1", r.getBooking("bc1")?.status, "ACTIVE");
    check("batch bc2", r.getBooking("bc2")?.status, "ACTIVE");
    check("batch bc3 skipped", r.getBooking("bc3"), null);
    check("batch bc4", r.getBooking("bc4")?.status, "ACTIVE");
  });

  level("Level 4 — Waitlist", () => {
    const r = new ResourceScheduler();
    r.addResource("room", 1);
    r.createBooking("occ", "room", 0, 10);
    // room full at [0,10) — add to waitlist
    check("waitlist", r.addToWaitlist("w1", "room", 0, 10, 5), true);
    check("waitlist 2", r.addToWaitlist("w2", "room", 0, 10, 10), true);
    check("waitlist dup", r.addToWaitlist("w1", "room", 0, 10, 3), false);
    check("waitlist bad res", r.addToWaitlist("w3", "nope", 0, 10, 1), false);
    // id collision with existing booking
    check("waitlist id clash", r.addToWaitlist("occ", "room", 0, 10, 1), false);
    // waitlist sorted by priority desc, then id asc
    check("get waitlist", r.getWaitlist("room"), ["w2", "w1"]);
    // can't process yet — room full
    check("process full", r.processWaitlist("room"), []);
    // cancel the blocking booking
    r.cancelBooking("occ");
    // process: w2 (priority 10) first → books [0,10), room full again
    // w1 (priority 5) can't fit → stays on waitlist
    check("process", r.processWaitlist("room"), ["w2"]);
    check("w2 booked", r.getBooking("w2"),
      { resourceId: "room", start: 0, end: 10, status: "ACTIVE" });
    check("remaining", r.getWaitlist("room"), ["w1"]);
    // cancel w2, process again → w1 gets booked
    r.cancelBooking("w2");
    check("process again", r.processWaitlist("room"), ["w1"]);
    check("empty waitlist", r.getWaitlist("room"), []);
    check("w1 booked", r.getBooking("w1"),
      { resourceId: "room", start: 0, end: 10, status: "ACTIVE" });

    // waitlist with different time slots
    r.addResource("gym", 1);
    r.createBooking("g1", "gym", 0, 10);
    r.addToWaitlist("gw1", "gym", 0, 10, 3);   // conflicts with g1
    r.addToWaitlist("gw2", "gym", 10, 20, 1);  // no conflict
    // gw2 has no conflict, gw1 does — both processed in priority order
    // gw1 (priority 3) first but can't book, gw2 (priority 1) second and can
    check("process mixed", r.processWaitlist("gym"), ["gw2"]);
    check("gym waitlist left", r.getWaitlist("gym"), ["gw1"]);
  });
}

function main(): void {
  console.log("\nResource Scheduler\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
