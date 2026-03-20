/*
Drill 10 — Event Scheduler / Subscription Notifications

Schedule notifications for subscriptions based on lifecycle events.
This is a commonly reported Stripe problem that tests date arithmetic
and event-driven processing.

Target time: 30 minutes for all 4 levels.

────────────────────────────────────────
Level 1 — Basic Notification Scheduling (8 min)

  scheduleNotifications(subscriptions): Notification[]

  Given subscriptions: { id: string, name: string, plan: string,
                         startDay: number, duration: number }

  Schedule three notification types per subscription:
    - "welcome"      → on startDay
    - "expiring_soon" → on (startDay + duration - 15), only if duration > 15
    - "expired"      → on (startDay + duration)

  Return a sorted array of:
    { day: number, type: string, name: string, plan: string }

  Sort by day ascending, then by subscription id for ties.

────────────────────────────────────────
Level 2 — Plan Changes (10 min)

  applyPlanChanges(subscriptions, changes): Notification[]

  Accept plan changes: { name: string, newPlan: string, changeDay: number }

  On changeDay, emit a "changed" notification (with the newPlan).
  Recalculate remaining notifications: the change doesn't extend or
  shorten — the expiry date stays the same (original startDay + duration).
  Cancel any future notifications for the old plan that fall on or after
  changeDay, and schedule new ones for the new plan.

────────────────────────────────────────
Level 3 — Renewals (8 min)

  applyRenewals(subscriptions, renewals): Notification[]

  Accept renewals: { name: string, extensionDays: number, renewDay: number }

  On renewDay, emit a "renewed" notification.
  Extend the subscription duration by extensionDays.
  Reschedule "expiring_soon" and "expired" notifications based on the
  new end date. Cancel old future notifications that are now stale.

────────────────────────────────────────
Level 4 — Penalty Calculation (8 min)

  bestClosingTime(log: string): number

  Given a log string of 'Y' (customer present) and 'N' (no customer)
  for each hour:
    - Calculate penalty for a given closing time:
        +1 for each 'N' before closing, +1 for each 'Y' after closing.
    - Closing time i means close at the start of hour i
      (0 = close before any hour, log.length = close after all hours).
    - Return the optimal closing time (minimum penalty, earliest if tied).

  processLogs(input: string): number[]

  Process multiple logs separated by BEGIN / END markers.
  Return the optimal closing time for each log.

  This is Stripe's "Store Closing Penalty" problem (LeetCode 2483).
*/

interface Subscription {
  id: string;
  name: string;
  plan: string;
  startDay: number;
  duration: number;
}

interface Notification {
  day: number;
  type: string;
  name: string;
  plan: string;
}

interface PlanChange {
  name: string;
  newPlan: string;
  changeDay: number;
}

interface Renewal {
  name: string;
  extensionDays: number;
  renewDay: number;
}

// Level 1
function scheduleNotifications(subscriptions: Subscription[]): Notification[] {
  throw new Error("TODO: implement scheduleNotifications");
}

// Level 2
function applyPlanChanges(subscriptions: Subscription[], changes: PlanChange[]): Notification[] {
  throw new Error("TODO: implement applyPlanChanges");
}

// Level 3
function applyRenewals(subscriptions: Subscription[], renewals: Renewal[]): Notification[] {
  throw new Error("TODO: implement applyRenewals");
}

// Level 4
function bestClosingTime(log: string): number {
  throw new Error("TODO: implement bestClosingTime");
}

function processLogs(input: string): number[] {
  throw new Error("TODO: implement processLogs");
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
  level("Level 1 — Basic Notification Scheduling", () => {
    const subs: Subscription[] = [
      { id: "s1", name: "Alice", plan: "basic", startDay: 10, duration: 30 },
      { id: "s2", name: "Bob", plan: "pro", startDay: 10, duration: 10 },
    ];
    const result = scheduleNotifications(subs);

    // Alice: welcome on 10, expiring_soon on 25 (10+30-15), expired on 40
    // Bob: welcome on 10, no expiring_soon (duration <= 15), expired on 20
    // Day 10: Alice welcome, Bob welcome (sorted by id: s1 < s2)
    // Day 20: Bob expired
    // Day 25: Alice expiring_soon
    // Day 40: Alice expired
    check("total notifications", result.length, 5);
    check("first is Alice welcome", result[0], { day: 10, type: "welcome", name: "Alice", plan: "basic" });
    check("second is Bob welcome (tie broken by id)", result[1], { day: 10, type: "welcome", name: "Bob", plan: "pro" });
    check("Bob expired on day 20", result[2], { day: 20, type: "expired", name: "Bob", plan: "pro" });
    check("Alice expiring_soon on day 25", result[3], { day: 25, type: "expiring_soon", name: "Alice", plan: "basic" });
    check("Alice expired on day 40", result[4], { day: 40, type: "expired", name: "Alice", plan: "basic" });
  });

  level("Level 2 — Plan Changes", () => {
    const subs: Subscription[] = [
      { id: "s1", name: "Alice", plan: "basic", startDay: 10, duration: 30 },
    ];
    const changes: PlanChange[] = [
      { name: "Alice", newPlan: "pro", changeDay: 20 },
    ];
    const result = applyPlanChanges(subs, changes);

    // Alice original: welcome@10(basic), expiring_soon@25(basic), expired@40(basic)
    // Change on day 20: cancel future basic notifs (expiring_soon@25, expired@40)
    // Add: changed@20(pro), expiring_soon@25(pro), expired@40(pro)
    // Final: welcome@10(basic), changed@20(pro), expiring_soon@25(pro), expired@40(pro)
    check("total after plan change", result.length, 4);
    check("welcome stays basic", result[0], { day: 10, type: "welcome", name: "Alice", plan: "basic" });
    check("changed notification", result[1], { day: 20, type: "changed", name: "Alice", plan: "pro" });
    check("expiring_soon now pro", result[2], { day: 25, type: "expiring_soon", name: "Alice", plan: "pro" });
    check("expired still on day 40 as pro", result[3], { day: 40, type: "expired", name: "Alice", plan: "pro" });
  });

  level("Level 3 — Renewals", () => {
    const subs: Subscription[] = [
      { id: "s1", name: "Alice", plan: "basic", startDay: 10, duration: 30 },
    ];
    const renewals: Renewal[] = [
      { name: "Alice", extensionDays: 20, renewDay: 35 },
    ];
    const result = applyRenewals(subs, renewals);

    // Original: welcome@10, expiring_soon@25, expired@40
    // Renewal on day 35: new end = 10 + 30 + 20 = 60
    // Cancel stale future notifs (expiring_soon@25 stays since 25 < 35,
    //   but expired@40 is stale — reschedule)
    // After renewal: welcome@10, expiring_soon@25, renewed@35,
    //   expiring_soon@45(new), expired@60(new)
    // Actually: old expiring_soon@25 is before renewDay so it stays.
    //   old expired@40 is after renewDay so it gets cancelled.
    //   new expiring_soon = 60-15 = 45, new expired = 60.
    check("total after renewal", result.length, 5);
    check("welcome unchanged", result[0], { day: 10, type: "welcome", name: "Alice", plan: "basic" });
    check("original expiring_soon kept", result[1], { day: 25, type: "expiring_soon", name: "Alice", plan: "basic" });
    check("renewed notification", result[2], { day: 35, type: "renewed", name: "Alice", plan: "basic" });
    check("new expiring_soon at day 45", result[3], { day: 45, type: "expiring_soon", name: "Alice", plan: "basic" });
    check("new expired at day 60", result[4], { day: 60, type: "expired", name: "Alice", plan: "basic" });
  });

  level("Level 4 — Penalty Calculation", () => {
    // "YYNY": close at 0 → penalty = Y+Y+N+Y = 3 (all Y after)
    //         close at 1 → penalty = Y+N+Y = 2
    //         close at 2 → penalty = N+Y = 1+1 = 2  (wait, let me recalc)
    // Actually: penalty(i) = count of 'N' in [0,i) + count of 'Y' in [i, n)
    // "YYNY":
    //   close@0: 0 N before + 3 Y after = 3
    //   close@1: 0 N before + 2 Y after = 2
    //   close@2: 0 N before + 1 Y after = 1
    //   close@3: 1 N before + 1 Y after = 2
    //   close@4: 1 N before + 0 Y after = 1
    //   Min is 1 at close@2 (earliest)
    check("bestClosingTime YYNY", bestClosingTime("YYNY"), 2);

    // "NYNY":
    //   close@0: 0N before + 2Y after = 2
    //   close@1: 1N before + 1Y after = 2
    //   close@2: 1N before + 1Y after = 2
    //   close@3: 2N before + 0Y after = 2
    //   close@4: 2N before + 0Y after = 2
    //   Min is 2 at close@0 (all tied, earliest wins)
    check("bestClosingTime NYNY", bestClosingTime("NYNY"), 0);

    // "YYYY": best to close at end (no penalty for N, 0 Y after)
    //   close@4: 0 N before + 0 Y after = 0
    check("bestClosingTime YYYY", bestClosingTime("YYYY"), 4);

    const input = "BEGIN\nYYNY\nEND\nBEGIN\nNNNN\nEND";
    // "YYNY" → 2, "NNNN" → 0
    check("processLogs two logs", processLogs(input), [2, 0]);
  });
}

function main(): void {
  console.log("\nEvent Scheduler / Subscription Notifications\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
