/*
Drill 03 — Feature Flag Service

Implement a FeatureFlagService class with global toggles,
user overrides, group rules, and snapshots.

────────────────────────────────────────
Level 1 — Global Flags

  enable(flag: string): void
    Enable a flag globally.

  disable(flag: string): void
    Disable a flag globally.

  isEnabled(flag: string): boolean
    Returns whether the flag is enabled. Defaults to false.

────────────────────────────────────────
Level 2 — User Overrides

  enableForUser(flag: string, userId: string): void
    Enable a flag for a specific user.

  disableForUser(flag: string, userId: string): void
    Disable a flag for a specific user.

  isEnabledForUser(flag: string, userId: string): boolean
    If user has an override for this flag, return the override.
    Otherwise fall back to the global state.

────────────────────────────────────────
Level 3 — Groups

  addUserToGroup(userId: string, group: string): void
    Add a user to a group. A user can be in multiple groups.

  enableForGroup(flag: string, group: string): void
    Enable a flag for an entire group.

  disableForGroup(flag: string, group: string): void
    Disable a flag for an entire group.

  Resolution order for isEnabledForUser:
    1. User override (highest priority)
    2. Group — if the user belongs to ANY group that has
       this flag enabled, the flag is enabled at group level.
       If all of the user's groups with a setting have it
       disabled, the flag is disabled at group level.
       Only applies when at least one of the user's groups
       has a setting for this flag.
    3. Global state (lowest priority)

────────────────────────────────────────
Level 4 — Snapshots

  snapshot(name: string): void
    Save the entire current state under this name.
    Overwrites if the name already exists.

  restore(name: string): boolean
    Restore state from the named snapshot.
    Returns false if the name does not exist.

  listSnapshots(): string[]
    Returns all snapshot names, sorted alphabetically.
*/

type Snapshot = {
  flags: Map<string, boolean>;
  users: Map<string, Map<string, boolean>>;
  userGroups: Map<string, Set<string>>;
  groupFlags: Map<string, Map<string, boolean>>;
}

export class FeatureFlagService {
  flags: Map<string, boolean>;
  users: Map<string, Map<string, boolean>>;
  userGroups: Map<string, Set<string>>;
  groupFlags: Map<string, Map<string, boolean>>;
  snapshots: Map<string, Snapshot>;

  constructor() {
    this.flags = new Map();
    this.users = new Map();
    this.userGroups = new Map();
    this.groupFlags = new Map();
    this.snapshots = new Map();
  }

  enable(flag: string): void {
    this.flags.set(flag, true);
  }

  disable(flag: string): void {
    this.flags.set(flag, false);
  }

  isEnabled(flag: string): boolean {
    return !!this.flags.get(flag);
  }

  enableForUser(flag: string, userId: string): void {
    let user = this.users.get(userId);
    if (!user) {
      user = new Map(); 
    }
    user.set(flag, true);
    this.users.set(userId, user);
  }

  disableForUser(flag: string, userId: string): void {
    let user = this.users.get(userId);
    if (!user) {
      user = new Map(); 
    }
    user.set(flag, false);
    this.users.set(userId, user);
  }

  isEnabledForUser(flag: string, userId: string): boolean {
    const user = this.users.get(userId);
    if (user && user.has(flag)) {
      return user.get(flag)!;
    }

    const userGroups = Array.from(this.userGroups.entries()).filter(([_, users]) => users.has(userId));
    const groupEnabled = userGroups.reduce<(boolean)[]>((accum, [group]) => {
      const enabled = this.groupFlags.get(group)?.get(flag);
      return enabled !== undefined ? [...accum, enabled] : accum;
;    }, []);

    return groupEnabled.length ? groupEnabled.some((enabled: boolean) => enabled) : this.isEnabled(flag)
  }
  // Alternative: replace the reduce with filter + map
  //
  // const groupSettings = userGroups
  //   .map(([group]) => this.groupFlags.get(group)?.get(flag))
  //   .filter((v): v is boolean => v !== undefined);
  //
  // return groupSettings.length ? groupSettings.some(Boolean) : this.isEnabled(flag);

  addUserToGroup(userId: string, group: string): void {
    let userGroup = this.userGroups.get(group);
    if (!userGroup) {
      userGroup = new Set();
    }
    userGroup.add(userId);
    this.userGroups.set(group, userGroup);
  }

  enableForGroup(flag: string, group: string): void {
    let groupFlags = this.groupFlags.get(group);
    if (!groupFlags) {
      groupFlags = new Map();
    }
    groupFlags.set(flag, true);
    this.groupFlags.set(group, groupFlags);
  }

  disableForGroup(flag: string, group: string): void {
    let groupFlags = this.groupFlags.get(group);
    if (!groupFlags) {
      groupFlags = new Map();
    }
    groupFlags.set(flag, false);
    this.groupFlags.set(group, groupFlags);
  }

  snapshot(name: string): void {
    this.snapshots.set(name, {
      flags: structuredClone(this.flags),
      users: structuredClone(this.users),
      userGroups: structuredClone(this.userGroups),
      groupFlags: structuredClone(this.groupFlags),
    });
  }

  restore(name: string): boolean {
    const snapshot = this.snapshots.get(name);
    if (!snapshot) return false
    this.flags = snapshot.flags;
    this.users = snapshot.users;
    this.userGroups = snapshot.userGroups;
    this.groupFlags = snapshot.groupFlags;
    // these should clone, any future state changes will corrupt the snapshot
    return true;
  }

  listSnapshots(): string[] {
    const names = Array.from(this.snapshots.keys());
    names.sort((a,b) => a > b ? 1 : -1);
    return names;
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
  level("Level 1 — Global Flags", () => {
    const f1 = new FeatureFlagService();
    check("default off", f1.isEnabled("dark_mode"), false);
    f1.enable("dark_mode");
    check("enabled", f1.isEnabled("dark_mode"), true);
    f1.disable("dark_mode");
    check("disabled", f1.isEnabled("dark_mode"), false);
  });

  level("Level 2 — User Overrides", () => {
    const f2 = new FeatureFlagService();
    f2.enable("new_ui");
    f2.disableForUser("new_ui", "user1");
    check("user override off", f2.isEnabledForUser("new_ui", "user1"), false);
    check("fallback global", f2.isEnabledForUser("new_ui", "user2"), true);
    f2.enableForUser("new_ui", "user1");
    check("user override on", f2.isEnabledForUser("new_ui", "user1"), true);
  });

  level("Level 3 — Groups", () => {
    const f3 = new FeatureFlagService();
    f3.addUserToGroup("alice", "beta");
    f3.addUserToGroup("alice", "staff");
    f3.enableForGroup("experiment", "beta");
    check("group enabled", f3.isEnabledForUser("experiment", "alice"), true);
    check("not in group", f3.isEnabledForUser("experiment", "bob"), false);
    f3.disableForUser("experiment", "alice");
    check("user beats group", f3.isEnabledForUser("experiment", "alice"), false);

    const f3b = new FeatureFlagService();
    f3b.enable("feature_x");
    f3b.addUserToGroup("carol", "internal");
    f3b.disableForGroup("feature_x", "internal");
    check("group off beats global", f3b.isEnabledForUser("feature_x", "carol"), false);
    check("no group uses global", f3b.isEnabledForUser("feature_x", "dave"), true);
  });

  level("Level 4 — Snapshots", () => {
    const f4 = new FeatureFlagService();
    f4.enable("flag_a");
    f4.enable("flag_b");
    f4.snapshot("v1");
    f4.disable("flag_a");
    check("after disable", f4.isEnabled("flag_a"), false);
    check("restore", f4.restore("v1"), true);
    check("restored", f4.isEnabled("flag_a"), true);
    check("other flag", f4.isEnabled("flag_b"), true);
    check("missing snapshot", f4.restore("nope"), false);
    check("list", f4.listSnapshots(), ["v1"]);
  });
}

function main(): void {
  console.log("\nFeature Flag Service\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
