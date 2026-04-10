/**
 * Drill 04 — Permission Manager
 *
 * Implement a PermissionManager class for Dropbox-style access control.
 * Access control is core to how Dash handles cross-app search results.
 *
 * ----------------------------------------
 * Level 1 — Basic Permissions (10 min)
 *
 *   grant(user: string, resource: string, level: string): void
 *     Grant access. Level is one of "read", "write", "admin".
 *     Granting replaces any existing permission for that user/resource pair.
 *
 *   check(user: string, resource: string): string
 *     Return permission level, or "none" if no access.
 *     "admin" implies "write" which implies "read".
 *
 *   revoke(user: string, resource: string): boolean
 *     Remove permission. Return false if no permission existed.
 *
 *   getResources(user: string): string[]
 *     Return resources the user has any access to, sorted alphabetically.
 *
 * ----------------------------------------
 * Level 2 — Hierarchical Resources (10 min)
 *
 *   Resources can be hierarchical paths:
 *     "/docs", "/docs/team", "/docs/team/q1"
 *
 *   Permissions inherit downward: a grant on "/docs" applies to
 *   "/docs/team" and "/docs/team/q1".
 *
 *   A more specific grant overrides a parent grant:
 *     grant("alice", "/docs", "write")
 *     grant("alice", "/docs/team", "read")
 *     check("alice", "/docs/team") -> "read"
 *
 *   check must resolve the most specific applicable permission.
 *
 * ----------------------------------------
 * Level 3 — Groups (10 min)
 *
 *   createGroup(group: string): boolean
 *     Create a group. Return false if it already exists.
 *
 *   addToGroup(user: string, group: string): boolean
 *     Add user to group. Return false if group doesn't exist.
 *
 *   grant can accept a group name (prefixed with "@") as the user
 *   parameter:
 *     grant("@engineers", "/code", "write")
 *
 *   check must resolve permissions with this priority:
 *     direct user permission > group permission > inherited permission
 *
 *   If a user is in multiple groups with different permissions on
 *   the same resource, use the highest level.
 *
 * ----------------------------------------
 * Level 4 — Audit Trail (15 min)
 *
 *   getAuditLog(resource: string): string[]
 *     Return chronological list of permission changes for a resource.
 *     Format: "GRANT user level" or "REVOKE user"
 *     (uses the exact user/group string passed to grant/revoke)
 *
 *   effectivePermissions(resource: string): Record<string, string>
 *     Return dict of {user: level} for all users who have access to
 *     this resource, including via groups and inheritance.
 *     Resolve the final effective level per user.
 */

class PermissionManager {
  constructor() {
    // TODO: initialize your data structures
  }

  grant(user: string, resource: string, level: string): void {
    throw new Error("TODO: grant");
  }

  check(user: string, resource: string): string {
    throw new Error("TODO: check");
  }

  revoke(user: string, resource: string): boolean {
    throw new Error("TODO: revoke");
  }

  getResources(user: string): string[] {
    throw new Error("TODO: getResources");
  }

  createGroup(group: string): boolean {
    throw new Error("TODO: createGroup");
  }

  addToGroup(user: string, group: string): boolean {
    throw new Error("TODO: addToGroup");
  }

  getAuditLog(resource: string): string[] {
    throw new Error("TODO: getAuditLog");
  }

  effectivePermissions(resource: string): Record<string, string> {
    throw new Error("TODO: effectivePermissions");
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
    const pm = new PermissionManager();

    // no permission
    _check("no access", pm.check("alice", "/docs"), "none");

    // grant and check
    pm.grant("alice", "/docs", "read");
    _check("read access", pm.check("alice", "/docs"), "read");

    // overwrite permission
    pm.grant("alice", "/docs", "admin");
    _check("upgraded to admin", pm.check("alice", "/docs"), "admin");

    // different user has no access
    _check("bob no access", pm.check("bob", "/docs"), "none");

    // revoke
    _check("revoke success", pm.revoke("alice", "/docs"), true);
    _check("after revoke", pm.check("alice", "/docs"), "none");
    _check("revoke nonexistent", pm.revoke("alice", "/docs"), false);

    // getResources
    pm.grant("carol", "/a", "read");
    pm.grant("carol", "/c", "write");
    pm.grant("carol", "/b", "admin");
    _check("resources sorted", pm.getResources("carol"), ["/a", "/b", "/c"]);
    _check("no resources", pm.getResources("nobody"), []);
  }

  _level("Level 1 \u2014 Basic Permissions", level1);

  function level2(): void {
    const pm = new PermissionManager();

    // inheritance
    pm.grant("alice", "/docs", "write");
    _check("inherit to child", pm.check("alice", "/docs/team"), "write");
    _check("inherit to grandchild", pm.check("alice", "/docs/team/q1"), "write");

    // override with more specific
    pm.grant("alice", "/docs/team", "read");
    _check("specific overrides parent", pm.check("alice", "/docs/team"), "read");
    _check("grandchild uses specific", pm.check("alice", "/docs/team/q1"), "read");
    _check("parent unchanged", pm.check("alice", "/docs"), "write");

    // no inheritance upward
    const pm2 = new PermissionManager();
    pm2.grant("bob", "/docs/team", "admin");
    _check("no upward inherit", pm2.check("bob", "/docs"), "none");
    _check("child access", pm2.check("bob", "/docs/team"), "admin");

    // unrelated path
    _check("unrelated path", pm.check("alice", "/photos"), "none");

    // sibling not affected
    pm.grant("alice", "/docs/personal", "admin");
    _check("sibling independent", pm.check("alice", "/docs/personal"), "admin");
    _check("team still read", pm.check("alice", "/docs/team"), "read");
  }

  _level("Level 2 \u2014 Hierarchical Resources", level2);

  function level3(): void {
    const pm = new PermissionManager();

    // create group
    _check("create group", pm.createGroup("engineers"), true);
    _check("duplicate group", pm.createGroup("engineers"), false);

    // add to group
    _check("add to group", pm.addToGroup("alice", "engineers"), true);
    _check("add to nonexistent", pm.addToGroup("bob", "ghosts"), false);

    // group permission
    pm.grant("@engineers", "/code", "write");
    _check("group grant", pm.check("alice", "/code"), "write");

    // direct overrides group
    pm.grant("alice", "/code", "read");
    _check("direct overrides group", pm.check("alice", "/code"), "read");

    // highest group wins
    pm.createGroup("admins");
    pm.addToGroup("carol", "engineers");
    pm.addToGroup("carol", "admins");
    pm.grant("@admins", "/code", "admin");
    _check("highest group wins", pm.check("carol", "/code"), "admin");

    // non-member unaffected
    _check("non-member no access", pm.check("dave", "/code"), "none");

    // group with hierarchy
    pm.grant("@engineers", "/docs", "read");
    _check("group inherits down", pm.check("alice", "/docs/specs"), "read");
  }

  _level("Level 3 \u2014 Groups", level3);

  function level4(): void {
    const pm = new PermissionManager();

    // audit log
    pm.grant("alice", "/docs", "read");
    pm.grant("bob", "/docs", "write");
    pm.revoke("alice", "/docs");
    pm.grant("alice", "/docs", "admin");
    _check("audit log", pm.getAuditLog("/docs"), [
      "GRANT alice read",
      "GRANT bob write",
      "REVOKE alice",
      "GRANT alice admin",
    ]);

    // audit log for different resource
    _check("empty audit log", pm.getAuditLog("/photos"), []);

    // group grants in audit log
    pm.createGroup("team");
    pm.addToGroup("carol", "team");
    pm.grant("@team", "/shared", "read");
    _check("group in audit", pm.getAuditLog("/shared"), ["GRANT @team read"]);

    // effectivePermissions
    const pm2 = new PermissionManager();
    pm2.grant("alice", "/docs", "write");
    pm2.grant("bob", "/docs", "read");
    pm2.grant("carol", "/docs/team", "admin");
    _check("effective on /docs", pm2.effectivePermissions("/docs"), {
      alice: "write",
      bob: "read",
    });

    // effective with inheritance
    _check("effective with inherit", pm2.effectivePermissions("/docs/team"), {
      alice: "write",
      bob: "read",
      carol: "admin",
    });

    // effective with groups
    const pm3 = new PermissionManager();
    pm3.createGroup("devs");
    pm3.addToGroup("dan", "devs");
    pm3.addToGroup("eve", "devs");
    pm3.grant("@devs", "/repo", "read");
    pm3.grant("eve", "/repo", "admin");
    _check("effective with groups", pm3.effectivePermissions("/repo"), {
      dan: "read",
      eve: "admin",
    });

    // effective empty resource
    _check("effective no perms", pm3.effectivePermissions("/nothing"), {});
  }

  _level("Level 4 \u2014 Audit Trail", level4);
}

function main(): void {
  console.log("\nPermission Manager\n");
  _runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) {
    console.log("All tests passed.");
  }
}

main();
