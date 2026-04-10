"""
Drill 04 — Permission Manager

Implement a PermissionManager class for Dropbox-style access control.
Access control is core to how Dash handles cross-app search results.

────────────────────────────────────────
Level 1 — Basic Permissions (10 min)

  grant(user: str, resource: str, level: str) -> None
    Grant access. Level is one of "read", "write", "admin".
    Granting replaces any existing permission for that user/resource pair.

  check(user: str, resource: str) -> str
    Return permission level, or "none" if no access.
    "admin" implies "write" which implies "read".

  revoke(user: str, resource: str) -> bool
    Remove permission. Return False if no permission existed.

  get_resources(user: str) -> list[str]
    Return resources the user has any access to, sorted alphabetically.

────────────────────────────────────────
Level 2 — Hierarchical Resources (10 min)

  Resources can be hierarchical paths:
    "/docs", "/docs/team", "/docs/team/q1"

  Permissions inherit downward: a grant on "/docs" applies to
  "/docs/team" and "/docs/team/q1".

  A more specific grant overrides a parent grant:
    grant("alice", "/docs", "write")
    grant("alice", "/docs/team", "read")
    check("alice", "/docs/team") -> "read"

  check must resolve the most specific applicable permission.

────────────────────────────────────────
Level 3 — Groups (10 min)

  create_group(group: str) -> bool
    Create a group. Return False if it already exists.

  add_to_group(user: str, group: str) -> bool
    Add user to group. Return False if group doesn't exist.

  grant can accept a group name (prefixed with "@") as the user
  parameter:
    grant("@engineers", "/code", "write")

  check must resolve permissions with this priority:
    direct user permission > group permission > inherited permission

  If a user is in multiple groups with different permissions on
  the same resource, use the highest level.

────────────────────────────────────────
Level 4 — Audit Trail (15 min)

  get_audit_log(resource: str) -> list[str]
    Return chronological list of permission changes for a resource.
    Format: "GRANT user level" or "REVOKE user"
    (uses the exact user/group string passed to grant/revoke)

  effective_permissions(resource: str) -> dict[str, str]
    Return dict of {user: level} for all users who have access to
    this resource, including via groups and inheritance.
    Resolve the final effective level per user.
"""

from __future__ import annotations


class PermissionManager:
    def __init__(self) -> None:
        # TODO: initialize your data structures
        pass

    def grant(self, user: str, resource: str, level: str) -> None:
        raise NotImplementedError("TODO: implement grant")

    def check(self, user: str, resource: str) -> str:
        raise NotImplementedError("TODO: implement check")

    def revoke(self, user: str, resource: str) -> bool:
        raise NotImplementedError("TODO: implement revoke")

    def get_resources(self, user: str) -> list[str]:
        raise NotImplementedError("TODO: implement get_resources")

    def create_group(self, group: str) -> bool:
        raise NotImplementedError("TODO: implement create_group")

    def add_to_group(self, user: str, group: str) -> bool:
        raise NotImplementedError("TODO: implement add_to_group")

    def get_audit_log(self, resource: str) -> list[str]:
        raise NotImplementedError("TODO: implement get_audit_log")

    def effective_permissions(self, resource: str) -> dict[str, str]:
        raise NotImplementedError("TODO: implement effective_permissions")


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
        pm = PermissionManager()

        # no permission
        _check("no access", pm.check("alice", "/docs"), "none")

        # grant and check
        pm.grant("alice", "/docs", "read")
        _check("read access", pm.check("alice", "/docs"), "read")

        # overwrite permission
        pm.grant("alice", "/docs", "admin")
        _check("upgraded to admin", pm.check("alice", "/docs"), "admin")

        # different user has no access
        _check("bob no access", pm.check("bob", "/docs"), "none")

        # revoke
        _check("revoke success", pm.revoke("alice", "/docs"), True)
        _check("after revoke", pm.check("alice", "/docs"), "none")
        _check("revoke nonexistent", pm.revoke("alice", "/docs"), False)

        # get_resources
        pm.grant("carol", "/a", "read")
        pm.grant("carol", "/c", "write")
        pm.grant("carol", "/b", "admin")
        _check("resources sorted", pm.get_resources("carol"), ["/a", "/b", "/c"])
        _check("no resources", pm.get_resources("nobody"), [])

    _level("Level 1 \u2014 Basic Permissions", level_1)

    def level_2():
        pm = PermissionManager()

        # inheritance
        pm.grant("alice", "/docs", "write")
        _check("inherit to child", pm.check("alice", "/docs/team"), "write")
        _check("inherit to grandchild", pm.check("alice", "/docs/team/q1"), "write")

        # override with more specific
        pm.grant("alice", "/docs/team", "read")
        _check("specific overrides parent", pm.check("alice", "/docs/team"), "read")
        _check("grandchild uses specific", pm.check("alice", "/docs/team/q1"), "read")
        _check("parent unchanged", pm.check("alice", "/docs"), "write")

        # no inheritance upward
        pm2 = PermissionManager()
        pm2.grant("bob", "/docs/team", "admin")
        _check("no upward inherit", pm2.check("bob", "/docs"), "none")
        _check("child access", pm2.check("bob", "/docs/team"), "admin")

        # unrelated path
        _check("unrelated path", pm.check("alice", "/photos"), "none")

        # sibling not affected
        pm.grant("alice", "/docs/personal", "admin")
        _check("sibling independent", pm.check("alice", "/docs/personal"), "admin")
        _check("team still read", pm.check("alice", "/docs/team"), "read")

    _level("Level 2 \u2014 Hierarchical Resources", level_2)

    def level_3():
        pm = PermissionManager()

        # create group
        _check("create group", pm.create_group("engineers"), True)
        _check("duplicate group", pm.create_group("engineers"), False)

        # add to group
        _check("add to group", pm.add_to_group("alice", "engineers"), True)
        _check("add to nonexistent", pm.add_to_group("bob", "ghosts"), False)

        # group permission
        pm.grant("@engineers", "/code", "write")
        _check("group grant", pm.check("alice", "/code"), "write")

        # direct overrides group
        pm.grant("alice", "/code", "read")
        _check("direct overrides group", pm.check("alice", "/code"), "read")

        # highest group wins
        pm.create_group("admins")
        pm.add_to_group("carol", "engineers")
        pm.add_to_group("carol", "admins")
        pm.grant("@admins", "/code", "admin")
        _check("highest group wins", pm.check("carol", "/code"), "admin")

        # non-member unaffected
        _check("non-member no access", pm.check("dave", "/code"), "none")

        # group with hierarchy
        pm.grant("@engineers", "/docs", "read")
        _check("group inherits down", pm.check("alice", "/docs/specs"), "read")

    _level("Level 3 \u2014 Groups", level_3)

    def level_4():
        pm = PermissionManager()

        # audit log
        pm.grant("alice", "/docs", "read")
        pm.grant("bob", "/docs", "write")
        pm.revoke("alice", "/docs")
        pm.grant("alice", "/docs", "admin")
        _check("audit log", pm.get_audit_log("/docs"), [
            "GRANT alice read",
            "GRANT bob write",
            "REVOKE alice",
            "GRANT alice admin",
        ])

        # audit log for different resource
        _check("empty audit log", pm.get_audit_log("/photos"), [])

        # group grants in audit log
        pm.create_group("team")
        pm.add_to_group("carol", "team")
        pm.grant("@team", "/shared", "read")
        _check("group in audit", pm.get_audit_log("/shared"), ["GRANT @team read"])

        # effective_permissions
        pm2 = PermissionManager()
        pm2.grant("alice", "/docs", "write")
        pm2.grant("bob", "/docs", "read")
        pm2.grant("carol", "/docs/team", "admin")
        _check("effective on /docs", pm2.effective_permissions("/docs"), {
            "alice": "write",
            "bob": "read",
        })

        # effective with inheritance
        _check("effective with inherit", pm2.effective_permissions("/docs/team"), {
            "alice": "write",
            "bob": "read",
            "carol": "admin",
        })

        # effective with groups
        pm3 = PermissionManager()
        pm3.create_group("devs")
        pm3.add_to_group("dan", "devs")
        pm3.add_to_group("eve", "devs")
        pm3.grant("@devs", "/repo", "read")
        pm3.grant("eve", "/repo", "admin")
        _check("effective with groups", pm3.effective_permissions("/repo"), {
            "dan": "read",
            "eve": "admin",
        })

        # effective empty resource
        _check("effective no perms", pm3.effective_permissions("/nothing"), {})

    _level("Level 4 \u2014 Audit Trail", level_4)


def main() -> None:
    print("\nPermission Manager\n")
    _run_self_checks()
    total = _passed + _failed
    print(f"\n{_passed}/{total} passed")
    if _failed == 0 and total > 0:
        print("All tests passed.")


if __name__ == "__main__":
    main()
