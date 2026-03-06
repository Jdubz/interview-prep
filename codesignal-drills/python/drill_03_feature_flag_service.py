"""
Drill 03 — Feature Flag Service

Implement a FeatureFlagService class with global toggles,
user overrides, group rules, and snapshots.

────────────────────────────────────────
Level 1 — Global Flags

  enable(flag) -> None
    Enable a flag globally.

  disable(flag) -> None
    Disable a flag globally.

  is_enabled(flag) -> bool
    Returns whether the flag is enabled. Defaults to False.

────────────────────────────────────────
Level 2 — User Overrides

  enable_for_user(flag, user_id) -> None
    Enable a flag for a specific user.

  disable_for_user(flag, user_id) -> None
    Disable a flag for a specific user.

  is_enabled_for_user(flag, user_id) -> bool
    If user has an override for this flag, return the override.
    Otherwise fall back to the global state.

────────────────────────────────────────
Level 3 — Groups

  add_user_to_group(user_id, group) -> None
    Add a user to a group. A user can be in multiple groups.

  enable_for_group(flag, group) -> None
    Enable a flag for an entire group.

  disable_for_group(flag, group) -> None
    Disable a flag for an entire group.

  Resolution order for is_enabled_for_user:
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

  snapshot(name) -> None
    Save the entire current state under this name.
    Overwrites if the name already exists.

  restore(name) -> bool
    Restore state from the named snapshot.
    Returns False if the name does not exist.

  list_snapshots() -> list[str]
    Returns all snapshot names, sorted alphabetically.
"""

from __future__ import annotations


class FeatureFlagService:
    def __init__(self) -> None:
        # TODO: initialize your data structures
        pass

    def enable(self, flag: str) -> None:
        raise NotImplementedError("TODO: implement enable")

    def disable(self, flag: str) -> None:
        raise NotImplementedError("TODO: implement disable")

    def is_enabled(self, flag: str) -> bool:
        raise NotImplementedError("TODO: implement is_enabled")

    def enable_for_user(self, flag: str, user_id: str) -> None:
        raise NotImplementedError("TODO: implement enable_for_user")

    def disable_for_user(self, flag: str, user_id: str) -> None:
        raise NotImplementedError("TODO: implement disable_for_user")

    def is_enabled_for_user(self, flag: str, user_id: str) -> bool:
        raise NotImplementedError("TODO: implement is_enabled_for_user")

    def add_user_to_group(self, user_id: str, group: str) -> None:
        raise NotImplementedError("TODO: implement add_user_to_group")

    def enable_for_group(self, flag: str, group: str) -> None:
        raise NotImplementedError("TODO: implement enable_for_group")

    def disable_for_group(self, flag: str, group: str) -> None:
        raise NotImplementedError("TODO: implement disable_for_group")

    def snapshot(self, name: str) -> None:
        raise NotImplementedError("TODO: implement snapshot")

    def restore(self, name: str) -> bool:
        raise NotImplementedError("TODO: implement restore")

    def list_snapshots(self) -> list[str]:
        raise NotImplementedError("TODO: implement list_snapshots")


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
        f1 = FeatureFlagService()
        _check("default off", f1.is_enabled("dark_mode"), False)
        f1.enable("dark_mode")
        _check("enabled", f1.is_enabled("dark_mode"), True)
        f1.disable("dark_mode")
        _check("disabled", f1.is_enabled("dark_mode"), False)
    _level("Level 1 \u2014 Global Flags", level_1)

    def level_2():
        f2 = FeatureFlagService()
        f2.enable("new_ui")
        f2.disable_for_user("new_ui", "user1")
        _check("user override off", f2.is_enabled_for_user("new_ui", "user1"), False)
        _check("fallback global", f2.is_enabled_for_user("new_ui", "user2"), True)
        f2.enable_for_user("new_ui", "user1")
        _check("user override on", f2.is_enabled_for_user("new_ui", "user1"), True)
    _level("Level 2 \u2014 User Overrides", level_2)

    def level_3():
        f3 = FeatureFlagService()
        f3.add_user_to_group("alice", "beta")
        f3.add_user_to_group("alice", "staff")
        f3.enable_for_group("experiment", "beta")
        _check("group enabled", f3.is_enabled_for_user("experiment", "alice"), True)
        _check("not in group", f3.is_enabled_for_user("experiment", "bob"), False)
        f3.disable_for_user("experiment", "alice")
        _check("user beats group", f3.is_enabled_for_user("experiment", "alice"), False)

        f3b = FeatureFlagService()
        f3b.enable("feature_x")
        f3b.add_user_to_group("carol", "internal")
        f3b.disable_for_group("feature_x", "internal")
        _check("group off beats global", f3b.is_enabled_for_user("feature_x", "carol"), False)
        _check("no group uses global", f3b.is_enabled_for_user("feature_x", "dave"), True)
    _level("Level 3 \u2014 Groups", level_3)

    def level_4():
        f4 = FeatureFlagService()
        f4.enable("flag_a")
        f4.enable("flag_b")
        f4.snapshot("v1")
        f4.disable("flag_a")
        _check("after disable", f4.is_enabled("flag_a"), False)
        _check("restore", f4.restore("v1"), True)
        _check("restored", f4.is_enabled("flag_a"), True)
        _check("other flag", f4.is_enabled("flag_b"), True)
        _check("missing snapshot", f4.restore("nope"), False)
        _check("list", f4.list_snapshots(), ["v1"])
    _level("Level 4 \u2014 Snapshots", level_4)


def main() -> None:
    print("\nFeature Flag Service\n")
    _run_self_checks()
    total = _passed + _failed
    print(f"\n{_passed}/{total} passed")
    if _failed == 0 and total > 0:
        print("All tests passed.")


if __name__ == "__main__":
    main()
