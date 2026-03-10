# CodeSignal-Style Progressive Drills

Practice drills modeled after the CodeSignal Industry Coding Framework.

Each drill gives you a class to implement with methods that build on each other across 4 levels. The self-check harness instantiates your class and calls methods directly — just like the real assessment.

## How it works

1. Read the class spec at the top of the file.
2. Implement the class methods level by level.
3. Run the file. The self-checks call your methods and report pass/fail.
4. Fix failures and repeat until `All self-checks passed.`

## Drill overview

| # | Class | Skills tested |
|---|-------|---------------|
| 01 | **FileStorage** | Hash maps, sorting, capacity constraints, undo stack |
| 02 | **KeyValueStore** | CRUD, prefix scanning, nested transactions, history |
| 03 | **FeatureFlagService** | Layered state resolution, group membership, snapshots |
| 04 | **TaskScheduler** | Priority ordering, tag indexing, dependency graphs, completion history |
| 05 | **Leaderboard** | Ranking, sorted queries, team aggregation, score history |
| 06 | **InventoryTracker** | Stock constraints, category indexing, threshold alerts, change history |

## Level structure

- **Level 1:** Core methods — get the basics right
- **Level 2:** New methods that build on Level 1
- **Level 3:** Constraints/complexity that affect existing methods
- **Level 4:** History, undo, or rollback — the stretch goal

## Recommended cadence

- Warmup: 75–90 minutes (target all 4 levels)
- Interview mode: 45–60 minutes (target Levels 1–3)

## Recommended learning order

If you're learning Python alongside challenge strategy:

1. Do all drills in TypeScript first (`typescript/`)
2. Re-do the same drills in Python (`python/`)

## Project layout

```
typescript/
  drill_01_file_storage.ts
  drill_02_key_value_store.ts
  drill_03_feature_flag_service.ts
  drill_04_task_scheduler.ts
  drill_05_leaderboard.ts
  drill_06_inventory_tracker.ts
python/
  drill_01_file_storage.py
  drill_02_key_value_store.py
  drill_03_feature_flag_service.py
toggle-codesignal-mode.sh
COMMANDS.md
```

## Where to write code

Each file has a class with TODO method stubs. Implement those methods.
Do not edit the self-check harness at the bottom of each file.

**TypeScript:**
- `drill_01_file_storage.ts` — implement `FileStorage` class
- `drill_02_key_value_store.ts` — implement `KeyValueStore` class
- `drill_03_feature_flag_service.ts` — implement `FeatureFlagService` class
- `drill_04_task_scheduler.ts` — implement `TaskScheduler` class
- `drill_05_leaderboard.ts` — implement `Leaderboard` class
- `drill_06_inventory_tracker.ts` — implement `InventoryTracker` class

**Python:**
- `drill_01_file_storage.py` — implement `FileStorage` class
- `drill_02_key_value_store.py` — implement `KeyValueStore` class
- `drill_03_feature_flag_service.py` — implement `FeatureFlagService` class

## Run

```bash
# TypeScript
npx tsx codesignal-drills/typescript/drill_01_file_storage.ts
npx tsx codesignal-drills/typescript/drill_02_key_value_store.ts
npx tsx codesignal-drills/typescript/drill_03_feature_flag_service.ts
npx tsx codesignal-drills/typescript/drill_04_task_scheduler.ts
npx tsx codesignal-drills/typescript/drill_05_leaderboard.ts
npx tsx codesignal-drills/typescript/drill_06_inventory_tracker.ts

# Python
python3 codesignal-drills/python/drill_01_file_storage.py
python3 codesignal-drills/python/drill_02_key_value_store.py
python3 codesignal-drills/python/drill_03_feature_flag_service.py
```
