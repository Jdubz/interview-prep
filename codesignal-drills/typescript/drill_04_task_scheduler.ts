/*
Drill 04 — Task Scheduler

Implement a TaskScheduler class with task management, priority
ordering, dependencies, and completion history.

────────────────────────────────────────
Level 1 — Basic Tasks

  addTask(taskId: string, priority: number): boolean
    Add a task with given priority.
    Returns false if taskId already exists.

  getTask(taskId: string): { id: string; priority: number; done: boolean } | null
    Returns the task, or null if not found.

  complete(taskId: string): boolean
    Mark a task as done.
    Returns false if not found or already done.

  getPendingCount(): number
    Returns the number of uncompleted tasks.

────────────────────────────────────────
Level 2 — Priority Ordering

  getNext(): string | null
    Returns the id of the highest-priority pending task.
    Higher number = higher priority.
    Ties broken alphabetically by id.
    Returns null if no pending tasks.

  listPending(): string[]
    Returns all pending task ids, sorted by priority descending,
    then id ascending for ties.

  addTag(taskId: string, tag: string): boolean
    Add a tag to a task. Returns false if task not found.

  getByTag(tag: string): string[]
    Returns all pending task ids with this tag,
    sorted alphabetically.

────────────────────────────────────────
Level 3 — Dependencies

  addDependency(taskId: string, dependsOnId: string): boolean
    Declare that taskId cannot complete until dependsOnId is done.
    Returns false if either task doesn't exist or
    if taskId === dependsOnId.

  getReadyTasks(): string[]
    Returns pending tasks whose dependencies are all completed,
    sorted by priority descending, then id ascending.

  complete() now also returns false if the task has
  any uncompleted dependencies.

────────────────────────────────────────
Level 4 — Completion History

  undoComplete(taskId: string): boolean
    Revert a completed task to pending.
    Returns false if not found or not completed.

  getCompletionOrder(): string[]
    Returns task ids in the order they were completed.
    Undone tasks are removed from this list.
*/

type Task = {
  id: string;
  priority: number;
  completedAt?: Date;
  tags?: Set<string>;
  dependencyIds?: Set<string>;
};

export class TaskScheduler {
  tasks: Map<string, Task>;

  constructor() {
    this.tasks = new Map();
  }

  addTask(taskId: string, priority: number): boolean {
    if (this.tasks.has(taskId)) return false;
    this.tasks.set(taskId, { id: taskId, priority });
    return true;
  }

  getTask(taskId: string): { id: string; priority: number; done: boolean } | null {
    const task = this.tasks.get(taskId);
    return task ? { id: task.id, priority: task.priority, done: !!task.completedAt } : null;
  }

  complete(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || task.completedAt) return false;
    let incompleteDeps = false;
    if (task.dependencyIds) {
      task.dependencyIds.forEach(depId => {
        const dep = this.tasks.get(depId)!;
        if (!dep.completedAt) {
          incompleteDeps = true;
        }
      });
    }
    if (incompleteDeps) return false;
    task.completedAt = new Date();
    return true;
  }

  getPendingCount(): number {
    let pending = 0;
    this.tasks.forEach((task) => {
      if (!task.completedAt) {
        pending += 1;
      }
    })
    return pending;
  }

  getNext(): string | null {
    let highestPriority = 0;
    this.tasks.forEach((task) => {
      if (!task.completedAt && task.priority > highestPriority) {
        highestPriority = task.priority;
      }
    });
    const pending: Task[] = [];
    this.tasks.forEach((task) => {
      if (!task.completedAt && task.priority === highestPriority) {
        pending.push(task);
      }
    });
    if (!pending.length) return null;
    if (pending.length > 1) {
      pending.sort((a,b) => a.id > b.id ? 1 : -1);
    }
    return pending[0].id;
  }
  // Review: Two passes over `this.tasks` isn't necessary — you could
  // reuse listPending() and return the first element:
  //   const pending = this.listPending();
  //   return pending.length ? pending[0] : null;
  //
  // Also, initializing highestPriority to 0 would break for negative
  // priorities (e.g. priority -1 would never be found).

  listPending(): string[] {
    const pending: Task[] = [];
    this.tasks.forEach((task) => {
      if (!task.completedAt) {
        pending.push(task);
      }
    });
    pending.sort((a,b) => {
      const order = b.priority - a.priority;
      if (order !== 0) return order;
      return a.id > b.id ? 1 : -1;
    });
    return pending.map(task => task.id);
  }

  addTag(taskId: string, tag: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task) return false;
    if (!task.tags) {
      task.tags = new Set();
    }
    task.tags.add(tag);
    return true;
  }

  getByTag(tag: string): string[] {
    const tagCollection: Task[] = [];
    this.tasks.forEach(task => {
      if (task.tags && task.tags.has(tag)) {
        tagCollection.push(task);
      }
    });
    tagCollection.sort((a,b) => a.id > b.id ? 1 : -1);
    return tagCollection.map(task => task.id);
  }
  // Review: The spec says getByTag returns PENDING tasks with the tag,
  // but this returns all tasks (including completed ones). The tests
  // happen to pass because completed tasks don't have the tag in the
  // test data. Add `&& !task.completedAt` to the filter to be correct.

  addDependency(taskId: string, dependsOnId: string): boolean {
    if (!this.tasks.has(taskId) || !this.tasks.has(dependsOnId) || taskId === dependsOnId) return false;
    const task = this.tasks.get(taskId)!;
    if (!task.dependencyIds) {
      task.dependencyIds = new Set();
    }
    task.dependencyIds.add(dependsOnId);
    return true;
  }

  getReadyTasks(): string[] {
    const ready: Task[] = [];
    this.tasks.forEach(task => {
      if (!task.completedAt) {
        let incompleteDeps = false;
        if (task.dependencyIds) {
          task.dependencyIds.forEach(depId => {
            const dep = this.tasks.get(depId)!;
            if (!dep.completedAt) {
              incompleteDeps = true;
            }
          });
        }
        if (!incompleteDeps) {
          ready.push(task);
        }
      }
    });
    ready.sort((a,b) => {
      const order = b.priority - a.priority;
      if (order !== 0) return order;
      return a.id > b.id ? 1 : -1;
    });
    return ready.map(task => task.id);
  }
  // Review: The dependency-check logic is duplicated between complete()
  // and getReadyTasks(). Extract a helper:
  //   private hasUnmetDeps(task: Task): boolean {
  //     if (!task.dependencyIds) return false;
  //     for (const depId of task.dependencyIds) {
  //       if (!this.tasks.get(depId)?.completedAt) return true;
  //     }
  //     return false;
  //   }
  // Using `for...of` with early return is also cleaner than forEach
  // with a mutable flag.

  undoComplete(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || !task.completedAt) return false;
    delete task.completedAt;
    return true;
  }

  getCompletionOrder(): string[] {
    const completed: Task[] = [];
    this.tasks.forEach(task => {
      if (task.completedAt) {
        completed.push(task);
      }
    });
    completed.sort((a,b) => a.completedAt!.getTime() - b.completedAt!.getTime());
    return completed.map(task => task.id);
  }
  // Review: Using Date timestamps is clever, but tasks completed within
  // the same millisecond would have an unstable sort order. A simple
  // incrementing counter (completionOrder: number) or an array would be
  // more reliable and avoids the Date object overhead.
  //
  // Overall: Clean, well-structured solution. The Task type is a good
  // design choice. Main areas to tighten up:
  //   1. getByTag should filter out completed tasks
  //   2. Extract the dependency-check into a shared helper
  //   3. getNext can delegate to listPending
  //   4. Sort comparators should use localeCompare() instead of > / <
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
  level("Level 1 — Basic Tasks", () => {
    const s1 = new TaskScheduler();
    check("add", s1.addTask("deploy", 3), true);
    check("add dup", s1.addTask("deploy", 5), false);
    check("get priority", s1.getTask("deploy")?.priority, 3);
    check("get done", s1.getTask("deploy")?.done, false);
    check("get missing", s1.getTask("nope"), null);
    s1.addTask("test", 2);
    s1.addTask("build", 1);
    check("pending", s1.getPendingCount(), 3);
    check("complete", s1.complete("deploy"), true);
    check("complete again", s1.complete("deploy"), false);
    check("complete missing", s1.complete("nope"), false);
    check("pending after", s1.getPendingCount(), 2);
  });

  level("Level 2 — Priority Ordering", () => {
    const s2 = new TaskScheduler();
    s2.addTask("beta", 2);
    s2.addTask("alpha", 2);
    s2.addTask("gamma", 5);
    s2.addTask("delta", 1);
    check("next", s2.getNext(), "gamma");
    s2.complete("gamma");
    check("next tie", s2.getNext(), "alpha");
    check("list pending", s2.listPending(), ["alpha", "beta", "delta"]);
    check("add tag", s2.addTag("alpha", "urgent"), true);
    s2.addTag("delta", "urgent");
    check("tag missing", s2.addTag("nope", "x"), false);
    check("by tag", s2.getByTag("urgent"), ["alpha", "delta"]);
    check("tag empty", s2.getByTag("nope"), []);
  });

  level("Level 3 — Dependencies", () => {
    const s3 = new TaskScheduler();
    s3.addTask("compile", 3);
    s3.addTask("test", 2);
    s3.addTask("deploy", 1);
    check("add dep", s3.addDependency("test", "compile"), true);
    check("add dep 2", s3.addDependency("deploy", "test"), true);
    check("dep missing", s3.addDependency("test", "nope"), false);
    check("dep self", s3.addDependency("test", "test"), false);
    check("blocked", s3.complete("test"), false);
    check("ready", s3.getReadyTasks(), ["compile"]);
    s3.complete("compile");
    check("unblocked", s3.complete("test"), true);
    check("ready after", s3.getReadyTasks(), ["deploy"]);
  });

  level("Level 4 — Completion History", () => {
    const s4 = new TaskScheduler();
    s4.addTask("a", 1);
    s4.addTask("b", 2);
    s4.addTask("c", 3);
    s4.complete("b");
    s4.complete("c");
    check("order", s4.getCompletionOrder(), ["b", "c"]);
    check("undo", s4.undoComplete("b"), true);
    check("order after", s4.getCompletionOrder(), ["c"]);
    check("undo missing", s4.undoComplete("nope"), false);
    check("undo pending", s4.undoComplete("a"), false);
    check("pending restored", s4.getTask("b")?.done, false);
  });
}

function main(): void {
  console.log("\nTask Scheduler\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
