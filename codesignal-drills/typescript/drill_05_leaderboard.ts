/*
Drill 05 — Leaderboard

Implement a Leaderboard class with score management, rankings,
team grouping, and score history.

────────────────────────────────────────
Level 1 — Basic Scores

  addScore(playerId: string, score: number): void
    Set a player's score. Overwrites if player already exists.

  getScore(playerId: string): number | null
    Returns the player's score, or null if not found.

  removePlayer(playerId: string): boolean
    Removes the player. Returns true if removed, false if not found.

  getPlayerCount(): number
    Returns the number of players on the leaderboard.

────────────────────────────────────────
Level 2 — Rankings

  getTop(n: number): string[]
    Returns up to n player ids, sorted by score descending.
    Ties broken alphabetically by id.

  getRank(playerId: string): number | null
    Returns the 1-based rank of a player.
    Ties broken alphabetically by id (sequential ranks).
    Returns null if the player is not found.

  getPlayersInRange(minScore: number, maxScore: number): string[]
    Returns player ids with scores in [minScore, maxScore] inclusive.
    Sorted by score descending, then id ascending.

────────────────────────────────────────
Level 3 — Teams

  addPlayerToTeam(playerId: string, team: string): boolean
    Assign a player to a team. A player can only be on one team.
    Reassigns if already on a team.
    Returns false if the player doesn't exist.

  getTeamScore(team: string): number | null
    Returns the sum of all scores for players on this team.
    Returns null if no players are on this team.

  getTeamRanking(): string[]
    Returns team names sorted by total score descending,
    then name ascending for ties.

────────────────────────────────────────
Level 4 — Score History

  getScoreHistory(playerId: string): number[]
    Returns all scores ever set for this player, in order.
    Returns [] if never added.

  revertScore(playerId: string): boolean
    Reverts to the previous score (or removes the player
    if only one score was ever set).
    Returns false if no history exists for the player.
    Removes the reverted score from history.
*/

type Player = {
  playerId: string;
  score: number;
  history: number[];
  team?: string;
}

export class Leaderboard {
  players: Map<string, Player>;
  constructor() {
    this.players = new Map();
  }

  addScore(playerId: string, score: number): void {
    // dont clear history & team.
    let player = this.players.get(playerId);
    if (!player) {
      player = {
        playerId,
        score,
        history: [],
      }
    }
    player.score = score;
    player.history.push(score);
    this.players.set(playerId, player);
  }
  // REVIEW: Looks good. Correctly preserves history/team on overwrite.
  // Minor: the `score` field is set twice for new players (init + line 93).
  // You could omit `score` from the initial object and let line 93 handle it.

  getScore(playerId: string): number | null {
    const player = this.players.get(playerId);
    if (!player) return null;
    return player.score;
  }
  // REVIEW: Clean, no issues.

  removePlayer(playerId: string): boolean {
    if (!this.players.has(playerId)) return false
    this.players.delete(playerId);
    return true;
  }
  // REVIEW: Works. Missing semicolon on the `return false` line.

  getPlayerCount(): number {
    return this.players.size;
  }
  // REVIEW: Clean, no issues.

  getTop(n: number): string[] {
    const allPlayers = this._getSortedPlayers();
    const topPlayers = allPlayers.slice(0, n);
    return topPlayers.map(player => player.playerId);
  }
  // REVIEW: Works. Could simplify by chaining:
  //   return this._getSortedPlayers().slice(0, n).map(p => p.playerId);

  getRank(playerId: string): number | null {
    const allPlayers = this._getSortedPlayers();
    let rank;
    const player = allPlayers.find((player, index) => {
      if (player.playerId === playerId) {
        rank = index + 1;
        return true;
      }
    });
    return rank || null;
  }
  // REVIEW: Works but has a subtle edge case — `rank || null` would
  // return null if rank were 0, but since ranks are 1-based that's
  // fine here. However, `rank ?? null` would be more intentional.
  // Also, the `player` variable is assigned but never used.
  // Consider: `const idx = allPlayers.findIndex(p => p.playerId === playerId);`
  //           `return idx === -1 ? null : idx + 1;`

  getPlayersInRange(minScore: number, maxScore: number): string[] {
    const allPlayers = this._getSortedPlayers();
    const playersInRange = allPlayers.filter((player) => {
      return player.score >= minScore && player.score <= maxScore;
    });
    return playersInRange.map(player => player.playerId);
  }
  // REVIEW: Correct. Same chaining opportunity as getTop.

  addPlayerToTeam(playerId: string, team: string): boolean {
    const player = this.players.get(playerId);
    if (!player) return false;
    player.team = team;
    return true;
  }
  // REVIEW: Clean, no issues.

  getTeamScore(team: string): number | null {
    let score = 0;
    this.players.forEach(player => {
      if (player.team === team) {
        score += player.score;
      }
    });

    return score || null;
  }
  // REVIEW: BUG — `score || null` returns null when score is 0.
  // If all players on a team have score 0, this returns null instead of 0.
  // Fix: track a `found` boolean, or count matching players, and return
  // `found ? score : null` to distinguish "team exists with score 0"
  // from "team doesn't exist".

  getTeamRanking(): string[] {
    const teams: Set<string> = new Set();
    this.players.forEach(player => {
      if (player.team) {
        teams.add(player.team);
      }
    });
    const teamsArray = Array.from(teams);
    teamsArray.sort((a, b) => {
      const scoreA = this.getTeamScore(a)!;
      const scoreB = this.getTeamScore(b)!;
      return scoreA === scoreB ? a.localeCompare(b) : scoreB - scoreA;
    });
    return teamsArray;
  }
  // REVIEW: Works. getTeamScore is called O(t) times inside sort
  // (once per comparison), so scores are recomputed repeatedly.
  // For a drill this is fine, but in production you'd precompute
  // a Map<string, number> of team scores first.

  getScoreHistory(playerId: string): number[] {
    const player = this.players.get(playerId);
    return player?.history || [];
  }
  // REVIEW: Correct. Note this returns a reference to the internal
  // array — callers could mutate it. Fine for a drill, but worth
  // knowing (a defensive copy would be `[...player.history]`).

  revertScore(playerId: string): boolean {
    const player = this.players.get(playerId);
    if (!player) return false;
    player.history.pop();
    if (player.history.length === 0) {
      this.players.delete(playerId);
      return true;
    }
    player.score =  player.history[player.history.length - 1];
    return true;
  }
  // REVIEW: BUG (fixed) — previously returned false when removing
  // the player after popping the last history entry. Should return true
  // because a revert did occur. Also: extra space before `player.history`
  // on the score assignment line.

  _getSortedPlayers(): Player[] {
    const allPlayers = Array.from(this.players.values());
    allPlayers.sort((a, b) => {
      return a.score === b.score ? a.playerId.localeCompare(b.playerId) : b.score - a.score;
    });
    return allPlayers;
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
  level("Level 1 — Basic Scores", () => {
    const s1 = new Leaderboard();
    s1.addScore("alice", 100);
    s1.addScore("bob", 200);
    s1.addScore("carol", 150);
    check("get", s1.getScore("alice"), 100);
    check("get missing", s1.getScore("dave"), null);
    s1.addScore("alice", 300);
    check("overwrite", s1.getScore("alice"), 300);
    check("count", s1.getPlayerCount(), 3);
    check("remove", s1.removePlayer("bob"), true);
    check("remove missing", s1.removePlayer("bob"), false);
    check("count after", s1.getPlayerCount(), 2);
  });

  level("Level 2 — Rankings", () => {
    const s2 = new Leaderboard();
    s2.addScore("alice", 100);
    s2.addScore("bob", 200);
    s2.addScore("carol", 200);
    s2.addScore("dave", 50);
    check("top 2", s2.getTop(2), ["bob", "carol"]);
    check("top all", s2.getTop(10), ["bob", "carol", "alice", "dave"]);
    check("rank", s2.getRank("bob"), 1);
    check("rank 3", s2.getRank("alice"), 3);
    check("rank missing", s2.getRank("eve"), null);
    check("range", s2.getPlayersInRange(100, 200), ["bob", "carol", "alice"]);
    check("range empty", s2.getPlayersInRange(300, 400), []);
  });

  level("Level 3 — Teams", () => {
    const s3 = new Leaderboard();
    s3.addScore("alice", 100);
    s3.addScore("bob", 200);
    s3.addScore("carol", 150);
    check("assign", s3.addPlayerToTeam("alice", "red"), true);
    check("assign 2", s3.addPlayerToTeam("bob", "red"), true);
    check("assign 3", s3.addPlayerToTeam("carol", "blue"), true);
    check("assign missing", s3.addPlayerToTeam("dave", "red"), false);
    check("team score", s3.getTeamScore("red"), 300);
    check("team score 2", s3.getTeamScore("blue"), 150);
    check("team missing", s3.getTeamScore("green"), null);
    check("team ranking", s3.getTeamRanking(), ["red", "blue"]);
  });

  level("Level 4 — Score History", () => {
    const s4 = new Leaderboard();
    s4.addScore("alice", 10);
    s4.addScore("alice", 20);
    s4.addScore("alice", 30);
    check("history", s4.getScoreHistory("alice"), [10, 20, 30]);
    check("revert", s4.revertScore("alice"), true);
    check("after revert", s4.getScore("alice"), 20);
    check("history after", s4.getScoreHistory("alice"), [10, 20]);
    check("revert again", s4.revertScore("alice"), true);
    check("after revert 2", s4.getScore("alice"), 10);
    check("revert last", s4.revertScore("alice"), true);
    check("fully reverted", s4.getScore("alice"), null);
    check("no more", s4.revertScore("alice"), false);
    check("empty history", s4.getScoreHistory("alice"), []);
    check("never added", s4.getScoreHistory("bob"), []);
  });
}

function main(): void {
  console.log("\nLeaderboard\n");
  runSelfChecks();
  const total = _passed + _failed;
  console.log(`\n${_passed}/${total} passed`);
  if (_failed === 0 && total > 0) console.log("All tests passed.");
}

main();
