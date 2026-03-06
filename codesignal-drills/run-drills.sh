#!/usr/bin/env bash
set -uo pipefail

DRILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DRILL_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

track="${1:-all}"  # all | ts | py | 1 | 2 | 3

ts_drills=(
  "typescript/drill_01_file_storage.ts"
  "typescript/drill_02_key_value_store.ts"
  "typescript/drill_03_feature_flag_service.ts"
)

py_drills=(
  "python/drill_01_file_storage.py"
  "python/drill_02_key_value_store.py"
  "python/drill_03_feature_flag_service.py"
)

run_drill() {
  local file="$1"
  local path="$DRILL_DIR/$file"
  local name
  name=$(basename "$file" | sed 's/\.\(ts\|py\)$//')

  if [[ "$file" == *.ts ]]; then
    output=$(cd "$ROOT_DIR" && npx tsx "$path" 2>&1)
  else
    output=$(cd "$ROOT_DIR" && python3 "$path" 2>&1)
  fi

  if echo "$output" | grep -q "All self-checks passed"; then
    echo -e "  ${GREEN}✓${NC} $name"
    return 0
  elif echo "$output" | grep -q "TODO:"; then
    local todo
    todo=$(echo "$output" | head -1)
    echo -e "  ${YELLOW}○${NC} $name — $todo"
    return 1
  else
    local fail
    fail=$(echo "$output" | head -1)
    echo -e "  ${RED}✗${NC} $name — $fail"
    return 1
  fi
}

passed=0
failed=0
total=0

run_set() {
  local label="$1"
  shift
  local files=("$@")
  echo "$label"
  for f in "${files[@]}"; do
    total=$((total + 1))
    if run_drill "$f"; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
    fi
  done
  echo ""
}

filter_by_number() {
  local num="$1"
  shift
  local files=("$@")
  for f in "${files[@]}"; do
    if [[ "$f" == *"drill_0${num}"* ]]; then
      echo "$f"
    fi
  done
}

case "$track" in
  ts)
    run_set "TypeScript" "${ts_drills[@]}"
    ;;
  py)
    run_set "Python" "${py_drills[@]}"
    ;;
  1|2|3)
    matching_ts=( $(filter_by_number "$track" "${ts_drills[@]}") )
    matching_py=( $(filter_by_number "$track" "${py_drills[@]}") )
    [[ ${#matching_ts[@]} -gt 0 ]] && run_set "TypeScript" "${matching_ts[@]}"
    [[ ${#matching_py[@]} -gt 0 ]] && run_set "Python" "${matching_py[@]}"
    ;;
  all|"")
    run_set "TypeScript" "${ts_drills[@]}"
    run_set "Python" "${py_drills[@]}"
    ;;
  *)
    echo "Usage: bash run-drills.sh [all|ts|py|1|2|3]"
    exit 1
    ;;
esac

echo "───────────────"
echo -e "${GREEN}$passed${NC} passed, ${RED}$failed${NC} remaining ($total total)"
