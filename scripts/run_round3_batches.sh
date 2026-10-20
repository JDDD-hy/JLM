#!/usr/bin/env bash
# Sequential slow batches; stops at first non-zero exit. Usage: run_round3_batches.sh START_INDEX "kw1 kw2 kw3" "kw..." ...
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
i=$1; shift
for kws in "$@"; do
  name=$(printf "batch%02d" "$i"); out=data/raw/live-20261019-round3-$name
  if [ -e "$out/report.json" ]; then echo "skip $name (exists)"; i=$((i+1)); continue; fi
  echo "=== $name $(date +%H:%M:%S) keywords: $kws"
  # shellcheck disable=SC2086
  ./.venv/Scripts/python.exe scripts/collect_candidates.py --no-filters --max-posts 3 --max-comments 10 \
    --min-interval 15 --max-interval 25 --keywords $kws --output-dir "$out" \
    --permission-evidence data/reviewed/permissions/school-declaration-20261019.json 2>&1 | grep -E '^\{"(search|reading|id)'
  rc=${PIPESTATUS[0]}; echo "=== $name exit=$rc"
  if [ "$rc" -ne 0 ]; then echo "STOP: $name failed"; exit "$rc"; fi
  i=$((i+1))
  pause=$((60 + RANDOM % 30)); echo "pause ${pause}s"; sleep "$pause"
done
echo "ALL DONE"
