#!/usr/bin/env bash
# Sequential slow batches with a hard post cap. Usage: PREFIX=data/raw/live-20261020-round4 MAX_POSTS=30 run_collection_batches.sh START "kw kw kw" ...
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PREFIX=${PREFIX:?set PREFIX}; MAX_POSTS=${MAX_POSTS:?set MAX_POSTS}
count() { ./.venv/Scripts/python.exe -c "import json,glob;print(sum(len(json.load(open(p,encoding='utf-8'))['posts']) for p in glob.glob('$PREFIX-batch*/report.json')))"; }
i=$1; shift
for kws in "$@"; do
  used=$(count); left=$((MAX_POSTS - used))
  if [ "$left" -le 0 ]; then echo "CAP REACHED: $used/$MAX_POSTS"; exit 0; fi
  per=3; [ "$left" -lt 3 ] && per=$left
  name=$(printf "batch%02d" "$i"); out=$PREFIX-$name
  if [ -e "$out/report.json" ]; then echo "skip $name"; i=$((i+1)); continue; fi
  echo "=== $name $(date +%H:%M:%S) used=$used max-posts=$per keywords: $kws"
  # shellcheck disable=SC2086
  ./.venv/Scripts/python.exe scripts/collect_candidates.py --no-filters --max-posts $per --max-comments 10 \
    --min-interval 15 --max-interval 25 --keywords $kws --output-dir "$out" \
    --permission-evidence data/reviewed/permissions/school-declaration-20261019.json 2>&1 | grep -E '^\{"(search|reading|id)'
  rc=${PIPESTATUS[0]}; echo "=== $name exit=$rc"
  if [ "$rc" -ne 0 ]; then echo "STOP: $name failed"; exit "$rc"; fi
  i=$((i+1)); pause=$((60 + RANDOM % 30)); echo "pause ${pause}s"; sleep "$pause"
done
echo "ALL DONE used=$(count)/$MAX_POSTS"
