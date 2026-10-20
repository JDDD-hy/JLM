#!/usr/bin/env bash
# Re-run corpus_progress with the previous input manifest plus any new review_form paths given as args.
set -e
cd "$(dirname "$0")/.."
prev=$(./.venv/Scripts/python.exe -c "import json;print(' '.join(i['path'] for i in json.load(open('data/reviewed/pool-20261019/progress.json',encoding='utf-8'))['inputs']))")
new=""
for p in "$@"; do case " $prev " in *" $p "*) ;; *) new="$new $p";; esac; done
./.venv/Scripts/python.exe scripts/corpus_progress.py --inputs $prev $new --output-dir data/reviewed/pool-20261019 \
  | ./.venv/Scripts/python.exe -c "import json,sys;d=json.load(sys.stdin);print({k:d[k] for k in ('unique_candidates','remaining_to_1000','training_eligible','exact_duplicates','near_duplicate_pairs','unknown_scene_candidates')});print(dict(sorted(d['provisional_scene_counts'].items(),key=lambda x:int(x[0]))))"
