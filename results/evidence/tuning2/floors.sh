#!/bin/zsh
# usage: floors.sh <outdir>   code-only floors (no model) on every suite with the current code
cd /Users/ali/Desktop/Projects/retail-returns-desk
OUT=$1; mkdir -p $OUT
for mode in always-clarify fake-policy; do
  for s in pub var hold adv; do
    case $s in
      pub) IN=candidate_assets/public_cases.json;; var) IN=evals/variants/variants_v1.json;;
      hold) IN=evals/holdout/holdout_v2.json;; adv) IN=tests/review/business/cases/adversarial_v1.json;;
    esac
    .venv/bin/python -m evals.holdout.run_stub --mode $mode --input $IN --output $OUT/${mode}_${s}.json > $OUT/${mode}_${s}.log 2>&1
    .venv/bin/python var/tuning2/summarize.py $s $OUT/${mode}_${s}.json | head -1
  done
done
