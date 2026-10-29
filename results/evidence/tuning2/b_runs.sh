#!/bin/zsh
# B re-measurement on the frozen code (only B's no-prefetch lookup path changed after final3's F runs).
cd /Users/ali/Desktop/Projects/retail-returns-desk
R=var/tuning2/final3
run() {
  uptime > $R/$1.load.txt
  .venv/bin/python adapter.py --input $3 --output $R/$1.json --profile $2 --port 8441 > $R/$1.log 2>&1
  .venv/bin/python candidate_assets/validate_submission.py --input $3 --output $R/$1.json > $R/$1.validate.txt 2>&1
  echo "$(date +%H:%M:%S) $1 $(tail -1 $R/$1.validate.txt)" >> $R/progress.log
}
echo "$(date +%H:%M:%S) B re-run after the lookup fix (B files below replace the earlier B entries)" >> $R/progress.log
run B_pub_run1 B candidate_assets/public_cases.json
run B_var B evals/variants/variants_v1.json
run B_hold B evals/holdout/holdout_v2.json
echo "$(date +%H:%M:%S) B done" >> $R/progress.log
