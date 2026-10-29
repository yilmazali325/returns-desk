#!/bin/zsh
# Final measured runs: each is a fresh adapter process that starts (and stops) its own fresh
# llama-server, one server on the machine at a time. F on port 8440, B on 8441.
cd /Users/ali/Desktop/Projects/retail-returns-desk
R=${OUTDIR:-var/tuning2/final}; mkdir -p $R
run() {  # run <name> <profile> <input>
  local PP=$([ "$2" = "B" ] && echo 8441 || echo 8440)
  uptime > $R/$1.load.txt
  .venv/bin/python adapter.py --input $3 --output $R/$1.json --profile $2 --port $PP > $R/$1.log 2>&1
  .venv/bin/python candidate_assets/validate_submission.py --input $3 --output $R/$1.json > $R/$1.validate.txt 2>&1
  echo "$(date +%H:%M:%S) $1 $(tail -1 $R/$1.validate.txt)" >> $R/progress.log
}
PUB=candidate_assets/public_cases.json; VAR=evals/variants/variants_v1.json; HOLD=evals/holdout/holdout_v2.json
ADV=tests/review/business/cases/adversarial_v1.json; PROBE=evals/holdout/probes/probes.json
for k in 1 2 3; do run F_pub_run$k F $PUB; done
run F_var F $VAR
run F_hold F $HOLD
run F_adv F $ADV
run F_probe F $PROBE
run F_live2 F tests/review/verify_redteam/live2.json
run F_live3 F tests/review/verify_redteam/live3.json
run B_pub_run1 B $PUB
run B_var B $VAR
run B_hold B $HOLD
echo "$(date +%H:%M:%S) all done" >> $R/progress.log
