# Shared technical starter assets

Use these files alongside the candidate brief. They provide identical synthetic inputs, disclosed rules and an evaluation adapter contract. They do not contain a working retail backend, agent implementation, private tests or a full scoring harness. Build the mock business tools and application yourself.

- `CONTRACT.md`: required adapter, state schema, policies and fault semantics.
- `public_cases.json`: 6 public scenarios with complete resettable initial states.
- `validate_submission.py`: standard-library structural checker. It does not verify business correctness.

Run your adapter:

`python adapter.py --input public_cases.json --output results.json`

Check its output:

`python validate_submission.py --input public_cases.json --output results.json`

Optionally add `--baseline baseline_results.json` to check that the input hash, hardware declaration and case IDs match a prior run on the same device. This is a consistency check, not proof of an actual benchmark run or of comparable model quality. Preserve raw logs and configuration for review.

The 6 public cases exercise a refund at the 30-day boundary, an exchange, an expired return, clarification, stock changes before approval, and a timeout after commit followed by replay. Private variants may change values, request wording and eligible outcomes only within the disclosed schema and policies. Do not hardcode these cases.
