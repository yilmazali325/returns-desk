"""Test doubles that run the real desk without a model (never used by the adapter or by default).

    fake_policy_llm.py   FakePolicyLLM: a deterministic, rule-based stand-in for the local LLM.
                         The UI server uses it only when DESK_LLM=fake-policy (E2E tests, demos
                         on a machine without the model files).
"""
