"""Returns Desk: a local agentic returns/exchanges desk.

The LLM agents propose; the deterministic core in this package (policy, store, tools)
disposes. Business code never reads the clock or uses randomness: dates come from
``state["as_of"]`` and IDs from per-session counters, so every run is reproducible.
"""
