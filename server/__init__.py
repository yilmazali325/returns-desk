"""The Returns Desk web UI: a FastAPI server (app.py) and a vanilla-JS single page (static/).

The server holds no business rules. It wraps the same `desk.session.CaseSession` the adapter
uses, streams every audit entry to the browser while a turn runs (Server-Sent Events), and
exposes the failure-lab controls (external stock change, timeout after the next commit).
Run it with `scripts/run.sh` (starts the model too) or `python -m server.app --help`.
"""
