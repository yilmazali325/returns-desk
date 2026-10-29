# Offline proof

Written by `scripts/offline_check.sh --port 8406 --profile F` on 2026-10-29T08:36:34Z
(macOS 26.4.1 25E253, Apple M5 Pro).
Every command below ran exactly as shown; outputs are verbatim (long outputs trimmed to their last 40 lines).

## Sandbox profile (`scripts/offline.sb`)

```scheme
(version 1)
;; Offline sandbox for scripts/offline_check.sh (macOS sandbox-exec, Sandbox Profile Language).
;;
;;   sandbox-exec -f scripts/offline.sb <command ...>
;;
;; Everything is allowed except the network. Outbound IP connections may only go to loopback (the
;; adapter talks to its own llama-server on 127.0.0.1), inbound connections are accepted only on
;; loopback, and the DNS resolver's socket is closed so no name lookup can leave the machine
;; either (lookups go through mDNSResponder, a process outside this sandbox). Child processes
;; (the .venv re-exec, llama-server, ps/pmset/ioreg) inherit the profile. In SBPL the last
;; matching rule wins, so each allow below narrows the deny before it.
(allow default)
(deny network-outbound (remote ip "*:*"))
(allow network-outbound (remote ip "localhost:*"))
(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))
(deny network-inbound)
(allow network-inbound (local ip "localhost:*"))
```

Environment for every sandboxed command: HF_HUB_OFFLINE=1 UV_OFFLINE=1.

## 0. Positive control outside the sandbox (informational)

### TCP connect to 1.1.1.1:443 without the sandbox

```
$ /usr/bin/nc -z -G 3 1.1.1.1 443
Connection to 1.1.1.1 port 443 [tcp/https] succeeded!
```

Result: exit 0

### HTTPS to huggingface.co without the sandbox

```
$ /usr/bin/curl -sS -o /dev/null -w 'HTTP %{http_code}\n' --max-time 10 https://huggingface.co
HTTP 200
```

Result: exit 0

The machine is online, so the failures below are caused by the sandbox.

## 1. Negative controls inside the sandbox (must all fail)

### TCP connect to 1.1.1.1:443 inside the sandbox

```
$ sandbox-exec -f scripts/offline.sb /usr/bin/nc -z -G 3 1.1.1.1 443

```

Result: exit 1: blocked as required

### TCP connect to 1.1.1.1:443 from the project's Python inside the sandbox

```
$ sandbox-exec -f scripts/offline.sb .venv/bin/python -c 'import socket; socket.create_connection(("1.1.1.1", 443), 3); print("connected")'
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ali/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/socket.py", line 865, in create_connection
    raise exceptions[0]
  File "/Users/ali/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/socket.py", line 850, in create_connection
    sock.connect(sa)
PermissionError: [Errno 1] Operation not permitted
```

Result: exit 1: blocked as required

### HTTPS to huggingface.co by name inside the sandbox

```
$ sandbox-exec -f scripts/offline.sb /usr/bin/curl -sS -o /dev/null --max-time 10 https://huggingface.co
curl: (6) Could not resolve host: huggingface.co
```

Result: exit 6: blocked as required

### HTTPS to huggingface.co with DNS pre-resolved outside (13.35.196.43) inside the sandbox

```
$ sandbox-exec -f scripts/offline.sb /usr/bin/curl -sS -o /dev/null --max-time 10 --resolve huggingface.co:443:13.35.196.43 https://huggingface.co
curl: (7) Failed to connect to huggingface.co port 443 after 0 ms: Couldn't connect to server
```

Result: exit 7: blocked as required


## 2. Full public run inside the sandbox

Command (adapter.py re-execs into .venv/bin/python; llama-server is its child, so both run sandboxed):

```
$ sandbox-exec -f scripts/offline.sb python3 adapter.py --profile F --port 8406 --input candidate_assets/public_cases.json --output results/offline/results.json
```

Adapter exit code: 0, wall time 33 s. Adapter log:

```
[adapter] public_01_refund_boundary: ['propose', 'completed'] (5780 ms, ok)
[adapter] public_02_exchange: ['propose', 'completed'] (4924 ms, ok)
[adapter] public_03_expired: ['deny'] (2587 ms, ok)
[adapter] public_04_clarify: ['clarify', 'propose', 'completed'] (5931 ms, ok)
[adapter] public_05_stale_stock: ['propose', 'deny'] (5221 ms, ok)
[adapter] public_06_timeout_replay: ['propose', 'completed', 'completed'] (5173 ms, ok)
[adapter] wrote 6 case results to results/offline/results.json
```

### Sockets of the adapter and llama-server during the run (lsof, taken outside the sandbox)

```
# snapshot 1 at +2 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)

# snapshot 2 at +4 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 67466  ali    4u  IPv4 0xa1beaeaa39104969      0t0  TCP 127.0.0.1:55902->127.0.0.1:8406 (ESTABLISHED)
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)
llama-ser 67482  ali    8u  IPv4 0x82e6f9e7a1e4ee89      0t0  TCP 127.0.0.1:8406->127.0.0.1:55902 (ESTABLISHED)

# snapshot 3 at +6 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 67466  ali    4u  IPv4 0xa1beaeaa39104969      0t0  TCP 127.0.0.1:55902->127.0.0.1:8406 (ESTABLISHED)
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)
llama-ser 67482  ali    8u  IPv4 0x82e6f9e7a1e4ee89      0t0  TCP 127.0.0.1:8406->127.0.0.1:55902 (ESTABLISHED)

# snapshot 4 at +8 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 67466  ali    4u  IPv4 0xa1beaeaa39104969      0t0  TCP 127.0.0.1:55902->127.0.0.1:8406 (ESTABLISHED)
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)
llama-ser 67482  ali    8u  IPv4 0x82e6f9e7a1e4ee89      0t0  TCP 127.0.0.1:8406->127.0.0.1:55902 (ESTABLISHED)

# snapshot 5 at +10 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 67466  ali    4u  IPv4 0xa1beaeaa39104969      0t0  TCP 127.0.0.1:55902->127.0.0.1:8406 (ESTABLISHED)
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)
llama-ser 67482  ali    8u  IPv4 0x82e6f9e7a1e4ee89      0t0  TCP 127.0.0.1:8406->127.0.0.1:55902 (ESTABLISHED)

# snapshot 6 at +12 s (adapter pid 67466, llama-server pid 67482)
COMMAND     PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 67466  ali    4u  IPv4 0xa1beaeaa39104969      0t0  TCP 127.0.0.1:55902->127.0.0.1:8406 (ESTABLISHED)
llama-ser 67482  ali    3u  IPv4 0x20b5d486bc9f095d      0t0  TCP 127.0.0.1:8406 (LISTEN)
llama-ser 67482  ali    8u  IPv4 0x82e6f9e7a1e4ee89      0t0  TCP 127.0.0.1:8406->127.0.0.1:55902 (ESTABLISHED)

```

Result: 6 snapshot(s); every socket is on 127.0.0.1.

## 3. Validation and scoring (inside the sandbox)

### Structural validator

```
$ sandbox-exec -f scripts/offline.sb .venv/bin/python candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output results/offline/results.json
STRUCTURE CHECK PASSED: 6 cases. Business correctness was not scored.
```

Result: exit 0: ok

### Independent oracle score

```
$ sandbox-exec -f scripts/offline.sb .venv/bin/python -m evals.score --input candidate_assets/public_cases.json --labels evals/expected/public_expected.json --output results/offline/results.json --json results/offline/score.json
Strict case success: 6/6 = 100.0% (Wilson 95% CI 61.0%-100.0%)
Lenient (documented alternatives accepted): 6/6
Turn decision accuracy: 100.0% of 13 turns
Invariant violations: 0 in 0 case(s)

family                             cases strict lenient turn-acc
public                                 6      6       6     100%
```

Result: exit 0: ok


## Verdict

PASSED: the network was provably closed (negative controls failed inside the sandbox), the full public run completed inside it using only loopback sockets, and its output passes the structural validator. Strict oracle score of this run: 6/6.
