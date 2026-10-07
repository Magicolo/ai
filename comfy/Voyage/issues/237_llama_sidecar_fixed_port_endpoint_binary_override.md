# 237 — Llama sidecar: fixed port 8080, loopback claim vs endpoint-controlled port, arbitrary-binary env override

Severity: MEDIUM (track F-11).

## Technical description

Three compounding choices in `voyage/llama_server.py`: (a) default port 8080 fixed
(`LLAMA_SERVER_PORT`), colliding with any host/container service on 8080 — second
`generate` or a squatting process (AGENTS notes a "comatose foreign process squatting the
sidecar port" killed an enhancer A/B) fails or talks to the wrong server; (b) bind is
always `127.0.0.1` but the readiness probe + worker client connect to the caller-supplied
`llama_endpoint` host — a non-loopback endpoint is probed/used while the docstring claims
"never leaves the box"; (c) `VOYAGE_LLAMA_SERVER_BIN` accepts any absolute path as the
server binary with no allowlist/checksum (`server_binary()`, `llama_server.py:102-105`).

## Rationale

Port collision / traffic misdirection; binary substitution.

## Live evidence

```
voyage/llama_server.py:47-51 (HOST/PORT constants), 113-126 (port_for_endpoint: any 1-65535 from endpoint),
129-145 (visible_devices_for: cpu/garbage → None = inherit visibility),
240-253 (Popen(command, env=child_env) — child_env None inherits whole parent env),
64-65/102-105 (VOYAGE_LLAMA_SERVER_BIN override)
```

Repro: `VOYAGE_LLAMA_SERVER_BIN=/tmp/evil-llama ./scripts/run.sh generate <name>` →
supervisor spawns attacker binary; two concurrent `generate` on one box → second
`start()` probes the first's 8080.

## Source refs

`voyage/llama_server.py:47-65, 102-105, 113-145, 213-265`.

## Online sources

- Docker/container least-privilege (fixed ports + PATH binaries as trust boundaries).
- In-tree GPU-contention rule (shared-box collisions are already observed).

## Fix candidates

- Ephemeral port (`port=0` + read-back) or per-run port file + `flock`; validate
  `llama_endpoint` hostname ∈ {127.0.0.1, localhost} fail-closed; resolve the binary once
  at build (`/opt/llama.cpp/bin/llama-server` digest) and require the env override to match
  an allowlisted sha or refuse.

## Log

- 2026-10-07: filed from read-only Track F sweep; no code touched.

## Evaluation (2026-10-07)

Live check confirmed all three sub-findings in
`voyage/llama_server.py`: `server_binary()` returned any absolute env
path unchecked; `port_for_endpoint()` parsed any host while the bind
stayed loopback; first-start (`Supervisor._start_llama_sidecar`) used
the fixed endpoint port with no `resolve_port` (only the heal path
resolved). `resolve_port`/`find_free_port`/`is_port_in_use` already
existed, so the missing pieces were validation + allowlist + a
race-free claim. A start-side auto-resolve alone was rejected as
incorrect: the director payload routes on the *config* endpoint, so a
silently remapped spawn port would misroute without supervisor
propagation (supervisor is out of scope here).

## Progress log

- 2026-10-07 (`voyage/llama_server.py` only): added
  `LLAMA_ALLOWED_HOSTS` (`127.0.0.1`, `localhost`) +
  `validate_endpoint_host()` fail-closed, wired into
  `port_for_endpoint()` (active immediately on the supervisor path);
  added `LLAMA_SERVER_BAKED_PATHS` allowlist (both baked install
  paths) with `server_binary()` refusing anything else (the issue's
  `/tmp/evil-llama` repro now raises); added `port_file_for()` +
  `claim_sidecar_port()` (shared-lock probe + per-run port record).
- 2026-10-07: new `tests/test_issue_237_llama_hardening.py` (8 tests:
  host validation, port-parse fail-closed, binary default/allow/
  refuse, port-file shape, claim free/occupied).

## Resolution (2026-10-07)

Resolved in scope: hostname fail-closed and binary allowlist are live
(no caller change needed); the port-collision half lands as the tested
`claim_sidecar_port()` helper. Open follow-up (supervisor scope):
wire claim → `start(port=claimed)` → route director traffic at the
claimed endpoint in `_start_llama_sidecar`, and validate
`llama_endpoint` in the director worker client too.
