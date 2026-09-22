# Voyage — Phase 0 skeleton

Autonomous infinite audiovisual voyage. Full design: `Voyage/DESIGN.md`.

## Quick start (all inside Docker — no host installs)

```bash
./scripts/build.sh     # build image + run gates inside it
./scripts/test.sh      # pytest inside the container
./scripts/gates.sh     # ruff + mypy + pytest inside the container

# Run a 2-segment fake voyage:
./scripts/run.sh init --output /tmp/demo --run-id demo \
  --style "pastel neon line-art, peaceful, slow cinematic motion" --force
./scripts/run.sh run --run /tmp/demo --segments 2
./scripts/run.sh validate --run /tmp/demo
./scripts/run.sh finalize --run /tmp/demo --output /tmp/demo/final.mp4
```

## Layout

- `voyage/` — supervisor package (config, state, RPC, workers, media, CLI).
  Never imports torch/transformers/diffusers (DESIGN §83).
- `tests/` — pytest suite, runs in-container.
- `scripts/` — build/test/gates/run helpers (thin docker wrappers).
