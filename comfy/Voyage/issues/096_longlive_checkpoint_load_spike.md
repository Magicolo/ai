# 096 — LongLive2 checkpoint load spikes to ~38 GB anon RSS, OOM-killed on a shared box

- Status: open (blocks the longlive2 excerpt; 2026-09-29 probe evidence below)
- Severity: high (backend unusable unless ~40 GB host RAM is free at worker-init time)
- Area: checkpoint load (`voyage/workers/video_longlive.py:705 load_generator_container`
  → `torch.load(..., map_location="cpu", weights_only=True)` on the 10 GB `model_bf16.pt`)
- Rank rationale: turns a 16 GB-card backend into a 40 GB-host-RAM backend;
  deterministic SIGKILL with no traceback, so every reporter re-investigates from zero.

## Technical description

`LongLiveSession` init verifies the manifest hash, prints
`loading generator checkpoint ...`, then `torch.load`s the 9.99 GB
`longlive2/model_bf16.pt` to CPU. The worker subprocess dies at that
exact line (stderr ends there, supervisor reports `closed stdout`, no
traceback, no error reply). Manual repro with the serve-map handshake
(`init` + `latent_shape [1,8,48,44,80]`, fp8, local 16/sink 8, `-u`,
`--user 1000:1000`, `--gpus all`) exits **RC=137 (SIGKILL)**.

The kernel OOM-killer record (`journalctl`, 2026-09-29 14:25) names the
victim: `python3 total-vm:62620504kB, anon-rss:39629176kB` — **~37.8 GB
anonymous RSS for a 10 GB file (~3.8x)**. No cgroup limit is set
(`memory.max: max`, docker `Total Memory: 61.62GiB`); this is a global
OOM (`constraint=CONSTRAINT_NONE`) on a box with 28 GB free, so the
spike itself is the bug, not the box.

## Why this is an issue

- The 2026-09-24 qual run (`qual-longlive2`, VALID 87f) used the same
  file and code path — it survived on an emptier box + swap. Any
  concurrent load (other agents' MMAudio/gates containers) makes
  longlive2 init a coin flip, and the failure mode (silent SIGKILL,
  supervisor `closed stdout`) misdirects every investigation toward
  box contention first (that cost this probe ~2 h: b4-era contention
  was real, but the quiet-box retry died identically).
- A plain `torch.load` of the same file in a CPU-only container as the
  same uid succeeds, so it is not permissions, not the `--user`
  migration, not a corrupt file — it is peak anon during the load.

## Live evidence

- `voyage/workers/video_longlive.py:715-717` (load path; uncommitted
  hunks in this file are 045/015 scope — none touch the load).
- `~/.cache/voyage-models/longlive2/model_bf16.pt` 9999853030 B.
- Image `voyage-video:latest` torch `2.8.0+cu128` (rebuilt 2026-09-29
  by another agent; torch version unchanged from qual).
- OOM record quoted above; `dmesg`/`journalctl -u systemd-oomd` show
  the kill, `docker inspect` cannot (supervisor-level death; the
  container exits cleanly afterwards).

## Repro

```bash
# manual worker init (fails, RC=137, stderr ends at "loading generator checkpoint ...")
docker run --rm -e PYTHONUNBUFFERED=1 -e VOYAGE_LONGLIVE_DIR=/tmp/ll-tree \
  -w /app --user=$(id -u):$(id -g) --gpus all \
  -v $PWD:/app -v /tmp:/tmp -v ~/.cache/voyage-models:/models \
  voyage-video:latest bash -c 'echo "{\"id\":\"1\",\"op\":\"init\",\"payload\":{\
  \"models_dir\":\"/models\",\"device\":\"cuda:0\",\
  \"latent_shape\":[1,8,48,44,80],\"quantization\":\"fp8\",\
  \"local_attn_size\":16,\"sink_size\":8}}" \
  | timeout 240 python3 -u -m voyage.workers.video_longlive 2>/tmp/ll.err \
  | head -c 300; echo "RC=${PIPESTATUS[1]}"'
```

(`VOYAGE_LONGLIVE_DIR=/tmp/ll-tree` with host symlink
`wan_models -> /models/wan_models` is the `--user` workaround for the
root-owned `/opt/longlive` checkout — orthogonal to this issue.)

## Fix candidates

1. `torch.load(..., mmap=True)` in `load_generator_container` (torch 2.x
   supports mmap+weights_only; storages stay file-backed until touched —
   kills the unpickle-copy spike if the spike is buffer+storage duplication).
2. `torch.load` with `mmap_location`/`LazyUnpickler` + immediate
   `share_memory_()`? No — single process; mmap alone should do.
3. If mmap is insufficient, stream-load: read file once via
   `torch.serialization` storage-level API, or shard the container.
4. Regression test: cap-addressable probe is hard in CI (10 GB file);
   at minimum assert `mmap=` kwarg presence + a small-file load-peak
   property (peak anon < 2x file size for a synthetic container).
5. Short-term docs: note the ~40 GB host-RAM requirement at longlive2
   init in `docs/BACKENDS.md` until fixed.
