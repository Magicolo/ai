# 102 — Finalize publish defects: unescaped concat paths, whole-file RAM staging, /tmp staging outside the §53 preflight fs

**Severity:** MEDIUM

**File:line (verified live 2026-09-30):**
- `voyage/media.py:1006-1011` (fast-path concat list: `f"file '{segment / 'video.mp4'}'\n"`, `-safe 0`)
- `voyage/media.py:1071-1072` (re-encode-path concat list: `f"file '{part}'\n"`, `-safe 0`)
- `voyage/media.py:1044` (fast path publish: `atomic_write_bytes(output_path, staged.read_bytes())`)
- `voyage/media.py:1113-1115` (re-encode path publish: same `staged.read_bytes()` pattern)
- `voyage/media.py:977` (`with tempfile.TemporaryDirectory() as tmp:` — bare: no `dir=`, no `prefix=`)
- `voyage/media.py:920` (`check_free_space(run_dir, min_free_space_gib)` — the §53 preflight measures the **run** fs)
- Related prior art (explicitly NOT re-filed here, cross-referenced): `Voyage/issues/043_final_publish_reads_whole_mp4.md` (the `read_bytes` RAM staging, rated HIGH, with the `atomic_copy` fix design) and `Voyage/issues/053_concat_quoting_and_ffmpeg_hygiene.md` (concat quoting + `-nostdin` hygiene).

**Description:**
This issue tracks the finalize-publish defects as one publish-path review because they share a root cause (the staging/publish half of `finalize_run` never got the hardening the commit path did), but two of the three legs already have owning issues — read the cross-refs first; the novel leg here is (3).

1. **Concat lists interpolate raw paths into single quotes (both paths).** Lines 1008-1010 and 1072 build `file '<path>'` lines with `-safe 0` and no escaping. The ffmpeg concat demuxer requires a `'` inside a quoted filename to be escaped as `'\''`; a run dir, segment name, or (via `video.mp4` under a user `--run-id`-derived path) any component containing `'` produces a malformed list and `final concat copy failed` / `final encode failed` (`MediaError`, lines 1041/1112). Proven live failure class with `'` in path (see evidence). Owning issue: **053** — this leg is listed here only so the publish path reads complete; do not double-fix.

2. **Whole-file `staged.read_bytes()` into RAM (both paths + SFX remux).** Lines 1044/1115 materialize the entire staged final before `atomic_write_bytes`, which copies it again into its own temp file — 2-3× the final's size in host RAM transiently. Multi-hundred-segment voyages at 1280×720@32 h264 stage hundreds of MB; the chunked/streaming finalize work elsewhere in the tree exists precisely because all-at-once doesn't scale. Owning issue: **043** (with the `atomic_copy` chunked-copy design) — listed here only for completeness; do not double-fix.

3. **Staging dir is bare `TemporaryDirectory()` → `$TMPDIR`/`/tmp`, while the §53 preflight measures the run fs (NOVEL — no owning issue).** Line 977 passes neither `dir=` nor even `prefix=` (every other `TemporaryDirectory` in the tree passes a `prefix="voyage-…"` — `media.py:291`, `cli.py:1455`, `sfx_finalize.py:493`, all worker bench/take sites). Consequences: (a) the free-space preflight at line 920 (`check_free_space(run_dir, …)`) validates the wrong filesystem — `/tmp` is routinely tmpfs (RAM-backed, typically half of host RAM) or a small root partition, so a run can pass preflight and then die mid-finalize with `ENOSPC` (or wedge the host by filling tmpfs) while staging per-segment parts + the concat + the final; (b) cross-filesystem `os.replace`/copy at publish time instead of a same-dir atomic rename; (c) unidentifiable `tmpXXXXXX` dirs in `/tmp` during forensics instead of greppable `voyage-*` names. The finalize staging holds the largest transient bytes in the whole pipeline (per-segment re-encoded parts + final), i.e. exactly the workload a preflight must cover.

**Rationale:**
Legs (1) and (2) are recorded here solely as pointers so the next reader of `finalize_run` sees the full publish-path risk in one place. Leg (3) is the live gap: disk-full handling is otherwise careful in this tree (`DiskSpaceError → PAUSED_DISK_FULL`, resumable; preflight "must never let a final metadata write be the thing that discovers a full disk", `media.py:61-72`), yet the finalize path stages gigabytes on a filesystem nobody measured. A tmpfs `/tmp` also converts a disk-space problem into a host-RAM problem — the worst possible exchange on a 16 GB GPU box where VRAM-adjacent host memory is already the binding constraint.

**Live evidence (current tree):**
```
media.py:1006-1011 (fast path):
    concat_list = tmpdir / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{segment / 'video.mp4'}'\n" for segment in usable),
        encoding="utf-8",
    )
media.py:1071-1072 (re-encode path):
    concat_list = tmpdir / "concat.txt"
    concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
media.py:977:    with tempfile.TemporaryDirectory() as tmp:      # bare — cf. prefix= everywhere else
media.py:920:        check_free_space(run_dir, min_free_space_gib)  # measures run fs, not $TMPDIR
media.py:1044 / 1115: atomic_write_bytes(output_path, staged.read_bytes())
```
Quoting probe (host stdlib, exact primitive):
```
$ python3 -c "p=\"/tmp/voyage o'brien/segments/000000/video.mp4\"; ..."
concat line:    file '/tmp/voyage o'brien/segments/000000/video.mp4'      <- demuxer parse error
ffmpeg-correct: file '/tmp/voyage o'\''brien/segments/000000/video.mp4'
```
`rg TemporaryDirectory voyage/media.py` → lines 291 (`prefix="voyage-assemble-"`) and 977 (bare): the finalize staging is the only site without even a prefix. `df /tmp` vs `df <run_dir>` on typical hosts shows different filesystems — preflight and staging disagree by construction.

**Repro (CPU, no GPU):**
1. Quoting: `init --run-id "o'brien"` (or rename any run dir to contain `'`), commit ≥1 segment with `fake` backends, `finalize` → `MediaError: final concat copy failed` (fast path) — ffmpeg chokes on the unescaped quote. (Owned by 053; repro recorded here for path-completeness.)
2. RAM staging: finalize a long fake run while sampling RSS — peak holds ~2-3× the final mp4. (Owned by 043.)
3. Wrong-fs staging (novel): set `TMPDIR` to a tiny tmpfs (e.g. `mount -t tmpfs -o size=16m tmpfs /tmp/vsmall; TMPDIR=/tmp/vsmall finalize`), with the run dir on a roomy fs — preflight at line 920 passes, then part-encoding dies with `ENOSPC` under `$TMPDIR`. Or invert: fill `/tmp` to near-full and watch a healthy-space run fail finalize despite passing preflight.

**Fix candidates:**
- (1) Escape `'` as `'\''` in both concat writers (or route both through one shared helper; 053's design). Do not implement here — 053 owns it.
- (2) `atomic_copy` chunked publish (`shutil.copyfileobj` 1 MiB + flush/fsync/replace/`fsync_dir`); 043 owns the design. Do not implement here — 043 owns it.
- (3, novel — implement here): `TemporaryDirectory(prefix="voyage-final-", dir=<run_dir/"tmp"|output parent>)` so staging lives on the preflight-measured filesystem and publishes via same-dir `os.replace`; `mkdir -p` the staging parent first (note: same-dir staging leaves `.partial`-style temp names out of `validate_run`'s orphan scan scope — check the scan's patterns when choosing the dir, or clean the staging dir explicitly on success/failure).
- Test: finalize with `TMPDIR` pointed at a 16 MB tmpfs and a roomy run dir → passes after the fix; concat unit test with `'`/space/`$()` paths (coordinate with 053 to avoid duplicate tests).

**Refs:**
- Overlaps with 043 (leg 2 owner — RAM staging, `atomic_copy` design) and 053 (leg 1 owner — concat quoting); do not double-fix legs 1-2, novel leg here is (3). Merge recommendation: keep three files, ownership as noted — or fold legs 1-2 here and close pointers.
- Owning issues: `Voyage/issues/043_final_publish_reads_whole_mp4.md` (RAM staging, HIGH, `atomic_copy` design) and `Voyage/issues/053_concat_quoting_and_ffmpeg_hygiene.md` (quoting + `-nostdin` hygiene, with the `o'brien` repro shape).
- In-tree: `voyage/media.py:61-72` (`check_free_space` contract), `:857-930` (`finalize_run` head + preflight), `:977-1116` (staging + both publish paths); `voyage/atomic.py:37-58` (publish pattern the fix should reuse).
- ffmpeg concat demuxer: quoted filenames escape `'` as `'\''` (see https://ffmpeg.org/ffmpeg-formats.html#concat); `-safe 0` (used at lines 1020-1021/1085-1086) only relaxes path-safety checks, it does not change quoting rules.
