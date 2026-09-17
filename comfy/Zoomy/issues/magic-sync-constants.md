# Magic sync constants duplicate `OUTPUT_SAMPLE_RATE`

- Severity: low (drift risk between SFX slice math and mux rate).
- Status: FIXED. `SOUND_EFFECT_SYNC_FRAMES_PER_SECOND = 25` (documented
  as unrelated to the coincidental `SOUND_EFFECT_STEPS = 25`) and
  `SOUND_EFFECT_SYNC_FRAME_PIXELS = 224` named in `engine_protocol.py`;
  both slice lines, both transform sizes, and the `soundfile.write` rate
  (now `final_assembly.OUTPUT_SAMPLE_RATE`) use them. Test:
  `test_sound_effect_sync_constants_are_named_once` pins all three values
  cross-module. The `1e-5` epsilon stays vendor-mirror inline, as noted.

## Evidence

```python
sync_frames = torch.stack(...)[: int(25 * duration_seconds)]
duration_seconds = sync_frames.shape[0] / 25.0
...
soundfile.write(str(stem_path), waveform[0].T.numpy(), 44100)
```

vs `final_assembly.OUTPUT_SAMPLE_RATE = 44100`. The literal `25` (sync
fps) appears twice with no name; the literal `44100` duplicates the
assembly constant. If the output rate ever changes, the SFX slice math
silently disagrees with the mux. The `224`px sync resize/crop
(`local_engine.py:1159-1162`) is docstring-only with no constant — same
class, smaller blast radius. The `1e-5` batch-norm epsilon
(`local_engine.py`, loader section) is vendor-mirror inline — acceptable
as-is, noted only.

## Fix

Import `OUTPUT_SAMPLE_RATE` from `final_assembly` for the `soundfile.write`
call; name the sync fps (`SOUND_EFFECT_SYNC_FRAMES_PER_SECOND = 25` next
to the other `SOUND_EFFECT_*` constants in `engine_protocol.py`) and use
it in both slice lines; name the `224`px sync size similarly.

## Verification

- `grep -rn "44100\|([^0-9]25[* /])" zoomy/` shows only the constant
  definitions; behavior unchanged (SFX e2e or stubbed test green).
- Gates: `Zoomy/scripts/quality-gates.sh` green.
