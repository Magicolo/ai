# PROMPTING — style charter, novelty, staged prompts

## Captions (video / music / SFX)

The director carries three caption families — per-block video stages,
a music caption for ACE-Step takes, and an SFX caption for MMAudio
windows — all derived from the same concept + charter and evolved
gradually with the general prompt. Full operator reference (pins
`--music-caption/--video-caption/--sfx-caption`, finalize windows,
ladder): `docs/SFX.md`.

## Style charter

`voyage generate --name <id> --style "..."` sets the run's permanent visual identity
(`config.style`, stored in the run manifest). It is injected into **every** video prompt, stage
prompt, and director proposal by code — never left to the model to
remember. Changing it mid-run changes the voyage's identity; prefer a
new run for a new look.

## Transition prompts

The director proposes a `destination_concept` and intermediate stages
between the current concept and the destination. Stage texts become the
per-block video prompts (`blocks_per_prompt_stage = 3` blocks share one
stage prompt), so a shot evolves gradually instead of jumping.

## Novelty system

`ConceptStore` (`voyage/concepts.py`, `novelty/` dir) keeps every accepted
concept with its MiniLM embedding. Since 2026-10-01 novelty is
steer-and-accept — it never rejects: the director prompt carries a
`NOVELTY STEERING` section (visited worlds listed under
`FORBIDDEN CONCEPT SUMMARY`, omitted entirely when revisits are allowed)
nudging the destination slightly away from history, every generation is
scored (`novelty_scored` metric: score vs `novelty_threshold = 0.85`,
embeddings on/fallback), and the first schema/style-valid generation
renders — a revisit simply carries `novelty_accepted = false`
(`novel`/`hold` in the console). Bounded retries still apply to
schema/style failures only (exhaustion falls back to the deterministic
director). Revisits are off by default (`allow_concept_revisit = false`);
the pre-2026-10-01 reject-and-retry regime is retired (pinned by
`tests/test_novelty_steer_accept.py`).

## Staged prompt design

`build_staged_prompt_plan` maps blocks → stage prompts → transition
texts so each of the N blocks in a segment knows exactly which prompt
and seed it renders (`video_seed(config.seed, number, block)` —
sequential draws continue one noise trajectory across blocks/segments).
Scene cuts fire only on destination change (new shot → upstream cut
prefix, zero-KV + sink re-pin).

## Why style is injected by code

Model proposals drift: an LLM left to itself rephrases, drops, or
"improves" the style within a few segments. Code-level injection
(style charter prepended to every prompt, schema-validated decisions,
`ProposalRejected` on violation) makes the charter load-bearing instead
of advisory. The visual inspector's `feedback_amendments` apply the same
way — post-validation, pre-style-check, never as free text into a prompt.
