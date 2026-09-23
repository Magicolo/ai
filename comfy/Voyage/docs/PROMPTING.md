# PROMPTING — style charter, novelty, staged prompts

## Style charter

`voyage init --style "..."` sets the run's permanent visual identity
(`config.style`). It is injected into **every** video prompt, stage
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
concept with its MiniLM embedding. A proposal whose cosine similarity to
history exceeds `novelty_threshold = 0.85` is rejected with feedback and
the director retries (bounded attempts, then the best candidate wins).
Revisits are off by default (`allow_concept_revisit = false`).

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
