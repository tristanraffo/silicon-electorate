# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green before starting the next milestone (188 tests, about 35 s; no
model is loaded). The real-model smoke test is opt-in:

```bash
uv run pytest -m hf -s
```

Try the CLI (mock backend, synthetic personas, writes to `results/`):

```bash
uv run scv elicit --config configs/runs/dev_smoke.yaml --dry-run
```

Check an election config (lists TODOs, exits non-zero until it is ready):

```bash
uv run scv data check anes2024_ts
```

Then begin **M6**.

## Done

- **M1 — `ballots.py`, `rules.py`** ✅
  - `src/scv/ballots.py`: `WeakOrder`, `from_scores`, `pairwise`, `is_strict`, `is_complete`,
    `linear_extensions`, `move_up`.
  - `src/scv/rules.py`: `Profile` (also `Profile.from_ranks` / `from_scores` for arrays; cached
    `ranks`, `pairwise`, `majority`), `pairwise_matrix`, `condorcet_winner` / `condorcet_loser`,
    `has_majority_cycle`, `smith_set`, `three_cycles`, `majority_cycles`, `condorcet_order`,
    six rules via `RULES` / `run_all_rules`, `rule_disagreement`, `irv_monotonicity_probe`.
- **M2 — `domains.py`, `nulls.py`** ✅
  - `src/scv/nulls.py`: `impartial_culture`, `spatial` / `spatial_1d` / `spatial_2d`,
    `permutation_null`.
  - `src/scv/domains.py`: `all_axes`, `is_single_peaked`, `is_single_plateaued`,
    `sp_consistency`, `sp_axes`, `sp_max_consistent`, `sp_deletion_share`,
    `single_peakedness`, `value_restriction`, `dimensionality`, `diversity`.
  - Calibration: IC cycle frequency 8.82% (target 8.77% ± 0.5 pp); IC SP deletion share 0.8350.
- **M3 — backends `mock` and `hf`, `prompts.py`, `elicit.py`** ✅
  - `configs/prompts/{persona,thermometer,ranking,pairwise}_v1.yaml`.
  - `src/scv/prompts.py`: `load_templates`, `Template.render`, content hashing,
    `render_persona` (drops missing fields).
  - `src/scv/backends/`: `base.py` (`Backend`, `ChatPrompt`, `BackendError`), `mock.py`
    (`MockBackend`: hidden utilities, noise, position bias, failure rate), `hf.py`
    (`HFBackend`: chat template, thinking off, per-prompt seeds, single-token label check,
    fail-fast load errors).
  - `src/scv/elicit.py`: `ElicitConfig`, `Respondent`, `plan_calls` → `Call` (with the §11
    `key`), `execute` (yields call-log record batches; retries; sequential chains),
    `run_elicitation` (in-memory), `parse_thermometer`, `parse_ranking`,
    `thermometer_table`, `ranking_table`, `pairwise_table`, `pairwise_coherence`.
  - Acceptance: mock end-to-end recovers 100% of hidden rankings at noise 0 in all three
    modes (`tests/test_elicit.py`); tiny-model CPU smoke test passes
    (`tests/test_hf_smoke.py`, see `DECISIONS.md` for the observed numbers).

- **M4 — `runner.py`, `cli.py`** ✅
  - `src/scv/runner.py`: `CallLog` (atomic Parquet shards, `CALL_SCHEMA`), `run_calls`
    (skip logged keys, flush every N, chains re-run whole), `stratified_subsample`,
    `RunConfig` / `load_run_config`, `load_models`, `plan_run`, `dry_run`, `make_backend`,
    `run_from_config`, `read_calls`, `synthetic_respondents`.
  - `src/scv/cli.py`: `scv elicit --config … [--dry-run] [--limit-respondents K]`.
  - `configs/models/models.yaml`, `configs/runs/dev_smoke.yaml`.
  - Acceptance: `tests/test_runner.py::test_sigkill_mid_run_then_resume_…` SIGKILLs a real
    `scv elicit` process while it is writing shards, reruns the same command, and finds
    every planned key exactly once.

- **M5 — `anes.py`, data check, personas** ✅
  - `src/scv/anes.py`: `ElectionConfig`, `load_election`, `check`, `require_ready`,
    `process` (exclusion log), `build`, `load_processed`, `label_value`, `persona_texts`.
  - CLI: `scv data check <election>`, `scv data build <election>`.
  - `configs/elections/{anes2024_ts,anes2016_ts,anes2016_pilot}.yaml` (TODOs except the
    three confirmed 2024 codes), `configs/runs/prod_example.yaml`.
  - Runner: `election: {source: anes, id: …}` via `runner.resolve_election`.
  - Fixture: `tests/fixtures/anes_synthetic.{csv,yaml}` + `make_anes_fixture.py`
    (synthetic, fake `SYN_*` codes). Tests: `tests/test_anes.py`.

## What Tristan has to do before real data runs

1. Put the ANES file in `data/raw/` and set `file:` in the election YAML.
2. Fill every `TODO` (weight, thermometer codes, persona variables, exact question
   text) and write `value_labels` that read naturally in `configs/prompts/persona_v1.yaml`.
   Optionally set `id_var`. `tests/fixtures/anes_synthetic.yaml` is a complete example.
3. `uv run scv data check <election>` until it prints OK, then `uv run scv data build <election>`.

## Next

- **M6 — `vllm_backend.py` and the vast.ai runbook** (SPEC §9, §13). Acceptance: the code
  path exists and mirrors `hf`; unit tests mock vLLM; the README has copy-paste commands
  (instance image, `uv sync --extra gpu`, HF login, `--dry-run`, run, `rsync` results
  back, **destroy the instance**).
  - Mirror `src/scv/backends/hf.py`: `VLLMBackend(hf_id, dtype=…, base=…)`, offline
    batched `LLM.generate`, `logprobs=20` for next-token probabilities, per-request seeds
    (E5), plain temperature sampling `top_k`/`top_p` off (E6), chat template with
    `enable_thinking=False`, single-token label check at startup, fail-fast load/OOM errors,
    logged model revision (E20). `runner.make_backend` already imports
    `scv.backends.vllm_backend.VLLMBackend`.
  - Verify every id in `configs/models/models.yaml` and its licence gating on the Hub (U11).
  - vLLM cannot be installed on the Mac: tests must inject a fake `vllm` module.
- Then M7 (SPEC §12): `electorates.py`, `analysis.py`, `scv analyze`, notebook.

## Known issues / notes for later milestones

- The `gpu` extra (`vllm`) is in the lockfile but has never been installed or run (M6).
- Dry-run throughput defaults are placeholders, not measurements (U9).
- M7 performance: compute `sp_consistency` once per electorate and reuse it across
  bootstraps and n-curve draws (D10); keep `irv_monotonicity_probe` out of the bootstrap loop.
- M7: decide which ranking is the ranking ballot (planned: the greedy one, E11). A mock run
  on the synthetic ANES fixture (`tests/test_anes.py::anes_run`) is the natural test input.
- Flagged for Tristan in `DECISIONS.md` (⚑): R6, R14, D7, E1 (prompt wording), E6
  (sampling), A2 (2016 candidate lists), A5 (zero-weight cases dropped).
- vLLM backend, analysis: not started.
