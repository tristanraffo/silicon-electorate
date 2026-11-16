# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green (225 tests, about 40 s; no model is loaded). The real-model
smoke test is opt-in:

```bash
uv run pytest -m hf -s
```

**All seven milestones (M1–M7) are built.** What remains is real data and a real GPU
run; see "What Tristan has to do" and "Not yet verified" below.

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

- **M6 — `vllm_backend.py`, runbook** ✅ (code path only; **never run on a GPU**)
  - `src/scv/backends/vllm_backend.py`: `VLLMBackend`, mirroring `hf.py`.
  - `src/scv/backends/_tokens.py`: rendering, encoding, single-token label check and
    revision lookup shared by `hf` and `vllm`.
  - `tests/test_vllm_backend.py`: unit tests against a fake `vllm` module.
  - `README.md`: commands table, election prep, vast.ai runbook (image, `uv sync --extra
    gpu`, HF login, dry run, pilot, run, rsync back, destroy).
  - `configs/models/models.yaml` ids verified on the Hub (Llama and Gemma are gated).

- **M7 — `electorates.py`, `analysis.py`, notebook** ✅
  - `src/scv/electorates.py`: `build_ballot_table`, `human_electorate`, `homogeneous`,
    `assign_models`, `heterogeneous`, `null_ic`, `null_permutation`.
  - `src/scv/analysis.py`: `point_metrics`, `bootstrap`, `n_curve`, `matched_comparisons`,
    `matched_groups`, `analyze`, `write_outputs`, `analyze_run`, `fig_cycle_ncurves`,
    `fig_sp_deletion`, `fig_pc1`.
  - CLI: `scv analyze --config <run.yaml>`. Notebook: `notebooks/01_results.ipynb`.
  - Acceptance: `tests/test_analysis.py::test_scv_analyze_writes_summary_and_three_figures`
    runs `scv analyze` on a mock-generated run over the synthetic ANES fixture.

## What Tristan has to do before real data runs

1. Put the ANES file in `data/raw/` and set `file:` in the election YAML.
2. Fill every `TODO` (weight, thermometer codes, persona variables, exact question
   text) and write `value_labels` that read naturally in `configs/prompts/persona_v1.yaml`.
   Optionally set `id_var`. `tests/fixtures/anes_synthetic.yaml` is a complete example.
3. `uv run scv data check <election>` until it prints OK, then `uv run scv data build <election>`.

## Not yet verified (do these before trusting results)

1. **vLLM on a GPU** has never run (`DECISIONS.md` V1, V2, V4, V7). Follow the README
   runbook; start with the 20-respondent pilot and check parse rates and
   `missing_label_count`.
2. **Real ANES data** has never been loaded: every test uses the synthetic fixture.
3. **Prompt wording** (E1) and the **sampling settings** (E6) should be reviewed before the
   first paid run; changing a template afterwards changes every call key.
4. **Dry-run throughput** defaults are placeholders (U9); calibrate from the pilot.
5. Full-scale timing of `scv analyze` (n ≈ 5,000, m = 7, B = 1,000, R = 50) has not been
   measured; the replicated electorates are the slow part (about 1.3 s per draw for the
   single-peakedness matrix). Use `--replicates 10` for a first look.
6. Decisions flagged ⚑ in `DECISIONS.md`: R6, R14, D7, E1, E6, A2, A5, V1/V2/V4/V7, M6.

## Possible follow-ups (not in the spec's milestones)

- H2 (dimensionality moderation) and H5 (training stage) need no new code: run several
  candidate sets (`election.candidates:` subsets) or the `training_stage` models as
  separate `run_id`s and concatenate their `summary.csv` files in the notebook.
- H4 per-voter coherence is in `summary.csv` as `h4_*` (source `pairwise`) and per voter
  in `coherence.parquet`; there is no figure for it yet.
