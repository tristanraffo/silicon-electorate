# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green (244 tests, about 45 s; no model is loaded). The real-model
smoke test is opt-in:

```bash
uv run pytest -m hf -s
```

**All seven milestones (M1–M7) are built, and the first real runs are done** (three
elections, 300 respondents each, four models, on a rented GPU). See "Where things stand"
below. `DECISIONS.md` sections C, S, P, Q and G record everything decided on 2026-11-16.

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

## Where things stand (2026-11-16, end of the build window)

**Data.** The three ANES files are in `data/raw/` and the processed tables in
`data/processed/` (both gitignored): 5,375 respondents for 2024, 3,415 for the 2016 Time
Series, 1,199 for the 2016 Pilot. The election configs are complete (`scv data check`
prints OK). Rebuild with `uv run scv data build <election>` after changing one.

**Runs** (call logs in `results/calls/`, gitignored, complete, `remaining = 0`):

| run_id | config | election |
|---|---|---|
| `anes2024_pilot_v4` | `configs/runs/prod_example.yaml` | 2024 Time Series |
| `anes2016_ts_300_v4` | `configs/runs/anes2016_ts_300_v4.yaml` | 2016 Time Series |
| `anes2016_pilot_300_v4` | `configs/runs/anes2016_pilot_300_v4.yaml` | 2016 Pilot |

Each: 300 respondents (stratified on party ID), Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B,
Gemma 2 9B, thermometer + ranking + pairwise. Older logs (`anes2024_pilot_v1`–`v3`,
`rank_trial_*`) are prompt trials; do not use them for results.

**Analyses** (in `results/metrics/`, each run at `--replicates 10`): the default
(`<run_id>/`) and the combined-ranking variant (`<run_id>__ranking_combined/`) for all
three, plus `anes2024_pilot_v4__ranking_without_mistral_7b/` (made before the pairwise
repair; its `h4_*` rows for Mistral are stale).

**What was verified on the GPU:** the vLLM backend end to end; thermometer parse rate
99.8–100%; pairwise readable for all four models after the Mistral fix (G13; under 2.5%
of answers are exactly 50/50); greedy ranking parse rate 97.7–100% except Mistral
(85–98%).

**What to keep in mind when reading results**
1. Thermometer-source results are the reliable ones. Ranking ballots have position
   effects (G14): across the three runs Llama and Qwen rank later-listed candidates
   higher (correlation of list position with rank −0.27 to −0.60), Mistral earlier-listed
   ones in two runs (+0.25, +0.30), Gemma about none. `--ranking-ballot combined` (G15)
   is the mitigation; it also introduces ties in about a third of ballots.
   `uv run python scripts/ranking_report.py <run_id> ...` prints these numbers.
2. Ranking ballots are strict orders (no ties) by design of the prompt (G10).
3. 2016 Time Series respondents are those who rated at least three of Clinton, Trump,
   Johnson, Stein (855 of 4,270 dropped, S2). 2016 Pilot personas have no income (P4).
4. The models strongly associate the 2024 race with Joe Biden (G7).
5. All models were trained after both elections.

## Next steps for Tristan

1. Destroy the vast.ai instance if it still exists.
2. Commit the work of 2026-11-16 (configs, prompts, backend fixes, scripts, tests).
3. Go through the ⚑ decisions in `DECISIONS.md` and confirm or change each: R6, R14, D7,
   E1, E6, A2, A5, M6 (from the build); C1, C2, C4, S2, P2, P4, P5 (data); Q3 (question
   wording unverified for the two Time Series); G4, G6, G7, G8, G10, G11, G13, G14, G15
   (GPU runs). Changing a prompt, a persona label or a candidate set means a new
   `run_id` and a rerun.
4. Read `ANALYSIS_PLAN.md` (which rows answer which hypothesis, how to read each metric,
   checks to make first) and accept or edit its proposed decision rules. Then look at
   the results: `notebooks/01_results.ipynb` reads `results/metrics/*/`. Analyses made
   before commit "paired differences" lack the `*_minus_human` rows and
   `analysis_settings.json`; rerun `scv analyze` to get them.
5. Re-run the analyses at full settings (drop `--replicates 10`; B = 1,000, R = 50) once
   the choices in step 3 are settled. Timing at full scale is still unmeasured.
0. **Before reading any result: ties (G20).** Mistral and Qwen give many completely flat
   thermometer ballots, which make the H1 measures look more monocultural mechanically.
   Built (G21): rerun `scv analyze` with `--ties break_random` and `--ties drop_flat`
   for each full run (done 2026-11-17 for all three), then
   `uv run python scripts/h1_table.py anes2024_ts_full_v4 anes2016_ts_full_v4
   anes2016_pilot_full_v4` and read H1 only where the `robust` column is True.
6. Full-scale runs (every respondent): configs in `configs/runs/full/`, commands in
   `FULL_RUNS.md`. As of 2026-11-17 the 2024 and 2016 Time Series runs are complete and
   on the Mac; the 2016 Pilot run stopped with 47,538 calls left (G19, now fixed) and
   needs resuming with the same command after copying the code up. Rerun `scv analyze`
   for every run afterwards: H4 changed (G19).
7. Not yet run at all: the training-stage models in `configs/models/models.yaml` (the
   OLMo checkpoints, including a base model that uses the plain-completion prompts, which
   have never met a GPU), and candidate-subset runs for H2. Both are new `run_id`s with
   no new code; start each with a 20-respondent pilot and the ranking report.
8. Dry-run time estimates still use placeholder speeds (U9).

Rules that still hold (SPEC §0): no analysis prose or paper from the assistant, apart
from the plan Tristan asked for (`ANALYSIS_PLAN.md`, G17), which contains no findings.

## Possible follow-ups (not in the spec's milestones)

- H2 (dimensionality moderation) and H5 (training stage) need no new code: run several
  candidate sets (`election.candidates:` subsets) or the `training_stage` models as
  separate `run_id`s and concatenate their `summary.csv` files in the notebook.
- H4 per-voter coherence is in `summary.csv` as `h4_*` (source `pairwise`) and per voter
  in `coherence.parquet`; there is no figure for it yet.
