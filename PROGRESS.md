# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green before starting the next milestone (152 tests, about 15 s; no
model is loaded). The real-model smoke test is opt-in:

```bash
uv run pytest -m hf -s
```

Then begin **M4**.

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

## Next

- **M4 — `runner.py` with resumability and dry run** (SPEC §11). Acceptance: a test kills
  the runner mid-run, restarts it, and finds zero duplicate keys and a complete call count.
  - Build on `elicit.plan_calls` (all keys known up front) and `elicit.execute` (yields
    record batches; accepts any subset of calls). Skip calls whose key is already in
    `results/calls/<run_id>/part-*.parquet`; add `run_id` to each row; flush every 500.
  - Sequential thermometer chains (`Call.chain`) must be re-run whole: if any turn of a
    chain is missing, run the chain and drop rows whose key is already logged.
  - Also in M4: `configs/models/models.yaml` plus a loader/factory for backends,
    `configs/runs/dev_smoke.yaml`, `cli.py` with `scv elicit --config … --dry-run`
    and `--limit-respondents K` (stratified on party_id_7), and the
    `[project.scripts] scv = "scv.cli:app"` entry point.
  - Respondents come from M5's `personas.parquet`; until then the runner can take a
    list of `Respondent` objects (tests use the mock backend).
- Then M5 → M7 in the order of SPEC §13.

## Known issues / notes for later milestones

- The `gpu` extra (`vllm`) is in the lockfile but has never been installed or run (M6).
  M6 must use the same plain temperature sampling as `hf.py` (`DECISIONS.md` E6) and
  per-request seeds (E5).
- M7 performance: compute `sp_consistency` once per electorate and reuse it across
  bootstraps and n-curve draws (D10); keep `irv_monotonicity_probe` out of the bootstrap loop.
- M7: decide which ranking is the ranking ballot (planned: the greedy one, E11).
- Flagged for Tristan in `DECISIONS.md` (⚑): R6, R14, D7, E1 (prompt wording), E6 (sampling).
- ANES loader, runner, CLI, vLLM backend, analysis: not started.
