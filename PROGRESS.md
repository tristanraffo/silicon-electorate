# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green before starting the next milestone (169 tests, about 30 s; no
model is loaded). The real-model smoke test is opt-in:

```bash
uv run pytest -m hf -s
```

Try the CLI (mock backend, synthetic personas, writes to `results/`):

```bash
uv run scv elicit --config configs/runs/dev_smoke.yaml --dry-run
```

Then begin **M5**.

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

## Next

- **M5 — `anes.py`, data check, persona rendering** (SPEC §8). Acceptance: runs on a
  synthetic ANES-shaped CSV generated in `tests/fixtures`; `scv data check <election>`
  lists every TODO and every variable missing from the file and exits non-zero.
  - Write `configs/elections/{anes2016_ts,anes2016_pilot,anes2024_ts}.yaml`. Confirmed
    codes (2024, pre-election): Harris `V241156PRE`, RFK Jr. `V241159PRE`, Shapiro
    `V241160PRE`. **Everything else stays `TODO`; never invent a code.**
  - Outputs: `data/processed/<election>/human_scores.parquet` and `personas.parquet`
    (labels must read naturally in `configs/prompts/persona_v1.yaml`; see its header).
  - `render_persona` already exists in `scv.prompts`.
  - Wire it into the runner: add `source: anes` to `runner.ElectionSpec` /
    `load_respondents` (respondent_id, rendered persona, `stratum` = party_id_7 label),
    then write `configs/runs/prod_example.yaml`.
- Then M6 → M7 in the order of SPEC §13.

## Known issues / notes for later milestones

- The `gpu` extra (`vllm`) is in the lockfile but has never been installed or run (M6).
  M6 must use the same plain temperature sampling as `hf.py` (`DECISIONS.md` E6) and
  per-request seeds (E5), and must verify the ids in `configs/models/models.yaml` (U11).
- Dry-run throughput defaults are placeholders, not measurements (U9).
- M7 performance: compute `sp_consistency` once per electorate and reuse it across
  bootstraps and n-curve draws (D10); keep `irv_monotonicity_probe` out of the bootstrap loop.
- M7: decide which ranking is the ranking ballot (planned: the greedy one, E11).
- Flagged for Tristan in `DECISIONS.md` (⚑): R6, R14, D7, E1 (prompt wording), E6 (sampling).
- ANES loader, vLLM backend, analysis: not started.
