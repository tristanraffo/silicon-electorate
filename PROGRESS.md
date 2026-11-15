# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green before starting the next milestone. Then begin **M2**.

## Done

- **M1 — `ballots.py`, `rules.py`** ✅ (56 tests)
  - `src/scv/ballots.py`: `WeakOrder`, `from_scores`, `pairwise`, `is_strict`, `is_complete`,
    `linear_extensions`, `move_up`.
  - `src/scv/rules.py`: `Profile` (cached `ranks`, `pairwise`, `majority` arrays),
    `pairwise_matrix`, `condorcet_winner` / `condorcet_loser`, `has_majority_cycle`,
    `smith_set`, `three_cycles`, `majority_cycles`, `condorcet_order`, the six rules
    (`plurality`, `borda`, `copeland`, `minimax`, `kemeny`, `irv`) via `RULES` /
    `run_all_rules`, `rule_disagreement`, `irv_monotonicity_probe`.
  - Tests: `tests/test_ballots.py`, `tests/test_rules.py` (every §5 required test, worked by
    hand in comments), `tests/test_rules_reference.py` (cross-check against slow reference
    implementations). Shared builders in `tests/helpers.py`.
  - Scaffolding: `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`, `.gitignore`.

## Next

- **M2 — `domains.py`, `nulls.py`** (SPEC §6, §7). Acceptance: §6 and §7 tests pass,
  including the IC 8.77% cycle calibration (m = 3, n = 1,001, 20,000 draws, ±0.5 pp).
  - Build on `Profile.ranks` (n × m, NaN = missing) and `scv.ballots.linear_extensions`.
  - `tests/helpers.single_peaked_profile` is a stand-in for the 1-D spatial generator;
    once `nulls.py` has the real one, point the M1 Kemeny property test at it.
  - Record the observed IC deletion share (m = 5, n = 1,000) in `DECISIONS.md`.
- Then M3 → M7 in the order of SPEC §13.

## Known issues / notes for later milestones

- The `gpu` extra (`vllm`) is in the lockfile but has never been installed or run (M6).
- No `scv` CLI entry point yet; add `[project.scripts] scv = "scv.cli:app"` when `cli.py` exists.
- `irv_monotonicity_probe` costs about 0.6 s at n = 5,000, m = 7. Do not put it inside the
  B = 1,000 bootstrap in M7 without subsampling; compute it on point estimates and n-curves.
- The probe perturbs single ballots, so it reads near 0 on large unit-weight electorates
  (`DECISIONS.md` R14). Flagged for Tristan.
- Borda treats a missing candidate as tied with everything (`DECISIONS.md` R6). Flagged for Tristan.
- ANES configs, prompts, backends, runner, analysis: not started.
