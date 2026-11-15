# PROGRESS

Read `SPEC.md` first. This file is enough to resume from a fresh session.

## Resume

```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

All three must be green before starting the next milestone (111 tests, about 10 s). Then
begin **M3**.

## Done

- **M1 — `ballots.py`, `rules.py`** ✅
  - `src/scv/ballots.py`: `WeakOrder`, `from_scores`, `pairwise`, `is_strict`, `is_complete`,
    `linear_extensions`, `move_up`.
  - `src/scv/rules.py`: `Profile` (also `Profile.from_ranks` / `from_scores` for arrays; cached
    `ranks`, `pairwise`, `majority`), `pairwise_matrix`, `condorcet_winner` / `condorcet_loser`,
    `has_majority_cycle`, `smith_set`, `three_cycles`, `majority_cycles`, `condorcet_order`,
    six rules (`plurality`, `borda`, `copeland`, `minimax`, `kemeny`, `irv`) via `RULES` /
    `run_all_rules`, `rule_disagreement`, `irv_monotonicity_probe`.
  - Tests: `tests/test_ballots.py`, `tests/test_rules.py` (every §5 required test, worked by
    hand in comments), `tests/test_rules_reference.py` (cross-check against slow reference
    implementations). Shared builders in `tests/helpers.py`.
- **M2 — `domains.py`, `nulls.py`** ✅
  - `src/scv/nulls.py`: `impartial_culture`, `spatial` / `spatial_1d` / `spatial_2d`
    (returns `SpatialElectorate` with `.profile`, `.scores`, `.axis`), `permutation_null`.
  - `src/scv/domains.py`: `all_axes`, `is_single_peaked`, `is_single_plateaued`,
    `sp_consistency`, `sp_axes`, `sp_max_consistent`, `sp_deletion_share`,
    `single_peakedness` (both checks), `value_restriction`, `dimensionality`, `diversity`,
    `kendall_tau_distance`.
  - Tests: `tests/test_domains.py`, `tests/test_nulls.py`. Observed calibration values:
    IC cycle frequency 8.82% (target 8.77% ± 0.5 pp); IC SP deletion share 0.8350.

## Next

- **M3 — backends `mock` and `hf`, plus `elicit.py`** (SPEC §9, §10). Acceptance: an
  end-to-end test with `mock` recovers the hidden utilities' ranking for more than 95% of
  mock voters at noise 0; the tiny-model smoke test (`Qwen/Qwen2.5-0.5B-Instruct`, about
  1 GB, under the 2 GB download limit) runs on CPU for 5 respondents × 3 candidates.
  - Needs `prompts.py` (template rendering and hashing) and the four
    `configs/prompts/*_v1.yaml` files first; `elicit.py` depends on them.
  - Elicitation output feeds `scv.ballots.from_scores` (thermometer) and `WeakOrder`
    (ranking); pairwise mode produces probabilities for the H4 metrics in §10.3.
- Then M4 → M7 in the order of SPEC §13.

## Known issues / notes for later milestones

- The `gpu` extra (`vllm`) is in the lockfile but has never been installed or run (M6).
- No `scv` CLI entry point yet; add `[project.scripts] scv = "scv.cli:app"` when `cli.py` exists.
- M7 performance: compute `sp_consistency` once per electorate and reuse it across
  bootstraps and n-curve draws (`DECISIONS.md` D10); keep `irv_monotonicity_probe`
  (0.6 s at n = 5,000) out of the B = 1,000 bootstrap loop.
- Flagged for Tristan in `DECISIONS.md` (⚑): R6 Borda with missing candidates; R14 the IRV
  probe moves single ballots; D7 value restriction is all-or-nothing, graded version added.
- ANES configs, prompts, backends, runner, analysis, CLI: not started.
