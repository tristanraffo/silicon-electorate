# DECISIONS

Choices made where `SPEC.md` is silent or ambiguous. Each has the default that was picked
and why. Tristan may override any of them; items marked **⚑** are the ones most worth a look.

## Process and environment

| # | Decision | Why |
|---|---|---|
| P1 | Commits go straight onto `master`. | The repo had no commits and no remote; §0.3 asks for one linear `M<N>:` commit per milestone. |
| P2 | `requires-python = ">=3.11,<3.12"`, plus `.python-version` = 3.11. | §2 says Python 3.11. |
| P3 | All §2 core dependencies are declared from M1 (including `torch`, `transformers`). | Keeps `uv.lock` stable across milestones instead of re-resolving at M3. |
| P4 | `pydantic>=2` and `gpu = ["vllm>=0.8; sys_platform == 'linux'"]`. | With no bounds, uv's universal resolution picked vLLM 0.2.5 and dragged the dev install down to pydantic 1.10. With the bounds the lock has vLLM 0.33.0. The Linux marker keeps vLLM off the Mac. |
| P5 | Dev torch is 2.14.0, not the newest 2.15.0. | vLLM pins torch, and uv keeps one torch across dev and prod. Same version on both runtimes is a feature. |
| P6 | The `gpu` extra is locked but has never been installed or tested. | That is M6. |
| P7 | Build backend `hatchling`; import package `scv` (per §3). Ruff rules `E,F,I,B,UP`, line length 100. | Simplest working setup. |
| P8 | No `scv` console script yet. | `cli.py` does not exist until a later milestone; a dangling entry point would just error. |

## M1 — `ballots.py`

| # | Decision | Why |
|---|---|---|
| B1 | `WeakOrder` stores tie groups as tuples (sorted within a group) and `missing` as a `frozenset`; lists/sets are accepted and normalised. | §4 says "list" and "set", but ballots must be hashable for diversity metrics and memoisation, and `A=B` must equal `B=A`. |
| B2 | `from_scores`: `None` **and NaN** mean missing; ±inf raises. | §10.1 stores NaN for a failed thermometer parse; it should flow through as missing. |
| B3 | `is_strict` means "no ties among ranked candidates" and ignores missing candidates. `is_complete` (no missing) is a separate function. | A respondent who strictly ranks 4 of 5 candidates is strict but incomplete; conflating the two loses information. |
| B4 | `pairwise` returns every ordered pair over ranked ∪ missing candidates. | §4: 0 means "tie or missing", so missing candidates need keys. |
| B5 | `linear_extensions` yields `WeakOrder` objects and only resolves ties; missing candidates stay missing. | A linear extension of the ranked part is the only thing the ballot determines. |
| B6 | Added helpers not named in §4: `WeakOrder.strict`, `WeakOrder.parse("A>B=C")`, `move_up`. | `move_up` is needed by the IRV probe; the other two keep tests readable. |

## M1 — `rules.py`

| # | Decision | Why |
|---|---|---|
| R1 | `Profile` takes the candidate list explicitly. A listed candidate that a ballot does not mention is treated as missing on that ballot; a ballot naming an unlisted candidate is an error. Empty profiles and zero total weight are errors. | §5 says the candidate list is part of the input. Failing loudly beats guessing. |
| R2 | `Profile` allows any m ≥ 2. The m ≤ 7 limit is enforced only where brute force needs it (`kemeny`, `linear_extensions`). | §4's 3 ≤ m ≤ 7 is the research range; a hard floor of 3 would only block harmless two-candidate checks. |
| R3 | **Tolerance.** Two weighted totals are equal if they differ by ≤ `1e-9 × max(1, total weight)`. Applies to majority comparisons, score ties, Kemeny optima and IRV eliminations. | ANES weights are floats: `0.1 + 0.2 ≠ 0.3` must not create a majority or a winner. |
| R4 | **Ties at the top are reported, not broken.** Every rule returns `winners` (all co-winners); `winner` is the unique winner or `None`. IRV is the exception (R10). | §4: ties are never broken at random. |
| R5 | Rule disagreement compares winner **sets**: a rule that ties {A,B} disagrees with one that elects A alone. | The only definition that needs no tie-break. |
| R6 ⚑ | **Borda with missing candidates:** a comparison involving a missing candidate counts as a tie (0.5 to each side). So Borda = `0.5·(W·(m−1) + Σ_b (N[a][b] − N[b][a]))`. | Keeps Borda consistent with the majority matrix, where missing counts for neither side. The alternative (missing scores 0) would penalise candidates fewer respondents rate — e.g. Shapiro in ANES 2024 — for being less known rather than less liked. |
| R7 | **Minimax** uses margins (Simpson–Kramer). Reported score is the candidate's worst margin `min_b (N[a][b] − N[b][a])`, so higher is better, like every other rule's scores. | Simplest and most common variant. With missing data the winning-votes and pairwise-opposition variants can differ. |
| R8 | **Kemeny with several optimal rankings:** `ranking` is the first optimum in lexicographic order of the candidate list; `details["optimal_rankings"]` lists all; `details["unique"]` flags it; `winners` is every candidate that tops some optimum. | Deterministic, and nothing is hidden. |
| R9 | **Smith set** = smallest non-empty set whose members each *strictly* beat every non-member. **Cycle** = a directed cycle of any length in the strict-majority relation (with pairwise ties a 4-cycle can exist with no 3-cycle; there is a test). | Standard definitions; "tournament" in §5.3 is not quite accurate once ties exist. |
| R10 | **IRV:** one elimination per round, run until one candidate is left (so there is a full ranking: reverse elimination order). Ties for lowest are broken by `numpy.random.default_rng(seed).permutation(m)`; the tied candidate earliest in that order is eliminated. Each tie-break is logged (`scv.rules`, INFO) and stored in `details["tie_breaks"]`; `details["winner_by_tiebreak"]` flags a tied final round. Ballots with no remaining ranked candidate are exhausted. | §5.4. |
| R11 | Added `condorcet_order(profile)`: the majority relation as a `WeakOrder` if the weak relation "beats or ties" is transitive, else `None`. | Needed to state the required Kemeny property test, and it is the direct "did this electorate escape Arrow" indicator. |
| R12 | **IRV probe, "move w up one position":** if w is tied with others, the tie is broken in w's favour; if w is alone in its group, it swaps with the whole group directly above (the ordinary adjacent swap on a strict ballot). | §5.6 does not define the move for weak orders. |
| R13 | **IRV probe, eligibility and denominators:** eligible = w is ranked and is not the sole first choice (so "tied for first" is eligible; "w missing" is not). `failure_fraction` = failures / eligible ballots (0.0 if none are eligible); `failure_fraction_all` = failures / all ballots. Both count ballots, not weight. | §5.6 says "the fraction of ballots where this happens" without a denominator; both are reported. |
| R14 ⚑ | **The probe moves one ballot row with its full weight.** In a unit-weight electorate that is a single voter, so the probe only fires on knife-edge profiles: the textbook failure needs a *block* of voters to move. `test_irv_monotonicity_probe_moves_one_row_...` shows the same electorate giving 1/3 as four weighted rows and 0 as 100 unit ballots. Expect values near 0 at large n; it is informative mainly on the small-n curves. | This is §5.6 as written. If a block-move probe is wanted instead, it is a small change — say so. |
| R15 | **IRV monotonicity test example** is constructed (39/35/26 → 49/25/26) and verified by hand in the test, not copied from a paper. The test cites Doron & Kronick (1977) and Fishburn & Brams (1983) for the phenomenon; those citations are from memory and were not checked online. | §5 allows "find one in the literature or construct one". |
| R16 | Added `tests/test_rules_reference.py`: every engine function is cross-checked against slow reference code written from the definitions (exact `Fraction` arithmetic) on hundreds of random profiles, with assertions that cycles, tie-breaks and probe failures actually occur in the sample. | §0.6: correctness over breadth. |
| R17 | The probe reruns IRV incrementally (cached whole-electorate tallies ± one ballot's contribution). | Full reruns took 8 s at n = 5,000, m = 7; now 0.6 s. Result is identical to the brute-force reference. |
