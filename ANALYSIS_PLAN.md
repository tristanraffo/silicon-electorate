# Analysis plan and metric guide

Written 2026-11-16 at Tristan's request. This is a plan and a reading guide: which
numbers answer which hypothesis, how to compare them, and what each metric means. It
contains **no findings**. The decision rules in §3 are proposed defaults for Tristan to
accept or change *before* looking at the full-scale results.

Everything named here is in `results/metrics/<run_id>/`:
`summary.csv` (one row per electorate × source × weighting × metric), `ncurves.parquet`,
`matched.parquet`, `matched_groups.parquet`, `coherence.parquet`, `winners.parquet`, and
`analysis_settings.json` (the options that produced the folder).

## 1. Design in one paragraph

Each ANES respondent has a human ballot (their feeling thermometers). A persona built
from that respondent's covariates is given to each model, which produces thermometer
scores, a ranking and pairwise choices for the same candidates. Electorates are then
assembled over the *same respondents with the same survey weights*: `human`; one per
model (`kind = homogeneous`); a mixed-model one (`heterogeneous`, each persona assigned
to one model, stratified on party ID, repeated R times); and two nulls. So a human–LLM
difference is a difference between two sets of ballots cast "by the same people".

## 2. Choices that apply to every comparison

| Choice | Default | Why |
|---|---|---|
| Ballot source | **`thermometer` is primary**; `ranking` is secondary | Rankings carry position effects for three of four models (DECISIONS G14). Report ranking results in both the default and the `__ranking_combined` variant, and treat disagreement between them as a caveat, not a result. |
| Weighting | **`weighted` is the headline**; `unweighted` is the robustness check | SPEC §14.3. A conclusion should not depend on which is used. |
| Sample | State which: 300-respondent pilot or full | The n-curves stop at the electorate size. |
| Elections | Each election is analysed separately | Different candidates, years and questionnaires. Agreement across elections is the replication. |

## 3. Hypotheses: what to compare, and what would count as support

### H1 (monoculture): LLM electorates are more single-peaked, lower-dimensional and less cyclic than the matched human electorate

| Outcome | Rows to read | H1 predicts |
|---|---|---|
| Single-peakedness (**primary**) | `sp_deletion_share_minus_human`, each model, `source = thermometer`, `weighted` | negative (the LLM electorate needs fewer deletions) |
| Dimensionality | `pc1_share_minus_human`, same rows | positive (one dimension explains more) |
| Cycles | `ncurves.parquet`: `cycle_freq` by `n`, model vs `human` | lower at each n, most visibly at small n |
| Supporting | `vr_share`, `mean_kendall_tau`, `ballot_entropy`, `has_cycle_boot_freq` | higher, lower, lower, lower |

- **Test statistic.** The `_minus_human` rows are the model's value minus the human
  value, with a 95% interval from resampling the same voters in both electorates
  (`interval = paired_bootstrap`). An interval that excludes zero in the predicted
  direction is evidence for H1 for that model and election.
- **Proposed decision rule.** H1 is *supported in an election* if the primary outcome's
  interval excludes zero in the predicted direction for at least three of the four pool
  models, and the dimensionality outcome agrees in sign. It is *supported overall* if
  that holds in at least two of the three elections. Anything else is reported as mixed.
- **Scale.** Read every value against the two nulls: `null_permutation` (same marginals,
  no correlation between candidates) and `null_ic` (uniformly random orders). A
  difference of 0.05 means little without knowing where human and nulls sit.
- **Cycles need the n-curves.** At the full electorate size `has_cycle` is almost always
  0 for everyone (SPEC §12), so the informative comparison is the cycle frequency in
  sub-electorates of 11 to 251 voters.
- **Multiplicity.** Four models × three elections × several outcomes is many intervals.
  One primary outcome is named above for that reason; the rest are supporting. Do not
  pick the best-looking model or metric after the fact.

### H2 (dimensionality moderation): the gap is largest where human preferences are most multidimensional

- **Unit of analysis:** a candidate set. For each, x = the human `pc1_share` (lower means
  more multidimensional) and y = the H1 gap (`sp_deletion_share_minus_human`, per model).
- **Prediction:** a positive relationship between (1 − human `pc1_share`) and the size
  of the gap.
- **What exists:** three candidate sets (one per election). Three points can show a
  direction and nothing more.
- **To test it properly:** run candidate subsets (`election.candidates:` in a run config,
  one `run_id` each; for example every 3- and 4-candidate subset of an election's
  thermometers) and plot gap against human `pc1_share` across them. Subsets of one
  election share respondents, so treat this as descriptive, not as independent points.

### H3 (heterogeneity): a mixed-model electorate is less homogeneous than single-model ones

| Outcome | Rows | H3 predicts |
|---|---|---|
| Diversity | `mean_kendall_tau`, `ballot_entropy`, `distinct_share`: `heterogeneous` vs each model | higher for `heterogeneous` |
| Structure | `sp_deletion_share`, `pc1_share`: same comparison | closer to the human value than the single-model electorates are |

- **Comparison.** The mixed electorate's value (mean over R assignments) against (a) each
  single-model electorate and (b) their average. H3 needs the mixed value to exceed the
  *average*, since a mix always falls somewhere among its components on some metrics.
- **Interval.** `interval = replicates` on `heterogeneous` rows is the spread over
  random model assignments with the voters held fixed. It is **not** a sampling interval
  and is not comparable to a bootstrap interval. No paired difference is computed for
  the mixed electorate; compare point values and state that.
- **Ranking source:** the mixed ranking electorate includes whichever models have
  ranking ballots; see `--exclude-ranking` and DECISIONS G11, G12.

### H4 (individual coherence): an LLM voter's own pairwise choices are intransitive or contradict its ranking

Rows: `h4_*` in `summary.csv` (`source = pairwise`, one per model); per voter in
`coherence.parquet`.

| Metric | Meaning | Reading |
|---|---|---|
| `h4_wst_violation_share` | Share of candidate triples where the voter's pairwise choice probabilities violate weak stochastic transitivity | 0 = coherent |
| `h4_any_wst_violation` | Share of voters with at least one such triple | 0 = coherent |
| `h4_strict_cycle_share`, `h4_any_strict_cycle` | The same, counting only strict majority cycles a > b > c > a in the voter's own choices | 0 = coherent |
| `h4_ranking_agreement` | Share of pairs where the pairwise choice agrees with the voter's own greedy ranking | 1 = consistent |
| `h4_abs_position_bias` | How much the choice depends on which option was listed first (0 = not at all; 0.5 = entirely) | diagnostic |

- **There is no human counterpart** (ANES has no pairwise items), so H4 is a statement
  about the models, read against 0 and 1 and against each other, not a human–LLM gap.
- **Benchmark for chance.** With four candidates there are four triples; a voter choosing
  at random would show a strict cycle in a quarter of triples. Use that, not zero, to
  judge whether a small share is small.
- **Position bias first.** Each pair is asked in both orders and averaged, which removes
  first-order position bias from the choice. A model with high `h4_abs_position_bias`
  has little information left after averaging; read its other H4 numbers with that in
  mind.
- **`h4_ranking_agreement` inherits the ranking problem.** It uses the greedy ranking,
  which has position effects (G14). A low value can mean the ranking is noisy, not that
  the pairwise choices are.

### H5 (training stage), exploratory

Not run. The OLMo checkpoints in `configs/models/models.yaml` (base, SFT, DPO, instruct)
would be separate runs. The base model uses a different prompt frame (plain completion,
no ranking), which is confounded with training stage; SPEC §10.4 says so. Compare the
same H1 outcomes across checkpoints, and call it exploratory.

## 4. What each metric means

"More Arrow-escaping" below means: closer to a domain where majority rule is transitive.

| Metric | Definition | Range | More Arrow-escaping is | Caution |
|---|---|---|---|---|
| `sp_deletion_share` | Smallest weight share of voters to remove so the rest are single-peaked on one common ordering of the candidates | 0–1 | lower | With few candidates there are few orderings, so even random ballots score well below 1; read against `null_ic`. Tied ballots pass more easily (see §6). |
| `sp_plateau_deletion_share` | The same under the stricter "single-plateaued" rule for ballots with ties | 0–1 | lower | Equals the above when ballots have no ties. |
| `vr_share` | Share of candidate triples satisfying Sen's value restriction (some candidate is never best, never middle or never worst) | 0–1 | higher | All-or-nothing per triple: one voter can break a triple. |
| `vr_mean_min_violation` | Graded version: per triple, the smallest weight share of voters violating the best-fitting restriction, averaged | 0–1 | lower | The usable companion to `vr_share` in large electorates. |
| `pc1_share`, `pc2_share` | Share of variance in thermometer scores (centred within voter) on the first and second principal components | 0–1 | higher `pc1_share` | Thermometer source only. Not a test of single-peakedness: one dimension can be a non-spatial one. |
| `has_cycle` | Whether the strict majority relation of the whole electorate contains a cycle | 0 or 1 | 0 | One electorate, one draw; nearly always 0 at full size. |
| `has_cycle_boot_freq` | Share of bootstrap resamples with a cycle | 0–1 | lower | A stability measure for the full-size electorate, not a small-n frequency. |
| `condorcet_winner_exists`, `condorcet_winner_boot_freq` | Whether some candidate beats every other by strict majority; and its bootstrap frequency | 0/1; 0–1 | 1; higher | No cycle does not imply a winner when majorities tie. |
| `condorcet_loser_exists` | Some candidate loses to every other | 0/1 | n/a | Descriptive. |
| `majority_transitive` | The whole majority relation is a transitive order | 0/1 | 1 | Stronger than "has a Condorcet winner". |
| `smith_size` | Size of the smallest set of candidates that beat everyone outside it | 1–m | 1 | 1 means a Condorcet winner. |
| `n_three_cycles` | Number of majority 3-cycles | 0+ | 0 | |
| `rule_disagreement_share` | Share of pairs of voting rules (plurality, Borda, Copeland, minimax, Kemeny, IRV) whose winners differ | 0–1 | lower | When all rules agree, the choice of rule does not matter for this electorate. |
| `borda_differs_from_condorcet` | Borda winner is not the Condorcet winner | 0/1 | 0 | Undefined without a Condorcet winner. |
| `irv_monotonicity_failure_fraction` | Share of ballots where moving the IRV winner up one place makes it lose | 0–1 | 0 | A probe, not an exhaustive search. |
| `mean_kendall_tau` | Mean normalised Kendall distance between two voters' ballots | 0–1 | lower | 0 = everyone identical; about 0.5 = unrelated ballots. A diversity measure: low diversity is what "monoculture" means, but it is not the same as single-peakedness. |
| `ballot_entropy` | Entropy (bits) of the distribution of distinct ballots | 0–log2(#ballots) | lower | Grows with the number of candidates; compare within an election only. |
| `distinct_share` | Distinct ballots divided by voters | 0–1 | lower | Depends on n. |
| `share_strict_ballots`, `share_complete_ballots` | Share of ballots with no ties; with no missing candidates | 0–1 | n/a | **Read these first.** They say how comparable two electorates' ballots are (§6). |
| `matched_tau_b_thermometer`, `matched_tau_b_ranking` | Per respondent, Kendall tau-b between the human ballot and the model's; mean | −1 to 1 | n/a | Individual-level fidelity. High fidelity is not required for H1, and aggregate structure can match with low fidelity. |
| `matched_top_agree_*` | Share of respondents whose top choice(s) overlap with the model's | 0–1 | n/a | Chance level is roughly 1/m, more with ties. |
| `matched_abs_error` | Mean absolute difference in thermometer scores, in points | 0–100 | n/a | Breakdowns by party and ideology are in `matched_groups.parquet`. |

## 5. Reading the uncertainty columns

| `interval` | What varies | Use it for | Do not use it for |
|---|---|---|---|
| `bootstrap` | Voters resampled with replacement (B times), weights kept | Sampling uncertainty of one electorate's value | Comparing two electorates by eye: they share voters, so overlapping intervals can still hide a clear difference |
| `paired_bootstrap` | The same resampled voters in the model and the human electorate | **The human–LLM comparison** | |
| `replicates` | Random model assignment (mixed electorate) or random null draw, voters fixed | How much the mix or the null moves by chance | Sampling uncertainty |
| (empty) | Nothing: a point value | | |

- **Model answers are treated as fixed.** The intervals reflect which respondents were
  sampled, not the randomness of the model's own sampling (thermometers are a median of
  five samples; rankings a single greedy answer).
- **Survey design.** The bootstrap resamples respondents independently and ignores the
  ANES strata and clusters, so intervals are somewhat too narrow for population claims.
  They are appropriate for the matched human–LLM difference on this sample.
- **n-curves** (`ncurves.parquet`): for each sub-electorate size, the share of random
  sub-electorates with a cycle, with a Condorcet winner, and the mean deletion share.
  Weighted curves draw voters in proportion to survey weight (DECISIONS M6).

## 6. Checks to make before interpreting anything

1. **Parse rates and position effects:** `uv run python scripts/ranking_report.py
   <run_id>`. A model with a low greedy parse rate or a large `pos_corr` has unreliable
   ranking ballots; pairwise answers that are exactly 50/50 should be near zero.
2. **Ties and missing ballots:** compare `share_strict_ballots` and
   `share_complete_ballots` between `human` and each model. Human thermometers heap on
   round numbers and so contain ties; a ballot with ties is counted as single-peaked if
   any way of breaking the ties is. If humans have many more (or fewer) ties than a
   model, part of a `sp_deletion_share` gap is an artefact of that. Use
   `sp_plateau_deletion_share` and `vr_mean_min_violation` as cross-checks.
3. **The nulls behave:** `null_ic` should have a high deletion share and
   `mean_kendall_tau` near 0.5; `null_permutation` should sit between it and `human`.
4. **Weighted and unweighted agree in sign.**
5. **Settings:** `analysis_settings.json` shows full settings (bootstrap 1,000, draws
   500, replicates 50) and the intended ballot variant.

## 7. Known limits to state alongside any result

- **Rankings:** strict by design (no ties, G10) and affected by list position for
  three of four models (G14); the combined ballot reduces that but introduces ties in
  about a third of ballots (G15).
- **Personas:** ten covariates; region not state (C4); no income in the 2016 Pilot
  (P4); attitudes are not in the persona, so a model cannot know anything about a
  respondent that these covariates do not imply.
- **Samples:** 2016 Time Series respondents are those who rated at least three of
  Clinton, Trump, Johnson and Stein (S2), who differ from those who did not.
- **Models:** all were trained after both elections and know the outcomes; they
  associate the 2024 race with Joe Biden (G7). Four models of 7–9 billion parameters
  are not "LLMs in general".
- **Thermometer wording** for the two Time Series was not checked against the
  questionnaires by the assistant (Q3).
- **Human ballots are thermometers,** not votes or stated rankings.

## 8. Order of work

1. Settle the flagged decisions in `DECISIONS.md` and accept or edit the rules in §3.
2. Full-scale runs: `FULL_RUNS.md`.
3. The checks in §6, for every run.
4. Read §3's primary rows, then the supporting ones, per election.
5. Repeat for the ranking source, default and `--ranking-ballot combined`, as a
   robustness section.
6. H3 and H4 tables. H2 and H5 only after their extra runs.
