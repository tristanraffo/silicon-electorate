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
| R18 | (Added in M2.) `Profile.from_ranks` / `Profile.from_scores` build a profile from an (n, m) array; `WeakOrder` objects are created only if `.ballots` is read. | Simulated and ANES-scale electorates should not pay for 5,000 Python objects per draw. The IC calibration runs 20,000 electorates of 1,001 voters through the real engine in about 4 s. |

## M2 — `nulls.py`

| # | Decision | Why |
|---|---|---|
| N1 | Every generator takes `rng` as a `numpy.random.Generator` or an int seed. `None` is a `TypeError`. | §2: everything is seeded. An unseeded default would be a silent reproducibility hole. |
| N2 | Spatial model: candidates ~ N(0, I), voters ~ N(0, `voter_sd`² I), Euclidean distance, `scores = −distance`. `spatial_1d` / `spatial_2d` are thin wrappers over `spatial(dim=…)`. Synthetic candidates are named `c1..cm`. | §7 fixes the model family, not the distributions. Normal voters reach beyond the outermost candidates, which is what pins the axis down. |
| N3 | `permutation_null` takes and returns a score **matrix**. Survey weights stay with the voter row; missing scores are permuted along with their column. | Keeps each candidate's marginal distribution exactly, including its missing rate. |
| N4 | **IC calibration observed: 8.82%** cycle frequency (m = 3, n = 1,001, 20,000 draws, seed 1785) against Guilbaud's 8.77%. | Within the ±0.5 pp required by §7. |

## M2 — `domains.py`

| # | Decision | Why |
|---|---|---|
| D1 | Ballots with zero weight are ignored by every diagnostic. | Lets a bootstrap or sub-electorate be expressed as a weight vector. |
| D2 | **Missing candidates:** a ballot is tested on the axis restricted to the candidates it ranks. | Equivalent to "some way of filling in the missing candidates is single-peaked", and consistent with §4 (missing is neither above nor below anything). |
| D3 | **Weak ballots, "peaked":** some linear extension is single-peaked (§6.2). Implemented as the equivalent "no valley" test — no ranked candidate is strictly worse than a candidate on each side of it — and checked against brute-force enumeration of linear extensions on >10,000 (ballot, axis) pairs. | Brute force over extensions × 2,520 axes × thousands of ballots is too slow at m = 7. |
| D4 | **"Plateaued" (stricter):** strictly improving up to one contiguous block of top-ranked candidates, then strictly worsening. Ties are allowed inside the plateau and between two candidates on opposite sides of it, nowhere else. Identical to "peaked" for strict ballots. | The standard single-plateaued definition; §6.2 names it without defining it. |
| D5 | An axis and its reversal are one axis; the orientation kept is the one whose first candidate comes earlier in the candidate list. When several axes tie for best, the first in enumeration order is returned. | Deterministic output. |
| D6 | **IC deletion share observed: 0.8350** (m = 5, n = 1,000, seed 0). | §6 asks for the value. Sanity check: only 2^(m−1) = 16 of 120 strict orders fit any one axis, so the expectation is a little below 1 − 16/120 = 0.867. |
| D7 ⚑ | **Value restriction** uses strictly determined positions only: on a ballot, a candidate is best / worst if strictly above / below both others in the triple, middle if strictly between. Ties and missing candidates place nobody. In addition to the §6.3 share of triples satisfied, each triple gets `min_violation`: the smallest weight share that must be removed for it to satisfy VR. | §6.3 says to use strict relations. The extra number is **not in the spec**: exact VR is all-or-nothing, so with thousands of voters it will read 0 for humans and LLMs alike; `min_violation` is the graded analogue of `sp_deletion_share`. Drop it if unwanted. |
| D8 | **Dimensionality:** complete cases only (voters with any missing score are dropped and counted in `n_dropped`). PCA is done in numpy (weighted column means and covariance, eigenvalues) rather than with scikit-learn. Shares are NaN if fewer than 2 voters remain or there is no variance. | scikit-learn's PCA has no sample weights, and §14.3 makes weighted the headline. With unit weights the result equals `sklearn.decomposition.PCA` to 1e-10 (tested); scikit-learn is used only in that test. |
| D9 | **Diversity:** entropy is in bits. Kendall-tau distance is normalised to [0, 1] by the number of candidate pairs; a pair that exactly one of the two ballots leaves undecided (tie or missing) counts 0.5. The mean is the weighted mean over pairs of *different* voters. | Consistent with the §4 pairwise convention. Consequence: for this one metric, weight 2 is not the same as two copies (two copies form a pair at distance 0). |
| D10 | `sp_consistency(profile)` exposes the (n × n_axes) boolean matrix. | It costs about 1.2 s at n = 5,000, m = 7. M7 should compute it once per electorate and reuse it for every bootstrap and sub-electorate (`weights @ matrix`). |

## M3 — prompts, backends, `elicit.py`

| # | Decision | Why |
|---|---|---|
| E1 ⚑ | **Prompt wording is mine** (`configs/prompts/*_v1.yaml`); only the ranking instruction and the pairwise question are verbatim from §10. The thermometer prompt is: persona, "Answer the survey question below exactly as this person would.", the ANES question text, the candidate, then "Respond with a single whole number from 0 to 100 and nothing else." | §14.2 leaves wording open. Review before the first prod run: changing a template changes its hash and therefore every call key. |
| E2 | The persona goes in the **user** turn; no system turn is used. | Gemma's chat template rejects a system role. §10.5 requires identical text across models. |
| E3 | `persona_v1` is one sentence per field in a fixed order; the YAML header documents how value labels must read ("a strong Democrat", "voted for Joe Biden"). Unknown field names are an error. | M5 has to write value labels that fit these sentences. |
| E4 | Templates are hashed over their whole parsed content. `prompt_hash` of a call covers the mode template, the persona template, the question text, chat-vs-completion format and independent-vs-sequential. Loading a template that contains a §10.5 forbidden word fails; so does a config whose question text or candidate names contain one. | "Templates are versioned: the hash is stored with every call." |
| E5 | `Backend.generate(seed=…)` takes one seed **per prompt** (an int expands to consecutive seeds). The HF backend samples each prompt on its own with its own seed; greedy prompts share batches. | A prompt's samples must not depend on what else is in the batch, or a resumed run would differ from an uninterrupted one. vLLM supports per-request seeds natively (M6). |
| E6 ⚑ | **HF sampling is plain temperature sampling**: `top_k=0, top_p=1.0, repetition_penalty=1.0`, overriding the model's own `generation_config` (Qwen ships top-p 0.8 / top-k 20). | "T = 1.0" should mean the same thing for every model. M6 must set the same in vLLM `SamplingParams`. |
| E7 | Call-log rows have one field beyond §11: `target` (the candidate, `"a|b"` for a pair in candidate-list order, or empty for a ranking). `parsed` is JSON: an integer, a list of tie groups, or P("1") for pairwise; `null` on failure. | The key needs "candidate or pair", and one Parquet column cannot hold three types. |
| E8 | **Thermometer parsing** follows §10.1 literally: the first integer token in 0–100. So `"-5"` → 5, `"3.5"` → 3, `"150, or 60"` → 60. `max_tokens` is 8. | Known hazard: a model that echoes "on a 0-100 scale…" parses as 0. The 8-token cap and the "nothing else" instruction make that unlikely; check `raw_text` in the pilot. |
| E9 | **Ranking parsing** accepts list markers, `>` separators, case differences and markdown emphasis, and fails on anything else (a preamble line, a missing or repeated candidate). | A failure triggers the repair retry, which is cheaper than guessing. |
| E10 | **Retries** continue the conversation: the bad answer, then the template's repair message, at the same temperature with a derived seed. The row stores the last attempt's text and the number of retries. After 2 retries: `parsed = null`, `parse_ok = False`. | §10.1. |
| E11 | Thermometer `score` is the median of the sampled answers that parsed (NaN if none did); the greedy answer is kept separately as `greedy`. `ranking_table` returns all four rankings per respondent; which one is "the" ranking ballot is left to M7 (planned default: the greedy one). | §10.1 defines the stored score; §10.2 does not say which ranking is the ballot. |
| E12 | **Sequential thermometer** (`sequential: true`): one multi-turn conversation per sample (5 sampled + 1 greedy), candidates in a per-sample random order. History carries each question and its final answer, not the repair exchange. A chain must be executed whole. | §10.1. The runner (M4) has to treat a chain as atomic. |
| E13 | **Base models:** completion prompts, thermometer and pairwise only, never sequential. For pairwise the answer label is checked as `" 1"` / `" 2"` (with the leading space a completion model writes after `A:`). | §10.4. The format difference is a known confound and is part of `prompt_hash`. |
| E14 | **H4 metrics** are counted per candidate triple: a triple has a WST violation if any ordering of it violates WST, and is a strict cycle if p > .5 all the way round. `ranking_agreement` is over pairs the ranking orders strictly; ranking ties are excluded and p = .5 exactly counts as disagreement. | §10.3 gives the conditions but not the unit of counting. |
| E15 | Presentation orders and seeds come from SHA-256 of `(run_seed, respondent_id, candidate set, call type[, sample])` and never involve the model. | §10.5: same text for every model. |
| E16 | Two respondents whose persona text is identical are still separate requests with separate seeds. | Found by test: they were briefly merged. Note for analysis: identical personas get identical greedy answers by construction, so coarse persona variables mechanically reduce LLM-electorate diversity. |
| E17 | The mock backend learns who it is from `prompt.meta`, but reads the candidates and their order from the rendered text. Noise applies to sampled generations and pairwise logits, not to greedy generations. | Tests then fail if the text shown disagrees with the order logged. |
| E18 | The tiny-model test is opt-in: `uv run pytest -m hf -s`. The default suite never loads a model. **Observed** (`Qwen/Qwen2.5-0.5B-Instruct`, CPU, float32, 5 respondents × 3 candidates, 140 calls, about 3 min): thermometer parse rate 1.00, ranking parse rate 0.60 after retries, mean absolute position bias 0.031. | Keeps the default suite at about 15 s with no network. The 0.60 is a 0.5B model's formatting ability, not a pipeline fault; watch the ranking parse rate per model in the pilot. |
| E20 | The logged `revision` is the commit hash of the snapshot actually loaded, read from the Hugging Face cache path when the model config does not carry it. | §2: a run is determined by its config plus the model revision hash. |
| E19 | `configs/models/models.yaml` is deferred to M4, where the runner needs a loader for it. | Nothing in M3 reads it. |

## M4 — `runner.py`, `cli.py`, run configs

| # | Decision | Why |
|---|---|---|
| U1 | A shard is written to `part-XXXXX.parquet.tmp`, fsynced and renamed into place. Leftover `.tmp` files are deleted on start. | A kill mid-write must never leave a half-written shard that breaks the resume. |
| U2 | Flushing happens when the buffer reaches `flush_every` rows (default 500), at the end, and when an exception (including Ctrl-C) unwinds. A hard kill loses at most the rows since the last flush; they are run again. Shards can exceed `flush_every` by up to one execution batch. | §11. `execute` yields whole batches. |
| U3 | A sequential thermometer conversation that is missing any turn is re-run whole; rows whose key is already logged are dropped, never rewritten. | E12: later turns depend on earlier answers. |
| U4 | An interrupted-and-resumed run gives the same answers as an uninterrupted one (tested with the mock at noise > 0). | Seeds are per prompt and derived from the call, not from batch position (E5). One exception: if a kill splits the 5 samples of one prompt, the missing samples are drawn with a seed that depends on which were missing. |
| U5 | Resuming a `run_id` with a model at a different revision than the rows already logged is refused. | §2: a run is determined by config plus model revision. Use a new `run_id`. |
| U6 | A model whose calls are all logged is not loaded at all on resume. | Saves minutes of GPU time per finished model. |
| U7 | `--limit-respondents K` (or `limit_respondents:` in the config; the flag wins) takes a proportional share of each party-id stratum, largest remainders first, and within a stratum the respondents with the smallest SHA-256 of `(seed, id)`. | Deterministic, independent of input order, and a larger pilot contains the smaller one. |
| U8 | Run configs are validated strictly: an unknown or misspelled field is an error. | A typo in a prod config should fail before the GPU is rented. |
| U9 | **Dry-run estimate:** prompt tokens = characters / `chars_per_token` (default 4), counted once per request (samples sharing a prompt share a request); output tokens assume every call uses its full `max_tokens`; time = prompt tokens / `prompt_tokens_per_s` (default 20,000) + output tokens / `output_tokens_per_s` (default 2,500). Retries are not included. It also shows what is already logged (`remaining`). | §11 makes throughput a config parameter. The defaults are placeholders, **not measurements**; calibrate them from the first pilot's `latency_ms`. |
| U10 | The run config's `election:` block currently supports only `source: synthetic` (made-up personas with a party stratum; the mock gets random hidden utilities). The ANES source arrives with M5. | §13 puts ANES after the runner; the runner still needs something to run on. |
| U11 | `configs/models/models.yaml` is the §9 list verbatim. The ids and licence gating are **not yet verified** on the Hub (M6). `backend: vllm` currently fails fast with a message pointing at `uv sync --extra gpu`. | §9 says to verify before finalizing. |
| U12 | `configs/runs/prod_example.yaml` (§3) is not written yet. | It needs the ANES election source (M5) and the vLLM backend (M6) to be meaningful. |

## M5 — `anes.py`, election configs, `scv data`

| # | Decision | Why |
|---|---|---|
| A1 | **No ANES code was invented.** The only codes in `configs/elections/` are the three confirmed in §8 (Harris, RFK Jr., Shapiro, 2024). A test fails if any other code appears. `process`, `build` and any run on an election with a TODO raise `ConfigError`. | §0.4. |
| A2 ⚑ | The **candidate lists** in `anes2016_ts.yaml` (Clinton, Trump, Johnson, Stein) and `anes2016_pilot.yaml` (Clinton, Sanders, Trump, Cruz, Rubio) are suggestions with TODO codes. | §8 only names 2024 candidates. Replace them with whoever actually has a thermometer in each file. |
| A3 | Election configs have one field beyond §8: optional `id_var` (respondent-id column). `null` uses the 1-based row number in the file. | Matching humans to LLM ballots needs a stable id; I did not want to guess the ANES case-id code. Set it once you have the codebook open. |
| A4 | `scv data check` reports: every TODO; a missing or unsupported data file; every configured variable absent from the file; a code used twice; persona fields the template does not know; `min_rated` out of range; fewer than 3 candidates. Exit code 1 if anything is listed, 2 if the config itself cannot be loaded. | §8, plus the slips most likely when filling in codes by hand. |
| A5 | **Processing order:** (1) drop respondents with a missing or non-positive weight; (2) set thermometers outside `valid_range` to missing; (3) drop respondents who rated fewer than `min_rated` candidates. Counts go to the log and to `exclusions.json`. | §8. ⚑ Step 1 means the unweighted analysis also excludes zero-weight cases. If you want them in the unweighted electorate, say so. |
| A6 | **Value labels.** A persona field with no `value_labels` entry is used as is, with negative numbers and blanks treated as missing. With an entry, a listed code gets its label and an **unlisted code is missing**; a `"*": "{value}"` key lets unlisted non-negative values through (for age: `{80: "at least 80", "*": "{value}"}`). Codes match whether they arrive as 1, 1.0 or "1". | ANES uses many special codes per variable (−9, −8, 99 …). Defaulting unlisted codes to missing means a forgotten code drops a sentence rather than printing "My political views are 99." |
| A7 | `personas.parquet` stores labels per field, not rendered text; the text is rendered at run time from the current persona template. | The template hash is part of every call key; storing text would let the two drift apart. |
| A8 | Files are read by extension: `.csv` with pandas, `.dta` / `.sav` with pyreadstat; only the configured columns are loaded. Tested on all three with the same data. | §2: ANES ships in all three. |
| A9 | The test fixture (`tests/fixtures/anes_synthetic.csv`, 400 rows, with its generator and election YAML) uses fake `SYN_*` codes and fictional candidates. Thermometers come from a one-dimensional latent position plus noise. | §13 M5. It is not ANES data and should never be read as results. |
| A10 | In a run config, `election: {source: anes, id: …}` takes candidates and question text from the election config; `candidates:` may name a subset (a candidate set, for H2). The stratum for `--limit-respondents` is the party_id_7 label. With `backend: mock`, the mock's hidden utilities are each respondent's own thermometer scores (50 where missing). | A mock run on real personas is then a known, noisy copy of the human electorate — a useful end-to-end check of M7 before spending GPU time. |
| A11 | `configs/runs/prod_example.yaml` is written but cannot run until the 2024 TODOs are filled and M6 lands; a test checks that it parses and that it is refused for the right reason. | §3 lists the file. |
