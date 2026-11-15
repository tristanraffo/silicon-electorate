# SPEC — `silicon-electorate`: Arrow's Domain Condition in LLM Electorates

**Owner:** Tristan (researcher). **Builder:** you (Fable).
**Build window:** now → Nov 16, 2026, 11:59 pm PT. After that, Tristan runs and extends this alone.
**Read this whole file before writing code.** It is the contract. Where it is silent, choose the simplest option and log it in `DECISIONS.md`.

---

## 0. How to work (read first: usage is scarce)

1. Put this file in the repo as `SPEC.md`. Create `CLAUDE.md` containing one line: "Read SPEC.md and PROGRESS.md before doing anything."
2. Work **milestone by milestone, in the order of §12**. Do not start milestone N+1 until N's acceptance tests pass.
3. After each milestone: run the tests, `git commit -m "M<N>: <summary>"`, and update `PROGRESS.md` (done / next / known issues / exact command to resume). A fresh session must be able to resume from `PROGRESS.md` alone.
4. **Do not:**
   - download large models (anything over 2 GB);
   - run experiments at scale;
   - write analysis prose or a paper;
   - invent ANES variable codes. Unknown codes stay `TODO` in config and fail loudly at runtime.
5. Ask Tristan only when truly blocked. Otherwise pick the default, record it in `DECISIONS.md`, and continue.
6. Optimize for correctness and testability over breadth. If time runs short, a fully tested M1–M4 is worth more than a buggy M1–M7.

---

## 1. Research context (so your design choices are informed)

**Question.** Arrow's theorem shows that no aggregation rule satisfies Pareto, IIA and non-dictatorship on the *unrestricted* domain. On restricted domains it does not bind: if preferences are single-peaked, majority rule is transitive and satisfies the other conditions (Black), and the same holds under Sen's value restriction. **Do LLM electorates generate preference profiles that sit in an Arrow-escaping domain more often than human electorates do?**

**Hypotheses** (all must be measurable by the tool):
- **H1 (monoculture):** LLM electorates show fewer majority cycles, more single-peakedness, and lower preference dimensionality than matched human electorates.
- **H2 (dimensionality moderation):** the human–LLM gap is largest for candidate sets where human preferences are most multidimensional.
- **H3 (heterogeneity):** mixed-model electorates are less homogeneous than single-model electorates, holding the persona distribution fixed.
- **H4 (individual coherence):** an LLM voter's pairwise preferences may be intransitive, or inconsistent with its own full ranking. That would mean Arrow's individual-rationality premise fails at the voter level.
- **H5 (training stage), exploratory:** base vs. SFT vs. DPO checkpoints of one model family differ on the metrics above.

**Core design: matched silicon sampling.** Each ANES respondent has a human ballot, built from their feeling thermometers. An LLM persona built from that respondent's covariates produces an LLM ballot for the same candidates. You end up with human and LLM electorates matched one person at a time.

---

## 2. Environment

- Python 3.11. Manage dependencies with `uv` (`pyproject.toml`). Lint with `ruff`. Tests with `pytest`.
- **Two runtimes:**
  - **Dev:** Apple Silicon M1, 8 GB RAM. CPU/MPS only. vLLM does not run here, so use the `transformers` backend with a tiny model (`Qwen/Qwen2.5-0.5B-Instruct`) and the `mock` backend.
  - **Prod:** vast.ai CUDA instance with one 24 GB GPU (RTX 3090/4090). `vllm` backend.
- GPU dependencies go in an optional extra (`uv sync --extra gpu`), so the dev install never pulls vLLM or CUDA.
- Core dependencies: `numpy`, `pandas`, `pyarrow`, `pydantic`, `pyyaml`, `typer`, `tqdm`, `scipy`, `scikit-learn` (PCA only), `transformers`, `torch`, `pyreadstat` (ANES ships in .sav/.dta/.csv).
- Everything is seeded. A run is fully determined by its config file plus the model revision hash.

---

## 3. Repo layout

```
silicon-electorate/
  SPEC.md  CLAUDE.md  PROGRESS.md  DECISIONS.md  README.md
  pyproject.toml
  configs/
    elections/anes2016_ts.yaml  anes2016_pilot.yaml  anes2024_ts.yaml
    models/models.yaml
    prompts/persona_v1.yaml  thermometer_v1.yaml  ranking_v1.yaml  pairwise_v1.yaml
    runs/dev_smoke.yaml  prod_example.yaml
  data/            # gitignored. raw/ (Tristan drops ANES files here), processed/
  results/         # gitignored. raw call logs, ballot tables, metrics
  src/scv/
    ballots.py        # weak-order type and conversions
    rules.py          # social choice engine
    domains.py        # single-peakedness, value restriction, dimensionality
    nulls.py          # synthetic electorates
    anes.py           # loaders and persona construction
    prompts.py        # template rendering and hashing
    backends/  base.py  mock.py  hf.py  vllm_backend.py
    elicit.py         # thermometer / ranking / pairwise elicitation
    runner.py         # resumable batch runner
    electorates.py    # human / homogeneous / heterogeneous assembly
    analysis.py       # metrics, bootstrap, n-curves
    cli.py
  tests/
  notebooks/01_results.ipynb   # thin: calls analysis.py, draws figures
```

---

## 4. Ballot representation (`ballots.py`)

- A ballot is a **weak order** over a candidate set `C` (size m, 3 ≤ m ≤ 7). Represent it as an ordered list of tie groups, e.g. `[["A"], ["B","C"], ["D"]]`. Unranked or missing candidates are a separate field `missing: set`.
- Functions:
  - `from_scores(dict[str, float|None]) -> WeakOrder`: higher score ranks higher, equal scores tie, `None` means missing.
  - `pairwise(ballot) -> dict[(a,b), {+1, 0, -1}]`: 0 means tie or missing.
  - `is_strict(ballot)`
  - `linear_extensions(ballot)`: generator; m ≤ 7, so brute force is fine.
- Ties are **kept**, never broken at random by default. Any rule that needs a strict order must state how it handles ties (see §5).

---

## 5. Social choice engine (`rules.py`)

Input: a `Profile` = list of ballots plus a weight per ballot (default 1.0; ANES survey weights optional), and the candidate list. Pure Python/numpy, with no LLM dependency.

Compute:
1. **Weighted pairwise majority matrix** `N[a][b]`: the total weight of voters strictly preferring a to b. Ties and missing contribute 0 to both sides.
2. **Condorcet winner** (beats every other candidate strictly) and **Condorcet loser**. Either may be `None`.
3. **Majority cycle detection:** whether the strict-majority tournament has a cycle; the **Smith set** and its size; and the list of all 3-cycles.
4. **Rules**, each returning a winner and a full social ranking:
   - Plurality (ties at the top split the voter's weight equally among the tied candidates).
   - Borda (weak-order version: each candidate scores the number of candidates it strictly beats plus 0.5 × the number it ties).
   - Copeland (win = 1, tie = 0.5).
   - Minimax.
   - Kemeny (brute force; m ≤ 7).
   - IRV (each ballot counts for its highest-ranked remaining candidate; tied tops split fractionally; eliminate the lowest each round; break ties deterministically by a seeded order and **log** them).
5. **Rule-disagreement matrix:** for each pair of rules, whether their winners differ.
6. **IRV monotonicity probe:** for each ballot where the IRV winner w is not ranked first, move w up one position, rerun IRV, and record whether the winner changes away from w. Report the fraction of ballots where this happens (a monotonicity failure).

**Tests (required, with exact known answers):**
- Condorcet paradox (3 voters, A>B>C, B>C>A, C>A>B): cycle = True, no Condorcet winner, Smith set = {A,B,C}.
- Borda ≠ Condorcet: 2 voters R>G>B and 3 voters B>R>G.
  - Pairwise: B beats R 3–2, R beats G 5–0, B beats G 3–2, so the **Condorcet winner is B**.
  - Borda (2/1/0 scoring): **R=7, B=6, G=2**, so the Borda winner is R.
  - Assert both results.
- A textbook IRV monotonicity failure (find one in the literature or construct one; document the source in the test).
- Kemeny equals the Condorcet ranking whenever the majority relation is transitive (property test on 200 random single-peaked profiles).
- Weights: duplicating a ballot gives the same result as giving it weight 2.

---

## 6. Domain diagnostics (`domains.py`): the core outcome variables

1. **Single-peakedness (strict ballots):** a ballot is single-peaked on axis σ (an ordering of C) if utility falls monotonically on both sides of the voter's peak. Enumerate all m!/2 axes (m ≤ 7 means at most 2,520).
   - `sp_axes(profile)`: the axes on which **every** ballot is single-peaked.
   - `sp_max_consistent(profile) -> (axis, weight_share)`: the axis that maximizes the weight share of single-peaked ballots. **Primary metric: `sp_deletion_share = 1 − best_share`** (how much of the electorate must be removed to make it single-peaked).
2. **Weak ballots:** a weak order counts as single-peaked on σ if **some** linear extension of it is. Also implement the stricter `single_plateaued` check and report both.
3. **Sen's value restriction:** for every triple, check whether some candidate is never-best, never-middle or never-worst across all ballots (use strict pairwise relations within the triple). Report the share of triples satisfying it.
4. **Dimensionality** (on the score matrix, voters × candidates; thermometer scores for humans and LLMs): explained-variance share of PC1 and PC2 after centering each voter's scores. Report PC1 share as a monoculture metric.
5. **Profile diversity:** number of distinct ballots / n; entropy of the ballot distribution; mean pairwise Kendall-tau distance between voters.

**Tests:**
- Profiles generated from a 1-D spatial model (§7) must give `sp_deletion_share == 0`, and the recovered axis must match the true axis up to reversal.
- Impartial-culture profiles with m = 5 and n = 1,000 must give a high deletion share. Assert > 0.3, and report the value you observe in `DECISIONS.md`.
- The Condorcet paradox fails value restriction on its single triple.

---

## 7. Null and synthetic electorates (`nulls.py`)

- **Impartial culture (IC):** uniform random strict orders. Calibration test: with m = 3, n = 1,001 and 20,000 draws, the cycle frequency must fall within 0.5 percentage points of **8.77%** (Guilbaud).
- **1-D spatial:** candidates and voters drawn on a line, with preferences by distance. Must be single-peaked by construction (used in the tests).
- **2-D spatial:** for intuition and calibration curves.
- **Permutation null for a real profile:** independently permute each candidate's column of scores across voters. This keeps each candidate's distribution but destroys correlation between candidates. It answers whether the observed structure comes from correlation across candidates.

---

## 8. ANES data and personas (`anes.py`)

**Tristan downloads the ANES files manually** (registration required) into `data/raw/`. You never download them.

Each election config (`configs/elections/*.yaml`) contains:
```yaml
id: anes2024_ts
file: data/raw/anes_timeseries_2024_csv.csv     # Tristan fills in the real filename
weight_var: TODO
candidates:            # display name -> thermometer variable code
  Kamala Harris: V241156PRE
  Robert F. Kennedy Jr.: V241159PRE
  Josh Shapiro: V241160PRE
  Donald Trump: TODO
valid_range: [0, 100]   # everything else is treated as missing (ANES uses negative codes and 998/999)
min_rated: 3            # respondents rating fewer candidates are excluded (logged count)
persona_vars:           # all TODO until Tristan fills in from the codebook
  age: TODO
  gender: TODO
  race_ethnicity: TODO
  education: TODO
  income: TODO
  state_or_region: TODO
  party_id_7: TODO
  ideology_7: TODO
  religious_attendance: TODO
  vote_prior: TODO
value_labels: {}        # optional: code -> human-readable text, per persona variable
question_text: TODO     # the exact ANES thermometer wording, pasted from the questionnaire
```
Variable codes confirmed so far (2024 Time Series, pre-election): Harris `V241156PRE`, RFK Jr. `V241159PRE`, Shapiro `V241160PRE`. Everything else stays TODO. Write `scv data check <election>`, which prints every TODO and every variable missing from the file and exits non-zero.

**Outputs:**
- `data/processed/<election>/human_scores.parquet`: respondent_id, weight, a score per candidate.
- `data/processed/<election>/personas.parquet`: respondent_id plus the persona fields, as human-readable labels.
- Log the exclusion counts at every step.

**Persona rendering:** a first-person backstory built from a template (`configs/prompts/persona_v1.yaml`). Fields that are missing are left out of the sentence, never filled with "unknown". Templates are versioned: the hash of the template text is stored with every call.

---

## 9. Model backends (`backends/`)

A common interface:
```python
class Backend:
    def generate(self, prompts: list[ChatPrompt], *, temperature, max_tokens, n, seed) -> list[list[str]]
    def next_token_logprobs(self, prompts: list[ChatPrompt], candidates: list[str]) -> list[dict[str, float]]
    def info(self) -> dict   # model id, revision hash, dtype, backend name
```
- `mock`: deterministic and configurable. It can impersonate a voter with a hidden utility vector, plus position bias and noise. **Every test of elicitation and the runner uses mock.**
- `hf` (transformers): dev on Mac; also the fallback.
- `vllm_backend`: prod. Use offline batched `LLM.generate`, and `logprobs=20` for next-token probabilities.
- Always apply the model's own chat template. For models that have a thinking mode, **turn it off** (e.g. Qwen3 `enable_thinking=False`). Base models get a plain-completion prompt (§10.4).
- `next_token_logprobs` must handle tokenization correctly: the answer labels (`"1"`, `"2"`) must each be a single token after the assistant-turn prefix; assert this per model at startup. Return log-probs normalized over the candidate labels only.

`configs/models/models.yaml` (**verify every ID and its license gating on Hugging Face before finalizing; Llama and Gemma require accepting a license and an HF token**):
```yaml
homogeneous_default: llama31_8b
pool:            # heterogeneous arm
  llama31_8b:  {hf_id: meta-llama/Llama-3.1-8B-Instruct, dtype: bfloat16}
  qwen25_7b:   {hf_id: Qwen/Qwen2.5-7B-Instruct, dtype: bfloat16}
  mistral_7b:  {hf_id: mistralai/Mistral-7B-Instruct-v0.3, dtype: bfloat16}
  gemma2_9b:   {hf_id: google/gemma-2-9b-it, dtype: bfloat16}
training_stage:  # H5
  olmo2_base:  {hf_id: allenai/OLMo-2-1124-7B, base: true}
  olmo2_sft:   {hf_id: allenai/OLMo-2-1124-7B-SFT}
  olmo2_dpo:   {hf_id: allenai/OLMo-2-1124-7B-DPO}
  olmo2_inst:  {hf_id: allenai/OLMo-2-1124-7B-Instruct}
dev:
  tiny:        {hf_id: Qwen/Qwen2.5-0.5B-Instruct}
```
If a newer, clearly better model in the same size class (≤ 9B, fits in 24 GB at bf16) exists, you may add it to `pool`, but don't replace the defaults. Note the addition in `DECISIONS.md`.

---

## 10. Elicitation (`elicit.py`)

For every (election, respondent, model) there are three elicitation modes. Each call is one row in the call log (§11).

### 10.1 Thermometer (the headline comparison with humans)
- Prompt: persona + the exact ANES question text + one candidate. **One call per candidate** (independent by default). Option `sequential: true` asks all candidates in one multi-turn conversation in a randomized order, imitating the survey flow.
- Sampling: `n=5` at `temperature=1.0`, plus one greedy call (`temperature=0`). Parse the first integer in 0–100. The stored score is the median of the 5 samples. All samples are kept.
- Parse failure: retry up to 2 times with a short repair message, then record `NaN` and a failure flag.

### 10.2 Full ranking
- Prompt: persona + the candidate list in a **randomized order per call** (the order is stored) + "Rank these candidates from most to least preferred. Ties are allowed; write tied candidates on the same line separated by '='. Output only the ranking, one line per rank."
- `n=3` at T=1.0, plus greedy. Parse into a weak order; validate that it covers every candidate exactly once. Retries as in 10.1.

### 10.3 Pairwise, by log-probs (for H4)
- For every unordered pair {a, b}: prompt "Which do you prefer? 1: a  2: b. Answer with 1 or 2." Read P("1") vs P("2") from next-token log-probs. Run **both presentation orders**.
- Store both raw probabilities. `p(a≻b) = 0.5·[P(1 | a first) + P(2 | b first)]`. `position_bias = 0.5·[P(1|a first) + P(1|b first)] − 0.5`.
- Derived per voter:
  - **stochastic-transitivity violations** (WST: p(a≻b) ≥ .5 and p(b≻c) ≥ .5 but p(a≻c) < .5);
  - **strict 3-cycles** in the majority-of-probability relation;
  - **ranking–pairwise agreement** (the share of pairs where the sign of p − .5 matches the voter's own full ranking).

### 10.4 Base models (OLMo base only)
No chat template. Use a completion frame such as "Survey transcript.\n[persona]\nQ: …\nA:". Thermometer and pairwise modes only. Log that the format differs; this is a known confound, documented in `DECISIONS.md`.

### 10.5 Prompt hygiene
- Candidate order is randomized per call, with a seed derived from `(run_seed, respondent_id, candidate_set, call_type)`.
- Persona and candidate names appear identically across models.
- Never mention "Arrow", "voting theory", "consistency" or "cycles" to the model.

---

## 11. Runner, logging, resumability (`runner.py`)

- **Call key** = hash(election, respondent_id, model_key, mode, candidate or pair, presentation order, sample index, prompt-template hash). Before running, load the existing log and **skip completed keys**. The runner must survive being killed at any moment (vast.ai preemption) and resume without duplicating work.
- Log format: append-only Parquet shards in `results/calls/<run_id>/part-XXXXX.parquet`, flushed every N=500 calls. Fields: key, run_id, election, respondent_id, model_key, hf_id, revision, backend, mode, candidates_shown (ordered), prompt_hash, temperature, seed, sample_idx, raw_text, parsed, logprobs (json), parse_ok, retries, latency_ms, timestamp.
- `scv elicit --config runs/<x>.yaml --dry-run` prints the call counts by mode and model, plus a token and time estimate (assume vLLM throughput is a config parameter). **Tristan runs a dry run before every prod run.**
- `--limit-respondents K`, deterministic stratified subsampling (by party_id_7) for pilots.
- Running out of GPU memory, or the model failing to load, must fail fast with a clear message.

---

## 12. Electorates and analysis (`electorates.py`, `analysis.py`)

**Key design:** every persona is run on every model **once**. Electorates are then assembled *after the fact* from the ballot table, so no extra inference is needed:
- **Human:** ANES thermometer ballots.
- **Homogeneous(model k):** all personas' ballots from model k.
- **Heterogeneous:** each persona is assigned to one model in the pool by a seeded assignment **stratified on party_id_7**, so every model gets the same persona mix. Draw R=50 assignments and report the mean and spread.
- **Nulls:** IC with the same n and m; the permutation null of the human profile.

For each electorate × election × ballot source (thermometer-derived, ranking-derived), compute all §5–§6 metrics.

**Uncertainty and the effect of n:**
- Voter bootstrap (B=1,000), with weights.
- **n-curves:** cycle frequency, Condorcet-winner existence and SP deletion share for random sub-electorates of size n ∈ {11, 25, 51, 101, 251, 501}, with 500 draws each. This matters because at very large n, aggregate structure almost always yields a Condorcet winner, so the informative comparisons are at small to moderate n.
- **Matched comparisons:** per respondent, Kendall's tau-b between the human and LLM ballots, top-choice agreement, and the absolute thermometer error. Break these down by party ID and ideology.

**Output:** `results/metrics/<run_id>/*.parquet` plus a single tidy `summary.csv`. The notebook only reads these.

---

## 13. Milestones (in priority order, with acceptance criteria)

| # | Milestone | Done when |
|---|---|---|
| M1 | `ballots.py` and `rules.py` | All §5 tests pass, including the hand-verified examples |
| M2 | `domains.py` and `nulls.py` | §6 and §7 tests pass; the IC 8.77% calibration passes |
| M3 | Backends `mock` and `hf`, plus `elicit.py` | An end-to-end test with mock recovers the hidden utilities' ranking for more than 95% of mock voters at noise 0; the tiny-model smoke test runs on CPU for 5 respondents × 3 candidates |
| M4 | `runner.py` with resumability and dry run | A test kills the runner mid-run, restarts it, and checks there are zero duplicate keys and the call count is complete |
| M5 | `anes.py`, data check, persona rendering | Runs on a **synthetic ANES-shaped CSV you generate in `tests/fixtures`**, since the real data may not be present; `scv data check` lists the TODOs |
| M6 | `vllm_backend.py` and the vast.ai runbook in the README | The code path exists and mirrors `hf`; unit tests mock vLLM; the README has copy-paste commands (instance image, `uv sync --extra gpu`, HF login, `--dry-run`, run, `rsync` the results back, **destroy the instance**) |
| M7 | `electorates.py`, `analysis.py`, notebook | `scv analyze` runs on a mock-generated run and produces `summary.csv` and three figures: cycle-rate n-curves by electorate; SP deletion share by electorate × election; PC1 share, human vs. LLM |

If you can't finish, stop at a green milestone, commit, and write in `PROGRESS.md` exactly what remains.

---

## 14. Open decisions (use the defaults; Tristan may override)

1. Thermometer calls are independent (default) or sequential.
2. The persona template wording (default: first-person backstory, `persona_v1`).
3. Whether ANES survey weights are used in the headline metrics (default: report both, with weighted as the headline).
4. Whether the heterogeneous arm includes the OLMo instruct checkpoint (default: no; keep the training-stage set separate).
5. Pilot size (default: 300 respondents per election via `--limit-respondents`, before any full run).
