# silicon-electorate

Do LLM electorates generate preference profiles that sit in an Arrow-escaping domain
(single-peaked, value-restricted, acyclic) more often than matched human electorates do?

`SPEC.md` is the contract. `PROGRESS.md` says what is built and how to resume.
`ANALYSIS_PLAN.md` says which numbers answer which hypothesis and how to read each
metric. `FULL_RUNS.md` has the commands for the full-scale runs (every respondent), which are
kept apart from the 300-respondent pilots.
`DECISIONS.md` records every choice the spec left open.

## Setup (dev, Apple Silicon)

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check .
```

The default test suite loads no model. The real-model smoke test (tiny Qwen, CPU, about
3 minutes) is opt-in:

```bash
uv run pytest -m hf -s
```

## Commands

| Command | What it does |
|---|---|
| `uv run scv data check <election>` | Lists every TODO and every configured variable missing from the data file. Non-zero exit until the config is ready. |
| `uv run scv data build <election>` | Writes `data/processed/<election>/human_scores.parquet` and `personas.parquet`. |
| `uv run scv elicit --config <run.yaml> --dry-run` | Call counts by model and mode, token and time estimate, what is already logged. |
| `uv run scv elicit --config <run.yaml>` | Runs or resumes the run. Safe to kill and rerun. |
| `… --limit-respondents K` | Pilot on K respondents, stratified on party ID. |
| `uv run scv analyze --config <run.yaml>` | Writes `results/metrics/<run_id>/`: `summary.csv`, Parquet tables and three figures. Options: `--bootstrap`, `--draws`, `--replicates`, `--seed`, `--no-figures`. The options used are saved in `analysis_settings.json`. |
| `… --ranking-ballot combined` | The same, with each voter's ranking ballot averaged over their four rankings instead of the greedy one alone (cancels position bias). Writes to `results/metrics/<run_id>__ranking_combined/`. |
| `… --ties break_random` / `--ties drop_flat` | Tie-robustness checks: break every tie at random, or leave out respondents with a flat thermometer ballot anywhere. Writes to `results/metrics/<run_id>__ties_<mode>/`. |
| `… --exclude-ranking <model>` | Sensitivity analysis: the same, with that model's ranking ballots left out. Writes to `results/metrics/<run_id>__ranking_without_<model>/`. |

`<election>` is an id from `configs/elections/` or a path to a YAML file.

## Trying the whole pipeline without data or a GPU

The dev config uses made-up personas and the mock backend (a noisy copy of made-up
"human" scores), so the numbers mean nothing, but every stage runs:

```bash
uv run scv elicit --config configs/runs/dev_smoke.yaml
```

```bash
uv run scv analyze --config configs/runs/dev_smoke.yaml --bootstrap 200 --draws 200 --replicates 10
```

`notebooks/01_results.ipynb` reads `results/metrics/*/` and redraws the figures.

## Preparing an election (once, on the Mac)

1. Put the ANES file in `data/raw/` and set `file:` in `configs/elections/<election>.yaml`.
2. Fill every `TODO` from the codebook, paste the exact thermometer wording into
   `question_text`, and write `value_labels` (`tests/fixtures/anes_synthetic.yaml` is a
   complete example). For a two-step question, `persona_fallbacks` names a second
   column to use where the first gives no label (see `configs/elections/anes2024_ts.yaml`).
3. Check, then build:

```bash
uv run scv data check anes2024_ts
```

```bash
uv run scv data build anes2024_ts
```

## Prod runbook: vast.ai, one 24 GB GPU

> **Status:** the vLLM backend is unit-tested against a fake `vllm` module only. It has
> not yet been run on a GPU. Do step 6 (the 20-respondent pilot) before anything larger,
> and expect to fix something the first time.

Throughout, `$HOST` and `$PORT` are the SSH address vast.ai shows for the instance (the
connect button on the instance card gives `ssh -p <PORT> root@<HOST>`), and `$REPO` is
this folder on the Mac. Set all three once per terminal window:

```bash
export REPO=/Users/tristanraffo/Downloads/code/Python/silicon-electorate PORT=<PORT> HOST=<HOST>
```

The copy commands below name `$REPO` in full, so they copy the same thing from any
folder, and they stop with an error if a variable is not set. (A bare `./` copies
whatever folder the terminal happens to be in; from a new window that is the whole home
directory.)

**1. Rent the instance.** One RTX 3090 or 4090 (24 GB), on-demand rather than
interruptible for the first run. Image: a plain CUDA 12 development image, for example
`nvidia/cuda:12.4.1-devel-ubuntu22.04` (the vast.ai "NVIDIA CUDA" template). Disk: at
least 120 GB; the four pool models are about 16 GB each. Python and torch come from
`uv`, so the image only needs the NVIDIA driver and CUDA libraries.

**The host's NVIDIA driver must support CUDA 13.0 or newer.** The locked torch and vLLM
builds are compiled for CUDA 13, and the driver belongs to the host machine, so it
cannot be updated from inside the instance. On the offer card look for "Max CUDA" and
pick 13.0 or higher. A host that is too old fails at step 3 with "The NVIDIA driver on
your system is too old (found version 12080)" (12080 means CUDA 12.8).

**2. Copy the repo and the processed data up.** The repo has no remote, so use rsync.
ANES data is registration-gated: only the processed tables go up, and they come back
down with the instance.

```bash
rsync -az -e "ssh -p ${PORT:?}" --exclude .venv --exclude results --exclude data --exclude .git "${REPO:?}/" root@${HOST:?}:/workspace/silicon-electorate/
```

```bash
rsync -az -e "ssh -p ${PORT:?}" "${REPO:?}/data/processed" root@${HOST:?}:/workspace/silicon-electorate/data/
```

The first copy takes seconds. If it runs for more than a minute, stop it (Ctrl-C) and
check `echo $REPO`.

**3. Install (on the instance).** `ssh -p $PORT root@$HOST`, then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

```bash
cd /workspace/silicon-electorate && uv sync --extra gpu
```

```bash
uv run python -c "import torch, vllm; print(torch.cuda.get_device_name(0), vllm.__version__)"
```

**4. Hugging Face login.** Llama 3.1 and Gemma 2 are gated: accept each licence on its
model page first (`meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`), with the
same account as the token. Qwen, Mistral and OLMo are open.

```bash
uv run hf auth login
```

**5. Dry run. Always.** Copy `configs/runs/prod_example.yaml`, give it a new `run_id`,
and read the counts and the time estimate before spending money. The throughput
numbers in the config are placeholders until the pilot has measured them.

```bash
uv run scv elicit --config configs/runs/prod_example.yaml --dry-run
```

**6. Pilot, then run.** Start inside `tmux` so a dropped SSH session does not stop it.
First a 20-respondent pilot to shake out the vLLM path, then the configured 300.

```bash
tmux new -s run
```

```bash
uv run scv elicit --config configs/runs/prod_example.yaml --limit-respondents 20
```

After the pilot, look at the parse rates and at `raw_text` for failures before going on:

```bash
uv run python -c "from scv.runner import read_calls; d = read_calls('results', 'anes2024_pilot_v4'); print(d.groupby(['model_key', 'mode'])[['parse_ok', 'retries', 'latency_ms']].mean())"
```

For the ranking mode, also check the greedy parse rate and how often a ranking simply
copies the order shown (chance is 1 in 24 with four candidates):

```bash
uv run python scripts/ranking_report.py anes2024_pilot_v4
```

```bash
uv run scv elicit --config configs/runs/prod_example.yaml
```

If the instance is preempted or the run crashes, run the same command again: finished
calls are skipped and a finished model is not reloaded. An out-of-memory error stops the
run with a message; lower `batch_size`, or add `gpu_memory_utilization` / `max_model_len`
under `backend_options` in the run config. The context length is capped at 4,096 tokens
by default, because vLLM otherwise reserves memory for a model's full context and Llama
3.1 (131,072 tokens) then does not fit on a 24 GB card.

**7. Bring the results back (from the Mac).** The call log is the only thing that
cannot be regenerated without paying again.

```bash
rsync -az -e "ssh -p ${PORT:?}" root@${HOST:?}:/workspace/silicon-electorate/results/ "${REPO:?}/results/"
```

Check that it arrived whole before the next step (this and the analysis run from the
project folder: `cd "$REPO"`):

```bash
uv run scv elicit --config configs/runs/prod_example.yaml --dry-run
```

`remaining` should be 0 for every model and mode.

After the results are back, analyse them on the Mac (no GPU needed):

```bash
uv run scv analyze --config configs/runs/prod_example.yaml
```

**8. Destroy the instance.** Stopping it still bills for disk. In the vast.ai console
choose **Destroy**, or with the CLI (`vastai show instances` gives the id):

```bash
vastai destroy instance <INSTANCE_ID>
```

Then confirm the instance list is empty.
