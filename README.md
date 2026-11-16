# silicon-electorate

Do LLM electorates generate preference profiles that sit in an Arrow-escaping domain
(single-peaked, value-restricted, acyclic) more often than matched human electorates do?

`SPEC.md` is the contract. `PROGRESS.md` says what is built and how to resume.
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

`<election>` is an id from `configs/elections/` or a path to a YAML file.

## Preparing an election (once, on the Mac)

1. Put the ANES file in `data/raw/` and set `file:` in `configs/elections/<election>.yaml`.
2. Fill every `TODO` from the codebook, paste the exact thermometer wording into
   `question_text`, and write `value_labels` (`tests/fixtures/anes_synthetic.yaml` is a
   complete example).
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

Throughout, `$HOST` and `$PORT` are the SSH address vast.ai shows for the instance.

**1. Rent the instance.** One RTX 3090 or 4090 (24 GB), on-demand rather than
interruptible for the first run. Image: a plain CUDA 12 development image, for example
`nvidia/cuda:12.4.1-devel-ubuntu22.04` (the vast.ai "NVIDIA CUDA" template). Disk: at
least 120 GB; the four pool models are about 16 GB each. Python and torch come from
`uv`, so the image only needs the NVIDIA driver and CUDA libraries.

**2. Copy the repo and the processed data up.** The repo has no remote, so use rsync.
ANES data is registration-gated: only the processed tables go up, and they come back
down with the instance.

```bash
rsync -az -e "ssh -p $PORT" --exclude .venv --exclude results --exclude data --exclude .git ./ root@$HOST:/workspace/silicon-electorate/
```

```bash
rsync -az -e "ssh -p $PORT" data/processed root@$HOST:/workspace/silicon-electorate/data/
```

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
uv run python -c "from scv.runner import read_calls; d = read_calls('results', 'anes2024_pilot_v1'); print(d.groupby(['model_key', 'mode'])[['parse_ok', 'retries', 'latency_ms']].mean())"
```

```bash
uv run scv elicit --config configs/runs/prod_example.yaml
```

If the instance is preempted or the run crashes, run the same command again: finished
calls are skipped and a finished model is not reloaded. An out-of-memory error stops the
run with a message; lower `batch_size`, or add `gpu_memory_utilization` / `max_model_len`
under `backend_options` in the run config.

**7. Bring the results back (from the Mac).** The call log is the only thing that
cannot be regenerated without paying again.

```bash
rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/silicon-electorate/results/ results/
```

Check that it arrived whole before the next step:

```bash
uv run scv elicit --config configs/runs/prod_example.yaml --dry-run
```

`remaining` should be 0 for every model and mode.

**8. Destroy the instance.** Stopping it still bills for disk. In the vast.ai console
choose **Destroy**, or with the CLI (`vastai show instances` gives the id):

```bash
vastai destroy instance <INSTANCE_ID>
```

Then confirm the instance list is empty.
