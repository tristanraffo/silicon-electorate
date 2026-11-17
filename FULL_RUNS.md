# Full-scale runs

These are the runs on **every respondent**, as opposed to the 300-respondent pilots.
Everything full-scale is kept apart from the pilots:

| | Pilots (done) | Full scale (this file) |
|---|---|---|
| Run configs | `configs/runs/*.yaml` | `configs/runs/full/*.yaml` |
| `run_id` | `anes2024_pilot_v4`, `anes2016_ts_300_v4`, `anes2016_pilot_300_v4` | `anes2024_ts_full_v4`, `anes2016_ts_full_v4`, `anes2016_pilot_full_v4` |
| Call logs | `results/calls/<pilot run_id>/` | `results/calls/<full run_id>/` |
| Analysis | `results/metrics/<pilot run_id>/` | `results/metrics/<full run_id>/` |

The full configs use the same seed, models, modes and prompts as the pilots; only
`limit_respondents` is gone. The pilots are not reused: a full run redoes its 300 people.

## Size and time

Times are measured from the pilot logs (one 24 GB GPU, inference only) and scaled by the
number of respondents. Add a few minutes per run for loading the four models.

| Config | Respondents | Calls | Estimated time |
|---|---|---|---|
| `full/anes2024_ts_full_v4.yaml` | 5,375 | 860,000 | about 2.9 h |
| `full/anes2016_ts_full_v4.yaml` | 3,415 | 546,400 | about 1.7 h |
| `full/anes2016_pilot_full_v4.yaml` | 1,199 | 258,984 | about 0.75 h |
| All three, one after another | | 1,665,384 | about 5.5 h |

The dry run prints its own estimate from placeholder speeds; it comes out about 15% low.
The call logs are small (about 60 MB for the largest run).

## Which GPU

- **Same kind of instance as the pilots, one run after another:** about 5.5 hours,
  unattended, safe to resume. Nothing new to go wrong. This is the default.
- **Three instances at once, one election each:** about 3 hours of waiting (the 2024 run
  sets the pace) for three setups. No code or config changes; the logs cannot collide
  because each election has its own `run_id`.
- **One larger GPU (48–80 GB):** should be faster, but by how much is not measured here.
  To benefit, raise `batch_size` in the config (256 is the number of prompts sent to the
  engine at a time) and probably vLLM's `max_num_seqs` under `backend_options`; neither
  has been tried. Do the 20-respondent check below first and compare speeds before
  committing to it.

Any instance needs a host with **Max CUDA 13.0 or higher** (README, runbook step 1),
about 120 GB of disk and at least 32 GB of RAM.

## Commands

Set up the instance exactly as in the README runbook, steps 1 to 4 (rent, copy the repo
and `data/processed` up, `uv sync --extra gpu`, `uv run hf auth login`). Then, on the
instance, in `/workspace/silicon-electorate`, inside `tmux`:

**1. Dry run** (no model is loaded; `calls` should match the table above):

```bash
for c in anes2024_ts_full_v4 anes2016_ts_full_v4 anes2016_pilot_full_v4; do uv run scv elicit --config configs/runs/full/$c.yaml --dry-run | tail -1; done
```

**2. Twenty-respondent check on this instance** (a few minutes; it confirms the machine
works and is the start of the real log, so nothing is wasted):

```bash
uv run scv elicit --config configs/runs/full/anes2016_pilot_full_v4.yaml --limit-respondents 20
```

```bash
uv run python scripts/ranking_report.py anes2016_pilot_full_v4
```

Parse rates should look like the pilots', and the last line (pairwise answers that are
exactly 50/50) should be near 0 for every model.

**3. The runs.** One instance, one after another, shortest first:

```bash
for c in anes2016_pilot_full_v4 anes2016_ts_full_v4 anes2024_ts_full_v4; do uv run scv elicit --config configs/runs/full/$c.yaml; done 2>&1 | tee full_runs.log
```

Or one election per instance:

```bash
uv run scv elicit --config configs/runs/full/anes2024_ts_full_v4.yaml 2>&1 | tee full_2024.log
```

```bash
uv run scv elicit --config configs/runs/full/anes2016_ts_full_v4.yaml 2>&1 | tee full_2016_ts.log
```

```bash
uv run scv elicit --config configs/runs/full/anes2016_pilot_full_v4.yaml 2>&1 | tee full_2016_pilot.log
```

Detach from tmux with `Ctrl-b` then `d`. If anything stops a run, give the same command
again: finished calls are skipped.

**4. Check before leaving the instance:**

```bash
uv run python scripts/ranking_report.py anes2024_ts_full_v4 anes2016_ts_full_v4 anes2016_pilot_full_v4
```

**5. Copy the results back** (on the Mac, with `REPO`, `PORT`, `HOST` set; once per
instance):

```bash
rsync -az -e "ssh -p ${PORT:?}" root@${HOST:?}:/workspace/silicon-electorate/results/ "${REPO:?}/results/"
```

Then, from the project folder on the Mac, confirm every log is whole (`remaining=0`):

```bash
for c in anes2024_ts_full_v4 anes2016_ts_full_v4 anes2016_pilot_full_v4; do uv run scv elicit --config configs/runs/full/$c.yaml --dry-run | tail -1; done
```

**6. Destroy the instance(s).**

## Analysis (on the Mac)

Analysis time at this size has not been measured, so time one run at reduced settings
first:

```bash
time uv run scv analyze --config configs/runs/full/anes2016_pilot_full_v4.yaml --replicates 10
```

Then the full settings (the defaults: bootstrap 1,000, draws 500, replicates 50), default
ballot and combined-ranking ballot:

```bash
for c in anes2024_ts_full_v4 anes2016_ts_full_v4 anes2016_pilot_full_v4; do uv run scv analyze --config configs/runs/full/$c.yaml; done
```

```bash
for c in anes2024_ts_full_v4 anes2016_ts_full_v4 anes2016_pilot_full_v4; do uv run scv analyze --config configs/runs/full/$c.yaml --ranking-ballot combined; done
```

Each output folder contains `analysis_settings.json`, which records the options, the
number of respondents and the time it was written.

## Before spending the GPU time

A full run fixes the prompts, persona labels and candidate sets it was made with.
Changing any of them afterwards means a new `run_id` and another full run. Settle the
flagged decisions in `DECISIONS.md` first (PROGRESS.md lists them), in particular the
candidate sets (S2, C3) and the ranking prompt (G10, G14).
