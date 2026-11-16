"""SPEC §11: runner, call log, resumability, dry run."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import Counter

import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from scv.backends.base import Backend, BackendError
from scv.backends.mock import MockBackend
from scv.cli import app
from scv.elicit import ElicitConfig, Respondent, plan_calls
from scv.runner import (
    CALL_SCHEMA,
    CallLog,
    RunConfig,
    call_log_dir,
    dry_run,
    load_models,
    load_respondents,
    load_run_config,
    plan_run,
    read_calls,
    run_calls,
    run_from_config,
    stratified_subsample,
    synthetic_respondents,
    synthetic_utilities,
)

CANDS = ["Alice Smith", "Bob Jones", "Carol White"]
QUESTION = "How would you rate this person? (test wording)"


def run_config(tmp_path, **overrides) -> RunConfig:
    base = {
        "run_id": "t",
        "seed": 3,
        "backend": "mock",
        "models": ["tiny"],
        "election": {
            "id": "syn",
            "candidates": CANDS,
            "question_text": QUESTION,
            "n_respondents": 30,
        },
        "flush_every": 50,
        "batch_size": 8,
        "results_dir": str(tmp_path / "results"),
    }
    return RunConfig.model_validate({**base, **overrides})


class Crash(RuntimeError):
    pass


class FlakyBackend(Backend):
    """Dies (as if preempted) after a fixed number of backend calls."""

    def __init__(self, inner: Backend, die_after: int):
        self.inner, self.left = inner, die_after

    def _tick(self):
        if self.left <= 0:
            raise Crash("preempted")
        self.left -= 1

    def info(self):
        return self.inner.info()

    def generate(self, prompts, **kw):
        self._tick()
        return self.inner.generate(prompts, **kw)

    def next_token_logprobs(self, prompts, candidates):
        self._tick()
        return self.inner.next_token_logprobs(prompts, candidates)


def world(n=30, sequential=False):
    respondents = synthetic_respondents(n, seed=1)
    cfg = ElicitConfig(
        election="syn", candidates=tuple(CANDS), question_text=QUESTION, sequential=sequential
    )
    calls = plan_calls(cfg, respondents, "tiny")
    return calls, synthetic_utilities(respondents, CANDS, seed=1)


# --------------------------------------------------------------------------------------
# M4 acceptance: kill mid-run, restart, zero duplicate keys, complete call count
# --------------------------------------------------------------------------------------


def test_sigkill_mid_run_then_resume_is_complete_with_no_duplicates(tmp_path):
    """A real process is SIGKILLed while writing shards, then the same command is run
    again. The log must end up with every planned key exactly once."""
    cfg = {
        "run_id": "killed",
        "seed": 0,
        "backend": "mock",
        "models": ["tiny", "qwen25_7b"],
        "election": {
            "id": "syn",
            "candidates": CANDS,
            "question_text": QUESTION,
            "n_respondents": 60,
        },
        "flush_every": 40,
        "batch_size": 8,
        "results_dir": str(tmp_path / "results"),
        "backend_options": {"delay_s": 0.01, "noise": 5.0},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, "-m", "scv.cli", "elicit", "--config", str(path)]
    log_dir = tmp_path / "results" / "calls" / "killed"
    planned = {c.key for calls in plan_run(load_run_config(path)).values() for c in calls}
    assert len(planned) == 2 * 60 * 28

    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while len(list(log_dir.glob("part-*.parquet"))) < 3:
            assert proc.poll() is None, "run finished before it could be killed"
            assert time.time() < deadline, "no shards appeared"
            time.sleep(0.01)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.wait()
    assert proc.returncode == -signal.SIGKILL

    partial = read_calls(tmp_path / "results", "killed")
    assert 0 < len(partial) < len(planned)  # really was interrupted part-way
    assert partial["key"].is_unique
    for shard in log_dir.glob("part-*.parquet"):  # every shard on disk is whole
        assert pq.read_table(shard).schema.equals(CALL_SCHEMA)

    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    final = read_calls(tmp_path / "results", "killed")
    assert len(final) == len(planned)  # complete call count
    assert final["key"].is_unique  # zero duplicate keys
    assert set(final["key"]) == planned
    assert not list(log_dir.glob("*.tmp"))
    # rows written before the kill were not touched by the resume
    merged = partial.merge(final, on="key", suffixes=("_before", "_after"))
    assert (merged["raw_text_before"] == merged["raw_text_after"]).all()
    assert (merged["timestamp_before"] == merged["timestamp_after"]).all()

    again = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert "written=0" in again.stdout and len(read_calls(tmp_path / "results", "killed")) == len(
        planned
    )


@pytest.mark.parametrize("die_after", [0, 1, 7, 23, 60])
def test_crash_and_resume_matches_an_uninterrupted_run(tmp_path, die_after):
    calls, utilities = world()
    clean = CallLog(tmp_path / "clean")
    run_calls(
        calls, MockBackend(utilities, noise=5.0), clean, "clean", flush_every=50, batch_size=8
    )
    reference = clean.read().set_index("key")
    assert len(reference) == len(calls)

    log = CallLog(tmp_path / "crashy")
    with pytest.raises(Crash):
        run_calls(
            calls,
            FlakyBackend(MockBackend(utilities, noise=5.0), die_after),
            log,
            "crashy",
            flush_every=50,
            batch_size=8,
        )
    partial = len(log.read())
    assert partial < len(calls)

    second = MockBackend(utilities, noise=5.0)
    stats = run_calls(calls, second, log, "crashy", flush_every=50, batch_size=8)
    if partial >= 100:  # the resume asks the model for strictly less than a full run
        full = MockBackend(utilities, noise=5.0)
        run_calls(calls, full, CallLog(tmp_path / "full"), "full", batch_size=8)
        assert sum(second.calls.values()) < sum(full.calls.values())
    assert (stats.planned, stats.skipped, stats.written) == (
        len(calls),
        partial,
        len(calls) - partial,
    )
    resumed = log.read()
    assert resumed["key"].is_unique and set(resumed["key"]) == set(reference.index)
    # determinism: interrupted + resumed gives the same answers as never interrupted
    resumed = resumed.set_index("key").loc[reference.index]
    assert (resumed["parsed"] == reference["parsed"]).all()
    assert (resumed["raw_text"] == reference["raw_text"]).all()

    idle = MockBackend(utilities)
    stats = run_calls(calls, idle, log, "crashy")
    assert stats.written == 0 and stats.skipped == len(calls)
    assert idle.calls == {"generate": 0, "logprobs": 0}  # finished work is never redone


def test_sequential_chains_resume_without_duplicates(tmp_path):
    """Kill the run at many different points inside the multi-turn conversations. A
    conversation cut off part-way is re-run whole, and its already-logged turns are not
    written twice."""
    calls, utilities = world(n=10, sequential=True)
    planned = {c.key for c in calls}
    chain_of = {c.key: c.chain for c in calls if c.chain}
    cut_mid_conversation = 0
    for die_after in range(27, 47):
        log = CallLog(tmp_path / f"seq{die_after}")
        with pytest.raises(Crash):
            run_calls(
                calls,
                FlakyBackend(MockBackend(utilities), die_after),
                log,
                "seq",
                flush_every=10,
                batch_size=4,
            )
        logged = set(log.read()["key"])
        started = {chain_of[k] for k in logged if k in chain_of}
        unfinished = {chain_of[k] for k in planned - logged if k in chain_of}
        cut_mid_conversation += bool(started & unfinished)

        backend = MockBackend(utilities)
        stats = run_calls(calls, backend, log, "seq", flush_every=10, batch_size=4)
        final = log.read()
        assert final["key"].is_unique, die_after
        assert set(final["key"]) == planned
        assert stats.written == len(planned) - len(logged)
    assert cut_mid_conversation >= 10  # the interesting case really is exercised


# --------------------------------------------------------------------------------------
# Call log mechanics
# --------------------------------------------------------------------------------------


def test_shards_are_flushed_every_n_rows_with_the_spec_schema(tmp_path):
    calls, utilities = world(n=10)  # 280 calls
    log = CallLog(tmp_path / "log")
    stats = run_calls(calls, MockBackend(utilities), log, "r1", flush_every=100, batch_size=8)
    assert stats.written == 280
    names = [p.name for p in log.shards()]
    assert names == [f"part-{i:05d}.parquet" for i in range(len(names))] and len(names) >= 2
    sizes = [pq.read_metadata(p).num_rows for p in log.shards()]
    assert sum(sizes) == 280 and all(100 <= s < 100 + 8 * 6 for s in sizes[:-1])
    for p in log.shards():
        assert pq.read_table(p).schema.equals(CALL_SCHEMA)
    df = log.read()
    assert list(df.columns) == CALL_SCHEMA.names
    assert set(df["run_id"]) == {"r1"}
    assert list(df["candidates_shown"].iloc[0]) != []
    assert log.keys() == {c.key for c in calls}


def test_empty_log_and_stale_temp_files(tmp_path):
    d = tmp_path / "log"
    d.mkdir()
    (d / "part-00007.parquet.tmp").write_bytes(b"half-written garbage")
    log = CallLog(d)
    assert not list(d.glob("*.tmp"))  # leftovers of a killed write are discarded
    assert log.keys() == set() and len(log.read()) == 0
    assert list(log.read().columns) == CALL_SCHEMA.names
    assert log.write([]) is None


def test_mixing_model_revisions_in_one_run_is_refused(tmp_path):
    calls, utilities = world(n=4)
    log = CallLog(tmp_path / "log")
    run_calls(calls[:20], MockBackend(utilities), log, "r")

    class Newer(MockBackend):
        def info(self):
            return {**super().info(), "revision": "deadbeef"}

    with pytest.raises(BackendError, match="mixing revisions"):
        run_calls(calls, Newer(utilities), log, "r")


# --------------------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------------------


def test_stratified_subsample_is_proportional_and_deterministic():
    strata = ["D"] * 50 + ["I"] * 30 + ["R"] * 20
    people = [Respondent(f"r{i:03d}", "p", stratum=s) for i, s in enumerate(strata)]
    sample = stratified_subsample(people, 10, seed=0)
    assert Counter(r.stratum for r in sample) == {"D": 5, "I": 3, "R": 2}
    assert [r.respondent_id for r in sample] == sorted(r.respondent_id for r in sample)

    assert stratified_subsample(people, 10, seed=0) == sample
    assert (
        stratified_subsample(people[::-1], 10, seed=0) == sample[::-1]
    )  # input order is irrelevant
    assert stratified_subsample(people, 10, seed=1) != sample
    # a bigger pilot contains the smaller one
    bigger = {r.respondent_id for r in stratified_subsample(people, 20, seed=0)}
    assert {r.respondent_id for r in sample} <= bigger and len(bigger) == 20

    # uneven split: 7 of 100 -> 3.5 / 2.1 / 1.4 -> 4 / 2 / 1 by largest remainder
    assert Counter(r.stratum for r in stratified_subsample(people, 7, seed=0)) == {
        "D": 4,
        "I": 2,
        "R": 1,
    }
    assert stratified_subsample(people, None, seed=0) == people
    assert stratified_subsample(people, 500, seed=0) == people
    with pytest.raises(ValueError):
        stratified_subsample(people, 0, seed=0)
    unlabelled = [Respondent(f"u{i}", "p") for i in range(10)]
    assert len(stratified_subsample(unlabelled, 4, seed=0)) == 4


def test_limit_respondents_applies_from_config_and_flag(tmp_path):
    cfg = run_config(tmp_path, limit_respondents=12)
    assert len(load_respondents(cfg)) == 12
    assert len(load_respondents(cfg, limit=5)) == 5  # the CLI flag wins
    assert len(load_respondents(run_config(tmp_path))) == 30
    assert {r.respondent_id for r in load_respondents(cfg, 5)} <= {
        r.respondent_id for r in load_respondents(cfg)
    }


# --------------------------------------------------------------------------------------
# Config, models, dry run, CLI
# --------------------------------------------------------------------------------------


def test_models_registry_and_config_validation(tmp_path):
    models = load_models("configs/models/models.yaml")
    assert set(models) == {
        "llama31_8b", "qwen25_7b", "mistral_7b", "gemma2_9b",
        "olmo2_base", "olmo2_sft", "olmo2_dpo", "olmo2_inst", "tiny",
    }  # fmt: skip
    assert models["tiny"]["hf_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert models["olmo2_base"]["base"] is True and "base" not in models["olmo2_sft"]
    assert models["llama31_8b"]["section"] == "pool"

    shipped = load_run_config("configs/runs/dev_smoke.yaml")
    assert shipped.run_id == "dev_smoke" and shipped.backend == "mock"
    with pytest.raises(ValueError):
        run_config(tmp_path, flush_evry=10)  # typos in a config must not pass silently
    with pytest.raises(ValueError, match="not in"):
        plan_run(run_config(tmp_path, models=["gpt-17"]))


def test_dry_run_counts_match_what_is_then_run(tmp_path):
    cfg = run_config(tmp_path, models=["tiny", "olmo2_base"], limit_respondents=10)
    table = dry_run(cfg)
    by = {(r.model, r.mode): r for r in table.itertuples()}
    # chat model: 10 x (3 x 6 thermometer, 4 ranking, 6 pairwise)
    assert (by[("tiny", "thermometer")].calls, by[("tiny", "ranking")].calls) == (180, 40)
    assert by[("tiny", "pairwise")].calls == 60
    # base model: no ranking mode at all
    assert ("olmo2_base", "ranking") not in by and by[("olmo2_base", "thermometer")].calls == 180
    assert (table["remaining"] == table["calls"]).all()
    assert by[("tiny", "thermometer")].requests == 60  # 30 prompts x (sampled, greedy)
    assert by[("tiny", "thermometer")].output_tokens == 180 * 8
    assert (table["prompt_tokens"] > 0).all() and (table["est_seconds"] > 0).all()
    assert not call_log_dir(cfg).exists()  # a dry run writes nothing

    stats = run_from_config(cfg)
    assert stats["tiny"].written == 280 and stats["olmo2_base"].written == 240
    calls = read_calls(cfg.results_dir, cfg.run_id)
    counts = calls.groupby(["model_key", "mode"]).size().to_dict()
    assert counts == {(r.model, r.mode): r.calls for r in table.itertuples()}
    after = dry_run(cfg)
    assert (after["remaining"] == 0).all() and (after["est_seconds"] == 0).all()

    # doubling the assumed throughput halves the estimate
    fast = run_config(
        tmp_path,
        run_id="fresh",
        limit_respondents=10,
        throughput={"prompt_tokens_per_s": 40_000, "output_tokens_per_s": 5_000},
    )
    slow = run_config(tmp_path, run_id="fresh", limit_respondents=10)
    assert dry_run(fast)["est_seconds"].sum() == pytest.approx(
        dry_run(slow)["est_seconds"].sum() / 2, abs=0.2
    )


def test_a_finished_model_is_not_loaded_again(tmp_path):
    cfg = run_config(tmp_path, limit_respondents=5)
    run_from_config(cfg)

    def explode(*_):
        raise AssertionError("backend loaded although nothing was left to do")

    stats = run_from_config(cfg, backend_factory=explode)
    assert stats["tiny"].written == 0 and stats["tiny"].skipped == stats["tiny"].planned


def test_cli_dry_run_and_run(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(run_config(tmp_path, limit_respondents=6).model_dump()))
    cli = CliRunner()
    out = cli.invoke(app, ["elicit", "--config", str(path), "--dry-run"])
    assert out.exit_code == 0, out.output
    assert "thermometer" in out.output and "TOTAL calls=168 remaining=168" in out.output
    assert not (tmp_path / "results").exists()

    out = cli.invoke(
        app, ["elicit", "--config", str(path), "--dry-run", "--limit-respondents", "2"]
    )
    assert "TOTAL calls=56 " in out.output

    out = cli.invoke(app, ["elicit", "--config", str(path)])
    assert out.exit_code == 0, out.output
    assert "tiny: planned=168 already_done=0 written=168" in out.output
    out = cli.invoke(app, ["elicit", "--config", str(path)])
    assert "tiny: planned=168 already_done=168 written=0" in out.output


def test_cli_fails_fast_when_the_backend_cannot_load(tmp_path):
    path = tmp_path / "run.yaml"
    cfg = run_config(tmp_path, backend="vllm", limit_respondents=2)
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    out = CliRunner().invoke(app, ["elicit", "--config", str(path)])
    assert out.exit_code == 1
    assert "vllm backend is not available" in out.output and "uv sync --extra gpu" in out.output
    assert len(read_calls(cfg.results_dir, cfg.run_id)) == 0
