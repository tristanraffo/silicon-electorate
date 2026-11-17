"""SPEC §8: ANES loading, data check and personas, on a synthetic ANES-shaped file."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import pytest
import yaml
from typer.testing import CliRunner

from scv import anes
from scv.cli import app
from scv.elicit import GREEDY, thermometer_table
from scv.runner import (
    RunConfig,
    load_respondents,
    plan_run,
    read_calls,
    resolve_election,
    run_from_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
CSV = FIXTURES / "anes_synthetic.csv"
YAML = FIXTURES / "anes_synthetic.yaml"
CANDS = ["Avery Adams", "Blake Brooks", "Casey Clark", "Devon Diaz", "Emery Evans"]
FT = ["SYN_FT_A", "SYN_FT_B", "SYN_FT_C", "SYN_FT_D", "SYN_FT_E"]


@pytest.fixture(scope="module")
def config() -> anes.ElectionConfig:
    return anes.load_election(YAML)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def processed(config) -> anes.Processed:
    return anes.process(config)


def variant(tmp_path, **changes) -> anes.ElectionConfig:
    data = yaml.safe_load(YAML.read_text())
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return anes.ElectionConfig.model_validate(data)


# --------------------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------------------


def test_fixture_is_reproducible_and_anes_shaped(raw):
    spec = importlib.util.spec_from_file_location("make_fixture", FIXTURES / "make_anes_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regenerated = pd.read_csv(pd.io.common.StringIO(module.make().to_csv(index=False)))
    pd.testing.assert_frame_equal(regenerated, raw)

    assert len(raw) == 400 and all(c.startswith("SYN_") for c in raw.columns)  # fake codes
    values = raw[FT].to_numpy()
    assert (values == -9).any() and (values == 998).any() and (values == 999).any()
    assert (raw["SYN_WT"] == 0).any() and raw["SYN_WT"].isna().any()


# --------------------------------------------------------------------------------------
# scv data check
# --------------------------------------------------------------------------------------


def test_check_passes_on_the_fixture(config):
    assert anes.check(config) == []
    anes.require_ready(config)


def test_shipped_configs_list_their_todos_and_invent_no_codes():
    # 2024 and 2016 Time Series: filled from the documentation shipped with the data;
    # Tristan pasted the question wording from the questionnaires.
    for election, harris_or_clinton in (("anes2024_ts", "V241156"), ("anes2016_ts", "V161086")):
        c = anes.load_election(election)
        assert harris_or_clinton in c.candidates.values()
        assert "TODO" not in c.variables().values()
        assert all(not code.endswith(("PRE", "POST")) for code in c.variables().values())
        assert set(c.value_labels) == set(c.persona_vars) == set(anes.PERSONA_FIELDS)
        assert "religious_attendance" in c.persona_fallbacks
        assert c.question_text.endswith("How would you rate:")
        assert "TODO" not in anes.check(c) and "\n" not in c.question_text

    # 2016 pilot: also filled. Its documentation does not give the income labels, so that
    # field is left out rather than guessed.
    c = anes.load_election("anes2016_pilot")
    assert c.candidates["Hillary Clinton"] == "fthrc" and "TODO" not in c.variables().values()
    assert set(c.persona_vars) == set(anes.PERSONA_FIELDS) - {"income"}
    assert set(c.value_labels) == set(c.persona_vars)
    assert c.value_labels["age"][1997] == "19" and c.value_labels["age"][1921] == "95"
    regions = set(c.value_labels["state_or_region"].values())
    assert regions == {"the Northeast", "the Midwest", "the South", "the West"}
    assert len(c.value_labels["state_or_region"]) == 51
    assert c.question_text.endswith("How would you rate:")
    assert not any(p.startswith("TODO") for p in anes.check(c))


def test_check_reports_variables_missing_from_the_file(tmp_path):
    c = variant(tmp_path, weight_var="SYN_NOPE", candidates={"Emery Evans": "SYN_FT_Z"})
    problems = anes.check(c)
    assert "variable not in data file: SYN_NOPE (weight_var)" in problems
    assert "variable not in data file: SYN_FT_Z (candidates.Emery Evans)" in problems
    assert len(problems) == 2

    mixed = variant(tmp_path, weight_var="TODO", persona_vars={"age": "SYN_GONE"})
    assert anes.check(mixed) == [
        "TODO: weight_var",
        "variable not in data file: SYN_GONE (persona_vars.age)",
    ]


def test_check_reports_config_slips(tmp_path):
    assert any("not found" in p for p in anes.check(variant(tmp_path, file="data/raw/nope.csv")))
    assert any("unsupported" in p for p in anes.check(variant(tmp_path, file=str(YAML))))
    assert any("more than once" in p for p in anes.check(variant(tmp_path, weight_var="SYN_AGE")))
    assert any("min_rated" in p for p in anes.check(variant(tmp_path, min_rated=9)))
    assert any(
        "value_labels" in p
        for p in anes.check(variant(tmp_path, value_labels={"zodiac": {1: "x"}}))
    )
    bad = variant(tmp_path, persona_vars={"shoe_size": "SYN_UNRELATED_1"})
    assert any("shoe_size" in p for p in anes.check(bad))
    with pytest.raises(ValueError):  # a misspelled key is rejected outright
        variant(tmp_path, wieght_var="SYN_WT")
    with pytest.raises(anes.ConfigError, match="no election config"):
        anes.load_election("anes1789")


def test_cli_data_check_and_build(tmp_path):
    cli = CliRunner()
    ok = cli.invoke(app, ["data", "check", str(YAML)])
    assert ok.exit_code == 0 and "anes_synthetic: OK" in ok.output

    unfilled = tmp_path / "unfilled.yaml"
    data = yaml.safe_load(YAML.read_text())
    data.update(file="TODO", weight_var="TODO")
    data["candidates"]["Emery Evans"] = "TODO"
    unfilled.write_text(yaml.safe_dump(data))
    todo = cli.invoke(app, ["data", "check", str(unfilled)])
    assert todo.exit_code == 1  # non-zero
    assert "TODO: weight_var" in todo.output and "TODO: candidates.Emery Evans" in todo.output
    assert "TODO: file" in todo.output
    missing = tmp_path / "missing.yaml"
    data = yaml.safe_load(YAML.read_text())
    data["file"] = str(tmp_path / "nowhere.csv")
    missing.write_text(yaml.safe_dump(data))
    assert "data file not found" in cli.invoke(app, ["data", "check", str(missing)]).output

    broken = tmp_path / "broken.yaml"
    data = yaml.safe_load(YAML.read_text())
    data["candidates"]["Emery Evans"] = "SYN_FT_Z"
    broken.write_text(yaml.safe_dump(data))
    out = cli.invoke(app, ["data", "check", str(broken)])
    assert out.exit_code == 1 and "variable not in data file: SYN_FT_Z" in out.output
    assert cli.invoke(app, ["data", "check", "anes1789"]).exit_code == 2

    built = cli.invoke(app, ["data", "build", str(YAML), "--out", str(tmp_path / "processed")])
    assert built.exit_code == 0, built.output
    assert "rated fewer than 3 candidates: dropped" in built.output
    assert (tmp_path / "processed" / "anes_synthetic" / "human_scores.parquet").exists()
    refused = cli.invoke(app, ["data", "build", str(unfilled), "--out", str(tmp_path / "p2")])
    assert refused.exit_code == 1 and "not ready" in refused.output
    assert not (tmp_path / "p2").exists()


# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------


def test_exclusions_match_an_independent_count(raw, processed):
    has_weight = raw["SYN_WT"].notna() & (raw["SYN_WT"] > 0)
    valid = raw[FT].where((raw[FT] >= 0) & (raw[FT] <= 100))
    enough = valid.notna().sum(axis=1) >= 3
    assert processed.exclusions == [
        {"step": "rows in file", "dropped": 0, "remaining": 400},
        {
            "step": "missing or non-positive weight",
            "dropped": int((~has_weight).sum()),
            "remaining": int(has_weight.sum()),
        },
        {
            "step": "rated fewer than 3 candidates",
            "dropped": int((has_weight & ~enough).sum()),
            "remaining": int((has_weight & enough).sum()),
        },
    ]
    assert (~has_weight).sum() == 10 and (has_weight & ~enough).sum() >= 10

    human = processed.human_scores
    kept = raw[has_weight & enough]
    assert list(human.columns) == ["respondent_id", "weight", *CANDS]
    assert list(human["respondent_id"]) == [str(i) for i in kept["SYN_ID"]]
    assert np.allclose(human["weight"], kept["SYN_WT"]) and (human["weight"] > 0).all()
    expected = valid[has_weight & enough].to_numpy(dtype=float)
    assert np.array_equal(human[CANDS].to_numpy(), expected, equal_nan=True)


def test_scores_outside_the_valid_range_become_missing(processed):
    scores = processed.human_scores[CANDS]
    assert scores.isna().to_numpy().any()  # -9 / 998 / 999 became missing, not scores
    assert scores.min().min() >= 0 and scores.max().max() <= 100
    assert (scores.notna().sum(axis=1) >= 3).all()
    assert list(processed.personas["respondent_id"]) == list(
        processed.human_scores["respondent_id"]
    )


def test_valid_range_and_min_rated_are_configurable(tmp_path, processed):
    narrow = anes.process(variant(tmp_path, valid_range=[0, 50]))
    assert narrow.human_scores[CANDS].max().max() <= 50
    assert len(narrow.human_scores) < len(processed.human_scores)
    strict = anes.process(variant(tmp_path, min_rated=5))
    assert strict.human_scores[CANDS].notna().all().all()
    assert len(strict.human_scores) < len(processed.human_scores)


def test_row_number_ids_and_duplicate_ids(tmp_path, raw, processed):
    by_row = anes.process(variant(tmp_path, id_var=None))
    assert len(by_row.human_scores) == len(processed.human_scores)
    offset = by_row.human_scores["respondent_id"].astype(int) + 100000
    assert list(offset.astype(str)) == list(processed.human_scores["respondent_id"])
    with pytest.raises(anes.ConfigError, match="duplicate"):
        anes.process(variant(tmp_path, id_var="SYN_UNRELATED_1"))


@pytest.mark.parametrize("suffix", [".dta", ".sav"])
def test_stata_and_spss_files_give_the_same_tables(tmp_path, raw, processed, suffix):
    path = tmp_path / f"anes{suffix}"
    writer = pyreadstat.write_dta if suffix == ".dta" else pyreadstat.write_sav
    writer(raw, str(path))
    c = variant(tmp_path, file=str(path))
    assert anes.check(c) == []
    other = anes.process(c)
    pd.testing.assert_frame_equal(other.human_scores, processed.human_scores)
    pd.testing.assert_frame_equal(other.personas, processed.personas)
    assert other.exclusions == processed.exclusions


def test_build_writes_tables_and_exclusion_log(tmp_path, config, processed):
    out = anes.build(config, tmp_path)
    assert out == tmp_path / "anes_synthetic"
    scores, personas = anes.load_processed("anes_synthetic", tmp_path)
    pd.testing.assert_frame_equal(scores, processed.human_scores)
    assert list(personas.columns) == ["respondent_id", *anes.PERSONA_FIELDS]
    assert json.loads((out / "exclusions.json").read_text()) == processed.exclusions
    with pytest.raises(anes.ConfigError, match="scv data build"):
        anes.load_processed("anes_synthetic", tmp_path / "elsewhere")


# --------------------------------------------------------------------------------------
# Personas
# --------------------------------------------------------------------------------------


def test_label_value_rules():
    party = {1: "a strong Democrat", 7: "a strong Republican"}
    assert anes.label_value(1, party) == "a strong Democrat"
    assert anes.label_value(1.0, party) == anes.label_value("1", party) == "a strong Democrat"
    assert anes.label_value(4, party) is None  # unlisted code: missing, not "4"
    assert anes.label_value(-9, party) is None
    assert anes.label_value(None, party) is None and anes.label_value(float("nan"), party) is None
    # no labels: used as is, but negative codes and blanks are missing
    assert anes.label_value(45, None) == "45" and anes.label_value(45.0, None) == "45"
    assert anes.label_value(-9, None) is None and anes.label_value("  ", None) is None
    assert anes.label_value("Ohio", None) == "Ohio"
    # "*" lets unlisted non-negative values through
    age = {80: "at least 80", "*": "{value}"}
    assert anes.label_value(80, age) == "at least 80"
    assert anes.label_value(37, age) == "37" and anes.label_value(-9, age) is None


def test_personas_hold_labels_and_leave_missing_values_empty(raw, processed):
    personas = processed.personas.set_index("respondent_id")
    source = raw.set_index(raw["SYN_ID"].astype(str)).loc[personas.index]

    assert set(personas["gender"].dropna()) == {"a man", "a woman"}
    assert personas["gender"].isna().sum() == (source["SYN_GENDER"] == -9).sum() > 0
    assert set(personas["party_id_7"].dropna()) <= set(
        yaml.safe_load(YAML.read_text())["value_labels"]["party_id_7"].values()
    )
    assert personas["party_id_7"].isna().sum() == (source["SYN_PID7"] == -8).sum()
    assert personas["ideology_7"].isna().sum() == (source["SYN_IDEO7"] == 99).sum() > 20
    assert personas["state_or_region"].isna().sum() == source["SYN_STATE"].isna().sum() > 0
    assert personas["age"].isna().sum() == (source["SYN_AGE"] == -9).sum()
    assert (personas.loc[source["SYN_AGE"] == 80, "age"] == "at least 80").all()
    young = source["SYN_AGE"].between(18, 79)
    assert list(personas.loc[young, "age"]) == [str(a) for a in source.loc[young, "SYN_AGE"]]
    for column in personas:  # no raw missing-data code ever leaks into a label
        assert not personas[column].dropna().astype(str).str.fullmatch(r"-\d+|99|998|999").any()


def test_persona_text_is_first_person_and_omits_missing_fields(processed):
    texts = anes.persona_texts(processed.personas)
    assert list(texts.index) == list(processed.personas["respondent_id"])
    assert texts.str.len().gt(0).all()
    for text in texts:
        assert not re.search(r"unknown|\bnan\b|None|-\d|\b99\b|TODO", text), text
        assert text.count("I ") + text.count("My ") >= 1

    row = processed.personas.iloc[0].to_dict()
    row.pop("respondent_id")
    full = processed.personas.dropna().iloc[0]
    text = texts[full["respondent_id"]]
    assert text == (
        f"I am {full['age']} years old. I am {full['gender']}. My race or ethnicity is "
        f"{full['race_ethnicity']}. My highest level of education is {full['education']}. "
        f"My household income is {full['income']} a year. I live in {full['state_or_region']}. "
        f"Politically, I think of myself as {full['party_id_7']}. My political views are "
        f"{full['ideology_7']}. I {full['religious_attendance']}. "
        f"In the last presidential election, I {full['vote_prior']}."
    )
    # a respondent with a missing field simply has one sentence fewer
    partial = processed.personas[processed.personas["ideology_7"].isna()].iloc[0]
    assert "My political views are" not in texts[partial["respondent_id"]]
    n_sentences = texts.str.count(r"\.(?:\s|$)")
    n_fields = processed.personas.drop(columns="respondent_id").notna().sum(axis=1)
    assert list(n_sentences) == list(n_fields)


# --------------------------------------------------------------------------------------
# ANES source in the runner
# --------------------------------------------------------------------------------------


def anes_run(tmp_path, **overrides) -> RunConfig:
    anes.build(anes.load_election(YAML), tmp_path / "processed")
    base = {
        "run_id": "anes_mock",
        "backend": "mock",
        "models": ["tiny"],
        "election": {
            "source": "anes",
            "id": "anes_synthetic",
            "config": str(YAML),
            "processed_dir": str(tmp_path / "processed"),
        },
        "limit_respondents": 40,
        "results_dir": str(tmp_path / "results"),
    }
    return RunConfig.model_validate({**base, **overrides})


def test_runner_reads_the_processed_election(tmp_path, processed):
    cfg = anes_run(tmp_path)
    election = resolve_election(cfg)
    assert election.id == "anes_synthetic" and election.candidates == tuple(CANDS)
    assert "Synthetic test wording" in election.question_text
    assert len(election.respondents) == len(processed.human_scores)
    by_id = processed.personas.set_index("respondent_id")
    for r in election.respondents[:50]:
        label = by_id.loc[r.respondent_id, "party_id_7"]
        assert r.stratum == (None if pd.isna(label) else label)
        assert r.persona.startswith("I ") or r.persona.startswith("My ")

    sample = load_respondents(cfg)
    assert len(sample) == 40
    share = lambda rs: pd.Series([r.stratum for r in rs]).value_counts(normalize=True, dropna=False)  # noqa: E731
    assert (share(sample) - share(election.respondents)).abs().max() < 0.02  # stratified

    stats = run_from_config(cfg)
    assert stats["tiny"].written == 40 * (5 * 6 + 4 + 20)
    calls = read_calls(cfg.results_dir, cfg.run_id)
    assert set(calls["election"]) == {"anes_synthetic"}
    # the mock answers with each respondent's own human thermometer score
    thermo = thermometer_table(calls[calls["sample_idx"] == GREEDY].assign(sample_idx=0))
    human = processed.human_scores.set_index("respondent_id")
    for row in thermo.itertuples():
        truth = human.loc[row.respondent_id, row.candidate]
        assert row.score == (50.0 if pd.isna(truth) else truth)


def test_runner_candidate_subsets_and_errors(tmp_path):
    election = {
        "source": "anes",
        "id": "anes_synthetic",
        "config": str(YAML),
        "processed_dir": str(tmp_path / "processed"),
    }
    subset = anes_run(tmp_path, election={**election, "candidates": CANDS[:3]})
    plan = plan_run(subset)["tiny"]
    assert {c for call in plan for c in call.candidates_shown} == set(CANDS[:3])
    assert len(plan) == 40 * (3 * 6 + 4 + 6)

    with pytest.raises(ValueError, match="not in election"):
        resolve_election(
            anes_run(tmp_path, election={**election, "candidates": ["Zed Zulu", "Avery Adams"]})
        )
    with pytest.raises(anes.ConfigError, match="scv data build"):
        resolve_election(
            anes_run(tmp_path, election={**election, "processed_dir": str(tmp_path / "x")})
        )
    # The raw data file is not needed for a run (it is never copied to the GPU box).
    no_raw = tmp_path / "no_raw.yaml"
    no_raw.write_text(yaml.safe_dump({**yaml.safe_load(YAML.read_text()), "file": "gone.csv"}))
    moved = {**election, "config": str(no_raw)}
    assert len(resolve_election(anes_run(tmp_path, election=moved)).respondents) == 377
    with pytest.raises(anes.ConfigError, match="data file not found"):
        anes.require_ready(anes.load_election(no_raw))

    unfilled = tmp_path / "unfilled.yaml"
    unfilled.write_text(yaml.safe_dump({**yaml.safe_load(YAML.read_text()), "weight_var": "TODO"}))
    todo = {"source": "anes", "id": "anes_synthetic", "config": str(unfilled)}
    with pytest.raises(anes.ConfigError, match="not ready"):  # TODO codes stop a run cold
        resolve_election(anes_run(tmp_path, election=todo))


def test_prod_example_config_is_valid_and_needs_built_data(tmp_path):
    from scv.runner import load_run_config

    cfg = load_run_config("configs/runs/prod_example.yaml")
    assert cfg.backend == "vllm" and cfg.limit_respondents == 300
    assert cfg.election.source == "anes" and cfg.election.id == "anes2024_ts"
    # Without processed tables it refuses to plan: "run scv data build" where the raw
    # file is present, "not ready" (data file not found) on a checkout without data/raw.
    election = cfg.election.model_copy(update={"processed_dir": str(tmp_path / "none")})
    with pytest.raises(anes.ConfigError):
        plan_run(cfg.model_copy(update={"election": election}))


def test_persona_fallback_column_fills_only_where_the_first_gives_no_label(tmp_path):
    # Two-step questions: the field's own column wins; the fallback column is consulted
    # only where that gives no label, and only its listed codes produce one.
    config = variant(tmp_path, persona_fallbacks={
        "religious_attendance": {"var": "SYN_UNRELATED_1", "labels": {1: "never"}}
    })  # fmt: skip
    config.value_labels["religious_attendance"] = {1: "every week"}
    assert anes.check(config) == []
    got = anes.process(config).personas.set_index("respondent_id")["religious_attendance"]
    raw = pd.read_csv(config.path).assign(rid=lambda d: d["SYN_ID"].astype(str)).set_index("rid")
    raw = raw.loc[got.index]
    first, second = raw["SYN_RELIG"] == 1, raw["SYN_UNRELATED_1"] == 1
    assert (first & second).any() and (~first & second).any() and (~first & ~second).any()
    assert (got[first] == "every week").all()
    assert (got[~first & second] == "never").all()
    assert got[~first & ~second].isna().all()

    bad = variant(tmp_path, persona_fallbacks={"age": {"var": "SYN_NOPE", "labels": {1: "x"}}})
    assert "variable not in data file: SYN_NOPE (persona_fallbacks.age)" in anes.check(bad)
    stray = variant(tmp_path, persona_fallbacks={"zodiac": {"var": "SYN_AGE", "labels": {}}})
    assert any("persona_fallbacks given for fields" in p for p in anes.check(stray))
