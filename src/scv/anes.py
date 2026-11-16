"""ANES loaders and persona construction (SPEC §8).

Tristan downloads the ANES files by hand into ``data/raw/``; nothing here downloads
anything. Variable codes that are not known stay the literal string ``TODO`` in the
election config, and every entry point refuses to run while any remain.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from scv.prompts import Template, load_templates, render_persona

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ELECTIONS_DIR = REPO_ROOT / "configs" / "elections"
TODO = "TODO"
PASSTHROUGH = "*"

PERSONA_FIELDS = (
    "age",
    "gender",
    "race_ethnicity",
    "education",
    "income",
    "state_or_region",
    "party_id_7",
    "ideology_7",
    "religious_attendance",
    "vote_prior",
)


class ConfigError(ValueError):
    """An election config is incomplete (TODOs) or does not match its data file."""


class ElectionConfig(BaseModel):
    """One ``configs/elections/*.yaml`` file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    file: str
    weight_var: str
    # Respondent id column. None = use the 1-based row number in the file.
    id_var: str | None = None
    candidates: dict[str, str]  # display name -> thermometer variable code
    valid_range: tuple[float, float] = (0, 100)
    min_rated: int = 3
    persona_vars: dict[str, str]
    # persona field -> {code: label}. See `label_value` for the rules.
    value_labels: dict[str, dict[Any, str]] = Field(default_factory=dict)
    question_text: str

    @property
    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else REPO_ROOT / p

    def variables(self) -> dict[str, str]:
        """Every data-file column the config refers to: ``where it is used -> code``."""
        out = {"weight_var": self.weight_var}
        if self.id_var is not None:
            out["id_var"] = self.id_var
        out.update({f"candidates.{name}": code for name, code in self.candidates.items()})
        out.update({f"persona_vars.{name}": code for name, code in self.persona_vars.items()})
        return out


def election_path(election: str | Path) -> Path:
    """Accept an election id (``anes2024_ts``) or a path to a YAML file."""
    p = Path(election)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() or p.exists() else REPO_ROOT / p
    return ELECTIONS_DIR / f"{election}.yaml"


def load_election(election: str | Path) -> ElectionConfig:
    path = election_path(election)
    if not path.exists():
        raise ConfigError(f"no election config at {path}")
    with path.open(encoding="utf-8") as f:
        return ElectionConfig.model_validate(yaml.safe_load(f))


# --------------------------------------------------------------------------------------
# Data check
# --------------------------------------------------------------------------------------


def _columns(path: Path) -> list[str]:
    """Column names of a .csv / .dta / .sav file, without loading the data."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    import pyreadstat

    reader = {".dta": pyreadstat.read_dta, ".sav": pyreadstat.read_sav}.get(suffix)
    if reader is None:
        raise ConfigError(f"unsupported data file type {suffix!r} (use .csv, .dta or .sav)")
    _, meta = reader(str(path), metadataonly=True)
    return list(meta.column_names)


def check(config: ElectionConfig) -> list[str]:
    """Every problem that stops this election from being processed: each TODO, a missing
    data file, each configured variable absent from the file, and config slips. An empty
    list means it is ready."""
    problems = []
    todo_vars = set()
    for where, code in config.variables().items():
        if code == TODO:
            problems.append(f"TODO: {where}")
            todo_vars.add(where)
    for where, value in (("file", config.file), ("question_text", config.question_text)):
        if value.strip() == TODO:
            problems.append(f"TODO: {where}")

    unknown = sorted(set(config.persona_vars) - set(PERSONA_FIELDS))
    if unknown:
        problems.append(f"persona_vars has fields the persona template does not know: {unknown}")
    stray = sorted(set(config.value_labels) - set(config.persona_vars))
    if stray:
        problems.append(f"value_labels given for fields not in persona_vars: {stray}")
    if len(config.candidates) < 3:
        problems.append("fewer than 3 candidates")
    if not 1 <= config.min_rated <= max(1, len(config.candidates)):
        problems.append(f"min_rated={config.min_rated} is outside 1..{len(config.candidates)}")
    codes = [c for c in config.variables().values() if c != TODO]
    dupes = sorted({c for c in codes if codes.count(c) > 1})
    if dupes:
        problems.append(f"variable codes used more than once: {dupes}")

    if config.file.strip() != TODO:
        if not config.path.exists():
            problems.append(f"data file not found: {config.path}")
        else:
            try:
                present = set(_columns(config.path))
            except ConfigError as e:
                problems.append(str(e))
            else:
                for where, code in config.variables().items():
                    if where not in todo_vars and code not in present:
                        problems.append(f"variable not in data file: {code} ({where})")
    return problems


def require_ready(config: ElectionConfig) -> None:
    problems = check(config)
    if problems:
        raise ConfigError(
            f"election {config.id!r} is not ready ({len(problems)} problems):\n  "
            + "\n  ".join(problems)
        )


# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    import pyreadstat

    reader = {".dta": pyreadstat.read_dta, ".sav": pyreadstat.read_sav}[suffix]
    df, _ = reader(str(path), usecols=columns)
    return df


def _code(value: Any) -> Any:
    """Normalise a data value or a YAML key so that 1, 1.0 and "1" are the same code."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            value = float(value)
        except ValueError:
            return value
    number = float(value)
    return int(number) if number.is_integer() else number


def label_value(value: Any, labels: dict[Any, str] | None) -> str | None:
    """Human-readable label for one persona value, or ``None`` if it is missing.

    - No ``value_labels`` entry for the field: the value is used as is, except that
      negative numbers (ANES missing-data codes) and blanks are missing.
    - With an entry: a listed code gets its label; an unlisted code is missing. A
      ``"*"`` key opts unlisted non-negative values back in, as a format string with
      ``{value}`` (e.g. ``"*": "{value}"`` for age, alongside ``80: "80 or older"``).
    """
    code = _code(value)
    if code is None:
        return None
    if labels is not None:
        table = {_code(k): v for k, v in labels.items() if k != PASSTHROUGH}
        if code in table:
            return str(table[code])
        if PASSTHROUGH not in labels:
            return None
    if isinstance(code, (int, float)) and code < 0:
        return None
    text = str(code)
    return labels[PASSTHROUGH].format(value=text) if labels and PASSTHROUGH in labels else text


@dataclass(frozen=True)
class Processed:
    """Result of :func:`process`."""

    human_scores: pd.DataFrame  # respondent_id, weight, one column per candidate
    personas: pd.DataFrame  # respondent_id + persona fields as labels
    exclusions: list[dict[str, Any]]  # step, dropped, remaining


def process(config: ElectionConfig) -> Processed:
    """Build the human score table and the persona table, logging every exclusion.

    Steps: (1) drop respondents with a missing or non-positive weight; (2) set
    thermometer values outside ``valid_range`` to missing; (3) drop respondents who
    rated fewer than ``min_rated`` candidates. Persona values that are missing or have
    no label are left empty, never filled.
    """
    require_ready(config)
    raw = _read(config.path, sorted(set(config.variables().values())))
    exclusions: list[dict[str, Any]] = []

    def step(name: str, keep: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[keep]
        exclusions.append({"step": name, "dropped": int((~keep).sum()), "remaining": len(out)})
        logger.info("%s: %s dropped %d, %d remain", config.id, name, (~keep).sum(), len(out))
        return out

    exclusions.append({"step": "rows in file", "dropped": 0, "remaining": len(raw)})
    if config.id_var is None:
        raw = raw.assign(_respondent_id=[str(i + 1) for i in range(len(raw))])
    else:
        raw = raw.assign(_respondent_id=raw[config.id_var].map(lambda v: str(_code(v))))
        if raw["_respondent_id"].duplicated().any():
            raise ConfigError(f"id_var {config.id_var!r} has duplicate values")

    weight = pd.to_numeric(raw[config.weight_var], errors="coerce")
    raw = step("missing or non-positive weight", weight.notna() & (weight > 0), raw)

    lo, hi = config.valid_range
    scores = pd.DataFrame(index=raw.index)
    for name, code in config.candidates.items():
        values = pd.to_numeric(raw[code], errors="coerce")
        scores[name] = values.where((values >= lo) & (values <= hi))
    n_rated = scores.notna().sum(axis=1)
    keep = n_rated >= config.min_rated
    raw, scores = step(f"rated fewer than {config.min_rated} candidates", keep, raw), scores[keep]

    human = pd.concat(
        [
            raw["_respondent_id"].rename("respondent_id"),
            pd.to_numeric(raw[config.weight_var]).astype(float).rename("weight"),
            scores.astype(float),
        ],
        axis=1,
    ).reset_index(drop=True)

    personas = pd.DataFrame({"respondent_id": raw["_respondent_id"].to_numpy()})
    for field in PERSONA_FIELDS:
        if field in config.persona_vars:
            labels = config.value_labels.get(field)
            personas[field] = [label_value(v, labels) for v in raw[config.persona_vars[field]]]
    for field in config.persona_vars:
        missing = int(personas[field].isna().sum())
        logger.info("%s: persona field %s missing for %d respondents", config.id, field, missing)
    return Processed(human_scores=human, personas=personas, exclusions=exclusions)


def processed_dir(election_id: str, root: str | Path = "data/processed") -> Path:
    return Path(root) / election_id


def build(config: ElectionConfig, root: str | Path = "data/processed") -> Path:
    """Run :func:`process` and write ``human_scores.parquet``, ``personas.parquet`` and
    ``exclusions.json`` under ``<root>/<election id>/``."""
    result = process(config)
    out = processed_dir(config.id, root)
    out.mkdir(parents=True, exist_ok=True)
    result.human_scores.to_parquet(out / "human_scores.parquet", index=False)
    result.personas.to_parquet(out / "personas.parquet", index=False)
    (out / "exclusions.json").write_text(json.dumps(result.exclusions, indent=2) + "\n")
    return out


def load_processed(
    election_id: str, root: str | Path = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(human_scores, personas)`` written by :func:`build`."""
    d = processed_dir(election_id, root)
    if not (d / "personas.parquet").exists():
        raise ConfigError(
            f"no processed data for {election_id!r} in {d}; run `scv data build {election_id}`"
        )
    return pd.read_parquet(d / "human_scores.parquet"), pd.read_parquet(d / "personas.parquet")


def persona_texts(personas: pd.DataFrame, template: Template | None = None) -> pd.Series:
    """Rendered first-person backstory per respondent (index: respondent_id). Missing
    fields are left out of the text."""
    template = template or load_templates().persona
    fields = [c for c in personas.columns if c != "respondent_id"]
    texts = [
        render_persona(template, {f: getattr(row, f) for f in fields})
        for row in personas.itertuples(index=False)
    ]
    return pd.Series(texts, index=personas["respondent_id"].astype(str), name="persona")
