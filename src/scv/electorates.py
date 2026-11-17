"""Electorate assembly (SPEC §12).

Every persona is run on every model once. Electorates are then assembled *after the
fact* from the ballot table, with no further inference:

- **human**: the ANES thermometer ballots;
- **homogeneous(model)**: every persona's ballot from one model;
- **heterogeneous**: each persona assigned to one pool model, stratified on party id;
- **nulls**: impartial culture with the same n and m, and the permutation null of the
  human score matrix.

All electorates of a run cover the same respondents in the same order, with the same
survey weights, so human and LLM electorates are matched person for person.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scv.elicit import GREEDY, ranking_table, thermometer_table
from scv.nulls import as_rng, impartial_culture, permutation_null
from scv.rules import Profile, _dense_ranks

SOURCES = ("thermometer", "ranking")
RANKING_BALLOTS = ("greedy", "combined")


@dataclass(frozen=True)
class BallotTable:
    """Aligned human and LLM ballots for one run. Row i is the same respondent everywhere.

    ``human`` and ``llm_scores[model]`` are (n, m) thermometer scores (NaN = missing).
    ``llm_ranks[model]`` are (n, m) rank values from the greedy full ranking (lower is
    better; an all-NaN row is a ranking that failed to parse).
    """

    election: str
    candidates: tuple[str, ...]
    respondent_ids: np.ndarray
    weights: np.ndarray
    strata: np.ndarray  # party_id_7 label, "" where unknown
    human: np.ndarray
    llm_scores: dict[str, np.ndarray] = field(default_factory=dict)
    llm_ranks: dict[str, np.ndarray] = field(default_factory=dict)
    personas: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n(self) -> int:
        return len(self.respondent_ids)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.llm_scores, *self.llm_ranks]))


def build_ballot_table(
    election: str,
    candidates: Sequence[str],
    human_scores: pd.DataFrame,
    calls: pd.DataFrame,
    personas: pd.DataFrame | None = None,
    ranking_ballot: str = "greedy",
) -> BallotTable:
    """Line up the human score table with the call log of a run.

    Respondents are those that appear in the call log (a pilot covers a subset), in the
    order of ``human_scores``. A thermometer score is the median of the sampled answers
    (SPEC §10.1).

    ``ranking_ballot`` chooses the ranking ballot. ``"greedy"`` (default) is the greedy
    ranking alone; the sampled rankings stay in the call log. ``"combined"`` averages
    each candidate's rank over all of the voter's rankings that parsed (the greedy one
    and the sampled ones). Every ranking call lists the candidates in a different random
    order, so averaging cancels much of a model's position bias; candidates with the
    same average rank are tied.
    """
    if ranking_ballot not in RANKING_BALLOTS:
        raise ValueError(f"ranking_ballot must be one of {RANKING_BALLOTS}, not {ranking_ballot!r}")
    candidates = tuple(candidates)
    calls = calls[calls["election"] == election]
    ran = set(calls["respondent_id"].astype(str))
    human = human_scores[human_scores["respondent_id"].astype(str).isin(ran)].reset_index(drop=True)
    ids = human["respondent_id"].astype(str).to_numpy()
    if len(ids) == 0:
        raise ValueError(f"no respondent of election {election!r} appears in the call log")
    row = {rid: i for i, rid in enumerate(ids)}
    col = {c: j for j, c in enumerate(candidates)}

    llm_scores: dict[str, np.ndarray] = {}
    thermo = thermometer_table(calls)
    for model, g in thermo.groupby("model_key", sort=False):
        mat = np.full((len(ids), len(candidates)), np.nan)
        for rid, cand, score in zip(g["respondent_id"], g["candidate"], g["score"], strict=True):
            if rid in row and cand in col:
                mat[row[rid], col[cand]] = score
        llm_scores[model] = mat

    llm_ranks: dict[str, np.ndarray] = {}
    ranks = ranking_table(calls)
    if ranking_ballot == "greedy":
        ranks = ranks[ranks["sample_idx"] == GREEDY]
    for model, g in ranks.groupby("model_key", sort=False):
        total = np.zeros((len(ids), len(candidates)))
        count = np.zeros(len(ids))
        for rid, ballot in zip(g["respondent_id"], g["ballot"], strict=True):
            if rid in row and ballot is not None:
                for cand, r in ballot.rank_of().items():
                    total[row[rid], col[cand]] += r
                count[row[rid]] += 1
        with np.errstate(invalid="ignore"):
            mat = np.round(total / count[:, None], 9)  # rounded so equal means are ties
        mat[count == 0] = np.nan  # no ranking parsed: an empty ballot
        llm_ranks[model] = mat

    strata = np.full(len(ids), "", dtype=object)
    persona_rows = pd.DataFrame({"respondent_id": ids})
    if personas is not None and len(personas):
        persona_rows = persona_rows.merge(
            personas.assign(respondent_id=personas["respondent_id"].astype(str)),
            on="respondent_id",
            how="left",
        )
        if "party_id_7" in persona_rows:
            strata = persona_rows["party_id_7"].fillna("").astype(str).to_numpy(dtype=object)
    return BallotTable(
        election=election,
        candidates=candidates,
        respondent_ids=ids,
        weights=human["weight"].to_numpy(dtype=float),
        strata=strata,
        human=human[list(candidates)].to_numpy(dtype=float),
        llm_scores=llm_scores,
        llm_ranks=llm_ranks,
        personas=persona_rows,
    )


@dataclass(frozen=True)
class Electorate:
    """One assembled electorate.

    ``ranks`` is (n, m): lower is better, equal is tied, NaN is missing. ``scores`` is
    the (n, m) thermometer matrix when the ballots come from thermometers, else ``None``.
    """

    name: str
    kind: str  # human | homogeneous | heterogeneous | null_ic | null_permutation
    source: str  # thermometer | ranking
    ranks: np.ndarray
    weights: np.ndarray
    scores: np.ndarray | None = None
    model: str | None = None
    replicate: int | None = None

    @property
    def n(self) -> int:
        return len(self.weights)

    def profile(self, candidates: Sequence[str], weighted: bool = True) -> Profile:
        w = self.weights if weighted else np.ones(self.n)
        return Profile.from_ranks(self.ranks, candidates, w)


def _from_scores(scores: np.ndarray) -> np.ndarray:
    return _dense_ranks(-scores)


def human_electorate(table: BallotTable) -> Electorate:
    return Electorate(
        "human", "human", "thermometer", _from_scores(table.human), table.weights, table.human
    )


def homogeneous(table: BallotTable, model: str, source: str = "thermometer") -> Electorate:
    """All personas' ballots from one model."""
    if source == "thermometer":
        scores = table.llm_scores[model]
        ranks = _from_scores(scores)
    else:
        scores, ranks = None, _dense_ranks(table.llm_ranks[model])
    return Electorate(model, "homogeneous", source, ranks, table.weights, scores, model=model)


def assign_models(strata: np.ndarray, n_models: int, rng: np.random.Generator | int) -> np.ndarray:
    """Assign each persona to one of ``n_models`` models, stratified on ``strata``.

    Within each stratum the personas are shuffled and dealt round-robin from a random
    starting model, so every model gets the same persona mix (stratum counts differ by
    at most one between models).
    """
    rng = as_rng(rng)
    out = np.empty(len(strata), dtype=int)
    for s in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == s)
        idx = idx[rng.permutation(len(idx))]
        out[idx] = (int(rng.integers(n_models)) + np.arange(len(idx))) % n_models
    return out


def heterogeneous(
    table: BallotTable,
    models: Sequence[str],
    source: str,
    rng: np.random.Generator | int,
    replicate: int | None = None,
) -> Electorate:
    """One mixed-model electorate: each persona's ballot comes from its assigned model."""
    assignment = assign_models(table.strata, len(models), rng)
    rows = np.arange(table.n)
    if source == "thermometer":
        scores = np.stack([table.llm_scores[m] for m in models])[assignment, rows]
        ranks = _from_scores(scores)
    else:
        scores = None
        ranks = _dense_ranks(np.stack([table.llm_ranks[m] for m in models])[assignment, rows])
    return Electorate(
        "heterogeneous", "heterogeneous", source, ranks, table.weights, scores, replicate=replicate
    )


def null_ic(
    table: BallotTable, rng: np.random.Generator | int, replicate: int | None = None
) -> Electorate:
    """Impartial culture with the same n, m and weights as the human electorate."""
    profile = impartial_culture(table.n, len(table.candidates), rng, table.candidates)
    return Electorate(
        "null_ic", "null_ic", "thermometer", np.array(profile.ranks), table.weights, None,
        replicate=replicate,
    )  # fmt: skip


def null_permutation(
    table: BallotTable, rng: np.random.Generator | int, replicate: int | None = None
) -> Electorate:
    """Permutation null of the human profile: each candidate's scores shuffled across
    voters (weights stay with the voter)."""
    scores = permutation_null(table.human, rng)
    return Electorate(
        "null_permutation", "null_permutation", "thermometer", _from_scores(scores),
        table.weights, scores, replicate=replicate,
    )  # fmt: skip
