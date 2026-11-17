"""One self-contained HTML page with every result of one or more runs (``scv report``).

It reads the folders ``scv analyze`` wrote (``results/metrics/<run_id>[__variant]/``) and,
if present, the call logs, and draws everything as inline SVG: no internet, no
JavaScript libraries. Two switches at the top pick the weighting and the ballot source;
every chart and table follows them. The page reports and explains; it does not draw
conclusions (SPEC §0). Definitions follow ANALYSIS_PLAN.md.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------

TIE_VARIANTS = {
    "default": "",
    "break_random": "__ties_break_random",
    "drop_flat": "__ties_drop_flat",
}
#: The three ballot views offered by the switch at the top of the page.
BALLOTS = {
    "thermometer": ("thermometer", ""),
    "ranking": ("ranking", ""),
    "ranking_combined": ("ranking", "__ranking_combined"),
}
BALLOT_LABELS = {
    "thermometer": "Thermometer",
    "ranking": "Ranking (greedy)",
    "ranking_combined": "Ranking (combined)",
}
WEIGHTINGS = ("weighted", "unweighted")
LABELS = {
    "human": "Human (ANES)",
    "heterogeneous": "Mixed models",
    "null_permutation": "Null: permuted human",
    "null_ic": "Null: random orders",
}
REFERENCE = ("human", "null_ic", "null_permutation")  # exist as thermometer ballots only


@dataclass
class RunData:
    run_id: str
    election: str
    folders: dict[str, Path]  # variant suffix ("" = default) -> folder
    summary: pd.DataFrame  # all variants, with a "variant" column
    ncurves: pd.DataFrame
    matched_groups: pd.DataFrame
    winners: pd.DataFrame
    settings: dict[str, dict[str, Any]]
    quality: pd.DataFrame | None = None
    models: list[str] = field(default_factory=list)


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def load_run(run_id: str, results: str | Path = "results", with_calls: bool = True) -> RunData:
    """Every analysis folder of one run (default and variants) and, if wanted and present,
    the call-log quality numbers."""
    base = Path(results) / "metrics"
    folders = {
        p.name[len(run_id) :]: p
        for p in sorted(base.glob(f"{run_id}*"))
        if p.is_dir()
        and (p / "summary.csv").exists()
        and (p.name == run_id or p.name.startswith(f"{run_id}__"))
    }
    if "" not in folders:
        raise ValueError(f"no analysis of {run_id!r} in {base}; run `scv analyze` first")
    frames: dict[str, list[pd.DataFrame]] = {"summary": [], "ncurves": [], "mg": [], "win": []}
    settings = {}
    for suffix, folder in folders.items():
        for key, name in (
            ("summary", "summary.csv"),
            ("ncurves", "ncurves.parquet"),
            ("mg", "matched_groups.parquet"),
            ("win", "winners.parquet"),
        ):
            frame = _read(folder, name)
            if len(frame):
                frames[key].append(frame.assign(variant=suffix))
        sfile = folder / "analysis_settings.json"
        settings[suffix] = json.loads(sfile.read_text()) if sfile.exists() else {}
    summary = pd.concat(frames["summary"], ignore_index=True)
    election = str(summary["election"].iloc[0])
    models = sorted(summary.loc[summary["kind"] == "homogeneous", "electorate"].unique())
    quality = None
    if with_calls and (Path(results) / "calls" / run_id).exists():
        from scv.runner import read_calls

        quality = call_quality(read_calls(results, run_id))
    cat = lambda k: pd.concat(frames[k], ignore_index=True) if frames[k] else pd.DataFrame()  # noqa: E731
    return RunData(
        run_id, election, folders, summary, cat("ncurves"), cat("mg"), cat("win"),
        settings, quality, models,
    )  # fmt: skip


def call_quality(calls: pd.DataFrame) -> pd.DataFrame:
    """Per model: parse rates by mode, unreadable pairwise share, and the ranking
    position-bias measures of scripts/ranking_report.py."""
    from scipy.stats import spearmanr

    from scv.elicit import GREEDY, ranking_table, unreadable_pairwise

    rows = []
    ranks = ranking_table(calls)
    pair = calls[calls["mode"] == "pairwise"]
    unread = unreadable_pairwise(pair) if len(pair) else pd.Series(dtype=bool)
    for model in sorted(calls["model_key"].unique()):
        c = calls[calls["model_key"] == model]
        row: dict[str, Any] = {"model": model}
        for mode in ("thermometer", "ranking"):
            m = c[c["mode"] == mode]
            row[f"{mode}_parsed"] = float(m["parse_ok"].mean()) if len(m) else math.nan
        r = ranks[(ranks["model_key"] == model) & (ranks["sample_idx"] == GREEDY)]
        row["ranking_greedy_parsed"] = float(r["parse_ok"].mean()) if len(r) else math.nan
        ok = r[r["parse_ok"]]
        corr, first, same = [], [], 0
        for ballot, order in zip(ok["ballot"], ok["candidates_shown"], strict=True):
            rank = ballot.rank_of()
            pos = [rank[x] for x in order]
            first.append(pos[0] == min(pos))
            same += [list(g) for g in ballot.groups] == [[x] for x in order]
            if len(set(pos)) > 1:
                corr.append(spearmanr(range(len(order)), pos).statistic)
        m = len(ok["candidates_shown"].iloc[0]) if len(ok) else 0
        row["rank_first_listed_first"] = float(np.mean(first)) if first else math.nan
        row["rank_first_listed_chance"] = 1 / m if m else math.nan
        row["rank_same_as_shown"] = same / len(ok) if len(ok) else math.nan
        row["rank_same_as_shown_chance"] = 1 / math.factorial(m) if m else math.nan
        row["rank_position_corr"] = float(np.mean(corr)) if corr else math.nan
        p = pair["model_key"] == model
        row["pairwise_unreadable"] = float(unread[p].mean()) if p.any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# A small SVG kit. Colours are CSS variables, so light and dark themes both work.
# --------------------------------------------------------------------------------------

FONT = 12
VARIANT_STYLE = {  # identity of the three tie variants: colour slot and marker shape
    "default": ("var(--s1)", "circle", "As is"),
    "break_random": ("var(--s2)", "square", "Ties broken at random"),
    "drop_flat": ("var(--s3)", "diamond", "Flat ballots dropped"),
}


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{digits}f}".replace("-", "−")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, t = [], start
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def tick_label(t: float) -> str:
    s = f"{t:.2f}".rstrip("0").rstrip(".") if abs(t) < 100 else f"{t:,.0f}"
    return "0" if s in ("-0", "0") else s.replace("-", "−")


def marker(shape: str, x: float, y: float, color: str, title: str, r: float = 4.5) -> str:
    t = f"<title>{esc(title)}</title>"
    ring = 'stroke="var(--surface)" stroke-width="2"'
    if shape == "square":
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" fill="{color}" {ring}>{t}</rect>'
    if shape == "diamond":
        d = r * 1.3
        pts = f"{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} {x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}"
        return f'<polygon points="{pts}" fill="{color}" {ring}>{t}</polygon>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" {ring}>{t}</circle>'


def legend(items: Iterable[tuple[str, str, str]]) -> str:
    """HTML legend: (colour, shape or 'line', label)."""
    parts = []
    for color, shape, label in items:
        if shape == "line":
            key = f'<svg width="18" height="10" aria-hidden="true"><line x1="1" y1="5" x2="17" y2="5" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>'
        else:
            key = f'<svg width="12" height="12" aria-hidden="true">{marker(shape, 6, 6, color, label, 4)}</svg>'
        parts.append(f'<span class="key">{key}{esc(label)}</span>')
    return f'<div class="legend">{"".join(parts)}</div>'


def dot_panel(
    title: str,
    rows: Sequence[dict[str, Any]],
    *,
    xlim: tuple[float, float] | None = None,
    ref: float | None = None,
    ref_label: str = "",
    width: int = 440,
    row_h: int = 22,
    label_w: int = 150,
    subtitle: str = "",
) -> str:
    """Horizontal dot plot. Each row: label, list of points (value, lo, hi, colour,
    shape, tooltip), optional emphasis. Rows with several points spread them vertically."""
    vals = [
        v
        for r in rows
        for p in r["points"]
        for v in (p[0], p[1], p[2])
        if v is not None and math.isfinite(v)
    ]
    if ref is not None:
        vals.append(ref)
    if not vals:
        return f'<figure class="panel"><figcaption>{esc(title)}</figcaption><p class="muted">No data.</p></figure>'
    lo, hi = xlim if xlim else (min(vals), max(vals))
    if not xlim:
        pad = (hi - lo) * 0.08 or 0.05
        lo, hi = lo - pad, hi + pad
    top, bottom = 8, 26
    h = top + bottom + row_h * len(rows)
    plot_w = width - label_w - 16
    sx = lambda v: label_w + (v - lo) / (hi - lo) * plot_w  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}">']
    ticks = nice_ticks(lo, hi)
    for t in ticks:
        x = sx(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}"/>')
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{h - bottom + 15}" text-anchor="middle">{tick_label(t)}</text>'
        )
    if ref is not None and lo <= ref <= hi:
        x = sx(ref)
        out.append(f'<line class="ref" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}"/>')
        if ref_label:
            out.append(f'<text class="tick" x="{x + 4:.1f}" y="{top + 9}">{esc(ref_label)}</text>')
    for i, r in enumerate(rows):
        yc = top + row_h * (i + 0.5)
        cls = "rowlab strong" if r.get("strong") else "rowlab"
        out.append(
            f'<text class="{cls}" x="{label_w - 8}" y="{yc + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>'
        )
        pts = r["points"]
        k = len(pts)
        for j, (v, a, b, color, shape, tip) in enumerate(pts):
            if v is None or not math.isfinite(v):
                continue
            y = yc + (j - (k - 1) / 2) * min(6, row_h / max(k, 1))
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                out.append(
                    f'<line x1="{sx(max(a, lo)):.1f}" y1="{y:.1f}" x2="{sx(min(b, hi)):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"><title>{esc(tip)}</title></line>'
                )
            out.append(marker(shape, sx(min(max(v, lo), hi)), y, color, tip))
    out.append("</svg>")
    sub = f'<div class="sub">{esc(subtitle)}</div>' if subtitle else ""
    return (
        f'<figure class="panel"><figcaption>{esc(title)}</figcaption>{sub}{"".join(out)}</figure>'
    )


def line_panel(
    title: str,
    series: Sequence[dict[str, Any]],
    *,
    xs: Sequence[float],
    ylim: tuple[float, float] | None = None,
    width: int = 300,
    height: int = 190,
    ylabel: str = "",
) -> str:
    """Line chart on a log x axis with dots at each x; series: label, colour, points
    [(x, y, tooltip)]."""
    left, right, top, bottom = 38, 10, 10, 30
    vals = [p[1] for s in series for p in s["points"] if p[1] is not None and math.isfinite(p[1])]
    if not vals:
        return f'<figure class="panel"><figcaption>{esc(title)}</figcaption><p class="muted">No data.</p></figure>'
    lo, hi = ylim if ylim else (0.0, max(vals) * 1.1 or 1.0)
    lx = [math.log(x) for x in xs]
    span = (max(lx) - min(lx)) or 1

    def sx(x: float) -> float:
        return left + (math.log(x) - min(lx)) / span * (width - left - right)

    sy = lambda y: top + (1 - (y - lo) / ((hi - lo) or 1)) * (height - top - bottom)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for t in nice_ticks(lo, hi, 4):
        y = sy(t)
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 5}" y="{y + 4:.1f}" text-anchor="end">{tick_label(t)}</text>'
        )
    for x in xs:
        out.append(
            f'<text class="tick" x="{sx(x):.1f}" y="{height - bottom + 15}" text-anchor="middle">{int(x)}</text>'
        )
    out.append(
        f'<line class="axis" x1="{left}" y1="{sy(lo):.1f}" x2="{width - right}" y2="{sy(lo):.1f}"/>'
    )
    for s in series:
        pts = [
            (sx(x), sy(y), tip) for x, y, tip in s["points"] if y is not None and math.isfinite(y)
        ]
        if len(pts) > 1:
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts)
            )
            out.append(
                f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for x, y, tip in pts:
            out.append(marker("circle", x, y, s["color"], tip, 3.5))
    out.append("</svg>")
    return (
        f'<figure class="panel small"><figcaption>{esc(title)}</figcaption>{"".join(out)}</figure>'
    )


def table_html(
    frame: pd.DataFrame,
    *,
    cls: str = "",
    heat: Callable[[str, Any], str] | None = None,
    digits: int = 3,
) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for c, v in row.items():
            style = heat(c, v) if heat else ""
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            cells.append(
                f'<td class="{"num" if num else ""}"{style}>{esc(fmt(v, digits)) if num or isinstance(v, (bool, np.bool_)) else esc(v)}</td>'
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _views(make: Callable[[str, str], str], ballots: Sequence[str] = tuple(BALLOTS)) -> str:
    """One copy of the content per (weighting, ballot) view; the page shows one at a time."""
    parts = []
    for w in WEIGHTINGS:
        for b in ballots:
            parts.append(f'<div class="view" data-w="{w}" data-b="{b}">{make(w, b)}</div>')
    return "".join(parts)


def _rows(
    run: RunData, weighting: str, ballot: str, metric: str, variant: str | None = None
) -> pd.DataFrame:
    """Summary rows for one view. Reference electorates (human, nulls) always come from
    the thermometer rows of the same folder."""
    source, suffix = BALLOTS[ballot]
    suffix = suffix if variant is None else variant
    s = run.summary[(run.summary["variant"] == suffix) & (run.summary["weighting"] == weighting)]
    s = s[s["metric"] == metric]
    return s[(s["source"] == source) | s["electorate"].isin(REFERENCE)]


def _electorate_order(run: RunData) -> list[str]:
    return ["human", *run.models, "heterogeneous", "null_permutation", "null_ic"]


def _ci(r: Any) -> str:
    if pd.notna(r.ci_low) and pd.notna(r.ci_high):
        kind = {"bootstrap": "95% bootstrap", "paired_bootstrap": "95% paired bootstrap",
                "replicates": "2.5–97.5% of draws"}.get(r.interval, str(r.interval))  # fmt: skip
        return f" [{fmt(r.ci_low)}, {fmt(r.ci_high)}] ({kind})"
    return ""


def section_overview(runs: Sequence[RunData]) -> str:
    rows = []
    for run in runs:
        s = run.settings.get("", {})
        variants = [v.strip("_").replace("__", ", ") or "default" for v in run.folders]
        rows.append({
            "election": run.election,
            "run_id": run.run_id,
            "respondents": s.get("n_respondents", math.nan),
            "candidates": ", ".join(s.get("candidates", [])),
            "models": ", ".join(run.models),
            "bootstrap / draws / replicates": f"{s.get('bootstrap', '?')} / {s.get('draws', '?')} / {s.get('replicates', '?')}",
            "analysis variants": ", ".join(sorted(variants)),
            "analysed": str(s.get("written", "?"))[:16].replace("T", " "),
        })  # fmt: skip
    return table_html(pd.DataFrame(rows))


def section_quality(runs: Sequence[RunData]) -> str:
    out = []
    for run in runs:
        parts = [f"<h3>{esc(run.election)}</h3>"]
        if run.quality is not None:
            q = run.quality.copy()
            q.columns = [
                "model", "thermometer parsed", "ranking parsed (all)", "ranking parsed (greedy)",
                "ranks first-listed first", "… chance", "ranking = order shown", "… chance",
                "position–rank correlation", "pairwise unreadable",
            ]  # fmt: skip
            parts.append(table_html(q))
        else:
            parts.append(
                '<p class="muted">Call log not on this machine: parse rates not shown.</p>'
            )
        s = run.summary[(run.summary["variant"] == "") & (run.summary["weighting"] == "weighted")]
        s = s[
            (s["source"] == "thermometer")
            & s["metric"].isin(["share_strict_ballots", "share_complete_ballots"])
        ]
        flat = run.settings.get("__ties_drop_flat", {}) or run.settings.get(
            "__ties_break_random", {}
        )
        counts = flat.get("flat_thermometer_ballots", {})
        n = flat.get("n_before")
        t = s.pivot_table(index="electorate", columns="metric", values="value")
        t = t.reindex([e for e in _electorate_order(run) if e in t.index])
        t.insert(0, "electorate", [LABELS.get(e, e) for e in t.index])
        if counts and n:
            t["flat ballots"] = [
                counts.get(e, math.nan) / n if e in counts else math.nan for e in t.index
            ]
        t = t.rename(
            columns={
                "share_strict_ballots": "ballots with no tie",
                "share_complete_ballots": "ballots rating every candidate",
            }
        )
        parts.append(table_html(t.reset_index(drop=True)))
        if "__ties_drop_flat" in run.settings:
            d = run.settings["__ties_drop_flat"]
            parts.append(
                f'<p class="muted">Dropping every respondent with a flat ballot anywhere keeps {fmt(d.get("n_after"))} of {fmt(d.get("n_before"))}.</p>'
            )
        out.append("".join(parts))
    return "".join(out)


def section_h1_gaps(runs: Sequence[RunData]) -> str:
    """Forest plots of the paired model-minus-human differences, across tie variants."""
    outcomes = [
        (
            "sp_deletion_share",
            "Single-peakedness: deletion share, model − human",
            "H1 predicts below 0",
        ),
        ("pc1_share", "Dimensionality: PC1 share, model − human", "H1 predicts above 0"),
    ]

    def make(w: str, b: str) -> str:
        source, ballot_suffix = BALLOTS[b]
        cards = []
        for run in runs:
            for metric, title, pred in outcomes:
                if source == "ranking" and metric == "pc1_share":
                    continue  # PC1 is computed from thermometer scores only
                rows = []
                for model in run.models:
                    pts = []
                    for var, suffix in TIE_VARIANTS.items():
                        folder = (ballot_suffix + suffix) if suffix else ballot_suffix
                        if ballot_suffix and suffix:
                            continue  # tie variants exist for the default ballots only
                        if folder not in run.folders:
                            continue
                        s = run.summary[
                            (run.summary["variant"] == folder) & (run.summary["weighting"] == w)
                            & (run.summary["source"] == source) & (run.summary["electorate"] == model)
                            & (run.summary["metric"] == f"{metric}_minus_human")
                        ]  # fmt: skip
                        if s.empty:
                            continue
                        r = next(s.itertuples())
                        n = int(r.n_voters)
                        color, shape, label = VARIANT_STYLE[var]
                        tip = f"{model} · {label}: {fmt(r.value)}{_ci(r)} · n = {n:,}"
                        pts.append((r.value, r.ci_low, r.ci_high, color, shape, tip))
                    rows.append({"label": model, "points": pts})
                cards.append(
                    dot_panel(f"{run.election}: {title}", rows, ref=0.0, subtitle=pred, row_h=30)
                )
        keys = [VARIANT_STYLE[v] for v in TIE_VARIANTS]
        lg = legend(keys) if not ballot_suffix else ""  # one series needs no legend
        return (
            lg
            + f'<div class="grid2">{"".join(cards)}</div>'
            + robust_table(runs, w, source, ballot_suffix)
        )

    return _views(make)


def robust_table(runs: Sequence[RunData], w: str, source: str, ballot_suffix: str) -> str:
    """The rule proposed in ANALYSIS_PLAN §6: every tie variant's interval excludes zero
    in the same direction."""
    if ballot_suffix:
        return ""
    rows = []
    for run in runs:
        for metric in ("sp_deletion_share", "pc1_share"):
            if source == "ranking" and metric == "pc1_share":
                continue
            for model in run.models:
                row: dict[str, Any] = {"election": run.election, "outcome": metric, "model": model}
                signs, excl = [], []
                for var, suffix in TIE_VARIANTS.items():
                    s = run.summary[
                        (run.summary["variant"] == suffix) & (run.summary["weighting"] == w)
                        & (run.summary["source"] == source) & (run.summary["electorate"] == model)
                        & (run.summary["metric"] == f"{metric}_minus_human")
                    ]  # fmt: skip
                    if s.empty:
                        row[var] = "–"
                        continue
                    r = next(s.itertuples())
                    row[var] = f"{fmt(r.value)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]"
                    signs.append(np.sign(r.value))
                    excl.append(r.ci_low > 0 or r.ci_high < 0)
                row["same sign"] = len(set(signs)) == 1 if signs else math.nan
                row["robust (all intervals exclude 0, same sign)"] = (
                    bool(signs)
                    and len(set(signs)) == 1
                    and all(excl)
                    and len(excl) == len(TIE_VARIANTS)
                )
                rows.append(row)
    t = pd.DataFrame(rows).rename(
        columns={
            "default": "as is",
            "break_random": "ties broken at random",
            "drop_flat": "flat ballots dropped",
        }
    )
    return "<h3>All three side by side</h3>" + table_html(t)


LEVEL_METRICS = [
    (
        "sp_deletion_share",
        "Single-peakedness: deletion share",
        (0, 1),
        "lower = more single-peaked",
    ),
    ("pc1_share", "Dimensionality: PC1 share", (0, 1), "higher = more one-dimensional"),
    (
        "vr_mean_min_violation",
        "Value restriction: mean violation",
        None,
        "lower = closer to value-restricted",
    ),
    (
        "mean_kendall_tau",
        "Diversity: mean Kendall distance",
        (0, 0.6),
        "lower = more alike; 0.5 = unrelated",
    ),
    ("ballot_entropy", "Diversity: ballot entropy (bits)", None, "lower = fewer distinct ballots"),
    (
        "condorcet_winner_boot_freq",
        "Condorcet winner in bootstrap samples",
        (0, 1),
        "share of resamples",
    ),
    (
        "rule_disagreement_share",
        "Voting rules disagreeing",
        (0, 1),
        "share of rule pairs with different winners",
    ),
]


def section_levels(runs: Sequence[RunData]) -> str:
    """Each electorate's own value, with the human value as a reference line (H1, H3)."""

    def make(w: str, b: str) -> str:
        blocks = []
        for run in runs:
            cards = []
            for metric, title, xlim, note in LEVEL_METRICS:
                if metric == "pc1_share" and BALLOTS[b][0] == "ranking":
                    continue  # computed from thermometer scores only
                s = _rows(run, w, b, metric)
                if s.empty or s["value"].isna().all():
                    continue
                human = s[s["electorate"] == "human"]["value"]
                rows = []
                for e in _electorate_order(run):
                    r = s[s["electorate"] == e]
                    if r.empty:
                        continue
                    r = next(r.itertuples())
                    if e == "human":
                        color = "var(--s1)"
                    elif e.startswith("null"):
                        color = "var(--muted)"
                    else:
                        color = "var(--ink2)"
                    tip = f"{LABELS.get(e, e)}: {fmt(r.value)}{_ci(r)} · n = {int(r.n_voters):,}"
                    rows.append({"label": LABELS.get(e, e), "strong": e == "human",
                                 "points": [(r.value, r.ci_low, r.ci_high, color, "circle", tip)]})  # fmt: skip
                ref = float(human.iloc[0]) if len(human) and pd.notna(human.iloc[0]) else None
                cards.append(
                    dot_panel(title, rows, xlim=xlim, ref=ref, ref_label="human", subtitle=note)
                )
            blocks.append(f'<h3>{esc(run.election)}</h3><div class="grid3">{"".join(cards)}</div>')
        return "".join(blocks)

    return _views(make)


def section_ncurves(runs: Sequence[RunData]) -> str:
    metrics = [
        ("cycle_freq", "Share of sub-electorates with a majority cycle"),
        ("condorcet_winner_freq", "Share with a Condorcet winner"),
        ("sp_deletion_mean", "Mean deletion share"),
    ]

    def make(w: str, b: str) -> str:
        source, suffix = BALLOTS[b]
        out = []
        for metric, mtitle in metrics:
            blocks = [f"<h3>{esc(mtitle)}</h3>"]
            for run in runs:
                nc = run.ncurves[
                    (run.ncurves["variant"] == suffix) & (run.ncurves["weighting"] == w)
                ]
                nc = nc[(nc["source"] == source) | nc["electorate"].isin(REFERENCE)]
                if nc.empty:
                    continue
                xs = sorted(nc["n"].unique())
                ymax = float(nc[metric].max()) if metric != "sp_deletion_mean" else 1.0
                ylim = (0.0, 1.0) if metric != "cycle_freq" else (0.0, max(0.05, ymax * 1.1))

                def series(
                    e: str, color: str, label: str, nc: pd.DataFrame = nc, metric: str = metric
                ) -> dict[str, Any]:
                    d = nc[nc["electorate"] == e].sort_values("n")
                    return {"color": color, "points": [
                        (r.n, getattr(r, metric), f"{label}, n = {r.n}: {fmt(getattr(r, metric))}") for r in d.itertuples()
                    ]}  # fmt: skip

                panels = []
                for e in [*run.models, "heterogeneous"]:
                    if e not in set(nc["electorate"]):
                        continue
                    panels.append(line_panel(
                        LABELS.get(e, e),
                        [series("null_ic", "var(--muted)", LABELS["null_ic"]),
                         series("human", "var(--s1)", LABELS["human"]),
                         series(e, "var(--s2)", LABELS.get(e, e))],
                        xs=xs, ylim=ylim,
                    ))  # fmt: skip
                blocks.append(
                    f'<h4>{esc(run.election)}</h4><div class="grid5">{"".join(panels)}</div>'
                )
            out.append("".join(blocks))
        lg = legend(
            [
                ("var(--s1)", "line", "Human (ANES)"),
                ("var(--s2)", "line", "This electorate"),
                ("var(--muted)", "line", "Null: random orders"),
            ]
        )
        return lg + "".join(out)

    return _views(make)


H4 = [
    (
        "h4_strict_cycle_share",
        "Own pairwise choices form a cycle (share of triples)",
        0.25,
        "random choices",
    ),
    (
        "h4_wst_violation_share",
        "Weak stochastic transitivity violated (share of triples)",
        0.25,
        "random choices",
    ),
    ("h4_any_strict_cycle", "Voters with at least one cycle", None, ""),
    ("h4_ranking_agreement", "Pairwise choice agrees with own ranking", 0.5, "chance"),
    ("h4_abs_position_bias", "Position bias in pairwise choices (0 = none, 0.5 = total)", 0.0, ""),
]


def section_h4(runs: Sequence[RunData]) -> str:
    def make(w: str, b: str) -> str:
        blocks = []
        for run in runs:
            s = run.summary[
                (run.summary["variant"] == "")
                & (run.summary["weighting"] == w)
                & (run.summary["source"] == "pairwise")
            ]
            cards = []
            for metric, title, ref, ref_label in H4:
                d = s[s["metric"] == metric]
                if d.empty:
                    continue
                rows = [
                    {
                        "label": r.electorate,
                        "points": [
                            (
                                r.value,
                                None,
                                None,
                                "var(--ink2)",
                                "circle",
                                f"{r.electorate}: {fmt(r.value)} · n = {int(r.n_voters):,}",
                            )
                        ],
                    }
                    for r in d.sort_values("electorate").itertuples()
                ]
                xlim = (0.0, 1.0) if metric != "h4_abs_position_bias" else (0.0, 0.5)
                cards.append(
                    dot_panel(title, rows, xlim=xlim, ref=ref, ref_label=ref_label, label_w=110)
                )
            blocks.append(f'<h3>{esc(run.election)}</h3><div class="grid3">{"".join(cards)}</div>')
        return "".join(blocks)

    return _views(make, ballots=("thermometer",))


MATCHED = [
    ("tau_b_thermometer", "Kendall tau-b, human vs model thermometers", "div"),
    ("top_agree_thermometer", "Top choice agrees (thermometers)", "seq"),
    ("abs_error", "Mean absolute thermometer error (points)", "seq_rev"),
    ("tau_b_ranking", "Kendall tau-b, human thermometers vs model ranking", "div"),
    ("top_agree_ranking", "Top choice agrees (ranking)", "seq"),
]


def _heat(kind: str, lo: float, hi: float) -> Callable[[str, Any], str]:
    def style(col: str, v: Any) -> str:
        if (
            col in ("model", "group")
            or not isinstance(v, (int, float, np.floating))
            or not math.isfinite(v)
        ):
            return ""
        if kind == "div":
            t = max(-1.0, min(1.0, v))
            hue = "var(--pos)" if t >= 0 else "var(--neg)"
            return f' style="background: color-mix(in oklab, {hue} {abs(t) * 70:.0f}%, var(--surface))"'
        t = (v - lo) / ((hi - lo) or 1)
        t = 1 - t if kind == "seq_rev" else t
        return f' style="background: color-mix(in oklab, var(--pos) {max(0, min(1, t)) * 60:.0f}%, var(--surface))"'

    return style


def _group_order(groups: Iterable[str]) -> list[str]:
    order = [
        "all", "a strong Democrat", "a not very strong Democrat", "an independent who leans Democratic",
        "an independent", "an independent who leans Republican", "a not very strong Republican",
        "a strong Republican", "extremely liberal", "very liberal", "liberal", "somewhat liberal",
        "slightly liberal", "closer to liberal than conservative", "moderate",
        "neither liberal nor conservative", "slightly conservative", "closer to conservative than liberal",
        "somewhat conservative", "conservative", "very conservative", "extremely conservative", "(missing)",
    ]  # fmt: skip
    g = list(dict.fromkeys(groups))
    return sorted(g, key=lambda x: order.index(x) if x in order else len(order))


def section_matched(runs: Sequence[RunData]) -> str:
    def make(w: str, b: str) -> str:
        col = "weighted_mean" if w == "weighted" else "unweighted_mean"
        blocks = []
        for run in runs:
            mg = run.matched_groups[run.matched_groups["variant"] == ""]
            if mg.empty:
                continue
            parts = [f"<h3>{esc(run.election)}</h3>"]
            for metric, title, kind in MATCHED:
                for gv, gtitle in (("party_id_7", "by party ID"), ("ideology_7", "by ideology")):
                    d = mg[(mg["metric"] == metric) & mg["group_var"].isin(["all", gv])]
                    if d.empty:
                        continue
                    t = d.pivot_table(index="model", columns="group", values=col)
                    t = t[[g for g in _group_order(t.columns)]].reset_index()
                    vals = t.drop(columns="model").to_numpy(dtype=float)
                    lo, hi = np.nanmin(vals), np.nanmax(vals)
                    parts.append(
                        f"<h4>{esc(title)}, {gtitle}</h4>"
                        + table_html(t, cls="heat", heat=_heat(kind, lo, hi), digits=2)
                    )
            blocks.append("".join(parts))
        return "".join(blocks)

    return _views(make, ballots=("thermometer",))


def section_winners(runs: Sequence[RunData]) -> str:
    def make(w: str, b: str) -> str:
        source, suffix = BALLOTS[b]
        blocks = []
        for run in runs:
            win = run.winners[(run.winners["variant"] == suffix) & (run.winners["weighting"] == w)]
            win = win[(win["source"] == source) | win["electorate"].isin(REFERENCE)]
            win = win[~win["electorate"].isin(["heterogeneous", "null_ic", "null_permutation"])]
            if win.empty:
                continue
            t = win.pivot_table(
                index="electorate", columns="rule", values="winners", aggfunc="first"
            )
            t = t.reindex([e for e in _electorate_order(run) if e in t.index])
            t = t.reindex(
                columns=[
                    r
                    for r in ("plurality", "borda", "copeland", "minimax", "kemeny", "irv")
                    if r in t.columns
                ]
            )
            t.insert(0, "electorate", [LABELS.get(e, e) for e in t.index])
            blocks.append(
                f"<h3>{esc(run.election)}</h3>"
                + table_html(t.reset_index(drop=True).replace("|", " = ", regex=False))
            )
        return "".join(blocks)

    return _views(make)


def section_all(runs: Sequence[RunData]) -> str:
    frames = []
    for run in runs:
        s = run.summary.copy()
        s["analysis"] = s["variant"].map(lambda v: v.strip("_").replace("__", ", ") or "default")
        frames.append(s)
    s = pd.concat(frames, ignore_index=True)
    cols = [
        "election",
        "analysis",
        "electorate",
        "source",
        "weighting",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "interval",
        "n_voters",
    ]
    s = s[cols]
    return (
        '<input id="filter" type="search" placeholder="Filter rows, e.g. 2024 human sp_deletion weighted" aria-label="Filter the table">'
        f'<p class="muted" id="count">{len(s):,} rows</p>' + table_html(s, cls="all")
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------

CSS = """
:root { color-scheme: light;
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --pos:#2a78d6; --neg:#e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pos:#3987e5; --neg:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --pos:#3987e5; --neg:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header.bar { position: sticky; top:0; z-index:5; background:var(--surface);
  border-bottom:1px solid var(--border); padding:10px 16px; display:flex; flex-wrap:wrap;
  gap:8px 20px; align-items:center; }
header.bar h1 { font-size:16px; margin:0 12px 0 0; font-weight:600; }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.seg button { font:inherit; font-size:13px; background:none; color:var(--ink2); border:0;
  padding:5px 10px; cursor:pointer; }
.seg button[aria-pressed="true"] { background:var(--s1); color:#fff; }
.seg button:focus-visible { outline:2px solid var(--s1); outline-offset:-2px; }
.ctl { display:flex; align-items:center; gap:6px; color:var(--ink2); font-size:13px; }
nav.toc { padding:8px 16px; display:flex; flex-wrap:wrap; gap:4px 14px; font-size:13px; }
nav.toc a { color:var(--ink2); }
main { padding:0 16px 48px; max-width:1400px; margin:0 auto; }
section { background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:16px 18px; margin:16px 0; }
h2 { font-size:18px; margin:0 0 4px; } h3 { font-size:15px; margin:18px 0 6px; }
h4 { font-size:13px; margin:14px 0 4px; color:var(--ink2); font-weight:600; }
.lead { color:var(--ink2); max-width:80ch; margin:4px 0 8px; }
details.read { margin:6px 0 10px; color:var(--ink2); max-width:90ch; }
details.read summary { cursor:pointer; color:var(--ink); }
.muted { color:var(--muted); }
.grid2 { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(420px, 100%), 1fr)); gap:12px; }
.grid3 { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(340px, 100%), 1fr)); gap:12px; }
.grid5 { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(180px, 100%), 1fr)); gap:10px; }
figure.panel { margin:0; } figure.panel svg { width:100%; height:auto; display:block; }
figcaption { font-weight:600; font-size:13px; } .sub { color:var(--muted); font-size:12px; }
svg text { fill:var(--ink2); font-size:12px; } svg text.tick { fill:var(--muted); font-size:11px; }
svg text.rowlab.strong { fill:var(--ink); font-weight:600; }
svg .grid { stroke:var(--grid); stroke-width:1; } svg .axis { stroke:var(--axis); stroke-width:1; }
svg .ref { stroke:var(--ink2); stroke-width:1; }
.legend { display:flex; flex-wrap:wrap; gap:4px 16px; margin:4px 0 10px; font-size:13px; color:var(--ink2); }
.key { display:inline-flex; align-items:center; gap:6px; }
.tablewrap { overflow-x:auto; margin:6px 0 12px; max-width:100%; }
table { border-collapse:collapse; font-size:12.5px; }
th, td { padding:4px 10px; border-bottom:1px solid var(--grid); text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; position:sticky; top:0; background:var(--surface); }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
table.all { width:100%; } .tablewrap:has(table.all) { max-height:70vh; overflow:auto; }
#filter { font:inherit; padding:6px 10px; width:min(560px,100%); border:1px solid var(--border);
  border-radius:8px; background:var(--page); color:var(--ink); }
.view { display:none; }
"""


def _view_css() -> str:
    rules = [
        f'body[data-w="{w}"][data-b="{b}"] .view[data-w="{w}"][data-b="{b}"]'
        for w in WEIGHTINGS
        for b in BALLOTS
    ]
    # sections that ignore the ballot switch carry only the thermometer copy
    rules += [f'body[data-w="{w}"] .view.wonly[data-w="{w}"]' for w in WEIGHTINGS]
    return ",\n".join(rules) + " { display:block; }\n"


JS = """
const body = document.body;
function setView(key, value) {
  body.dataset[key] = value;
  document.querySelectorAll(`[data-set="${key}"]`).forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.value === value)));
  try { localStorage.setItem('scv-' + key, value); } catch (e) {}
}
document.querySelectorAll('[data-set]').forEach(b =>
  b.addEventListener('click', () => setView(b.dataset.set, b.dataset.value)));
for (const k of ['w', 'b']) {
  let v = null; try { v = localStorage.getItem('scv-' + k); } catch (e) {}
  if (v && document.querySelector(`[data-set="${k}"][data-value="${v}"]`)) setView(k, v);
}
const f = document.getElementById('filter');
if (f) f.addEventListener('input', () => {
  const words = f.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let n = 0;
  document.querySelectorAll('table.all tbody tr').forEach(tr => {
    const t = tr.textContent.toLowerCase();
    const show = words.every(w => t.includes(w));
    tr.style.display = show ? '' : 'none'; n += show;
  });
  document.getElementById('count').textContent = n.toLocaleString() + ' rows';
});
"""

READ = {
    "overview": "Which runs and analysis folders this page was built from, and with which settings.",
    "quality": (
        "The checks of ANALYSIS_PLAN §6, to read before any result. <b>Ranking</b> columns: "
        "a model with no position bias ranks the first-listed candidate first about as often "
        "as the chance column says, copies the list order about as rarely as its chance "
        "column says, and has a position–rank correlation near 0 (negative = later-listed "
        "candidates ranked higher). <b>Flat ballots</b> give every candidate the same score; "
        "they count as single-peaked on every ordering, which is why the tie variants exist "
        "(DECISIONS G20, G21)."
    ),
    "h1": (
        "Each mark is a model's value minus the human value for the same respondents, with "
        "a 95% interval from resampling the same voters in both. The three marks per model "
        "are the default analysis and the two tie-robustness variants. The proposed rule "
        "(ANALYSIS_PLAN §3, §6) counts a gap as robust only if all three intervals exclude "
        "0 on the same side. PC1 share barely changes when ties are broken at random, by "
        "construction, so for that outcome only the dropped-flat variant is a real check. "
        "The ranking views show the default analysis only, and compare the models' ranking "
        "ballots, which never contain ties, with the human thermometer ballots, which often "
        "do: the two are not like for like (ANALYSIS_PLAN §6.2)."
    ),
    "levels": (
        "Each electorate's own value, with the human value as a vertical line and the two "
        "nulls as a scale: <b>random orders</b> (impartial culture) is what no structure "
        "looks like; <b>permuted human</b> keeps each candidate's human ratings but breaks "
        "the link between candidates. Intervals: voter bootstrap for human and single-model "
        "electorates; spread over 50 random draws for the mixed and null electorates (not a "
        "sampling interval). H3 compares <b>Mixed models</b> with the single-model rows."
    ),
    "ncurves": (
        "Random sub-electorates of n voters (500 per size). Cycles are rare in large "
        "electorates whatever the preferences, so the small sizes are where electorates "
        "differ. Each panel shows one electorate against the human one and the random-order "
        "null. The human and null curves always come from thermometer ballots."
    ),
    "h4": (
        "From each voter's own pairwise answers (both presentation orders averaged), per "
        "model; there is no human counterpart. The reference lines are what random choices "
        "would give: a strict cycle or a weak-stochastic-transitivity violation in about a "
        "quarter of triples, agreement with the own ranking half the time. Ranking agreement "
        "uses the greedy ranking, which carries position effects. Unreadable answers are "
        "left out (DECISIONS G19)."
    ),
    "matched": (
        "Person-by-person agreement between each respondent's human thermometers and the "
        "model's answers for their persona, as means over respondents in each group. "
        "Kendall tau-b: 1 = same order, 0 = unrelated, −1 = reversed. Top choice agrees "
        "when the two top sets share a candidate. Absolute error is in thermometer points."
    ),
    "winners": (
        "Winner under each voting rule for the whole electorate. Tied winners are joined with “=”."
    ),
    "all": (
        "Every row of every summary.csv on this page. Type words to filter (all must match). "
        "`_minus_human` rows are the paired differences."
    ),
}


def build_report(runs: Sequence[RunData]) -> str:
    sections = [
        ("overview", "Runs", section_overview(runs), False),
        ("quality", "Checks before reading", section_quality(runs), False),
        ("h1", "H1: model minus human, with tie checks", section_h1_gaps(runs), True),
        ("levels", "Levels: every electorate (H1, H3)", section_levels(runs), True),
        ("ncurves", "Sub-electorates of n voters (H1)", section_ncurves(runs), True),
        ("h4", "H4: each voter's own coherence", section_h4(runs), "w"),
        ("matched", "Matched comparisons, by group", section_matched(runs), "w"),
        ("winners", "Winners by rule", section_winners(runs), True),
        ("all", "All numbers", section_all(runs), False),
    ]
    toc = "".join(f'<a href="#{k}">{esc(t)}</a>' for k, t, _, _ in sections)
    seg_w = "".join(
        f'<button data-set="w" data-value="{w}" aria-pressed="{str(w == "weighted").lower()}">{w.capitalize()}</button>'
        for w in WEIGHTINGS
    )
    seg_b = "".join(
        f'<button data-set="b" data-value="{b}" aria-pressed="{str(b == "thermometer").lower()}">{esc(BALLOT_LABELS[b])}</button>'
        for b in BALLOTS
    )
    body = []
    for key, title, content, switched in sections:
        note = ""
        if switched == "w":
            note = '<p class="muted">Follows the weighting switch only.</p>'
            content = content.replace('class="view"', 'class="view wonly"')
        elif not switched:
            note = ""
        body.append(
            f'<section id="{key}"><h2>{esc(title)}</h2>'
            f'<details class="read"><summary>How to read this</summary><p>{READ[key]}</p></details>'
            f"{note}{content}</section>"
        )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silicon electorate results</title>
<style>{CSS}{_view_css()}</style></head>
<body data-w="weighted" data-b="thermometer">
<header class="bar"><h1>Silicon electorate results</h1>
<span class="ctl">Weighting <span class="seg" role="group" aria-label="Weighting">{seg_w}</span></span>
<span class="ctl">Ballots <span class="seg" role="group" aria-label="Ballots">{seg_b}</span></span>
<span class="muted" style="font-size:12px">Built {stamp} by <code>scv report</code>. No conclusions are drawn on this page.</span>
</header>
<nav class="toc" aria-label="Sections">{toc}</nav>
<main>{"".join(body)}</main>
<script>{JS}</script></body></html>"""


def write_report(
    run_ids: Sequence[str],
    out: str | Path,
    results: str | Path = "results",
    with_calls: bool = True,
) -> Path:
    runs = [load_run(r, results, with_calls) for r in run_ids]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(runs), encoding="utf-8")
    return out
