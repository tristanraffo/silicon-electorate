"""Command line interface: ``scv <command>``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from scv import anes
from scv.backends.base import BackendError
from scv.runner import call_log_dir, dry_run, load_run_config, run_from_config

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """silicon-electorate: Arrow's domain condition in LLM electorates."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@app.command()
def elicit(
    config: Annotated[Path, typer.Option(help="Run config (configs/runs/*.yaml).")],
    dry_run_: Annotated[
        bool,
        typer.Option("--dry-run", help="Print call counts and a token/time estimate, then exit."),
    ] = False,
    limit_respondents: Annotated[
        int | None,
        typer.Option(help="Pilot on K respondents (deterministic, stratified on party id)."),
    ] = None,
) -> None:
    """Run or resume an elicitation run. Always do a --dry-run first."""
    cfg = load_run_config(config)
    if dry_run_:
        table = dry_run(cfg, limit_respondents)
        typer.echo(f"run_id: {cfg.run_id}   backend: {cfg.backend}   log: {call_log_dir(cfg)}")
        typer.echo(table.to_string(index=False))
        total = table[["calls", "remaining", "prompt_tokens", "output_tokens"]].sum()
        typer.echo(
            f"TOTAL calls={total['calls']} remaining={total['remaining']} "
            f"prompt_tokens={total['prompt_tokens']} output_tokens={total['output_tokens']} "
            f"est_time={table['est_seconds'].sum() / 60:.1f} min "
            f"(assumes {cfg.throughput.output_tokens_per_s:g} output tok/s; retries excluded)"
        )
        return
    try:
        stats = run_from_config(cfg, limit_respondents, progress=True)
    except BackendError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    for model, s in stats.items():
        typer.echo(f"{model}: planned={s.planned} already_done={s.skipped} written={s.written}")
    typer.echo(f"call log: {call_log_dir(cfg)}")


@app.command()
def analyze(
    config: Annotated[Path, typer.Option(help="Run config of a finished run.")],
    bootstrap: Annotated[int, typer.Option(help="Voter-bootstrap replicates (B).")] = 1000,
    draws: Annotated[int, typer.Option(help="Sub-electorates per size for the n-curves.")] = 500,
    replicates: Annotated[
        int, typer.Option(help="Heterogeneous assignments and null draws (R).")
    ] = 50,
    seed: Annotated[int, typer.Option(help="Seed for all resampling.")] = 0,
    no_figures: Annotated[bool, typer.Option("--no-figures", help="Tables only.")] = False,
    exclude_ranking: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-ranking",
            help="Sensitivity analysis: leave this model's ranking ballots out (repeatable). "
            "Writes to results/metrics/<run_id>__ranking_without_<models>/.",
        ),
    ] = None,
    ranking_ballot: Annotated[
        str,
        typer.Option(
            "--ranking-ballot",
            help="'greedy' (default) or 'combined': average each voter's rankings, which "
            "cancels position bias. Writes to results/metrics/<run_id>__ranking_combined/.",
        ),
    ] = "greedy",
    ties: Annotated[
        str,
        typer.Option(
            "--ties",
            help="'as_is' (default), 'break_random' (break every tie at random) or "
            "'drop_flat' (leave out respondents with a flat thermometer ballot anywhere). "
            "Writes to results/metrics/<run_id>__ties_<mode>/.",
        ),
    ] = "as_is",
) -> None:
    """Compute every metric for a run: summary.csv, Parquet tables and three figures."""
    from scv.analysis import AnalysisOptions, analyze_run

    cfg = load_run_config(config)
    opts = AnalysisOptions(bootstrap=bootstrap, draws=draws, replicates=replicates, seed=seed)
    try:
        out = analyze_run(
            cfg,
            opts,
            figures=not no_figures,
            exclude_ranking=tuple(exclude_ranking or ()),
            ranking_ballot=ranking_ballot,
            ties=ties,
        )
    except (anes.ConfigError, ValueError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"wrote {out}/summary.csv")
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "summary.csv":
            typer.echo(f"  {path.relative_to(out)}")


@app.command()
def report(
    run_ids: Annotated[list[str], typer.Argument(help="Run ids to include, in page order.")],
    out: Annotated[Path, typer.Option(help="HTML file to write.")] = Path(
        "results/report/index.html"
    ),
    results: Annotated[Path, typer.Option(help="Results folder.")] = Path("results"),
    no_calls: Annotated[
        bool, typer.Option("--no-calls", help="Skip the call logs (no parse-rate table).")
    ] = False,
) -> None:
    """One self-contained HTML page with every analysed result of the given runs."""
    from scv.report import write_report

    try:
        path = write_report(run_ids, out, results, with_calls=not no_calls)
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


data_app = typer.Typer(no_args_is_help=True, help="ANES data: check configs, build tables.")
app.add_typer(data_app, name="data")


@data_app.command("check")
def data_check(
    election: Annotated[str, typer.Argument(help="Election id or path to its YAML.")],
) -> None:
    """List every TODO and every configured variable missing from the data file.
    Exits non-zero if there is anything to fix."""
    try:
        config = anes.load_election(election)
    except (anes.ConfigError, ValueError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e
    problems = anes.check(config)
    if not problems:
        typer.echo(f"{config.id}: OK ({len(config.variables())} variables found in {config.file})")
        return
    typer.echo(f"{config.id}: {len(problems)} problems")
    for p in problems:
        typer.echo(f"  {p}")
    raise typer.Exit(code=1)


@data_app.command("build")
def data_build(
    election: Annotated[str, typer.Argument(help="Election id or path to its YAML.")],
    out: Annotated[Path, typer.Option(help="Root for processed tables.")] = Path("data/processed"),
) -> None:
    """Write human_scores.parquet and personas.parquet for one election."""
    try:
        config = anes.load_election(election)
        directory = anes.build(config, out)
    except (anes.ConfigError, ValueError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e
    for line in json.loads((directory / "exclusions.json").read_text()):
        typer.echo(f"  {line['step']}: dropped {line['dropped']}, {line['remaining']} remain")
    typer.echo(f"wrote {directory}/human_scores.parquet and personas.parquet")


if __name__ == "__main__":
    app()
