"""Command line interface: ``scv <command>``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

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


if __name__ == "__main__":
    app()
