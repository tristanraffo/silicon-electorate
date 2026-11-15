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

The GPU extra (`uv sync --extra gpu`) is for the vast.ai box only; the runbook lands with M6.
