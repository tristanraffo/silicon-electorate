"""Prompt templates: loading, rendering and hashing (SPEC §8, §10).

Templates live in ``configs/prompts/*.yaml`` and are versioned by content: the hash of
a template's full parsed content is stored with every call, so editing a template in
place still changes the hash (and therefore the call keys).
"""

from __future__ import annotations

import hashlib
import json
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"

#: Words that must never reach a model (SPEC §10.5).
FORBIDDEN_WORDS = ("arrow", "voting theory", "consistency", "cycles")


def content_hash(obj: Any) -> str:
    """Stable 16-hex-digit hash of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Template:
    """One prompt template file. ``data`` is the parsed YAML; ``hash`` covers all of it."""

    id: str
    data: Mapping[str, Any]
    hash: str

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def has(self, key: str) -> bool:
        return key in self.data

    def render(self, key: str, **values: str) -> str:
        """Fill ``data[key]``. Missing or unused placeholders are errors, not blanks."""
        text = self.data[key]
        wanted = {name for _, name, _, _ in string.Formatter().parse(text) if name}
        if wanted != set(values):
            raise KeyError(
                f"template {self.id!r}.{key}: placeholders {sorted(wanted)} "
                f"but values given for {sorted(values)}"
            )
        return text.format(**values)


def load_template(path: str | Path) -> Template:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "id" not in data:
        raise ValueError(f"{path}: a prompt template must be a mapping with an 'id'")
    for text in _strings(data):
        lowered = text.lower()
        for word in FORBIDDEN_WORDS:
            if word in lowered:
                raise ValueError(f"{path}: template contains the forbidden word {word!r}")
    return Template(id=str(data["id"]), data=data, hash=content_hash(data))


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


@dataclass(frozen=True)
class Templates:
    """The template set used by one run."""

    persona: Template
    thermometer: Template
    ranking: Template
    pairwise: Template

    def for_mode(self, mode: str) -> Template:
        return {
            "thermometer": self.thermometer,
            "ranking": self.ranking,
            "pairwise": self.pairwise,
        }[mode]


def load_templates(
    directory: str | Path = PROMPTS_DIR,
    persona: str = "persona_v1",
    thermometer: str = "thermometer_v1",
    ranking: str = "ranking_v4_strict",
    pairwise: str = "pairwise_v1",
) -> Templates:
    d = Path(directory)
    return Templates(
        persona=load_template(d / f"{persona}.yaml"),
        thermometer=load_template(d / f"{thermometer}.yaml"),
        ranking=load_template(d / f"{ranking}.yaml"),
        pairwise=load_template(d / f"{pairwise}.yaml"),
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def render_persona(template: Template, fields: Mapping[str, Any]) -> str:
    """First-person backstory. A sentence whose field is missing (None, NaN or blank) is
    left out; nothing is ever filled with "unknown". Unknown field names are an error."""
    known = {s["field"] for s in template["sentences"]}
    extra = set(fields) - known
    if extra:
        raise KeyError(f"persona fields not in template {template.id!r}: {sorted(extra)}")
    parts = []
    for s in template["sentences"]:
        value = fields.get(s["field"])
        if not _is_missing(value):
            parts.append(s["text"].format(**{s["field"]: str(value).strip()}))
    return template["joiner"].join(parts)
