"""Rule-based classification: document type, period and language.

Rules are data (regular expressions declared in the recipe). The first matching rule
wins. Candidates that no rule can type are left undecided (``type=None``, score 0);
an LLM fallback for them comes in a later milestone.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_PATTERN_LENGTH = 500
_ID = r"^[a-z][a-z0-9_]*$"

FieldName = Literal[
    "url",
    "text",
    "title",
    "link_text",
    "link_title",
    "context",
    "heading",
    "row_header",
    "column_header",
]
"""Texts a rule can read. ``url`` is the decoded URL path; ``text`` is title + context."""
V1_TARGETS = frozenset({"url", "text", "any"})
JOIN = " | "


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str = Field(min_length=1, max_length=MAX_PATTERN_LENGTH)
    target: Literal["any"] | FieldName | list[FieldName] = "url"
    """One field, ``any`` (url, then text), or a list of fields searched as one string
    joined with ``" | "`` (e.g. ``[row_header, heading]`` -> ``"Q2 | 2025"``)."""
    ignore_case: bool = True

    @field_validator("target")
    @classmethod
    def _short_list(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not 1 <= len(value) <= 4:
            raise ValueError("a target list holds 1 to 4 fields")
        return value

    @property
    def uses_v1_targets_only(self) -> bool:
        return isinstance(self.target, str) and self.target in V1_TARGETS

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression {value!r}: {exc}") from exc
        return value

    @cached_property
    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE if self.ignore_case else 0)

    def search(self, fields: Mapping[str, str | None]) -> re.Match[str] | None:
        """Search the targeted field(s); ``fields`` must include ``url`` and ``text``."""
        if self.target == "any":
            return self.regex.search(fields.get("url") or "") or self.regex.search(
                fields.get("text") or ""
            )
        if isinstance(self.target, list):
            return self.regex.search(JOIN.join(fields.get(n) or "" for n in self.target))
        return self.regex.search(fields.get(self.target) or "")

    def describe(self) -> str:
        target = "+".join(self.target) if isinstance(self.target, list) else self.target
        return f"{target} /{self.pattern}/"


class TypeRule(_Rule):
    type: str = Field(pattern=_ID)


class PeriodRule(_Rule):
    """Named groups: ``year`` (required) and optionally ``q`` (quarter), ``h`` (half),
    ``fy`` (full year). ``q``/``h`` may carry a prefix such as ``t3``, ``q3``, ``s1``."""

    @model_validator(mode="after")
    def _has_year_group(self) -> PeriodRule:
        if "year" not in self.regex.groupindex:
            raise ValueError(f"period rule {self.pattern!r} needs a named group 'year'")
        return self


class LanguageRule(_Rule):
    """Either a named group ``lang`` or a fixed ``language`` value."""

    language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")

    @model_validator(mode="after")
    def _has_value(self) -> LanguageRule:
        if self.language is None and "lang" not in self.regex.groupindex:
            raise ValueError(f"language rule {self.pattern!r} needs group 'lang' or 'language'")
        return self


class ClassifyRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: list[TypeRule] = []
    period: list[PeriodRule] = []
    language: list[LanguageRule] = []

    def all_rules(self) -> list[_Rule]:
        return [*self.type, *self.period, *self.language]


@dataclass(frozen=True)
class Classification:
    type: str | None
    period: str | None
    period_year: int | None
    language: str | None
    score: float
    justification: list[str] = field(default_factory=list)


def classify(
    rules: ClassifyRules,
    *,
    url: str,
    text: str = "",
    fields: Mapping[str, str | None] | None = None,
    default_language: str | None = None,
) -> Classification:
    """Apply the rules. ``fields`` are the candidate's named texts (see ``FieldName``);
    when omitted, only ``text`` is available besides the URL."""
    values: dict[str, str | None] = {"text": text, **(fields or {})}
    values["url"] = unquote(urlsplit(url).path)
    notes: list[str] = []

    doc_type = None
    for type_rule in rules.type:
        if type_rule.search(values):
            doc_type = type_rule.type
            notes.append(f"type={doc_type} by {type_rule.describe()}")
            break

    period = period_year = None
    for period_rule in rules.period:
        if m := period_rule.search(values):
            period, period_year = normalize_period(m)
            notes.append(f"period={period} by {period_rule.describe()}")
            break

    language = None
    for lang_rule in rules.language:
        if m := lang_rule.search(values):
            language = lang_rule.language or m.group("lang").lower()
            notes.append(f"language={language} by {lang_rule.describe()}")
            break
    if language is None and default_language:
        language = default_language
        notes.append(f"language={language} by recipe default")

    if doc_type is None:
        notes.append("type undecided by rules")
    return Classification(
        type=doc_type,
        period=period,
        period_year=period_year,
        language=language,
        score=1.0 if doc_type else 0.0,
        justification=notes,
    )


def normalize_period(match: re.Match[str]) -> tuple[str, int]:
    """Canonical period (``2025-Q3``, ``2025-H1``, ``2025-FY`` or ``2025``) and its year."""
    groups = match.groupdict()
    year = int(groups["year"])
    if len(groups["year"]) == 2:
        year += 2000
    if groups.get("q"):
        return f"{year}-Q{_digit(groups['q'])}", year
    if groups.get("h"):
        return f"{year}-H{_digit(groups['h'])}", year
    if groups.get("fy"):
        return f"{year}-FY", year
    return str(year), year


def _digit(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if not digits:
        raise ValueError(f"no digit in period component {value!r}")
    return digits
