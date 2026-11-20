"""Rule-based classification: document type, period and language.

Rules are data (regular expressions declared in the recipe). The first matching rule
wins. Candidates that no rule can type are left undecided (``type=None``, score 0);
an LLM fallback for them comes in a later milestone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_PATTERN_LENGTH = 500
_ID = r"^[a-z][a-z0-9_]*$"


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str = Field(min_length=1, max_length=MAX_PATTERN_LENGTH)
    target: Literal["url", "text", "any"] = "url"
    """``url``: decoded URL path; ``text``: title and surrounding text; ``any``: both."""
    ignore_case: bool = True

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

    def search(self, url_text: str, text: str) -> re.Match[str] | None:
        if self.target in ("url", "any") and (m := self.regex.search(url_text)):
            return m
        if self.target in ("text", "any") and (m := self.regex.search(text)):
            return m
        return None

    def describe(self) -> str:
        return f"{self.target} /{self.pattern}/"


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
    text: str,
    default_language: str | None = None,
) -> Classification:
    url_text = unquote(urlsplit(url).path)
    notes: list[str] = []

    doc_type = None
    for type_rule in rules.type:
        if type_rule.search(url_text, text):
            doc_type = type_rule.type
            notes.append(f"type={doc_type} by {type_rule.describe()}")
            break

    period = period_year = None
    for period_rule in rules.period:
        if m := period_rule.search(url_text, text):
            period, period_year = normalize_period(m)
            notes.append(f"period={period} by {period_rule.describe()}")
            break

    language = None
    for lang_rule in rules.language:
        if m := lang_rule.search(url_text, text):
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
