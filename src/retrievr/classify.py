"""Rule-based classification: document type, period and language.

Rules are data (regular expressions declared in the recipe). The first matching rule
wins. Candidates that no rule can type are left undecided (``type=None``, score 0).

The LLM fallback (:func:`classify_with_llm`) is a cold path: it only sees documents the
rules leave undecided, and its decisions are stored so later runs do not ask again.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retrievr.llm import UNTRUSTED_NOTICE, StructuredLLM, untrusted
from retrievr.taxonomy import DocumentType, canonical_period, taxonomy_text

if TYPE_CHECKING:
    from retrievr.recipe import FoundDocument

MAX_PATTERN_LENGTH = 500
_ID = r"^[a-z][a-z0-9_]*$"

FieldName = Literal[
    "url",
    "text",
    "title",
    "link_text",
    "link_title",
    "file_name",
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
    ``fy`` (full year). ``q``/``h`` may carry a prefix such as ``t3``, ``q3``, ``s1``.

    Schema v3: ``period`` gives a fixed sub-period (``H1``, ``Q3``, ``FY``) for patterns
    that cannot carry it in a group, e.g. "au 30 juin (?P<year>20\\d{2})" -> ``H1``."""

    period: str | None = Field(default=None, pattern=r"^(Q[1-4]|H[12]|FY)$")

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
            try:
                period, period_year = normalize_period(m)
                if period_rule.period is not None:
                    period = f"{period_year}-{period_rule.period}"
            except (ValueError, IndexError):
                continue  # e.g. a "q" group without a digit: try the next rule
            notes.append(f"period={period} by {period_rule.describe()}")
            break

    language = None
    for lang_rule in rules.language:
        if m := lang_rule.search(values):
            language = lang_rule.language or (m.group("lang") or "").lower() or None
            if language is None:
                continue
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


_ORDINALS = (
    ("1", r"first|1st|premi[eè]re?|primer|primero|erste[snr]?"),
    ("2", r"second|2nd|deuxi[eè]me|seconde?|segundo|zweite[snr]?"),
    ("3", r"third|3rd|troisi[eè]me|tercer|tercero|dritte[snr]?"),
    ("4", r"fourth|4th|quatri[eè]me|cuarto|vierte[snr]?"),
)


def _digit(value: str) -> str:
    """The number of a quarter or half: ``Q3``, ``3Q``, ``T3``, or a word (``third``,
    ``troisième``, ``tercer``...)."""
    digits = re.sub(r"\D", "", value)
    if digits:
        return digits
    for digit, words in _ORDINALS:
        if re.search(rf"\b(?:{words})\b", value, re.IGNORECASE):
            return digit
    raise ValueError(f"no digit in period component {value!r}")


# -- LLM fallback ----------------------------------------------------------------------

FALLBACK_BATCH = 50
_TEXT_FIELDS = ("link_text", "title", "heading", "row_header", "column_header", "link_title")


@dataclass(frozen=True)
class Decision:
    """Classification of one document decided outside the recipe rules."""

    type: str | None
    period: str | None
    fiscal_year: int | None
    language: str | None
    source: str
    """Who decided, e.g. ``llm:learn`` or ``llm:classify``."""


class DocumentLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: DocumentType
    period: str | None
    """2025-Q3, 2025-H1, 2025-FY or 2025; null when the document gives no period."""
    language: str | None
    """ISO 639-1 code of the document, or null."""

    @field_validator("period")
    @classmethod
    def _canonical(cls, value: str | None) -> str | None:
        return canonical_period(value)

    @field_validator("language")
    @classmethod
    def _lang(cls, value: str | None) -> str | None:
        value = (value or "").strip().lower()
        return value if re.fullmatch(r"[a-z]{2,3}", value) else None

    def decision(self, source: str) -> Decision:
        year = int(self.period[:4]) if self.period else None
        return Decision(self.type, self.period, year, self.language, source)


class LabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[DocumentLabel]


def describe_document(doc_id: str, doc: FoundDocument) -> dict[str, str]:
    """Compact, prompt-friendly view of a document (texts that rules can read)."""
    path = unquote(urlsplit(doc.url).path)
    out = {"id": doc_id, "file": path.rsplit("/", 1)[-1][:120], "path": path[-160:]}
    query = urlsplit(doc.url).query
    if query:
        out["query"] = unquote(query)[:120]
    seen = {out["file"]}
    for name in _TEXT_FIELDS:
        value = " ".join((doc.fields.get(name) or "").split())[:160]
        if value and value not in seen:
            out[name] = value
            seen.add(value)
    context = " ".join((doc.fields.get("context") or "").split())
    if context and context not in seen:
        out["context"] = context[:200]
    return out


def document_table(docs: Sequence[FoundDocument], prefix: str = "D") -> tuple[str, list[str]]:
    """JSON lines describing ``docs`` and the ids used for them."""
    ids = [f"{prefix}{i + 1}" for i in range(len(docs))]
    lines = [
        json.dumps(describe_document(i, d), ensure_ascii=False)
        for i, d in zip(ids, docs, strict=True)
    ]
    return "\n".join(lines), ids


FALLBACK_SYSTEM = f"""You classify documents listed on investor-relations web pages.
For each document, give its type, the period it covers and its language, from its file
name, link text, headings and table headers. Use only the listed type ids.
{UNTRUSTED_NOTICE}

"""


def classify_with_llm(
    llm: StructuredLLM,
    docs: Sequence[FoundDocument],
    *,
    issuer: str,
    step: str = "classify",
    batch_size: int = FALLBACK_BATCH,
) -> dict[str, Decision]:
    """Ask the LLM for the documents the rules could not fully classify."""
    decisions: dict[str, Decision] = {}
    for start in range(0, len(docs), batch_size):
        batch = list(docs[start : start + batch_size])
        table, ids = document_table(batch)
        messages = [
            {"role": "system", "content": FALLBACK_SYSTEM + taxonomy_text()},
            {
                "role": "user",
                "content": f"Issuer: {issuer}\nGive one label per document id.\n"
                + untrusted("documents", table),
            },
        ]
        reply = llm.structured(messages, LabelBatch, step=step)
        by_id = {label.id: label for label in reply.labels}
        for doc_id, doc in zip(ids, batch, strict=True):
            if (label := by_id.get(doc_id)) is not None:
                decisions[doc.normalized_url] = label.decision(f"llm:{step}")
    return decisions


def needs_decision(doc: FoundDocument) -> bool:
    return doc.type is None or doc.period is None


def apply_decisions(
    docs: Iterable[FoundDocument], decisions: Mapping[str, Decision]
) -> list[FoundDocument]:
    """Fill what the rules left undecided; rules always win where they decided."""
    out = []
    for doc in docs:
        dec = decisions.get(doc.normalized_url)
        if dec is None or (not needs_decision(doc) and doc.language is not None):
            out.append(doc)
            continue
        changes: dict[str, object] = {}
        notes = []
        if doc.type is None and dec.type is not None:
            changes["type"] = dec.type
            changes["score"] = 0.8
            notes.append(f"type={dec.type} by {dec.source}")
        if doc.period is None and dec.period is not None:
            changes["period"] = dec.period
            changes["fiscal_year"] = dec.fiscal_year
            notes.append(f"period={dec.period} by {dec.source}")
        if doc.language is None and dec.language is not None:
            changes["language"] = dec.language
            notes.append(f"language={dec.language} by {dec.source}")
        if notes:
            changes["justification"] = "; ".join(
                [
                    *(p for p in doc.justification.split("; ") if p != "type undecided by rules"),
                    *notes,
                ]
            )
        out.append(dataclasses.replace(doc, **changes) if changes else doc)  # type: ignore[arg-type]
    return out
