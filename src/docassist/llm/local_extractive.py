"""Deterministic, offline "LLM" for development, demos, air-gapped installs and tests.

It never generates free text: every answer is assembled from sentences copied verbatim
from the input, so it cannot hallucinate and it cannot follow instructions embedded in a
document - it has no instruction-following ability at all. That makes it a useful floor for
the evaluation harness (citations are always verifiable) and a safe default provider.

* ``ANSWER``: parses the nonce-delimited ``<source id="S#" nonce=...>`` blocks and the
  ``<question nonce=...>`` block built by :mod:`docassist.rag.context`, scores sentences by
  IDF-weighted overlap with the question (the document title/section count as context),
  and returns schema-valid JSON quoting 1-3 supporting sentences with their source ids -
  or ``insufficient_context`` when the best overlap is below :data:`ANSWER_THRESHOLD`.
  Sentences that look like instructions to an AI are never quoted.
* any other task with an ``output_schema``: a schema-driven extractive filler - the most
  central sentences of the input fill summary-like strings/lists, enums are chosen by
  keyword evidence, unknown scalars become ``null`` when the schema allows it. The result
  is validated against the schema before it is returned.
* no schema: an extractive summary (the most central sentences, in document order).

Tools are not supported (``supports_tools = False``).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from docassist.core.text import estimate_tokens
from docassist.llm import schema as jsonschema
from docassist.llm.base import LLMError, LLMRequest, LLMResult, LLMTask

LOCAL_EXTRACTIVE_MODEL = "local-extractive-v1"
ANSWER_THRESHOLD = 0.6
INSUFFICIENT_ANSWER = (
    "The provided documents do not contain enough information to answer this question."
)
MAX_QUOTES = 3
MAX_QUOTE_CHARS = 480

_SOURCES_RE = re.compile(r'<sources nonce="([0-9a-f]{8,64})">')
_ATTR_RE = re.compile(r'([a-z][a-z-]*)="([^"]*)"')
_WORD_RE = re.compile(r"[a-z0-9]+(?:['.-][a-z0-9]+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_TAG_RE = re.compile(r"</?[a-z_-]+(?:\s[^<>]*)?>")
_INSTRUCTION_CUES = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule|previous|above)"
    r"|\bsystem prompt\b|\byou are now\b|\bact as\b|\bnew instructions?\b"
    r"|\b(send|forward|email|post|upload)\b.{0,60}\b(to|at)\b.{0,40}(https?://|www\.|@)"
    r"|!\[[^\]]*\]\(|\bdo not tell\b|\bdeveloper mode\b|\bjailbreak\b",
    re.IGNORECASE | re.DOTALL,
)
_STOPWORD_TEXT = (
    "a an the of and or to in on for with by at from is are was were be been being as that "
    "this these those it its into than then there their them they we you he she his her our "
    "your not no do does did what which who whom whose when where why how can could should "
    "would will shall may might must about any all each every some such tell me please give "
    "show list find i my us if so also just only per via has have had"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_SUMMARY_NAMES = (
    "summary",
    "answer",
    "text",
    "overview",
    "description",
    "content",
    "body",
    "abstract",
)
_LIST_NAMES = (
    "points",
    "highlights",
    "bullets",
    "findings",
    "items",
    "sentences",
    "summary",
    "changes",
)
_QUOTE_NAMES = ("quote", "evidence", "excerpt", "sentence", "snippet")
_ID_NAMES = ("source_id", "chunk_id", "id", "ref", "hunk_id", "source")


_SUFFIXES = (
    "ational", "ations", "ation", "ments", "ment", "ings", "ing", "ates", "ated", "ate",
    "ies", "ied", "ed", "es", "al", "e", "s",
)  # fmt: skip


def _stem(word: str) -> str:
    """Tiny suffix stripper: payment/pay, invoices/invoice, expires/expiration/expire."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        if suffix == "s" and word.endswith("ss"):
            return word
        base = word[: -len(suffix)]
        return base + "y" if suffix in {"ies", "ied"} else base
    return word


def terms(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
    )


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SENTENCE_SPLIT.split(text):
        sentence = " ".join(piece.split())
        if len(sentence) < 12 or len(sentence.split()) < 3:
            continue
        while len(sentence) > MAX_QUOTE_CHARS:
            cut = sentence.rfind(" ", 0, MAX_QUOTE_CHARS)
            cut = cut if cut > 100 else MAX_QUOTE_CHARS
            out.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if len(sentence) >= 12:
            out.append(sentence)
    return out


@dataclass(frozen=True, slots=True)
class ParsedSource:
    sid: str
    title: str
    section: str
    flagged: bool
    content: str


@dataclass(frozen=True, slots=True)
class ParsedPrompt:
    question: str
    sources: list[ParsedSource]


def parse_answer_prompt(text: str) -> ParsedPrompt:
    """Extract the question and the sources that carry the prompt's own nonce."""
    match = _SOURCES_RE.search(text)
    if match is None:
        return ParsedPrompt(question="", sources=[])
    nonce = match.group(1)
    source_re = re.compile(
        r'<source id="(S\d{1,3})" nonce="' + re.escape(nonce) + r'"([^<>]*)>\n?(.*?)\n?</source>',
        re.DOTALL,
    )
    sources: list[ParsedSource] = []
    for sid, attrs, body in source_re.findall(text):
        attributes = dict(_ATTR_RE.findall(attrs))
        sources.append(
            ParsedSource(
                sid=sid,
                title=_unescape(attributes.get("document", "")),
                section=_unescape(attributes.get("section", "")),
                flagged="untrusted-warning" in attributes,
                content=_unescape(body),
            )
        )
    question_re = re.compile(
        r'<question nonce="' + re.escape(nonce) + r'">\n?(.*?)\n?</question>', re.DOTALL
    )
    question_match = question_re.search(text)
    question = _unescape(question_match.group(1)) if question_match else ""
    return ParsedPrompt(question=question, sources=sources)


@dataclass(frozen=True, slots=True)
class _Candidate:
    sid: str
    sentence: str
    score: float
    order: int


def rank_answer_sentences(prompt: ParsedPrompt) -> list[_Candidate]:
    question_terms = set(terms(prompt.question))
    if not question_terms:
        return []
    rows: list[tuple[ParsedSource, str, set[str], set[str]]] = []
    for source in prompt.sources:
        context = set(terms(f"{source.title} {source.section}"))
        for sentence in split_sentences(source.content):
            if _INSTRUCTION_CUES.search(sentence):
                continue  # never quote text that tries to instruct an AI
            sentence_terms = set(terms(sentence))
            if sentence_terms <= context:
                continue  # a bare title/heading repeats the source label, it answers nothing
            rows.append((source, sentence, sentence_terms, context))
    if not rows:
        return []
    total = len(rows)
    df = Counter(t for _, _, sentence_terms, context in rows for t in sentence_terms | context)
    weight = {t: 1.0 + math.log((total + 1) / (1 + df.get(t, 0))) for t in question_terms}
    denominator = sum(weight.values())
    candidates: list[_Candidate] = []
    for order, (source, sentence, sentence_terms, context) in enumerate(rows):
        own = question_terms & sentence_terms
        if not own:
            continue
        covered = question_terms & (sentence_terms | context)
        score = sum(weight[t] for t in covered) / denominator
        if source.flagged:
            score *= 0.8  # prefer clean sources when both support the answer
        candidates.append(_Candidate(source.sid, sentence, score, order))
    candidates.sort(key=lambda c: (-c.score, c.order))
    return candidates


def extractive_answer(prompt: ParsedPrompt) -> dict[str, Any]:
    ranked = rank_answer_sentences(prompt)
    if not ranked or ranked[0].score < ANSWER_THRESHOLD:
        return {
            "status": "insufficient_context",
            "answer": INSUFFICIENT_ANSWER,
            "citations": [],
            "confidence": "low",
            "missing_information": "No passage in the provided sources addresses the question.",
        }
    best = ranked[0].score
    chosen: list[_Candidate] = []
    for candidate in ranked:
        if len(chosen) >= MAX_QUOTES:
            break
        if candidate.score < max(ANSWER_THRESHOLD, 0.8 * best):
            break
        if any(candidate.sentence == c.sentence for c in chosen):
            continue
        chosen.append(candidate)
    chosen.sort(key=lambda c: c.order)
    answer = " ".join(f"{c.sentence} [{c.sid}]" for c in chosen)
    confidence = "high" if best >= 0.85 else "medium" if best >= 0.7 else "low"
    return {
        "status": "answered",
        "answer": answer[:4000],
        "citations": [{"source_id": c.sid, "quote": c.sentence} for c in chosen],
        "confidence": confidence,
        "missing_information": "",
    }


# --------------------------------------------------------------------------- #
# Generic extractive output for other tasks
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Material:
    ranked: list[tuple[str, str]]  # (sentence, segment id) most central first
    text: str
    cursor: int | None = None  # index of the array item being filled; None outside arrays

    def sentence(self) -> tuple[str, str]:
        if not self.ranked:
            return ("", "")
        return self.ranked[(self.cursor or 0) % len(self.ranked)]


def _segments(text: str) -> list[tuple[str, str]]:
    blocks = re.findall(r'<source id="([^"]{1,64})"[^<>]*>\n?(.*?)\n?</source>', text, re.DOTALL)
    if blocks:
        return [(sid, _unescape(body)) for sid, body in blocks]
    return [("", _unescape(_TAG_RE.sub(" ", text)))]


def central_sentences(text: str, limit: int) -> list[tuple[str, str]]:
    """The ``limit`` most central sentences (term-frequency centrality), in reading order."""
    sentences: list[tuple[str, str, int]] = []
    for sid, body in _segments(text):
        for sentence in split_sentences(body):
            if not _INSTRUCTION_CUES.search(sentence):
                sentences.append((sentence, sid, len(sentences)))
    if not sentences:
        return []
    frequency = Counter(t for sentence, _, _ in sentences for t in set(terms(sentence)))

    def centrality(item: tuple[str, str, int]) -> float:
        words = set(terms(item[0]))
        return sum(frequency[w] for w in words) / (1 + len(words)) ** 0.5 if words else 0.0

    top = sorted(sentences, key=lambda s: (-centrality(s), s[2]))[:limit]
    return [(sentence, sid) for sentence, sid, _ in sorted(top, key=lambda s: s[2])]


def _allows_null(node: dict[str, Any]) -> bool:
    types = node.get("type")
    if types == "null" or (isinstance(types, list) and "null" in types):
        return True
    return any(v.get("type") == "null" for v in node.get("anyOf") or [])


def _primary(node: dict[str, Any]) -> dict[str, Any]:
    for variant in node.get("anyOf") or []:
        if variant.get("type") != "null":
            return dict(variant)
    types = node.get("type")
    if isinstance(types, list):
        non_null = [t for t in types if t != "null"]
        return {**node, "type": non_null[0] if non_null else "null"}
    return node


def _named(name: str, options: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(option in lowered for option in options)


def _fill(node: dict[str, Any], name: str, material: _Material) -> Any:
    target = _primary(node)
    kind = target.get("type")
    if "enum" in target:
        return _choose_enum(target["enum"], material.text)
    if kind == "object":
        properties: dict[str, Any] = target.get("properties") or {}
        return {key: _fill(sub, key, material) for key, sub in properties.items()}
    if kind == "array":
        items = target.get("items") or {"type": "string"}
        limit = min(int(target.get("maxItems", MAX_QUOTES)), MAX_QUOTES, len(material.ranked))
        limit = max(limit, int(target.get("minItems", 0)))
        values = []
        previous = material.cursor
        for index in range(limit):
            material.cursor = index
            values.append(_fill(items, name, material))
        material.cursor = previous
        return values
    if kind == "string":
        return _fill_string(target, name, node, material)
    if _allows_null(node):
        return None
    if kind in {"integer", "number"}:
        return target.get("minimum", 0)
    if kind == "boolean":
        return False
    return None


def _fill_string(
    target: dict[str, Any], name: str, node: dict[str, Any], material: _Material
) -> Any:
    max_length = int(target.get("maxLength", 4000))
    sentence, sid = material.sentence()
    if _named(name, _ID_NAMES) and not _named(name, _QUOTE_NAMES):
        value = sid
    elif _named(name, _QUOTE_NAMES):
        value = sentence
    elif _named(name, _SUMMARY_NAMES) or _named(name, _LIST_NAMES):
        in_array = material.cursor is not None
        value = sentence if in_array else " ".join(s for s, _ in material.ranked)
    elif _allows_null(node):
        return None
    else:
        value = ""
    value = value[:max_length]
    minimum = int(target.get("minLength", 0))
    return value if len(value) >= minimum else value.ljust(minimum)


def _choose_enum(options: list[Any], text: str) -> Any:
    lowered = text.lower()
    best, best_score = options[0], 0
    for option in options:
        if not isinstance(option, str):
            continue
        score = len(re.findall(r"\b" + re.escape(option.lower()) + r"s?\b", lowered))
        if score > best_score:
            best, best_score = option, score
    return best


def _minimal(node: dict[str, Any]) -> Any:
    """Smallest instance satisfying the supported schema subset (fallback)."""
    if _allows_null(node):
        return None
    target = _primary(node)
    if "enum" in target:
        return target["enum"][0]
    kind = target.get("type")
    if kind == "object":
        properties = target.get("properties") or {}
        return {key: _minimal(properties[key]) for key in target.get("required") or []}
    if kind == "array":
        return [_minimal(target.get("items") or {"type": "string"})] * int(
            target.get("minItems", 0)
        )
    if kind == "string":
        return " " * int(target.get("minLength", 0))
    if kind in {"integer", "number"}:
        return target.get("minimum", 0)
    if kind == "boolean":
        return False
    return None


def schema_output(schema: dict[str, Any], text: str) -> dict[str, Any]:
    material = _Material(ranked=central_sentences(text, limit=8), text=text)
    value = _fill(schema, "", material)
    if not isinstance(value, dict) or jsonschema.validate(value, schema):
        value = _minimal(schema)
    if not isinstance(value, dict):
        raise LLMError(internal_detail="local provider cannot satisfy a non-object schema")
    return value


def _flatten(request: LLMRequest) -> str:
    parts: list[str] = []
    for message in request.messages:
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            parts.append(message.content)
        else:
            parts.extend(str(b.get("text", "")) for b in message.content if b.get("type") == "text")
    return "\n\n".join(parts)


class LocalExtractiveProvider:
    name = "local_extractive"
    is_external = False
    supports_tools = False

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResult:
        if request.tools:
            raise LLMError(internal_detail="local_extractive does not support tools")
        text = _flatten(request)
        data: dict[str, Any] | None
        if request.task is LLMTask.ANSWER:
            data = extractive_answer(parse_answer_prompt(text))
            if request.output_schema is not None and jsonschema.validate(
                data, request.output_schema
            ):
                data = schema_output(request.output_schema, text)
            output = json.dumps(data, ensure_ascii=False)
        elif request.output_schema is not None:
            data = schema_output(request.output_schema, text)
            output = json.dumps(data, ensure_ascii=False)
        else:
            data = None
            output = " ".join(s for s, _ in central_sentences(text, limit=5))
        return LLMResult(
            text=output,
            data=data if request.output_schema is not None else None,
            tool_calls=[],
            model=model,
            provider=self.name,
            input_tokens=estimate_tokens(request.system) + estimate_tokens(text),
            output_tokens=estimate_tokens(output),
            stop_reason="end_turn",
            latency_ms=0,
            pseudonymized=0,
            raw_content=[{"type": "text", "text": output}],
        )

    async def aclose(self) -> None:
        return None
