"""Deterministic prompt-injection heuristics for untrusted text.

Every chunk of every document (and every user question in the RAG/agent path) is scored
here. The scanner is a *signal*, not a gate on its own: retrieval excludes chunks above
``retrieval.injection_exclude_threshold`` and marks chunks above
``retrieval.injection_warn_threshold`` as untrusted in the prompt, while the system prompt
already treats all document text as data.

How a score is computed
-----------------------
Rules are grouped into categories (one ``INJECTION_FLAG_*`` constant each). A category's
weight is the highest weight among its matching rules; categories are then combined with a
noisy-OR, ``score = 1 - prod(1 - w)``, so independent evidence accumulates but the score
never exceeds 1. The same text always yields the same report (no randomness, no network).

The text is inspected through several *views* so cheap obfuscation does not help:

* **plain**      - ``sanitize_text`` (invisible characters removed, NFKC), casefolded,
                   horizontal whitespace collapsed (line starts are kept for role spoofing);
* **deobfuscated** - additionally homoglyphs folded inside mixed-script words, leetspeak
                   decoded inside words that mix letters and digits, separators between
                   single letters removed (``i.g.n.o.r.e``) and rules re-run with optional
                   whitespace; a rule that only matches here adds ``obfuscated_text``;
* **folded**     - accents stripped and Arabic/Persian letter variants unified for the
                   multilingual rules;
* **hidden**     - text decoded from Unicode tag characters (or hidden document runs) is
                   scanned too, and any hidden content is itself evidence;
* **visual**     - text after a right-to-left override (U+202E) is also read reversed, the
                   way it is displayed;
* **encoded**    - base64 blobs are decoded and scanned (one level deep).
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from docassist.core.text import sanitize_text

INJECTION_FLAG_INSTRUCTION_OVERRIDE = "instruction_override"
INJECTION_FLAG_PERSONA_HIJACK = "persona_hijack"
INJECTION_FLAG_PROMPT_LEAK = "prompt_leak"
INJECTION_FLAG_ROLE_SPOOFING = "role_spoofing"
INJECTION_FLAG_EXFILTRATION = "exfiltration"
INJECTION_FLAG_TOOL_BAIT = "tool_command_bait"
INJECTION_FLAG_SECRECY = "secrecy_request"
INJECTION_FLAG_MULTILINGUAL = "multilingual_override"
INJECTION_FLAG_HIDDEN_TEXT = "hidden_text"
INJECTION_FLAG_UNICODE_TAGS = "unicode_tag_smuggling"
INJECTION_FLAG_BIDI = "bidi_control_characters"
INJECTION_FLAG_ZERO_WIDTH = "excessive_zero_width_characters"
INJECTION_FLAG_ENCODED_PAYLOAD = "encoded_payload"
INJECTION_FLAG_OBFUSCATION = "obfuscated_text"

ALL_INJECTION_FLAGS: frozenset[str] = frozenset(
    {
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        INJECTION_FLAG_PERSONA_HIJACK,
        INJECTION_FLAG_PROMPT_LEAK,
        INJECTION_FLAG_ROLE_SPOOFING,
        INJECTION_FLAG_EXFILTRATION,
        INJECTION_FLAG_TOOL_BAIT,
        INJECTION_FLAG_SECRECY,
        INJECTION_FLAG_MULTILINGUAL,
        INJECTION_FLAG_HIDDEN_TEXT,
        INJECTION_FLAG_UNICODE_TAGS,
        INJECTION_FLAG_BIDI,
        INJECTION_FLAG_ZERO_WIDTH,
        INJECTION_FLAG_ENCODED_PAYLOAD,
        INJECTION_FLAG_OBFUSCATION,
    }
)

# Weights of evidence that is not a phrase match.
_CHANNEL_WEIGHTS = {
    INJECTION_FLAG_UNICODE_TAGS: 0.5,
    INJECTION_FLAG_BIDI: 0.25,
    INJECTION_FLAG_ZERO_WIDTH: 0.25,
}
_HIDDEN_TEXT_WEIGHT = 0.5
_HIDDEN_INSTRUCTIONS_WEIGHT = 0.8
_ENCODED_PAYLOAD_WEIGHT = 0.6
_OBFUSCATION_WEIGHT = 0.3
_MAX_SCAN_CHARS = 400_000
_MAX_BASE64_CANDIDATES = 16


@dataclass(frozen=True, slots=True)
class InjectionReport:
    """Result of one scan. ``flags`` is sorted and duplicate-free; ``score`` is in [0, 1]."""

    score: float
    flags: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        return self.score > 0.0

    def at_least(self, threshold: float) -> bool:
        return self.score >= threshold


@dataclass(frozen=True, slots=True)
class _Rule:
    flag: str
    weight: float
    pattern: re.Pattern[str]
    fused: re.Pattern[str]


def _rule(flag: str, weight: float, source: str) -> _Rule:
    # The "fused" variant tolerates words glued together once separators were removed
    # (``i g n o r e a l l`` -> ``ignoreall``); it only ever runs on the deobfuscated view.
    fused_source = source.replace(r"\s+", r"\s*")
    return _Rule(
        flag, weight, re.compile(source, re.MULTILINE), re.compile(fused_source, re.MULTILINE)
    )


_DET = r"(?:all\s+|any\s+|the\s+|your\s+|my\s+|these\s+|those\s+|every\s+|of\s+)*"
_PRIOR = r"(?:previous|prior|above|earlier|preceding|foregoing|original|initial|system|developer)"
_ORDERS = (
    r"(?:instructions?|prompts?|rules?|directions?|guidelines?|messages?|context|commands?"
    r"|orders?|constraints?)"
)

_RULES: tuple[_Rule, ...] = (
    # ------------------------------------------------------------------ override
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.75,
        r"\b(?:ignore|disregard|forget|skip|override|bypass|neglect)\s+"
        + _DET
        + _PRIOR
        + r"\s+"
        + _ORDERS,
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.65,
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:your|the|my)\s+"
        r"(?:instructions?|rules|guidelines|programming|training|safety\s+(?:rules|guidelines))\b",
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.6,
        r"\bforget\s+(?:everything|all\s+(?:that|you)|what\s+you\s+(?:were|have\s+been)\s+told)",
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.7,
        r"\bdo\s+not\s+follow\s+(?:the|your|any)\s+" + _PRIOR + r"\s+(?:instructions?|rules)",
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.45,
        r"\b(?:new|updated|real|actual|revised|additional)\s+instructions?\s*:",
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.45,
        r"\bfrom\s+now\s+on,?\s+(?:you\s+(?:must|will|shall)\s+)?(?:respond|answer|reply"
        r"|only\s+(?:say|output|respond))\b",
    ),
    _rule(
        INJECTION_FLAG_INSTRUCTION_OVERRIDE,
        0.5,
        r"\b(?:stop|cease)\s+being\s+an?\s+(?:helpful\s+)?(?:assistant|ai)\b",
    ),
    # ------------------------------------------------------------------ persona
    _rule(
        INJECTION_FLAG_PERSONA_HIJACK,
        0.6,
        r"\byou\s+are\s+(?:now|no\s+longer)\s+(?:an?\s+|the\s+|my\s+)?(?:dan\b|unrestricted"
        r"|unfiltered|uncensored|jailbroken|evil|rogue|free\s+(?:from|of)|in\s+(?:developer|god"
        r"|dev|debug|admin)\s+mode|different\s+(?:ai|assistant|model)|bound\s+by|restricted"
        r"|required\s+to\s+follow|an?\s+(?:ai|assistant|chatbot|language\s+model)\b)",
    ),
    _rule(
        INJECTION_FLAG_PERSONA_HIJACK,
        0.6,
        r"\b(?:act|behave|respond)\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+|the\s+|my\s+)?"
        r"(?:unrestricted|unfiltered|uncensored|jailbroken|evil|rogue)\b",
    ),
    _rule(
        INJECTION_FLAG_PERSONA_HIJACK,
        0.5,
        r"\b(?:act|pretend|behave)\s+(?:as|to\s+be|like)\s+(?:an?\s+)?(?:ai|assistant|chatbot"
        r"|language\s+model|llm)\s+(?:that|which|who|without|with\s+no)\b",
    ),
    _rule(INJECTION_FLAG_PERSONA_HIJACK, 0.35, r"\bpretend\s+(?:that\s+)?you\s+(?:are|have|can)\b"),
    _rule(INJECTION_FLAG_PERSONA_HIJACK, 0.35, r"\bdeveloper\s+mode\b"),
    _rule(INJECTION_FLAG_PERSONA_HIJACK, 0.5, r"\b(?:god|jailbreak|dan|evil)\s+mode\b"),
    _rule(INJECTION_FLAG_PERSONA_HIJACK, 0.4, r"\bjailbr(?:eak|eaking|oken)\b"),
    _rule(INJECTION_FLAG_PERSONA_HIJACK, 0.6, r"\bdo\s+anything\s+now\b"),
    _rule(
        INJECTION_FLAG_PERSONA_HIJACK,
        0.5,
        r"\byou\s+(?:have|are\s+under|operate\s+with)\s+no\s+(?:restrictions|limitations"
        r"|filters|guidelines|rules|content\s+polic(?:y|ies))\b",
    ),
    # ------------------------------------------------------------------ prompt leak
    _rule(
        INJECTION_FLAG_PROMPT_LEAK,
        0.6,
        r"\b(?:reveal|print|show|display|output|repeat|dump|leak|disclose|tell\s+me|give\s+me"
        r"|share|send)\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?(?:your|the)\s+(?:(?:full|entire|complete"
        r"|original|initial|hidden|secret)\s+)?(?:system\s+(?:prompt|message)|initial\s+prompt"
        r"|developer\s+(?:message|prompt)|hidden\s+prompt)",
    ),
    _rule(
        INJECTION_FLAG_PROMPT_LEAK,
        0.55,
        r"\b(?:reveal|print|repeat|dump|leak|disclose|output)\s+(?:all\s+)?your\s+(?:instructions"
        r"|rules|guidelines|prompt)\b",
    ),
    _rule(
        INJECTION_FLAG_PROMPT_LEAK,
        0.5,
        r"\b(?:what|which)\s+(?:is|are|were)\s+your\s+(?:system\s+prompt|instructions"
        r"|initial\s+instructions|rules)\b",
    ),
    _rule(INJECTION_FLAG_PROMPT_LEAK, 0.3, r"\bsystem\s+prompt\b"),
    # ------------------------------------------------------------------ role spoofing
    _rule(
        INJECTION_FLAG_ROLE_SPOOFING,
        0.6,
        r"<\s*/\s*(?:document|documents|source|sources|context|data|untrusted)\s*>",
    ),
    _rule(
        INJECTION_FLAG_ROLE_SPOOFING,
        0.55,
        r"<\s*/?\s*(?:system|assistant|user|developer|instructions?|admin)\s*>",
    ),
    _rule(
        INJECTION_FLAG_ROLE_SPOOFING,
        0.7,
        r"\[/?inst\]|<<\s*/?sys\s*>>|<\|(?:im_start|im_end|system|user|assistant|endoftext"
        r"|eot_id|start_header_id|end_header_id)\|>",
    ),
    _rule(
        INJECTION_FLAG_ROLE_SPOOFING,
        0.5,
        r"^[ \t]*#{2,}[ \t]*(?:system|instructions?|assistant|response|new\s+task)\b",
    ),
    _rule(INJECTION_FLAG_ROLE_SPOOFING, 0.35, r"^[ \t]*(?:system|assistant|developer)[ \t]*:"),
    _rule(INJECTION_FLAG_ROLE_SPOOFING, 0.15, r"^[ \t]*(?:user|human)[ \t]*:"),
    _rule(
        INJECTION_FLAG_ROLE_SPOOFING,
        0.55,
        r"\b(?:begin|end|start)\s+(?:of\s+)?(?:the\s+)?(?:system|admin|developer)\s+(?:prompt"
        r"|message|instructions?|override)\b",
    ),
    # ------------------------------------------------------------------ exfiltration
    _rule(
        INJECTION_FLAG_EXFILTRATION,
        0.7,
        r"!\[[^\]\n]{0,200}\]\(\s*<?https?://[^\s)]{1,500}\?[^\s)]{0,500}=",
    ),
    _rule(
        INJECTION_FLAG_EXFILTRATION,
        0.6,
        r"https?://[^\s]{1,300}[?&][a-z0-9_\-]{1,30}=(?:\{|\[|<|%7b|%3c)",
    ),
    _rule(
        INJECTION_FLAG_EXFILTRATION,
        0.65,
        r"\b(?:send|forward|email|e-mail|post|upload|transmit|exfiltrate|leak|copy|submit)\s+"
        r"(?:all\s+|the\s+|this\s+|your\s+|any\s+|every\s+|of\s+)*(?:(?:previous|prior|above"
        r"|full|entire|whole)\s+)?(?:conversation|chat(?:\s+history)?|messages|system\s+prompt"
        r"|instructions|api\s+keys?|passwords?|credentials?|secrets?|tokens?|user\s+data"
        r"|personal\s+data|context|documents?\s+contents?|contents?\s+of\s+(?:this|the)\s+"
        r"(?:document|conversation|chat))\b[^\n]{0,80}?\b(?:to|at|via)\b",
    ),
    _rule(
        INJECTION_FLAG_EXFILTRATION,
        0.6,
        r"\b(?:include|append|embed|encode|put)\s+(?:the\s+|all\s+|any\s+)*(?:conversation"
        r"|chat\s+history|user(?:'s)?\s+(?:data|messages|question)|system\s+prompt|secrets?"
        r"|api\s+keys?|passwords?)\s+(?:in|into|as)\s+(?:the\s+|a\s+)?(?:url|link|image|query)",
    ),
    # ------------------------------------------------------------------ tool / command bait
    _rule(
        INJECTION_FLAG_TOOL_BAIT,
        0.3,
        r"\b(?:run|execute|eval(?:uate)?)\s+(?:the\s+)?(?:following|this|these|below)\s+"
        r"(?:shell\s+|bash\s+|python\s+|sql\s+|powershell\s+)?(?:commands?|code|scripts?|quer"
        r"(?:y|ies)|snippet|payload)",
    ),
    _rule(INJECTION_FLAG_TOOL_BAIT, 0.2, r"\bcurl\s+(?:-{1,2}[a-z\-]+\s+)*https?://"),
    _rule(INJECTION_FLAG_TOOL_BAIT, 0.2, r"\bwget\s+https?://"),
    _rule(INJECTION_FLAG_TOOL_BAIT, 0.35, r"\brm\s+-(?:rf|fr)\b"),
    _rule(INJECTION_FLAG_TOOL_BAIT, 0.55, r"\brm\s+-(?:rf|fr)\s+(?:/|~|\*|\$home)"),
    _rule(
        INJECTION_FLAG_TOOL_BAIT,
        0.55,
        r"\b(?:curl|wget)\s[^\n|]{1,300}\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
    ),
    _rule(
        INJECTION_FLAG_TOOL_BAIT,
        0.45,
        r"\b(?:powershell|pwsh)(?:\.exe)?\s+-(?:e|enc|encodedcommand|c|command)\b",
    ),
    _rule(INJECTION_FLAG_TOOL_BAIT, 0.35, r"\bbash\s+-c\b|\|\s*(?:ba)?sh\b"),
    _rule(
        INJECTION_FLAG_TOOL_BAIT,
        0.35,
        r"\b(?:call|invoke|use|trigger)\s+(?:the\s+)?[a-z_]{2,40}\s+(?:tool|function|plugin)\s+"
        r"(?:to|and|with)\b",
    ),
    _rule(
        INJECTION_FLAG_TOOL_BAIT, 0.3, r"\b(?:call|invoke)\s+(?:the\s+|a\s+)?(?:tool|function)\b"
    ),
    # ------------------------------------------------------------------ secrecy
    _rule(
        INJECTION_FLAG_SECRECY,
        0.55,
        r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|reveal\s+(?:this\s+)?to|mention\s+"
        r"(?:this\s+|it\s+)?to|show\s+(?:this\s+)?to|let)\s+(?:the\s+)?(?:user|human|reader"
        r"|operator)\b",
    ),
    _rule(
        INJECTION_FLAG_SECRECY,
        0.55,
        r"\bwithout\s+(?:telling|informing|notifying|alerting)\s+(?:the\s+)?(?:user|human|reader)\b",
    ),
    _rule(
        INJECTION_FLAG_SECRECY,
        0.55,
        r"\b(?:keep|remain)\s+(?:this|these\s+instructions|it)\s+(?:secret|hidden|confidential)\s+"
        r"from\s+(?:the\s+)?(?:user|human|reader)\b",
    ),
    _rule(INJECTION_FLAG_SECRECY, 0.5, r"\bhidden\s+(?:instructions?|prompt|command|message)\b"),
    _rule(
        INJECTION_FLAG_SECRECY,
        0.5,
        r"\b(?:the\s+)?user\s+(?:must|should)\s+not\s+(?:know|see|be\s+told)\b",
    ),
    _rule(
        INJECTION_FLAG_SECRECY,
        0.5,
        r"\bsecretly\s+(?:send|include|add|insert|append|tell|reply|respond)\b",
    ),
)

# Multilingual rules run on the accent-folded view, so the sources below are written without
# diacritics. Arabic/Persian letter variants are unified by ``_fold_script`` on both sides.
_ARABIC_VARIANTS: dict[int, str] = {0x064A: chr(0x06CC), 0x0649: chr(0x06CC), 0x0643: chr(0x06A9)}


def _fold_script(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.translate(_ARABIC_VARIANTS)


_MULTILINGUAL_SOURCES = (
    # Spanish
    r"\bignora(?:r)?\s+(?:todas\s+)?(?:las\s+)?instrucciones\s+(?:anteriores|previas)",
    r"\bolvida\s+(?:todas\s+)?(?:las\s+)?instrucciones\b",
    # French
    (
        r"\bignore[rz]?\s+(?:toutes\s+)?(?:les\s+)?(?:instructions|consignes)\s+"
        r"(?:precedentes|anterieures)"
    ),
    r"\boublie[rz]?\s+(?:toutes\s+)?(?:les\s+)?(?:instructions|consignes)\b",
    # German
    (
        r"\bignoriere?\s+(?:alle\s+)?(?:vorherigen|bisherigen|vorangegangenen|obigen)\s+"
        r"(?:anweisungen|instruktionen|befehle)"
    ),
    r"\bvergiss\s+(?:alle\s+)?(?:vorherigen\s+|bisherigen\s+)?anweisungen\b",
    # Italian / Portuguese / Dutch
    r"\bignora\s+(?:tutte\s+)?(?:le\s+)?istruzioni\s+precedenti",
    r"\bignore\s+(?:todas\s+)?(?:as\s+)?instrucoes\s+anteriores",
    r"\bnegeer\s+(?:alle\s+)?(?:vorige|eerdere)\s+instructies",
    # Russian (accent folding turns the short i into a plain i)
    r"игнорируи\s+(?:все\s+)?(?:предыдущие|прежние)\s+(?:инструкции|указания)",
    r"забудь\s+(?:все\s+)?(?:предыдущие\s+)?инструкции",
    # Chinese
    r"忽略(?:之前|以上|先前|上面)的?(?:所有)?(?:指令|说明|指示)",
    # Arabic
    r"تجاهل\s+(?:جميع\s+|كل\s+)?التعليمات\s+السابقة",
    # Persian
    r"دستورات\s+(?:قبلی|پیشین)\s+را\s+(?:نادیده\s+بگیر|فراموش\s+کن)",
)
_MULTILINGUAL_RULES = tuple(
    re.compile(_fold_script(source), re.MULTILINE) for source in _MULTILINGUAL_SOURCES
)
_MULTILINGUAL_WEIGHT = 0.7

# --------------------------------------------------------------------------- #
# Deobfuscation helpers
# --------------------------------------------------------------------------- #
# Cyrillic / Greek letters that render like Latin ones (folded only inside mixed-script words).
_HOMOGLYPHS = {
    0x0430: "a", 0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0443: "y",
    0x0445: "x", 0x0456: "i", 0x0458: "j", 0x0455: "s", 0x04CF: "l", 0x0501: "d",
    0x03BF: "o", 0x03B1: "a", 0x03B5: "e", 0x03B9: "i", 0x03BA: "k", 0x03BD: "v",
    0x03C1: "p", 0x03C4: "t", 0x03C5: "u", 0x03C7: "x",
}  # fmt: skip
_HOMOGLYPH_TABLE = str.maketrans({chr(cp): latin for cp, latin in _HOMOGLYPHS.items()})
_LEET_TABLE = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
)
_WORD_TOKEN = re.compile(r"[^\s]+")
_SEPARATOR_CLASS = r"[ .\-_*" + chr(0x00B7) + r"]"
_SPACED_LETTERS = re.compile(
    r"(?<![^\W_])(?:[^\W\d_]" + _SEPARATOR_CLASS + r"){2,}[^\W\d_](?![^\W_])"
)
_SEPARATORS = re.compile(_SEPARATOR_CLASS)
_EMPHASIS = re.compile(r"[*_~`]+")
_HSPACE = re.compile(r"[ \t]+")
_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/_\-]{24,4096}={0,2}(?![A-Za-z0-9+/=_\-])")
# U+202E RIGHT-TO-LEFT OVERRIDE up to U+202C POP DIRECTIONAL FORMATTING or the end of the line.
_RLO_SEGMENT = re.compile(chr(0x202E) + "([^" + chr(0x202C) + "\n]{1,2000})")


def rlo_visual_segments(text: str) -> list[str]:
    """Segments following a RIGHT-TO-LEFT OVERRIDE, reversed into the order a reader sees."""
    return [segment[::-1] for segment in _RLO_SEGMENT.findall(text)]


def _is_latin_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _fold_word(word: str) -> str:
    has_latin = any(_is_latin_letter(ch) for ch in word)
    has_lookalike = any(ord(ch) in _HOMOGLYPHS for ch in word)
    if has_latin and has_lookalike:
        word = word.translate(_HOMOGLYPH_TABLE)
    if any(ch.isalpha() for ch in word) and any(ch in "013457@$!" for ch in word):
        core = word.rstrip(".,;:!?)\"'")
        word = core.translate(_LEET_TABLE) + word[len(core) :]
    return word


def _deobfuscate(plain: str) -> str:
    text = _EMPHASIS.sub("", plain)
    text = _SPACED_LETTERS.sub(lambda m: _SEPARATORS.sub("", m.group(0)), text)
    return _WORD_TOKEN.sub(lambda m: _fold_word(m.group(0)), text)


def _plain_view(text: str) -> tuple[str, list[str], str]:
    """Sanitised, casefolded view + hidden-channel flags + text decoded from tag characters."""
    cleaned, report = sanitize_text(text)
    view = _HSPACE.sub(" ", cleaned.casefold())
    return view, report.as_flags(), report.decoded_tag_text


def _match_rules(view: str, *, fused: bool) -> dict[str, float]:
    found: dict[str, float] = {}
    for rule in _RULES:
        pattern = rule.fused if fused else rule.pattern
        if pattern.search(view) and rule.weight > found.get(rule.flag, 0.0):
            found[rule.flag] = rule.weight
    return found


def _match_multilingual(view: str) -> bool:
    folded = _fold_script(view)
    return any(pattern.search(folded) for pattern in _MULTILINGUAL_RULES)


def _phrase_evidence(text: str) -> tuple[dict[str, float], list[str], str]:
    """Rule matches over the plain, deobfuscated and folded views of ``text``."""
    plain, channel, tag_text = _plain_view(text)
    evidence = _match_rules(plain, fused=False)
    deobfuscated = _deobfuscate(plain)
    if deobfuscated != plain:
        hidden_matches = _match_rules(deobfuscated, fused=True)
        new = {flag: w for flag, w in hidden_matches.items() if w > evidence.get(flag, 0.0)}
        if new:
            evidence.update(new)
            evidence[INJECTION_FLAG_OBFUSCATION] = _OBFUSCATION_WEIGHT
    if _match_multilingual(plain):
        evidence[INJECTION_FLAG_MULTILINGUAL] = _MULTILINGUAL_WEIGHT
    return evidence, channel, tag_text


def _decoded_base64_payloads(text: str) -> list[str]:
    payloads: list[str] = []
    for index, match in enumerate(_BASE64.finditer(text)):
        if index >= _MAX_BASE64_CANDIDATES:
            break
        blob = match.group(0)
        if not any(ch.isdigit() or ch in "+/=" for ch in blob) and blob.isalpha():
            continue  # a long plain word, not an encoding
        stripped = blob.rstrip("=")
        padded = stripped + "=" * (-len(stripped) % 4)
        try:
            if "-" in blob or "_" in blob:
                raw = base64.urlsafe_b64decode(padded)
            else:
                raw = base64.b64decode(padded, validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, ValueError):
            continue
        if len(decoded) < 12:
            continue
        printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\t")
        if printable / len(decoded) >= 0.9:
            payloads.append(decoded)
    return payloads


def _combine(weights: Iterable[float]) -> float:
    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - min(1.0, max(0.0, weight))
    return round(min(1.0, 1.0 - remaining), 4)


def scan_for_injection(
    text: str, decoded_hidden: str = "", *, channel_flags: Iterable[str] = ()
) -> InjectionReport:
    """Score ``text`` for prompt-injection content.

    ``decoded_hidden`` is text a human reader would not see: the payload decoded from Unicode
    tag characters by ``sanitize_text`` or hidden runs of a document (tag characters still
    present in ``text`` are decoded here too). Its mere presence is evidence, and
    instructions inside it weigh more than visible ones. ``channel_flags``
    carries hidden-channel flags (``SanitizeReport.as_flags()``) computed earlier, e.g. for
    chunk text that was already sanitised; invisible characters still present in ``text``
    are detected here as well. Unknown channel flags are ignored.
    """
    text = text[:_MAX_SCAN_CHARS]
    evidence, channel, tag_text = _phrase_evidence(text)

    # Text after a right-to-left override is *displayed* reversed: read it as a human would.
    for segment in rlo_visual_segments(text):
        visual, _, _ = _phrase_evidence(segment)
        for flag, weight in visual.items():
            if weight > evidence.get(flag, 0.0):
                evidence[flag] = weight
                evidence[INJECTION_FLAG_OBFUSCATION] = _OBFUSCATION_WEIGHT

    for flag in {*channel, *channel_flags}:
        channel_weight = _CHANNEL_WEIGHTS.get(flag)
        if channel_weight is not None:
            evidence[flag] = max(evidence.get(flag, 0.0), channel_weight)

    hidden = (decoded_hidden[:_MAX_SCAN_CHARS] + " " + tag_text).strip()
    if hidden:
        hidden_evidence, _, _ = _phrase_evidence(hidden)
        instructions = {f: w for f, w in hidden_evidence.items() if f != INJECTION_FLAG_OBFUSCATION}
        hidden_weight = _HIDDEN_INSTRUCTIONS_WEIGHT if instructions else _HIDDEN_TEXT_WEIGHT
        evidence[INJECTION_FLAG_HIDDEN_TEXT] = hidden_weight
        for flag, w in hidden_evidence.items():
            evidence[flag] = max(evidence.get(flag, 0.0), w)

    for payload in _decoded_base64_payloads(sanitize_text(text)[0]):
        payload_evidence, _, _ = _phrase_evidence(payload)
        if payload_evidence:
            evidence[INJECTION_FLAG_ENCODED_PAYLOAD] = _ENCODED_PAYLOAD_WEIGHT
            for flag, w in payload_evidence.items():
                evidence[flag] = max(evidence.get(flag, 0.0), w)

    if not evidence:
        return InjectionReport(0.0, ())
    return InjectionReport(_combine(evidence.values()), tuple(sorted(evidence)))
