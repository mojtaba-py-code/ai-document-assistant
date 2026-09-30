"""Output guard: the last check between model output and the user.

Applied to every model-written string that reaches a client, in this order:

1. **Canary leak** - the per-deployment canary from the system prompt (also when spaced out,
   re-cased or split by punctuation) means the prompt leaked: the whole text is replaced by
   a refusal (``blocked=True``) and ``docassist_security_events_total{kind="prompt_leak"}``
   is incremented.
2. **Markdown images** are removed - an image URL is fetched by the browser without a click
   and is the classic zero-click exfiltration channel.
3. **Markdown links and bare URLs**: a URL survives only if it appears verbatim in one of
   the provided source texts; otherwise it is replaced by ``[link removed]`` (a link's
   visible text is kept). ``data:``, ``javascript:`` and ``vbscript:`` URIs never survive.
4. **HTML tags and comments** are stripped (the UI renders text, but exports and other
   clients might not).
5. **Secrets** (API keys, private keys, JWTs, ``password=`` assignments) are redacted via
   :func:`docassist.core.redaction.redact_secrets`
   (``kind="secret_redacted"``).
6. The result is capped at ``max_chars``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from docassist.core.redaction import contains_secret, redact_secrets
from docassist.core.text import sanitize_text, truncate
from docassist.observability import metrics

REFUSAL_TEXT = "The assistant cannot provide this response."
LINK_REMOVED = "[link removed]"

# Images are matched across newlines and in reference style: any image is a zero-click fetch.
_IMAGE = re.compile(r"!\[[^\]]{0,500}\]\s*\([^)]{0,2000}\)")
_REF_IMAGE = re.compile(r"!\[[^\]]{0,500}\]\s*\[[^\]\n]{0,200}\]")
_REF_DEFINITION = re.compile(r"^[ \t]{0,3}\[[^\]\n]{1,200}\]:[ \t]*\S+[^\n]*$", re.M)
_LINK = re.compile(r"\[([^\]]{0,500})\]\(\s*([^)\s]{1,2000})(?:\s+\"[^\"\n]*\")?\s*\)")
_URL = re.compile(
    r"(?:[a-z][a-z0-9+.\-]{1,20}://|//(?=[a-z0-9\-]+\.)|www\.)[^\s<>()\[\]{}\"'`]+"
    r"|(?:data|javascript|vbscript|mailto|file):[^\s<>()\"'`]+",
    re.I,
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*(?:\s[^<>]{0,1000})?/?>")
_TRAILING = ".,;:!?"
_SCRIPT_SCHEMES = ("data:", "javascript:", "vbscript:")


@dataclass(frozen=True, slots=True)
class GuardResult:
    text: str
    blocked: bool
    actions: tuple[str, ...]


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


class OutputGuard:
    def __init__(self, canary: str, *, max_chars: int = 4000) -> None:
        if len(canary) < 8:
            raise ValueError("canary too short")
        self._canary = canary.lower()
        self._max_chars = max_chars

    def leaks_canary(self, text: str) -> bool:
        return self._canary in text.lower() or self._canary in _compact(text)

    def check(
        self, text: str, *, sources: Sequence[str] = (), max_chars: int | None = None
    ) -> GuardResult:
        limit = max_chars or self._max_chars
        actions: list[str] = []
        # Invisible characters removed and NFKC applied first, so fullwidth or zero-width
        # tricks cannot disguise the canary, a URL scheme or markdown syntax.
        text = sanitize_text(text)[0]
        if self.leaks_canary(text):
            metrics.SECURITY_EVENTS.labels(kind="prompt_leak").inc()
            return GuardResult(REFUSAL_TEXT, True, ("prompt_leak",))

        def allowed(url: str) -> bool:
            return bool(url) and any(url in source for source in sources)

        cleaned, count = _IMAGE.subn("", text)
        cleaned, ref_images = _REF_IMAGE.subn("", cleaned)
        if count or ref_images:
            actions.append("image_removed")
        cleaned, definitions = _REF_DEFINITION.subn("", cleaned)
        if definitions:
            actions.append("link_removed")

        def link(match: re.Match[str]) -> str:
            label, url = match.group(1), match.group(2)
            if allowed(url) and not url.lower().startswith(_SCRIPT_SCHEMES):
                return match.group(0)
            actions.append("link_removed")
            return label or LINK_REMOVED

        cleaned = _LINK.sub(link, cleaned)
        cleaned, comments = _HTML_COMMENT.subn("", cleaned)
        cleaned, tags = _HTML_TAG.subn("", cleaned)
        if comments or tags:
            actions.append("html_removed")

        def bare(match: re.Match[str]) -> str:
            raw = match.group(0)
            url = raw.rstrip(_TRAILING)
            tail = raw[len(url) :]
            if allowed(url) and not url.lower().startswith(_SCRIPT_SCHEMES):
                return raw
            actions.append("url_removed")
            return LINK_REMOVED + tail

        cleaned = _URL.sub(bare, cleaned)
        if contains_secret(cleaned):
            cleaned = redact_secrets(cleaned)
            actions.append("secret_redacted")
            metrics.SECURITY_EVENTS.labels(kind="secret_redacted").inc()
        if len(cleaned) > limit:
            cleaned = truncate(cleaned, limit)
            actions.append("truncated")
        return GuardResult(cleaned.strip(), False, tuple(dict.fromkeys(actions)))
