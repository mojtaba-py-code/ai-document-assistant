"""Static analysis of the web UI (``src/docassist/web``).

The SPA runs under a strict CSP (``script-src 'self'; style-src 'self';
require-trusted-types-for 'script'; trusted-types 'none'``). These tests make sure no code
path relies on something that policy would block or that would re-open an injection sink:

* no HTML-parsing sinks, string-evaluating APIs, inline handlers or inline styles;
* no external URLs/CDNs - every asset is served from ``/static``;
* the access token is never persisted in Web Storage or cookies;
* the module graph is closed (every relative import resolves to a file) and, when Node.js
  is available, every module links and the security-relevant helpers behave as specified.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "src" / "docassist" / "web"
STATIC = WEB / "static"
INDEX = WEB / "index.html"

JS_FILES = sorted(STATIC.rglob("*.js"))
CSS_FILES = sorted(STATIC.rglob("*.css"))
SVG_FILES = sorted(STATIC.rglob("*.svg"))
SVG_NAMESPACE = "http" + "://www.w3.org/2000/svg"


def _rel(path: Path) -> str:
    return path.relative_to(WEB).as_posix()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# (pattern, reason) - applied to every JS file, comments included.
FORBIDDEN_JS = [
    (r"\binnerHTML\b", "innerHTML parses HTML"),
    (r"\bouterHTML\b", "outerHTML parses HTML"),
    (r"\binsertAdjacentHTML\b", "insertAdjacentHTML parses HTML"),
    (r"\bdocument\s*\.\s*write(ln)?\b", "document.write parses HTML"),
    (r"\bcreateContextualFragment\b", "Range.createContextualFragment parses HTML"),
    (r"\bDOMParser\b", "DOMParser parses HTML"),
    (r"\beval\s*\(", "eval executes strings"),
    (r"\bnew\s+Function\b", "new Function executes strings"),
    (r"\b(setTimeout|setInterval)\s*\(\s*[\"'`]", "string timers execute code"),
    (r"\.\s*srcdoc\s*=|setAttribute\(\s*[\"']srcdoc", "srcdoc renders HTML"),
    (r"[\"'`]\s*javascript:", "script URLs"),
    (r"\.\s*on[a-z]+\s*=(?!=)", "inline event-handler properties (use addEventListener)"),
    (r"setAttribute\(\s*[\"']on", "inline event-handler attributes"),
    (r"\.\s*style\b", "inline styles (use classes)"),
    (r"setAttribute\(\s*[\"']style", "inline style attributes"),
    (r"\bcssText\b", "inline styles"),
    (
        r"\blocalStorage\b|\bsessionStorage\b|\bindexedDB\b",
        "tokens/data must not be persisted client-side",
    ),
    (r"\bdocument\s*\.\s*cookie\b", "cookies are HttpOnly and managed by the server"),
    (r"https?://", "external URLs are not allowed"),
    (r"\bimportScripts\b|\bnew\s+Worker\b", "workers need a worker-src policy"),
]


def test_assets_exist() -> None:
    assert INDEX.is_file()
    assert (STATIC / "app.js").is_file()
    assert (STATIC / "app.css").is_file()
    assert (STATIC / "api.js").is_file()
    assert (STATIC / "router.js").is_file()
    assert len(JS_FILES) >= 20, "expected the modular SPA (app, api, router, ui and views)"


@pytest.mark.parametrize("path", JS_FILES, ids=_rel)
def test_js_has_no_forbidden_sinks(path: Path) -> None:
    source = _read(path)
    problems = []
    for pattern, reason in FORBIDDEN_JS:
        for match in re.finditer(pattern, source):
            line = source.count("\n", 0, match.start()) + 1
            problems.append(f"{_rel(path)}:{line}: {match.group(0)!r} - {reason}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", [*JS_FILES, *CSS_FILES, *SVG_FILES, INDEX], ids=_rel)
def test_sources_are_ascii(path: Path) -> None:
    """ASCII-only sources cannot hide bidi overrides or invisible characters (Trojan Source)."""
    text = _read(path)
    offenders = [
        (i, hex(ord(c)))
        for i, c in enumerate(text)
        if ord(c) > 127 or (ord(c) < 32 and c not in "\n\r\t")
    ]
    assert not offenders, f"{_rel(path)} contains non-ASCII/control characters: {offenders[:5]}"


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.script_bodies: list[str] = []
        self._in_script = False
        self.style_blocks = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self._in_script = True
        if tag == "style":
            self.style_blocks += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.script_bodies.append(data)


def _parse_index() -> _IndexParser:
    parser = _IndexParser()
    parser.feed(_read(INDEX))
    return parser


def test_index_has_no_inline_code_or_styles() -> None:
    parser = _parse_index()
    raw = _read(INDEX)
    assert not parser.script_bodies, "inline <script> content is blocked by the CSP"
    assert parser.style_blocks == 0, "inline <style> blocks are blocked by style-src 'self'"
    for tag, attrs in parser.tags:
        if tag == "script":
            assert attrs.get("src"), "<script> without src"
        for name in attrs:
            assert not name.startswith("on"), f"inline handler {name}= on <{tag}>"
            assert name != "style", f"style= attribute on <{tag}>"
    assert not re.search(r"\son[a-z]+\s*=", raw, re.IGNORECASE)
    assert not re.search(r"\sstyle\s*=", raw, re.IGNORECASE)


def test_index_references_only_local_assets() -> None:
    parser = _parse_index()
    references = [
        value
        for _tag, attrs in parser.tags
        for name, value in attrs.items()
        if name in {"src", "href", "action", "poster", "data"} and value is not None
    ]
    assert references
    for value in references:
        assert value.startswith("#") or (value.startswith("/static/") and "//" not in value), value
        if value.startswith("/static/"):
            assert (STATIC / value.removeprefix("/static/")).is_file(), f"missing asset {value}"
    tags = parser.tags
    assert ("script", {"type": "module", "src": "/static/app.js"}) in tags
    assert any(
        t == "link" and a.get("rel") == "stylesheet" and a.get("href") == "/static/app.css"
        for t, a in tags
    )
    assert not re.search(r"https?://", _read(INDEX))


IMPORT_PATTERNS = (
    # import {a} from "./x.js";  export {a} from "./x.js";  (possibly multi-line)
    re.compile(
        r"""^(?:import|export)\s[^;]*?\sfrom\s*["']([^"']+)["']""", re.MULTILINE | re.DOTALL
    ),
    # import "./x.js";
    re.compile(r"""^import\s*["']([^"']+)["']""", re.MULTILINE),
    # import("./x.js")
    re.compile(r"""\bimport\(\s*["']([^"']+)["']\s*\)"""),
)


def _module_specifiers(source: str) -> list[str]:
    return [spec for pattern in IMPORT_PATTERNS for spec in pattern.findall(source)]


def test_import_scanner_sees_static_and_dynamic_imports() -> None:
    specifiers = _module_specifiers(_read(STATIC / "app.js"))
    assert "./api.js" in specifiers
    assert "./views/library.js" in specifiers


@pytest.mark.parametrize("path", JS_FILES, ids=_rel)
def test_module_imports_resolve_locally(path: Path) -> None:
    for specifier in _module_specifiers(_read(path)):
        assert specifier.startswith(("./", "../")), (
            f"{_rel(path)}: non-relative import {specifier!r}"
        )
        target = (path.parent / specifier).resolve()
        assert target.is_file(), f"{_rel(path)}: import {specifier!r} does not exist"
        assert STATIC.resolve() in target.parents, f"{_rel(path)}: import escapes /static"


def test_css_is_self_contained() -> None:
    for path in CSS_FILES:
        css = _read(path)
        assert "@import" not in css
        assert "expression(" not in css.lower()
        assert not re.search(r"https?://", css)
        for url in re.findall(r"url\(\s*[\"']?([^\"')]+)", css):
            assert url.startswith("/static/"), f"{_rel(path)}: non-local url({url})"
            assert (STATIC / url.removeprefix("/static/")).is_file(), f"{_rel(path)}: missing {url}"


def test_every_icon_used_in_js_has_a_css_class() -> None:
    css = "\n".join(_read(p) for p in CSS_FILES)
    used: set[str] = set()
    for path in JS_FILES:
        source = _read(path)
        used.update(re.findall(r"""\bicon\(\s*["']([a-z-]+)["']\s*\)""", source))
        used.update(re.findall(r"""\bicon:\s*["']([a-z-]+)["']""", source))
    assert used
    missing = sorted(
        name for name in used if f".icon-{name} " not in css and f".icon-{name}{{" not in css
    )
    assert not missing, f"icons without CSS: {missing}"


@pytest.mark.parametrize("path", SVG_FILES, ids=_rel)
def test_svg_icons_are_inert(path: Path) -> None:
    svg = _read(path)
    assert "<script" not in svg.lower()
    assert "foreignobject" not in svg.lower()
    assert not re.search(r"\son[a-z]+\s*=", svg, re.IGNORECASE)
    assert not re.search(r"href\s*=", svg, re.IGNORECASE)
    assert re.findall(r"https?://[^\s\"']+", svg) == [SVG_NAMESPACE]


def test_api_client_session_contract() -> None:
    api = _read(STATIC / "api.js")
    # Refresh via the HttpOnly cookie, CSRF header, same-origin credentials.
    assert '"/api/v1/auth/refresh"' in api
    assert 'credentials: "same-origin"' in api
    assert '"X-CSRF-Protection"' in api
    # Single-flight refresh (one shared promise) plus a cross-tab lock.
    assert "refreshInFlight" in api
    assert "navigator.locks" in api
    # The token is only ever held in a module variable.
    assert re.search(r"^let accessToken = null;$", api, re.MULTILINE)


def test_ui_csp_matches_what_the_spa_needs() -> None:
    from docassist.api.middleware import _ui_csp

    policy = _ui_csp(None, upgrade=False)
    directives = {
        part.split()[0]: part.split()[1:] for part in (p.strip() for p in policy.split(";")) if part
    }
    assert directives["script-src"] == ["'self'"]
    assert directives["style-src"] == ["'self'"]
    assert directives["require-trusted-types-for"] == ["'script'"]
    assert directives["trusted-types"] == ["'none'"]
    assert "'unsafe-inline'" not in policy and "'unsafe-eval'" not in policy


NODE = shutil.which("node")

# Imports every module except the bootstrap (which touches `document` at load time), so a
# missing named export or a syntax error fails the ESM link step, then exercises the pure,
# security-relevant helpers.
NODE_SCRIPT = r"""
const base = process.argv[1];
const load = (rel) => import(new URL(rel, base).href);
const files = JSON.parse(process.argv[2]);
for (const rel of files) await load(rel);
const dom = await load("dom.js");
const api = await load("api.js");
const router = await load("router.js");
const format = await load("format.js");
const results = {
  safe: ["#/documents", "/api/v1/x", "blob:abc"].map(dom.isSafeUrl),
  unsafe: ["//evil.example", "/\\evil", "javascript:alert(1)", "data:text/html,x", "http:x", "JAVASCRIPT:x"].map(dom.isSafeUrl),
  path: api.apiPath("/api/v1/documents", "../users", "a/b"),
  disposition: api.filenameFromDisposition("attachment; filename=\"a.pdf\"; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf"),
  page: api.pageOf({ items: [1, 2], next_cursor: "c1" }),
  pageArray: api.pageOf([3]),
  pageJunk: api.pageOf("nope"),
  pageRef: format.pageRef(4, 6),
  humanize: format.humanize("expiration_date"),
};
router.defineRoutes([{ path: "/documents/:id", load: null }]);
const q = new URLSearchParams();
results.goodParam = router.match("/documents/0190c2a1-aaaa", q)?.params ?? null;
results.traversal = router.match("/documents/..%2Fusers", q);
results.encodedSlash = router.match("/documents/a%2Fb", q);
results.quoteParam = router.match("/documents/%22onload%3D", q);
console.log(JSON.stringify(results));
"""


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_modules_link_and_security_helpers_behave() -> None:
    assert NODE is not None
    modules = [p.relative_to(STATIC).as_posix() for p in JS_FILES if p.name != "app.js"]
    base = STATIC.resolve().as_uri() + "/"
    completed = subprocess.run(  # fixed argv, no shell
        [NODE, "--input-type=module", "-e", NODE_SCRIPT, base, json.dumps(modules)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert results["safe"] == [True, True, True]
    assert results["unsafe"] == [False] * 6
    assert results["path"] == "/api/v1/documents/..%2Fusers/a%2Fb"
    assert results["disposition"] == "r" + chr(0xE9) + "sum" + chr(0xE9) + ".pdf"
    assert results["page"] == {"items": [1, 2], "next": "c1", "total": None}
    assert results["pageArray"]["items"] == [3]
    assert results["pageJunk"]["items"] == []
    assert results["pageRef"] == "pp. 4" + chr(0x2013) + "6"
    assert results["humanize"] == "Expiration date"
    assert results["goodParam"] == {"id": "0190c2a1-aaaa"}
    assert results["traversal"] is None
    assert results["encodedSlash"] is None
    assert results["quoteParam"] is None


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_bootstrap_module_parses() -> None:
    assert NODE is not None
    completed = subprocess.run(  # fixed argv, no shell
        [NODE, "--check", str(STATIC / "app.js")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
