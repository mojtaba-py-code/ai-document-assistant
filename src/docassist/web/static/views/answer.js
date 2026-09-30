/**
 * Rendering of grounded answers.
 *
 * Two visually and semantically separate regions:
 * 1. the **AI-generated answer** card (model output: status, text, confidence meter,
 *    warnings) and
 * 2. the **Source evidence** panel (verbatim quotes from the user's documents, each with
 *    document / version / page / section and a button that opens the preview there).
 *
 * Model output is untrusted: it is inserted only as text. Citation markers such as `[S1]`
 * or `[2]` become buttons that move focus to the matching evidence item; URLs stay plain
 * text (never links), and markdown/HTML is shown literally.
 *
 * @module views/answer
 */

import { h, uid } from "../dom.js";
import { displayValue, formatNumber, formatPercent, humanize, pageRef } from "../format.js";
import { navigate } from "../router.js";
import { putHandoff } from "../session.js";
import { badge, button, callout, copyButton } from "../ui.js";
import { aiLabel } from "./doc-intel.js";

const STATUS = {
  answered: { text: "Answered from your documents", tone: "success" },
  insufficient_context: { text: "Not enough information in your documents", tone: "warning" },
  refused: { text: "The assistant declined this request", tone: "danger" },
};

/**
 * Normalises an answer-like object (live answer, stored message or agent result).
 * @param {any} raw
 * @returns {{status: string, answer: string, confidence: number | null, confidenceLabel: string, citations: any[], evidence: any[], warnings: string[], missing: string, model: string, provider: string, latencyMs: number | null, steps: any[], rows: any[]}}
 */
export function normalizeAnswer(raw) {
  const data = raw && typeof raw === "object" ? raw : { answer: String(raw ?? "") };
  const confidence = Number(data.confidence);
  return {
    status: typeof data.status === "string" ? data.status : "answered",
    answer: String(data.answer ?? data.content ?? data.text ?? ""),
    confidence: Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : null,
    confidenceLabel: typeof data.confidence_label === "string" ? data.confidence_label : "",
    citations: Array.isArray(data.citations) ? data.citations : [],
    evidence: Array.isArray(data.evidence) ? data.evidence : [],
    warnings: Array.isArray(data.warnings) ? data.warnings.map((w) => (typeof w === "string" ? w : displayValue(w))) : [],
    missing: typeof data.missing_information === "string" ? data.missing_information : "",
    model: typeof data.model === "string" ? data.model : "",
    provider: typeof data.provider === "string" ? data.provider : "",
    latencyMs: Number.isFinite(Number(data.latency_ms)) ? Number(data.latency_ms) : null,
    steps: Array.isArray(data.steps) ? data.steps : Array.isArray(data.tool_calls) ? data.tool_calls : [],
    rows: Array.isArray(data.rows) ? data.rows : Array.isArray(data.deadlines) ? data.deadlines : [],
    cached: data.cached === true,
  };
}

function confidenceText(answer) {
  if (answer.confidence === null && !answer.confidenceLabel) return "";
  const label = answer.confidenceLabel
    ? humanize(answer.confidenceLabel)
    : answer.confidence >= 0.75
      ? "High"
      : answer.confidence >= 0.5
        ? "Medium"
        : "Low";
  return answer.confidence === null ? `${label} confidence` : `${label} confidence (${formatPercent(answer.confidence)})`;
}

/**
 * Splits answer text into text nodes and citation-marker buttons.
 * @param {string} text
 * @param {Map<string, {id: string, label: string}>} targets marker key \u2192 evidence element id
 * @returns {Node[]}
 */
function textWithMarkers(text, targets) {
  const nodes = [];
  const pattern = /\[(S?\d{1,3})\]/g;
  let last = 0;
  let found;
  while ((found = pattern.exec(text)) !== null) {
    const key = found[1].toUpperCase();
    const target = targets.get(key) || targets.get(key.replace(/^S/, ""));
    if (!target) continue;
    if (found.index > last) nodes.push(document.createTextNode(text.slice(last, found.index)));
    nodes.push(
      h(
        "button",
        {
          type: "button",
          class: "cite-ref",
          "aria-label": `Source ${target.label}`,
          on: {
            click: () => {
              const element = document.getElementById(target.id);
              if (element) {
                element.scrollIntoView({ block: "nearest" });
                element.focus();
              }
            },
          },
        },
        found[0],
      ),
    );
    last = found.index + found[0].length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  return nodes;
}

function openCitation(citation) {
  const docId = String(citation.document_id || "");
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(docId)) return;
  putHandoff(`highlight:${docId}`, { quote: String(citation.quote || citation.excerpt || ""), chunkId: citation.chunk_id || "" });
  const page = Number(citation.page_start);
  const query = { tab: "preview" };
  if (page > 1) query.page = page;
  if (citation.version_number) query.version = citation.version_number;
  navigate(`/documents/${docId}`, query);
}

/**
 * Renders the AI answer card and the separate source-evidence panel for one exchange.
 * @param {any} raw GroundedAnswer-like object
 * @returns {HTMLElement}
 */
export function answerBlock(raw) {
  const answer = normalizeAnswer(raw);
  const key = uid("ans");
  const targets = new Map();
  answer.citations.forEach((citation, index) => {
    const n = Number(citation.n) || index + 1;
    const id = `${key}-src-${n}`;
    const label = `${n}: ${citation.document_title || "document"}${citation.page_start ? `, ${pageRef(citation.page_start, citation.page_end)}` : ""}`;
    targets.set(String(n), { id, label });
    if (citation.source_id) targets.set(String(citation.source_id).toUpperCase(), { id, label });
  });

  const status = STATUS[answer.status] || { text: humanize(answer.status), tone: "neutral" };
  const confidence = confidenceText(answer);
  const paragraphs = answer.answer.split(/\n{2,}/).filter((p) => p.trim());

  const card = h(
    "article",
    { class: ["answer-card", `answer-${answer.status}`], "aria-labelledby": `${key}-label` },
    h(
      "header",
      { class: "answer-header" },
      h("span", { id: `${key}-label` }, aiLabel("AI-generated answer")),
      badge(status.text, status.tone),
    ),
    h(
      "div",
      { class: "answer-text" },
      paragraphs.length ? paragraphs.map((p) => h("p", null, textWithMarkers(p, targets))) : h("p", { class: "muted" }, "No answer text was returned."),
    ),
    answer.rows.length > 0 && rowsTable(answer.rows),
    answer.missing && h("p", { class: "answer-missing" }, h("strong", null, "What is missing: "), answer.missing),
    answer.warnings.length > 0 &&
      callout("warning", "Please note", h("ul", { class: "plain-list" }, answer.warnings.map((w) => h("li", null, w)))),
    confidence &&
      h(
        "div",
        { class: "confidence" },
        h("span", { class: "confidence-label", id: `${key}-conf` }, "Confidence"),
        answer.confidence !== null &&
          h("meter", {
            class: "confidence-meter",
            min: "0",
            max: "1",
            low: "0.5",
            high: "0.75",
            optimum: "1",
            value: String(answer.confidence),
            "aria-labelledby": `${key}-conf`,
            "aria-valuetext": confidence,
          }),
        h("span", { class: "confidence-text" }, confidence),
      ),
    answer.steps.length > 0 && stepsList(answer.steps),
    h(
      "footer",
      { class: "answer-footer" },
      h(
        "span",
        { class: "muted small" },
        [
          answer.model && `Model: ${answer.model}${answer.provider ? ` (${answer.provider})` : ""}`,
          answer.cached ? "Reused a recent identical answer" : answer.latencyMs !== null && `${formatNumber(Math.round(answer.latencyMs / 100) / 10)} s`,
          "Verify important details in the source evidence.",
        ]
          .filter(Boolean)
          .join(" \u00b7 "),
      ),
      answer.answer && copyButton(() => answer.answer, "Copy answer"),
    ),
  );

  return h("div", { class: "exchange-answer" }, card, evidencePanel(answer, key));
}

function rowsTable(rows) {
  const keys = [...new Set(rows.slice(0, 50).flatMap((row) => (row && typeof row === "object" ? Object.keys(row) : [])))]
    .filter((k) => !/(^|_)id$/.test(k))
    .slice(0, 6);
  if (!keys.length) return null;
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: "table table-compact" },
      h("caption", { class: "visually-hidden" }, "Answer details"),
      h("thead", null, h("tr", null, keys.map((k) => h("th", { scope: "col" }, humanize(k))))),
      h(
        "tbody",
        null,
        rows.slice(0, 50).map((row) => h("tr", null, keys.map((k) => h("td", { dataset: { label: humanize(k) } }, displayValue(row[k]))))),
      ),
    ),
  );
}

function stepsList(steps) {
  return h(
    "details",
    { class: "details steps" },
    h("summary", null, `How the assistant worked (${steps.length} step${steps.length === 1 ? "" : "s"})`),
    h(
      "ol",
      { class: "plain-list" },
      steps.slice(0, 20).map((step) => {
        const data = step && typeof step === "object" ? step : { tool: String(step) };
        const name = humanize(String(data.tool || data.name || "step"));
        const detail = data.detail ? String(data.detail) : data.arguments ? displayValue(data.arguments) : "";
        return h(
          "li",
          null,
          h("span", null, name),
          data.ok === false && h("span", null, " ", badge("Failed", "danger")),
          detail && h("span", { class: "muted small" }, ` \u2014 ${detail.slice(0, 200)}`),
          Number.isFinite(Number(data.duration_ms)) && h("span", { class: "muted small" }, ` (${formatNumber(data.duration_ms)} ms)`),
        );
      }),
    ),
  );
}

function evidencePanel(answer, key) {
  const cited = answer.citations;
  const citedIds = new Set(cited.map((c) => String(c.source_id || "")));
  const others = answer.evidence.filter((e) => !citedIds.has(String(e.source_id || "")));
  return h(
    "section",
    { class: "evidence-panel", "aria-labelledby": `${key}-ev` },
    h("h3", { class: "evidence-title", id: `${key}-ev` }, "Source evidence"),
    h(
      "p",
      { class: "evidence-intro" },
      cited.length
        ? "Verbatim quotes from your documents that support the answer. Open a source to read it in context."
        : "No passages were cited for this answer.",
    ),
    cited.length > 0 &&
      h(
        "ol",
        { class: "evidence-list" },
        cited.map((citation, index) => {
          const n = Number(citation.n) || index + 1;
          const meta = [
            citation.version_number && `Version ${citation.version_number}`,
            pageRef(citation.page_start, citation.page_end),
            citation.section && String(citation.section),
          ].filter(Boolean);
          const page = Number(citation.page_start);
          return h(
            "li",
            { class: "evidence-item", id: `${key}-src-${n}`, tabindex: "-1" },
            h(
              "div",
              { class: "evidence-head" },
              h("span", { class: "cite-num", "aria-hidden": "true" }, `[${n}]`),
              h("span", { class: "visually-hidden" }, `Source ${n}: `),
              h("strong", { class: "evidence-doc" }, String(citation.document_title || "Untitled document")),
            ),
            meta.length > 0 && h("p", { class: "evidence-meta" }, meta.join(" \u00b7 ")),
            citation.quote && h("blockquote", { class: "evidence-quote-block" }, String(citation.quote)),
            citation.document_id &&
              button(page > 0 ? `Open ${pageRef(page)}` : "Open document", {
                small: true,
                variant: "ghost",
                icon: "file",
                ariaLabel: `Open ${citation.document_title || "document"}${page > 0 ? ` at ${pageRef(page)}` : ""}`,
                onClick: () => openCitation(citation),
              }),
          );
        }),
      ),
    others.length > 0 &&
      h(
        "details",
        { class: "details" },
        h("summary", null, `Other passages the assistant read (${others.length})`),
        h(
          "ul",
          { class: "evidence-list evidence-list-secondary" },
          others.slice(0, 15).map((item) =>
            h(
              "li",
              { class: "evidence-item" },
              h("strong", { class: "evidence-doc" }, String(item.document_title || "Untitled document")),
              h("p", { class: "evidence-meta" }, [pageRef(item.page_start, item.page_end), item.section && String(item.section)].filter(Boolean).join(" \u00b7 ")),
              item.excerpt && h("p", { class: "evidence-excerpt" }, String(item.excerpt)),
              item.document_id &&
                button("Open", { small: true, variant: "ghost", onClick: () => openCitation({ ...item, quote: item.excerpt }) }),
            ),
          ),
        ),
      ),
  );
}
