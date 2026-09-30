/**
 * Document intelligence panels for the detail page: extracted fields, AI summary and the
 * document report. AI output is always labelled as such and rendered as plain text.
 *
 * @module views/doc-intel
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { displayValue, fieldLabel, formatDate, formatNumber, formatPercent, humanize, pageRef } from "../format.js";
import {
  badge,
  button,
  callout,
  card,
  copyButton,
  dataTable,
  emptyState,
  errorCallout,
  loadingBlock,
  openDialog,
  select,
  withBusy,
  toast,
} from "../ui.js";

/**
 * Formats an extracted field's value from whichever typed column is populated.
 * @param {any} item
 * @returns {string}
 */
export function fieldValue(item) {
  if (!item || typeof item !== "object") return displayValue(item);
  if (item.value_date) return formatDate(item.value_date);
  if (item.value_number !== null && item.value_number !== undefined && item.value_number !== "") {
    const amount = formatNumber(item.value_number);
    return item.currency ? `${amount} ${item.currency}` : amount;
  }
  if (item.value_text) return String(item.value_text);
  if (item.value !== undefined) return displayValue(item.value);
  return displayValue(null);
}

/**
 * Small "AI-generated" label used on every model-produced block.
 * @param {string} [text]
 * @returns {HTMLElement}
 */
export function aiLabel(text = "AI-generated") {
  return h("span", { class: "ai-label" }, h("span", { class: "icon icon-sparkle", "aria-hidden": "true" }), text);
}

/**
 * Extracted-fields panel.
 * @param {{documentId: string, canRun: boolean, signal: AbortSignal, openPage: (page: number, quote?: string, chunkId?: string) => void}} options
 * @returns {HTMLElement}
 */
export function fieldsPanel(options) {
  const list = h("div", { "aria-live": "polite" });
  const load = async () => {
    mount(list, loadingBlock("Loading extracted data\u2026"));
    try {
      const response = await api.get(apiPath("/api/v1/intelligence/documents", options.documentId, "fields"), { signal: options.signal });
      const items = pageOf(response, ["fields"]).items;
      mount(
        list,
        dataTable({
          caption: "Extracted fields",
          rows: items,
          empty: emptyState({
            title: "No data has been extracted yet.",
            text: "Dates, amounts, parties and payment terms are detected automatically when the document is processed.",
            icon: "table",
          }),
          columns: [
            { label: "Field", primary: true, render: (f) => fieldLabel(f.field || f.name) },
            { label: "Value", render: (f) => fieldValue(f) },
            {
              label: "Evidence",
              className: "col-evidence",
              render: (f) => (f.evidence ? h("q", { class: "evidence-quote" }, String(f.evidence)) : null),
            },
            {
              label: "Page",
              render: (f) =>
                Number(f.page) > 0
                  ? button(pageRef(f.page), {
                      variant: "link",
                      small: true,
                      ariaLabel: `Open page ${f.page} in the preview`,
                      onClick: () => options.openPage(Number(f.page), f.evidence, f.chunk_id),
                    })
                  : null,
            },
            { label: "Confidence", render: (f) => formatPercent(f.confidence) },
            {
              label: "Method",
              render: (f) => (f.method === "llm" ? badge("AI", "accent") : badge(f.method === "rules" ? "Rules" : humanize(f.method), "neutral")),
            },
          ],
        }),
      );
    } catch (error) {
      if (!options.signal.aborted) mount(list, errorCallout(error, { retry: load }));
    }
  };

  const schema = select(
    [
      { value: "contract", label: "Contract fields" },
      { value: "invoice", label: "Invoice fields" },
    ],
    { "aria-label": "What to extract", class: "input select select-sm" },
  );
  const run = button("Extract with AI", {
    small: true,
    icon: "sparkle",
    onClick: () =>
      withBusy(run, async () => {
        try {
          const result = await api.post(
            "/api/v1/intelligence/extract",
            { document_id: options.documentId, kind: schema.value },
            { timeoutMs: 180_000 },
          );
          toast(extractionMessage(result), "success");
          await load();
        } catch (error) {
          toast(errorMessageFor(error), "danger");
        }
      }, "Extracting\u2026"),
  });

  load();
  return card({
    title: "Extracted data",
    description: "Every value links to the passage it was found in. AI-extracted values are only kept when the quoted evidence exists in the document.",
    actions: options.canRun ? [schema, run] : [],
    body: list,
  });
}

/**
 * Plain-language outcome of an AI extraction run.
 * @param {any} result ExtractionResult
 * @returns {string}
 */
function extractionMessage(result) {
  if (!result || typeof result !== "object") return "Extraction finished.";
  if (result.method === "rules_only") return "The AI service was not available, so only automatically detected values are shown.";
  const kept = Number(result.persisted_count ?? (Array.isArray(result.fields) ? result.fields.length : 0));
  const dropped = Number(result.dropped_unverified || 0);
  return `Extraction finished: ${kept} value${kept === 1 ? "" : "s"} saved${dropped ? `, ${dropped} discarded because the quoted evidence was not found in the document` : ""}.`;
}

function errorMessageFor(error) {
  return error && error.detail ? error.detail : "The request failed. Please try again.";
}

/**
 * AI summary panel (generated on demand).
 * @param {{documentId: string, signal: AbortSignal, openPage: (page: number) => void}} options
 * @returns {HTMLElement}
 */
export function summaryPanel(options) {
  const output = h("div", { "aria-live": "polite" });
  const style = select(
    [
      { value: "brief", label: "Brief" },
      { value: "detailed", label: "Detailed" },
      { value: "executive", label: "Executive" },
    ],
    { "aria-label": "Summary style", class: "input select select-sm" },
  );
  const run = button("Summarise", {
    variant: "primary",
    small: true,
    icon: "sparkle",
    onClick: () =>
      withBusy(run, async () => {
        mount(output, loadingBlock("Reading the document and writing a summary\u2026"));
        try {
          const result = await api.post(
            "/api/v1/intelligence/summarize",
            { document_id: options.documentId, style: style.value },
            { timeoutMs: 300_000, signal: options.signal },
          );
          mount(output, renderSummary(result, options.openPage));
        } catch (error) {
          if (!options.signal.aborted) mount(output, errorCallout(error, { title: "Could not summarise" }));
        }
      }, "Summarising\u2026"),
  });
  mount(output, emptyState({ title: "No summary yet.", text: "Choose a style and select Summarise.", icon: "sparkle" }));
  return card({ title: "Summary", actions: [style, run], body: output });
}

function renderSummary(result, openPage) {
  const data = result && typeof result === "object" ? result : { summary: String(result ?? "") };
  const text = String(data.summary ?? data.text ?? "");
  const points = Array.isArray(data.key_points) ? data.key_points : [];
  const extractive = data.method === "extractive";
  return h(
    "article",
    { class: "ai-block" },
    h("header", { class: "ai-block-header" }, aiLabel(extractive ? "Automatic summary (extractive)" : "AI-generated summary"), data.model && h("span", { class: "muted small" }, String(data.model))),
    extractive && callout("info", null, "The AI service was not used for this summary, so it is built from sentences taken directly from the document."),
    data.truncated === true && callout("info", null, "The document is long, so the summary covers only part of it."),
    Array.isArray(data.warnings) && data.warnings.length > 0 &&
      callout("warning", null, h("ul", { class: "plain-list" }, data.warnings.map((w) => h("li", null, String(w))))),
    text && text.split(/\n{2,}/).map((para) => h("p", { class: "ai-text" }, para)),
    points.length > 0 &&
      h(
        "div",
        null,
        h("h3", { class: "subheading" }, "Key points"),
        h(
          "ul",
          { class: "key-points" },
          points.map((point) => {
            const pointText = typeof point === "string" ? point : String(point.text ?? point.point ?? "");
            const pages = collectPages(point);
            return h(
              "li",
              null,
              pointText,
              pages.map((p) => h("span", null, " ", button(pageRef(p), { variant: "link", small: true, ariaLabel: `Open page ${p}`, onClick: () => openPage(p) }))),
            );
          }),
        ),
      ),
    h("p", { class: "muted small" }, "Check important details against the document before relying on them."),
  );
}

function collectPages(point) {
  if (!point || typeof point !== "object") return [];
  const pages = new Set();
  const add = (value) => {
    const n = Number(value);
    if (Number.isInteger(n) && n > 0) pages.add(n);
  };
  if (Array.isArray(point.pages)) point.pages.forEach(add);
  add(point.page);
  add(point.page_start);
  if (Array.isArray(point.citations)) point.citations.forEach((c) => add(c && (c.page ?? c.page_start)));
  return [...pages].sort((a, b) => a - b).slice(0, 5);
}

/**
 * Opens the document report (metadata, key fields, deadlines, risk flags) in a dialog.
 * @param {string} documentId
 * @param {string} title
 */
export async function openReportDialog(documentId, title) {
  const body = h("div", null, loadingBlock("Building the report\u2026"));
  const handle = openDialog({ title: `Report: ${title}`, size: "lg", content: body });
  try {
    const report = await api.get(apiPath("/api/v1/intelligence/documents", documentId, "report"), { timeoutMs: 180_000 });
    const markdown = String((report && (report.markdown ?? report.text)) || "");
    const inner = report && report.report && typeof report.report === "object" ? report.report : report || {};
    const flags = Array.isArray(inner.risk_flags) ? inner.risk_flags : [];
    mount(
      body,
      flags.length > 0 &&
        card({
          title: "Points to review",
          level: 3,
          body: h(
            "ul",
            { class: "risk-list" },
            flags.map((flag) => {
              const data = flag && typeof flag === "object" ? flag : { code: String(flag) };
              const tone = data.severity === "high" ? "danger" : data.severity === "info" ? "info" : "warning";
              return h(
                "li",
                null,
                badge(humanize(data.severity || "review"), tone),
                " ",
                h("strong", null, String(data.title || humanize(data.code || "flag"))),
                data.detail ? ` \u2014 ${data.detail}` : "",
                Number(data.page) > 0 ? h("span", { class: "muted small" }, ` (${pageRef(data.page)})`) : null,
              );
            }),
          ),
        }),
      markdown
        ? h("pre", { class: "report-text" }, markdown)
        : h("pre", { class: "report-text" }, JSON.stringify(report, null, 2)),
    );
    handle.footer.append(copyButton(() => markdown || JSON.stringify(report, null, 2), "Copy report"));
  } catch (error) {
    mount(body, errorCallout(error));
  }
}
