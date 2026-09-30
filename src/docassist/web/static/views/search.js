/**
 * Search across readable documents (hybrid / keyword / semantic).
 *
 * The query lives in the URL hash so results can be revisited; snippets are plain text
 * with query terms highlighted by splitting text nodes (no HTML from the server is used).
 *
 * @module views/search
 */

import { api } from "../api.js";
import { h, highlightTerms, mount } from "../dom.js";
import { docTypeLabel, formatNumber, formatPercent, pageRef } from "../format.js";
import { updateQuery } from "../router.js";
import { can, getDepartments, putHandoff } from "../session.js";
import {
  badge,
  button,
  callout,
  checkbox,
  emptyState,
  errorCallout,
  field,
  input,
  linkButton,
  loadingBlock,
  pageHeader,
  segmented,
  select,
  toast,
  withBusy,
} from "../ui.js";
import { classificationOptions, departmentOptions, docTypeOptions, documentBadges } from "./common.js";

const MODES = [
  { value: "hybrid", label: "Best match" },
  { value: "keyword", label: "Exact words" },
  { value: "semantic", label: "Similar meaning" },
];
const MODE_NAMES = { hybrid: "best match", keyword: "exact words", semantic: "similar meaning" };

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function searchView(ctx) {
  const departments = await getDepartments();
  let mode = MODES.some((m) => m.value === ctx.query.get("mode")) ? ctx.query.get("mode") : "hybrid";
  let limit = 20;
  let lastRequest = null;

  const query = input({
    type: "search",
    name: "q",
    value: ctx.query.get("q") || "",
    maxlength: 1000,
    required: true,
    class: "input input-lg",
    placeholder: "e.g. termination notice period for suppliers",
    autocomplete: "off",
  });
  const docType = select(docTypeOptions({ any: "All types" }));
  const classification = select(classificationOptions({ any: "Any classification" }));
  const department = select(departmentOptions(departments, "All departments"));
  const after = input({ type: "date" });
  const before = input({ type: "date" });
  const oldVersions = checkbox("Include older versions");
  const submit = button("Search", { type: "submit", variant: "primary", icon: "search" });
  const results = h("div", { class: "results", "aria-live": "polite", "aria-busy": "false" });

  const filtersPayload = () => {
    const filters = {};
    if (docType.value) filters.doc_types = [docType.value];
    if (classification.value) filters.classifications = [classification.value];
    if (department.value) filters.department_ids = [department.value];
    if (after.value) filters.created_after = new Date(`${after.value}T00:00:00`).toISOString();
    if (before.value) filters.created_before = new Date(`${before.value}T23:59:59`).toISOString();
    if (oldVersions.querySelector("input").checked) filters.include_old_versions = true;
    return filters;
  };

  const run = async () => {
    const text = query.value.trim();
    if (!text) return;
    updateQuery({ q: text, mode: mode === "hybrid" ? null : mode });
    lastRequest = { query: text, mode, filters: filtersPayload(), limit };
    results.setAttribute("aria-busy", "true");
    mount(results, loadingBlock("Searching\u2026"));
    try {
      const response = await api.post("/api/v1/search", lastRequest, { signal: ctx.signal });
      renderResults(response, text);
    } catch (error) {
      if (!ctx.signal.aborted) mount(results, errorCallout(error, { retry: run }));
    } finally {
      results.setAttribute("aria-busy", "false");
    }
  };

  const renderResults = (response, text) => {
    const items = response && Array.isArray(response.results) ? response.results : [];
    const terms = text.split(/\s+/).filter((t) => t.length > 1 && !/^(the|and|for|with|of|to|in|a|an)$/i.test(t));
    const took = response && Number.isFinite(Number(response.took_ms)) ? ` in ${formatNumber(response.took_ms)} ms` : "";
    const modeUsed = response && response.mode_used ? MODE_NAMES[response.mode_used] || String(response.mode_used) : MODE_NAMES[mode];
    mount(
      results,
      response && response.degraded &&
        callout("warning", "Limited search", "Similar-meaning search is temporarily unavailable, so these results match your exact words only."),
      h(
        "div",
        { class: "results-bar" },
        h("p", { class: "results-summary" }, `${items.length} result${items.length === 1 ? "" : "s"} (${modeUsed})${took}`),
        items.length > 0 && can("export:create") &&
          button("Export results", {
            small: true,
            icon: "download",
            onClick: (event) => exportResults(event.currentTarget),
          }),
      ),
      items.length
        ? h("ol", { class: "result-list" }, items.map((item) => resultItem(item, terms)))
        : emptyState({
            title: "No matching passages.",
            text: "Try different words, switch to \u201csimilar meaning\u201d, or remove filters. Only documents you are allowed to read are searched.",
            icon: "search",
          }),
      items.length >= limit && limit < 50 &&
        h(
          "div",
          { class: "load-more" },
          button("Show more results", {
            onClick: () => {
              limit = 50;
              run();
            },
          }),
        ),
    );
  };

  const exportResults = (control) =>
    withBusy(control, async () => {
      try {
        await api.post("/api/v1/exports", { kind: "search_results", format: "csv", params: exportParams(lastRequest) });
        toast("Export started. You can download it from the Exports page when it is ready.", "success");
      } catch (error) {
        toast(error && error.detail ? error.detail : "The export could not be created.", "danger");
      }
    }, "Exporting\u2026");

  const form = h(
    "form",
    { class: "search-form", role: "search", "aria-label": "Search documents" },
    h(
      "div",
      { class: "search-row" },
      h("label", { for: "search-q", class: "visually-hidden" }, "Search your documents"),
      query,
      submit,
    ),
    segmented("Search mode", MODES, {
      value: mode,
      onChange: (value) => {
        mode = value;
        if (query.value.trim()) run();
      },
    }),
    h(
      "details",
      { class: "details" },
      h("summary", null, "Filters"),
      h(
        "div",
        { class: "filters filters-plain" },
        field("Type", docType),
        field("Classification", classification),
        departments.length > 0 && field("Department", department),
        field("Created after", after),
        field("Created before", before),
        h("div", { class: "filters-extra" }, oldVersions),
      ),
    ),
  );
  query.id = "search-q";
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    limit = 20;
    run();
  });

  if (query.value.trim()) run();
  else {
    mount(
      results,
      emptyState({
        title: "Search inside your documents",
        text: "Results show the exact passage and page. Only documents you are allowed to read are searched.",
        icon: "search",
      }),
    );
  }

  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Search", subtitle: "Find passages across every document you can read." }),
    form,
    results,
  );
}

function resultItem(item, terms) {
  const page = Number(item.page_start) || 0;
  const docId = String(item.document_id || "");
  const params = new URLSearchParams({ tab: "preview" });
  if (page > 1) params.set("page", String(page));
  if (item.is_current === false && item.version_number) params.set("version", String(item.version_number));
  const openHref = `#/documents/${encodeURIComponent(docId)}?${params.toString()}`;
  // Hand the snippet to the preview so the matching passage is highlighted (kept out of the URL).
  const rememberSnippet = () => {
    const quote = String(item.snippet || "").replace(/^[\s.\u2026]+|[\s.\u2026]+$/g, "");
    if (quote) putHandoff(`highlight:${docId}`, { quote, chunkId: "" });
  };
  const meta = [
    item.doc_type && docTypeLabel(item.doc_type),
    pageRef(item.page_start, item.page_end),
    item.section && String(item.section),
    item.version_number && `Version ${item.version_number}`,
  ].filter(Boolean);
  return h(
    "li",
    { class: "result" },
    h(
      "div",
      { class: "result-head" },
      h("a", { href: openHref, class: "result-title", on: { click: rememberSnippet } }, String(item.document_title || "Untitled document")),
      documentBadges(item),
      item.flagged && badge("Contains instruction-like text", "warning"),
    ),
    meta.length > 0 && h("p", { class: "result-meta" }, meta.join(" \u00b7 ")),
    h("p", { class: "result-snippet" }, highlightTerms(String(item.snippet || ""), terms)),
    h(
      "div",
      { class: "result-foot" },
      Number.isFinite(Number(item.score)) && h("span", { class: "muted small" }, `Relevance ${formatPercent(Math.min(1, Math.max(0, Number(item.score))))}`),
      h("a", { href: openHref, class: "btn btn-ghost btn-sm", on: { click: rememberSnippet } }, h("span", { class: "btn-label" }, page ? `Open ${pageRef(page)}` : "Open document")),
      can("assistant:use") && linkButton("Ask about this document", `#/assistant?document=${encodeURIComponent(docId)}`, { small: true, variant: "ghost", icon: "sparkle" }),
    ),
  );
}

/**
 * Maps a search request to the export API's parameters (which accept a subset of the
 * search filters: document types, documents and tags).
 * @param {{query: string, mode: string, filters?: Record<string, any>, limit?: number}} request
 * @returns {Record<string, any>}
 */
export function exportParams(request) {
  const filters = (request && request.filters) || {};
  const params = { query: request.query, mode: request.mode || "hybrid", limit: Math.min(50, Number(request.limit) || 50) };
  if (Array.isArray(filters.doc_types) && filters.doc_types.length) params.doc_types = filters.doc_types;
  if (Array.isArray(filters.document_ids) && filters.document_ids.length) params.document_ids = filters.document_ids;
  if (Array.isArray(filters.tags) && filters.tags.length) params.tags = filters.tags;
  return params;
}
