/**
 * Compare two versions of a document, or two documents.
 *
 * The deterministic difference (added / removed / changed passages and changed key
 * fields) is always shown; the optional AI change summary is labelled as AI-generated and
 * only describes the listed differences.
 *
 * @module views/compare
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { displayValue, EMPTY, fieldLabel, formatNumber, formatPercent, humanize, pageRef } from "../format.js";
import { updateQuery } from "../router.js";
import {
  badge,
  button,
  callout,
  card,
  checkbox,
  dataTable,
  emptyState,
  errorCallout,
  field,
  loadingBlock,
  pageHeader,
  segmented,
  select,
  withBusy,
} from "../ui.js";
import { documentPicker, fetchDocument, versionsOf } from "./common.js";
import { aiLabel, fieldValue } from "./doc-intel.js";

/**
 * Request body for `POST /api/v1/intelligence/compare`: two versions of one document
 * (`base_version` -> `target_version`) or the current versions of two documents.
 * @param {{mode: string, left: any, right: any, fromVersion: number | null, toVersion: number | null, summary: boolean}} state
 * @returns {Record<string, any>}
 */
export function compareRequestBody(state) {
  const body = { document_id: String(state.left.id), include_change_summary: Boolean(state.summary) };
  if (state.mode === "versions") {
    body.base_version = state.fromVersion;
    body.target_version = state.toVersion;
  } else {
    body.other_document_id = String(state.right.id);
  }
  return body;
}

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function compareView(ctx) {
  const state = {
    mode: ctx.query.get("other") ? "documents" : "versions",
    left: null,
    right: null,
    fromVersion: Number(ctx.query.get("from")) || null,
    toVersion: Number(ctx.query.get("to")) || null,
    summary: true,
  };
  const presetId = ctx.query.get("document");
  const preset = presetId ? await fetchDocument(presetId, ctx.signal) : null;
  const presetOther = ctx.query.get("other") ? await fetchDocument(ctx.query.get("other"), ctx.signal) : null;

  const setup = h("div", { class: "compare-setup" });
  const output = h("div", { "aria-live": "polite", "aria-busy": "false" });
  const summaryToggle = checkbox("Also write an AI summary of the changes", { checked: true });
  const run = button("Compare", { type: "submit", variant: "primary", icon: "compare" });

  const versionSelects = h("div", { class: "field-row" });
  const renderVersionSelects = (detail) => {
    const indexed = versionsOf(detail).filter((v) => v.status === "indexed");
    if (indexed.length < 2) {
      mount(versionSelects, callout("info", null, "This document has fewer than two processed versions, so there is nothing to compare yet."));
      return;
    }
    const options = indexed.map((v) => ({ value: String(v.version_number), label: `Version ${v.version_number}${v.change_note ? ` \u2014 ${String(v.change_note).slice(0, 40)}` : ""}` }));
    const from = select(options, { value: String(state.fromVersion || indexed[1].version_number) });
    const to = select(options, { value: String(state.toVersion || indexed[0].version_number) });
    state.fromVersion = Number(from.value);
    state.toVersion = Number(to.value);
    from.addEventListener("change", () => (state.fromVersion = Number(from.value)));
    to.addEventListener("change", () => (state.toVersion = Number(to.value)));
    mount(versionSelects, field("Older version", from), field("Newer version", to));
  };

  const renderSetup = () => {
    if (state.mode === "versions") {
      mount(
        setup,
        documentPicker({
          label: "Document",
          initial: preset,
          onSelect: async (doc) => {
            state.left = doc;
            const detail = doc.versions ? doc : await fetchDocument(String(doc.id), ctx.signal);
            renderVersionSelects(detail || doc);
          },
        }),
        versionSelects,
      );
    } else {
      mount(
        setup,
        h(
          "div",
          { class: "field-row" },
          documentPicker({ label: "First document", initial: preset, onSelect: (doc) => (state.left = doc) }),
          documentPicker({ label: "Second document", initial: presetOther, onSelect: (doc) => (state.right = doc) }),
        ),
      );
    }
  };

  const form = h(
    "form",
    { class: "form card" },
    segmented(
      "What to compare",
      [
        { value: "versions", label: "Two versions of a document" },
        { value: "documents", label: "Two different documents" },
      ],
      {
        value: state.mode,
        onChange: (value) => {
          state.mode = value;
          state.right = null;
          renderSetup();
        },
      },
    ),
    setup,
    summaryToggle,
    h("div", { class: "form-actions" }, run),
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    state.summary = summaryToggle.querySelector("input").checked;
    if (!state.left || (state.mode === "documents" && !state.right)) {
      mount(output, callout("warning", null, "Choose the documents to compare first.", { live: true }));
      return;
    }
    if (state.mode === "versions" && (!state.fromVersion || !state.toVersion || state.fromVersion === state.toVersion)) {
      mount(output, callout("warning", null, "Choose two different versions.", { live: true }));
      return;
    }
    updateQuery({
      document: state.left.id,
      other: state.mode === "documents" && state.right ? state.right.id : null,
      from: state.mode === "versions" ? state.fromVersion : null,
      to: state.mode === "versions" ? state.toVersion : null,
    });
    await withBusy(run, async () => {
      output.setAttribute("aria-busy", "true");
      mount(output, loadingBlock("Comparing\u2026"));
      try {
        const result = await api.post("/api/v1/intelligence/compare", compareRequestBody(state), { timeoutMs: 300_000, signal: ctx.signal });
        mount(output, renderComparison(result));
      } catch (error) {
        if (!ctx.signal.aborted) mount(output, errorCallout(error, { title: "Comparison failed" }));
      } finally {
        output.setAttribute("aria-busy", "false");
      }
    }, "Comparing\u2026");
  });

  renderSetup();

  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Compare", subtitle: "See exactly what changed between two versions or two documents." }),
    form,
    output,
  );
}

function listOf(result, keys) {
  for (const key of keys) {
    if (result && Array.isArray(result[key])) return result[key];
  }
  return [];
}

function versionLabel(ref) {
  if (!ref || typeof ref !== "object") return "";
  return `${String(ref.document_title || "Document")} (version ${ref.version_number ?? "?"})`;
}

function fieldValues(values) {
  if (!Array.isArray(values)) return displayValue(values);
  if (!values.length) return EMPTY;
  return values.map((v) => fieldValue(v)).join("; ");
}

function refPage(ref) {
  return ref && typeof ref === "object" ? pageRef(ref.page_start, ref.page_end) : "";
}

/**
 * Renders a comparison result (`ComparisonResult`), defensively.
 * @param {any} result
 * @returns {HTMLElement}
 */
export function renderComparison(result) {
  const hunks = listOf(result, ["hunks", "changes"]);
  const fields = listOf(result, ["field_changes"]);
  const stats = (result && result.stats) || {};
  const count = (kind) => (Number.isFinite(Number(stats[kind])) ? Number(stats[kind]) : hunks.filter((hunk) => hunk.kind === kind).length);
  const summary = result && result.change_summary && typeof result.change_summary === "object" ? result.change_summary : null;
  const hunkId = (id) => `hunk-${String(id).replace(/[^A-Za-z0-9_-]/g, "")}`;
  const jumpTo = (id) => {
    const target = document.getElementById(hunkId(id));
    if (target) {
      target.scrollIntoView({ block: "center" });
      target.focus();
    }
  };
  const warnings = Array.isArray(result && result.warnings) ? result.warnings : [];

  return h(
    "div",
    { class: "stack" },
    result && result.base && result.target &&
      h("p", { class: "muted" }, `Comparing ${versionLabel(result.base)} with ${versionLabel(result.target)}.`),
    h(
      "div",
      { class: "stat-row" },
      stat("Added", count("added"), "success"),
      stat("Removed", count("removed"), "danger"),
      stat("Changed", count("changed"), "warning"),
      Number.isFinite(Number(stats.similarity)) && stat("Similarity", formatPercent(stats.similarity), "info"),
    ),
    warnings.length > 0 && callout("warning", null, h("ul", { class: "plain-list" }, warnings.map((w) => h("li", null, String(w))))),
    result && result.truncated === true &&
      callout("info", null, `Showing the first ${formatNumber(hunks.length)} of ${formatNumber(result.hunks_total)} differences.`),
    summary &&
      h(
        "article",
        { class: "ai-block" },
        h("header", { class: "ai-block-header" }, aiLabel("AI-generated summary of changes"), summary.model && h("span", { class: "muted small" }, String(summary.model))),
        String(summary.summary || "")
          .split(/\n{2,}/)
          .filter(Boolean)
          .map((paragraph) => h("p", { class: "ai-text" }, paragraph)),
        Array.isArray(summary.changes) && summary.changes.length > 0 &&
          h(
            "ul",
            { class: "key-points" },
            summary.changes.map((change) =>
              h(
                "li",
                null,
                String(change.text || ""),
                (Array.isArray(change.refs) ? change.refs : []).slice(0, 6).map((ref) =>
                  h("span", null, " ", button(`#${ref}`, { variant: "link", small: true, ariaLabel: `Show difference ${ref}`, onClick: () => jumpTo(ref) })),
                ),
              ),
            ),
          ),
        h("p", { class: "muted small" }, "Based only on the differences listed below. Check them before relying on the summary."),
      ),
    fields.length > 0 &&
      card({
        title: "Changed key facts",
        body: dataTable({
          caption: "Changed key facts",
          rows: fields,
          columns: [
            { label: "Field", primary: true, render: (f) => fieldLabel(f.field) },
            { label: "Before", render: (f) => fieldValues(f.before) },
            { label: "After", render: (f) => fieldValues(f.after) },
            {
              label: "Change",
              render: (f) =>
                h(
                  "span",
                  null,
                  badge(humanize(f.change || "changed"), f.change === "added" ? "success" : f.change === "removed" ? "danger" : "warning"),
                  f.delta_days !== null && f.delta_days !== undefined && Number.isFinite(Number(f.delta_days))
                    ? h("span", { class: "muted small" }, ` ${Number(f.delta_days) > 0 ? "+" : ""}${f.delta_days} days`)
                    : null,
                ),
            },
          ],
        }),
      }),
    card({
      title: "Text differences",
      description: hunks.length ? `${formatNumber(hunks.length)} difference${hunks.length === 1 ? "" : "s"} found.` : undefined,
      body: hunks.length
        ? h("ol", { class: "diff-list" }, hunks.slice(0, 500).map((hunk) => renderHunk(hunk, hunkId)))
        : emptyState({ title: "No text differences found.", icon: "check" }),
    }),
  );
}

function renderHunk(hunk, hunkId) {
  const kind = String(hunk.kind || "changed");
  const before = hunk.before ?? "";
  const after = hunk.after ?? "";
  const section = (hunk.after_ref && hunk.after_ref.section) || (hunk.before_ref && hunk.before_ref.section) || "";
  const pages = [
    refPage(hunk.before_ref) && `Before: ${refPage(hunk.before_ref)}`,
    refPage(hunk.after_ref) && `After: ${refPage(hunk.after_ref)}`,
    section && String(section),
  ].filter(Boolean);
  const tone = kind === "added" ? "success" : kind === "removed" ? "danger" : "warning";
  const inline = Array.isArray(hunk.inline) ? hunk.inline : [];
  return h(
    "li",
    { class: ["diff-hunk", `diff-${kind}`], id: hunk.id ? hunkId(hunk.id) : undefined, tabindex: "-1" },
    h(
      "div",
      { class: "diff-head" },
      badge(humanize(kind), tone),
      hunk.id && h("span", { class: "muted small" }, `#${hunk.id}`),
      pages.length > 0 && h("span", { class: "muted small" }, pages.join(" \u00b7 ")),
    ),
    kind === "changed" && inline.length > 0
      ? h(
          "p",
          { class: "diff-inline" },
          inline.map((part) =>
            part.op === "insert"
              ? h("ins", null, h("span", { class: "visually-hidden" }, "[added] "), String(part.text))
              : part.op === "delete"
                ? h("del", null, h("span", { class: "visually-hidden" }, "[removed] "), String(part.text))
                : String(part.text),
          ),
        )
      : [
          before && h("del", { class: "diff-before" }, h("span", { class: "visually-hidden" }, "Removed text: "), String(before)),
          after && h("ins", { class: "diff-after" }, h("span", { class: "visually-hidden" }, "Added text: "), String(after)),
        ],
  );
}

function stat(label, value, tone) {
  return h("div", { class: ["stat", `stat-${tone}`] }, h("span", { class: "stat-value" }, String(value)), h("span", { class: "stat-label" }, label));
}
