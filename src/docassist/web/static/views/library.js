/**
 * Document library: filterable, paginated list with live ingestion status.
 *
 * Filters are mirrored in the URL (`#/documents?q=...&classification=...`) so a filtered
 * view survives reloads and can be bookmarked. Documents still processing are re-checked
 * in the background until they are ready (polling stops when the view is left).
 *
 * @module views/library
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { classificationLabel, classificationRank, docTypeLabel, DOCUMENT_STATUSES, relativeTime, formatDateTime } from "../format.js";
import { updateQuery } from "../router.js";
import { can, departmentName, getDepartments } from "../session.js";
import {
  announce,
  badge,
  button,
  callout,
  checkbox,
  classificationBadge,
  dataTable,
  debounce,
  emptyState,
  errorCallout,
  field,
  input,
  loadingBlock,
  loadMoreButton,
  pageHeader,
  poll,
  select,
  statusBadge,
} from "../ui.js";
import { classificationOptions, departmentOptions, docTypeOptions, documentLink, tagList } from "./common.js";
import { openUploadDialog } from "./upload.js";

const FILTER_KEYS = ["q", "doc_type", "classification", "department_id", "status", "tag", "owner"];
const PAGE_SIZE = 25;

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function libraryView(ctx) {
  const departments = await getDepartments();
  const filters = {};
  for (const key of FILTER_KEYS) filters[key] = ctx.query.get(key) || "";

  const results = h("div", { class: "results", "aria-live": "polite", "aria-busy": "false" });
  const summary = h("p", { class: "results-summary muted" });
  const banner = h("div");
  /** @type {Map<string, {doc: any, status: HTMLElement}>} */
  const tracked = new Map();
  let nextCursor = null;
  let loadSeq = 0;

  const columns = [
    {
      label: "Title",
      primary: true,
      className: "col-title",
      render: (doc) =>
        h(
          "div",
          { class: "cell-title" },
          documentLink(doc),
          suggestionFlag(doc),
          tagList(doc.tags),
        ),
    },
    { label: "Type", render: (doc) => docTypeLabel(doc.doc_type) },
    { label: "Classification", render: (doc) => classificationBadge(doc.classification) },
    {
      label: "Status",
      render: (doc) => {
        const holder = h("span", null, statusBadge(doc.status));
        if (doc.id) tracked.set(String(doc.id), { doc, status: holder });
        return holder;
      },
    },
    { label: "Department", render: (doc) => departmentName(departments, doc.department_id) },
    {
      label: "Updated",
      render: (doc) => {
        const when = doc.updated_at || doc.created_at;
        return h("time", { datetime: String(when || ""), title: formatDateTime(when) }, relativeTime(when));
      },
    },
  ];

  let table = null;

  const fetchPage = async (cursor) =>
    pageOf(
      await api.get("/api/v1/documents", {
        query: { ...filters, owner: filters.owner ? "me" : "", cursor, limit: PAGE_SIZE },
        signal: ctx.signal,
      }),
      ["documents"],
    );

  const renderBanner = (items) => {
    const needsReview = items.filter(
      (doc) => doc.can_manage && doc.suggested_classification && classificationRank(doc.suggested_classification) > classificationRank(doc.classification),
    );
    mount(
      banner,
      needsReview.length > 0 &&
        callout(
          "warning",
          `${needsReview.length === 1 ? "1 document may" : `${needsReview.length} documents may`} need a higher classification`,
          "Sensitive information (such as personal or financial data) was detected. Open the document to review the suggestion.",
        ),
    );
  };

  const load = async () => {
    const seq = ++loadSeq;
    tracked.clear();
    results.setAttribute("aria-busy", "true");
    mount(results, loadingBlock("Loading documents\u2026"));
    try {
      const page = await fetchPage(null);
      if (seq !== loadSeq) return;
      nextCursor = page.next;
      renderBanner(page.items);
      const hasFilters = FILTER_KEYS.some((k) => filters[k]);
      table = dataTable({
        caption: "Documents",
        columns,
        rows: page.items,
        rowClass: (doc) => (doc.status === "quarantined" ? "row-danger" : null),
        empty: emptyState(
          hasFilters
            ? { title: "No documents match these filters.", text: "Try removing a filter or searching for a different word.", icon: "search" }
            : {
                title: "No documents yet.",
                text: can("document:upload") ? "Upload your first document to get started." : "Documents shared with you will appear here.",
                icon: "folder",
                action: can("document:upload") ? button("Upload document", { variant: "primary", icon: "upload", onClick: startUpload }) : undefined,
              },
        ),
      });
      summary.textContent = page.items.length
        ? `Showing ${page.items.length}${page.total ? ` of ${page.total}` : ""} document${page.items.length === 1 ? "" : "s"}`
        : "";
      mount(
        results,
        table,
        nextCursor &&
          loadMoreButton(async () => {
            const more = await fetchPage(nextCursor);
            nextCursor = more.next;
            if (table && table.appendRows) table.appendRows(more.items);
            const shown = results.querySelectorAll("tbody tr").length;
            summary.textContent = `Showing ${shown} documents`;
            return Boolean(nextCursor);
          }),
      );
    } catch (error) {
      if (ctx.signal.aborted) return;
      mount(results, errorCallout(error, { retry: load }));
    } finally {
      results.setAttribute("aria-busy", "false");
    }
  };

  // Background status refresh for documents that are still being processed.
  poll(
    ctx.signal,
    async () => {
      const pending = [...tracked.entries()].filter(([, entry]) => entry.doc.status === "processing").slice(0, 10);
      for (const [id, entry] of pending) {
        try {
          const fresh = await api.get(apiPath("/api/v1/documents", id), { signal: ctx.signal });
          if (fresh && fresh.status && fresh.status !== entry.doc.status) {
            entry.doc.status = fresh.status;
            mount(entry.status, statusBadge(fresh.status));
            const title = String(fresh.title || entry.doc.title || "A document");
            if (fresh.status === "ready") announce(`"${title}" is ready.`);
            else if (fresh.status === "failed") announce(`Processing "${title}" failed.`);
            else if (fresh.status === "quarantined") announce(`"${title}" was held for security review.`);
          }
        } catch {
          // Transient errors are ignored; the next tick retries.
        }
      }
      return true;
    },
    4000,
  );

  const applyFilters = () => {
    updateQuery(Object.fromEntries(FILTER_KEYS.map((k) => [k, filters[k]])));
    load();
  };

  const bind = (control, key, { debounced = false } = {}) => {
    const handler = () => {
      filters[key] = control.type === "checkbox" ? (control.checked ? "me" : "") : control.value.trim();
      applyFilters();
    };
    control.addEventListener(control.type === "checkbox" || control.tagName === "SELECT" ? "change" : "input", debounced ? debounce(handler, 350) : handler);
    return control;
  };

  const search = bind(input({ type: "search", name: "q", value: filters.q, placeholder: "Search titles\u2026", maxlength: 200 }), "q", { debounced: true });
  const typeSelect = bind(select(docTypeOptions({ any: "All types" }), { value: filters.doc_type }), "doc_type");
  const classSelect = bind(select(classificationOptions({ any: "Any classification" }), { value: filters.classification }), "classification");
  const deptSelect = bind(select(departmentOptions(departments, "All departments"), { value: filters.department_id }), "department_id");
  const statusSelect = bind(
    select([{ value: "", label: "Any status" }, ...DOCUMENT_STATUSES.map((s) => ({ value: s.value, label: s.label }))], { value: filters.status }),
    "status",
  );
  const tagInput = bind(input({ type: "text", name: "tag", value: filters.tag, placeholder: "Tag", maxlength: 64 }), "tag", { debounced: true });
  const mine = checkbox("Only documents I own", { checked: filters.owner === "me" });
  bind(mine.querySelector("input"), "owner");

  const clear = button("Clear filters", {
    variant: "ghost",
    small: true,
    onClick: () => {
      for (const key of FILTER_KEYS) filters[key] = "";
      search.value = "";
      typeSelect.value = "";
      classSelect.value = "";
      deptSelect.value = "";
      statusSelect.value = "";
      tagInput.value = "";
      mine.querySelector("input").checked = false;
      applyFilters();
    },
  });

  const filterBar = h(
    "form",
    { class: "filters", role: "search", "aria-label": "Filter documents", on: { submit: (e) => e.preventDefault() } },
    field("Search", search, { className: "filter-grow" }),
    field("Type", typeSelect),
    field("Classification", classSelect),
    departments.length > 0 && field("Department", deptSelect),
    field("Status", statusSelect),
    field("Tag", tagInput),
    h("div", { class: "filters-extra" }, mine, clear),
  );

  function startUpload() {
    openUploadDialog({ departments, onUploaded: () => load() });
  }

  load();

  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Documents",
      subtitle: "Browse, upload and manage the documents you have access to.",
      actions: [
        can("document:upload") && button("Upload document", { variant: "primary", icon: "upload", onClick: startUpload }),
      ].filter(Boolean),
    }),
    filterBar,
    banner,
    summary,
    results,
  );
}

function suggestionFlag(doc) {
  if (!doc.suggested_classification) return null;
  if (classificationRank(doc.suggested_classification) <= classificationRank(doc.classification)) return null;
  return badge(`Suggested: ${classificationLabel(doc.suggested_classification)}`, "warning");
}
