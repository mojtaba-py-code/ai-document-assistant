/**
 * Helpers shared by several views: option lists, a document picker and small renderers.
 *
 * @module views/common
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount, uid } from "../dom.js";
import { CLASSIFICATIONS, classificationRank, DOC_TYPES, formatDate } from "../format.js";
import { session } from "../session.js";
import { badge, classificationBadge, debounce, errorMessage, input, spinner, statusBadge } from "../ui.js";

/**
 * Classification options up to the user's clearance (the API rejects anything higher).
 * @param {{any?: string}} [options] include a leading "any" option with this label
 * @returns {{value: string, label: string}[]}
 */
export function classificationOptions(options = {}) {
  const ceiling = session.user ? classificationRank(session.user.clearance) : CLASSIFICATIONS.length - 1;
  const list = CLASSIFICATIONS.filter((_, index) => ceiling < 0 || index <= ceiling).map((c) => ({
    value: c.value,
    label: c.label,
  }));
  return options.any ? [{ value: "", label: options.any }, ...list] : list;
}

/**
 * @param {{any?: string}} [options]
 * @returns {{value: string, label: string}[]}
 */
export function docTypeOptions(options = {}) {
  const list = DOC_TYPES.map((t) => ({ value: t.value, label: t.label }));
  return options.any !== undefined ? [{ value: "", label: options.any }, ...list] : list;
}

/**
 * Departments the user may file documents under: all for organisation admins, otherwise
 * their own memberships.
 * @param {any[]} departments
 * @returns {any[]}
 */
export function uploadableDepartments(departments) {
  if (!session.user) return [];
  if (session.user.role === "organization_admin") return departments;
  return departments.filter((d) => session.user.department_ids.includes(d.id));
}

/**
 * @param {any[]} departments
 * @param {string} emptyLabel
 * @returns {{value: string, label: string}[]}
 */
export function departmentOptions(departments, emptyLabel) {
  return [
    { value: "", label: emptyLabel },
    ...departments.filter((d) => d && d.id).map((d) => ({ value: String(d.id), label: String(d.name || "Department") })),
  ];
}

/**
 * Link to a document's detail page.
 * @param {{id?: string, title?: string, document_id?: string, document_title?: string}} doc
 * @param {Record<string, any>} [query]
 * @returns {HTMLElement}
 */
export function documentLink(doc, query) {
  const id = doc.document_id || doc.id;
  const title = String(doc.document_title || doc.title || "Untitled document");
  if (!id || !/^[A-Za-z0-9_-]{1,80}$/.test(String(id))) return h("span", null, title);
  const params = new URLSearchParams();
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    }
  }
  const qs = params.toString();
  return h("a", { href: `#/documents/${encodeURIComponent(String(id))}${qs ? `?${qs}` : ""}`, class: "doc-link" }, title);
}

/**
 * Tag chips.
 * @param {unknown} tags
 * @returns {HTMLElement | null}
 */
export function tagList(tags) {
  if (!Array.isArray(tags) || !tags.length) return null;
  return h(
    "ul",
    { class: "tags", "aria-label": "Tags" },
    tags.slice(0, 20).map((tag) => h("li", { class: "tag" }, String(tag))),
  );
}

/**
 * Compact metadata line for a document summary.
 * @param {any} doc
 * @returns {HTMLElement}
 */
export function documentBadges(doc) {
  return h(
    "span",
    { class: "badge-row" },
    doc.classification && classificationBadge(doc.classification),
    doc.status && doc.status !== "ready" && statusBadge(doc.status),
    doc.is_current === false && badge("Older version", "warning"),
  );
}

/**
 * Document picker: a search box listing matching documents the user can see.
 * Calls `onSelect(doc)` with the chosen summary.
 * @param {{label: string, onSelect: (doc: any) => void, initial?: any}} options
 * @returns {HTMLElement}
 */
export function documentPicker(options) {
  const listId = uid("picker");
  const statusId = uid("picker-status");
  const search = input({
    type: "search",
    placeholder: "Type part of a title\u2026",
    autocomplete: "off",
    "aria-controls": listId,
    "aria-describedby": statusId,
  });
  search.id = uid("picker-input");
  const results = h("ul", { class: "picker-results", id: listId });
  const statusLine = h("p", { class: "field-hint", id: statusId, "aria-live": "polite" });
  const chosen = h("div", { class: "picker-chosen" });
  let requestSeq = 0;

  const choose = (doc) => {
    mount(
      chosen,
      h("span", { class: "picker-chosen-label" }, "Selected: "),
      h("strong", null, String(doc.title || "Untitled document")),
      doc.classification && classificationBadge(doc.classification),
    );
    results.replaceChildren();
    statusLine.textContent = "";
    options.onSelect(doc);
  };

  const runSearch = debounce(async () => {
    const q = search.value.trim();
    const seq = ++requestSeq;
    if (q.length < 2) {
      results.replaceChildren();
      statusLine.textContent = q ? "Type at least two characters." : "";
      return;
    }
    mount(statusLine, spinner("Searching"), " Searching\u2026");
    try {
      const page = pageOf(await api.get("/api/v1/documents", { query: { q, limit: 10, status: "ready" } }), ["documents"]);
      if (seq !== requestSeq) return;
      statusLine.textContent = page.items.length ? `${page.items.length} matching documents` : "No matching documents.";
      mount(
        results,
        page.items.map((doc) =>
          h(
            "li",
            null,
            h(
              "button",
              { type: "button", class: "picker-option", on: { click: () => choose(doc) } },
              h("span", { class: "picker-title" }, String(doc.title || "Untitled document")),
              h("span", { class: "picker-meta" }, formatDate(doc.updated_at || doc.created_at)),
            ),
          ),
        ),
      );
    } catch (error) {
      if (seq === requestSeq) statusLine.textContent = errorMessage(error);
    }
  }, 300);
  search.addEventListener("input", runSearch);
  if (options.initial) choose(options.initial);

  return h(
    "div",
    { class: "picker" },
    h("label", { for: search.id, class: "field-label" }, options.label),
    search,
    statusLine,
    results,
    chosen,
  );
}

/**
 * Loads a document summary/detail by id (used to pre-fill pickers from the URL).
 * @param {string} id
 * @param {AbortSignal} [signal]
 * @returns {Promise<any | null>}
 */
export async function fetchDocument(id, signal) {
  if (!id || !/^[A-Za-z0-9_-]{1,80}$/.test(id)) return null;
  try {
    return await api.get(apiPath("/api/v1/documents", id), { signal });
  } catch {
    return null;
  }
}

/**
 * Versions of a document as a list sorted newest first.
 * @param {any} detail
 * @returns {any[]}
 */
export function versionsOf(detail) {
  const versions = detail && Array.isArray(detail.versions) ? detail.versions.slice() : [];
  versions.sort((a, b) => Number(b.version_number || 0) - Number(a.version_number || 0));
  return versions;
}

/**
 * The version currently served for search and the assistant (the latest indexed one).
 * @param {any} detail document detail
 * @returns {any | null}
 */
export function currentVersionOf(detail) {
  const versions = versionsOf(detail);
  return (
    versions.find((v) => v.is_current === true) ||
    versions.find((v) => v.id && detail && v.id === detail.current_version_id) ||
    versions.find((v) => v.status === "indexed") ||
    null
  );
}

/**
 * The version number considered "current" for a document detail.
 * @param {any} detail
 * @returns {number | null}
 */
export function currentVersionNumber(detail) {
  const current = currentVersionOf(detail);
  return current ? Number(current.version_number) : null;
}

