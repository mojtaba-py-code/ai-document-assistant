/**
 * Exports: create CSV/JSON exports (generated in the background) and download them while
 * they are valid. Downloads use the authenticated endpoint, so no token appears in a URL.
 *
 * @module views/exports
 */

import { api, apiPath, download, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { formatBytes, formatDateTime, formatNumber, humanize, relativeTime } from "../format.js";
import {
  button,
  card,
  dataTable,
  emptyState,
  errorCallout,
  field,
  formStatus,
  input,
  loadingBlock,
  pageHeader,
  poll,
  select,
  statusBadge,
  toast,
  withBusy,
} from "../ui.js";
import { docTypeOptions, documentPicker } from "./common.js";

const KINDS = [
  { value: "extracted_fields", label: "Extracted data (dates, amounts, parties)" },
  { value: "deadlines", label: "Upcoming deadlines" },
  { value: "document_report", label: "Report for one document" },
  { value: "search_results", label: "Search results" },
];
const KIND_LABELS = Object.fromEntries(KINDS.map((k) => [k.value, k.label.replace(/ \(.*\)$/, "")]));

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function exportsView(ctx) {
  const list = h("div", { "aria-live": "polite" });
  let exportsCache = [];

  const load = async () => {
    try {
      exportsCache = pageOf(await api.get("/api/v1/exports", { query: { limit: 50 }, signal: ctx.signal }), ["exports"]).items;
      mount(
        list,
        dataTable({
          caption: "Your exports",
          rows: exportsCache,
          empty: emptyState({ title: "No exports yet.", text: "Create one above. Exports expire automatically.", icon: "download" }),
          columns: [
            { label: "Export", primary: true, render: (e) => `${KIND_LABELS[e.kind] || humanize(e.kind)} (${String(e.format || "").toUpperCase()})` },
            { label: "Status", render: (e) => statusBadge(e.status) },
            { label: "Rows", render: (e) => formatNumber(e.row_count) },
            { label: "Size", render: (e) => formatBytes(e.size_bytes) },
            { label: "Created", render: (e) => formatDateTime(e.created_at) },
            { label: "Expires", render: (e) => (e.status === "expired" ? "Expired" : e.expires_at ? relativeTime(e.expires_at) : null) },
            {
              label: "Actions",
              render: (e) =>
                e.status === "ready"
                  ? button("Download", {
                      small: true,
                      icon: "download",
                      onClick: (event) =>
                        withBusy(event.currentTarget, async () => {
                          try {
                            await download(apiPath("/api/v1/exports", e.id, "download"), { fallbackName: `${e.kind || "export"}.${e.format || "csv"}` });
                            load();
                          } catch (error) {
                            toast(error && error.detail ? error.detail : "The download failed.", "danger");
                          }
                        }, "Downloading\u2026"),
                    })
                  : e.status === "failed"
                    ? h("span", { class: "muted small" }, e.error_code ? humanize(e.error_code) : "Failed")
                    : null,
            },
          ],
        }),
      );
    } catch (error) {
      if (!ctx.signal.aborted) mount(list, errorCallout(error, { retry: load }));
    }
  };

  poll(
    ctx.signal,
    async () => {
      if (exportsCache.some((e) => e.status === "pending")) await load();
      return true;
    },
    3000,
  );

  mount(list, loadingBlock("Loading exports\u2026"));
  load();

  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Exports", subtitle: "Download extracted data as CSV (Excel-safe) or JSON. Only documents you can read are included." }),
    card({ title: "New export", body: createForm(load) }),
    card({ title: "Your exports", description: "Only you can download your exports.", body: list }),
  );
}

function createForm(onCreated) {
  const status = formStatus();
  const kind = select(KINDS, { name: "kind" });
  const format = select(
    [
      { value: "csv", label: "CSV (opens in Excel)" },
      { value: "json", label: "JSON" },
    ],
    { name: "format" },
  );
  const paramsArea = h("div");
  let params = {};
  let chosenDocument = null;

  const renderParams = () => {
    params = {};
    chosenDocument = null;
    if (kind.value === "extracted_fields") {
      const docType = select(docTypeOptions({ any: "All document types" }));
      docType.addEventListener("change", () => (params.doc_type = docType.value || undefined));
      mount(paramsArea, field("Document type", docType));
    } else if (kind.value === "deadlines") {
      const within = select([30, 60, 90, 180, 365].map((d) => ({ value: String(d), label: `Next ${d} days` })), { value: "90" });
      params.within_days = 90;
      try {
        params.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
      } catch {
        params.timezone = "UTC";
      }
      within.addEventListener("change", () => (params.within_days = Number(within.value)));
      mount(paramsArea, field("Period", within));
    } else if (kind.value === "document_report") {
      mount(paramsArea, documentPicker({ label: "Document", onSelect: (doc) => (chosenDocument = doc) }));
    } else {
      const query = input({ maxlength: 1000, placeholder: "Words to search for", required: true });
      query.addEventListener("input", () => (params.query = query.value.trim()));
      mount(paramsArea, field("Search for", query, { hint: "Tip: you can also export directly from the Search page." }));
    }
  };
  kind.addEventListener("change", renderParams);
  renderParams();

  const submit = button("Create export", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    h("div", { class: "field-row" }, field("What to export", kind), field("Format", format)),
    paramsArea,
    status.element,
    h("div", { class: "form-actions" }, submit),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    const body = { kind: kind.value, format: format.value, params: { ...params } };
    if (kind.value === "document_report") {
      if (!chosenDocument) {
        status.info("Choose a document first.");
        return;
      }
      body.params.document_id = String(chosenDocument.id);
    }
    if (kind.value === "search_results") body.params.mode = "hybrid";
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/exports", body);
        status.success("Export started. It appears below and is ready to download in a moment.");
        onCreated();
      } catch (error) {
        status.error(error);
      }
    }, "Creating\u2026");
  });
  return form;
}
