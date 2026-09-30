/**
 * Background jobs (document processing, exports, index sync) with retry/cancel, plus a
 * system health summary for administrators. Only ids, kinds, statuses and error codes are
 * shown - job payload internals never reach the UI.
 *
 * @module views/admin-jobs
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { formatDateTime, formatNumber, humanize, JOB_STATUSES, relativeTime, shortId } from "../format.js";
import { updateQuery } from "../router.js";
import { can, hasRole } from "../session.js";
import {
  badge,
  button,
  card,
  checkbox,
  dataTable,
  emptyState,
  errorCallout,
  field,
  loadingBlock,
  loadMoreButton,
  pageHeader,
  poll,
  select,
  statusBadge,
  toast,
  withBusy,
} from "../ui.js";

const KIND_LABELS = {
  ingest_version: "Process document",
  export_generate: "Generate export",
  vector_sync_version: "Sync search index (version)",
  vector_sync_document: "Sync search index (document)",
  vector_delete_document: "Remove from search index",
};

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function jobsView(ctx) {
  const canManage = can("jobs:manage");
  const filters = { status: ctx.query.get("status") || "", kind: ctx.query.get("kind") || "" };
  const results = h("div", { "aria-live": "polite" });
  const autoRefresh = checkbox("Refresh automatically", { checked: true });
  let cursor = null;

  const act = async (control, job, action) => {
    await withBusy(control, async () => {
      try {
        await api.post(apiPath("/api/v1/jobs", job.id, action));
        toast(action === "retry" ? "The job was queued again." : "The job was cancelled.", "success");
        load();
      } catch (error) {
        toast(error && error.detail ? error.detail : "The action failed.", "danger");
      }
    });
  };

  const columns = [
    {
      label: "Job",
      primary: true,
      render: (j) => {
        const docId = j.resource_ids && typeof j.resource_ids === "object" ? j.resource_ids.document_id : null;
        return h(
          "div",
          { class: "cell-title" },
          h("span", null, KIND_LABELS[j.kind] || humanize(j.kind)),
          h("span", { class: "muted small mono" }, shortId(j.id)),
          docId && /^[A-Za-z0-9_-]{1,80}$/.test(String(docId)) && h("a", { href: `#/documents/${encodeURIComponent(String(docId))}`, class: "small" }, "Open document"),
        );
      },
    },
    { label: "Status", render: (j) => statusBadge(j.status) },
    { label: "Attempts", render: (j) => `${Number(j.attempts) || 0}${j.max_attempts ? ` / ${j.max_attempts}` : ""}` },
    { label: "Error", render: (j) => (j.last_error_code || j.error_code ? badge(humanize(j.last_error_code || j.error_code), "danger") : null) },
    { label: "Created", render: (j) => h("time", { title: formatDateTime(j.created_at) }, relativeTime(j.created_at)) },
    { label: "Finished", render: (j) => (j.finished_at ? formatDateTime(j.finished_at) : null) },
    {
      label: "Actions",
      render: (j) => {
        if (!canManage) return null;
        if (j.status === "dead") return button("Retry", { small: true, ariaLabel: `Retry job ${shortId(j.id)}`, onClick: (e) => act(e.currentTarget, j, "retry") });
        if (j.status === "queued") return button("Cancel", { small: true, variant: "ghost", ariaLabel: `Cancel job ${shortId(j.id)}`, onClick: (e) => act(e.currentTarget, j, "cancel") });
        return null;
      },
    },
  ];

  const fetchPage = async (after) =>
    pageOf(await api.get("/api/v1/jobs", { query: { ...filters, cursor: after, limit: 50 }, signal: ctx.signal }), ["jobs"]);

  let table = null;
  const load = async (quiet = false) => {
    updateQuery(filters);
    if (!quiet) mount(results, loadingBlock("Loading jobs\u2026"));
    try {
      const page = await fetchPage(null);
      cursor = page.next;
      table = dataTable({
        caption: "Background jobs",
        columns,
        rows: page.items,
        rowClass: (j) => (j.status === "dead" ? "row-danger" : null),
        empty: emptyState({ title: "No jobs match.", text: "Jobs appear when documents are uploaded or exports are created.", icon: "jobs" }),
      });
      mount(
        results,
        table,
        cursor &&
          loadMoreButton(async () => {
            const more = await fetchPage(cursor);
            cursor = more.next;
            table.appendRows(more.items);
            return Boolean(cursor);
          }),
      );
    } catch (error) {
      if (!ctx.signal.aborted && !quiet) mount(results, errorCallout(error, { retry: () => load() }));
    }
  };

  poll(
    ctx.signal,
    async () => {
      if (autoRefresh.querySelector("input").checked && !cursor) await load(true);
      return true;
    },
    5000,
  );

  const status = select([{ value: "", label: "Any status" }, ...JOB_STATUSES], { value: filters.status });
  status.addEventListener("change", () => {
    filters.status = status.value;
    load();
  });
  const kind = select([{ value: "", label: "Any kind" }, ...Object.entries(KIND_LABELS).map(([value, label]) => ({ value, label }))], { value: filters.kind });
  kind.addEventListener("change", () => {
    filters.kind = kind.value;
    load();
  });

  load();
  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Background jobs", subtitle: "Document processing, exports and index maintenance for your organisation." }),
    (hasRole("organization_admin") || can("platform:health")) && healthCard(ctx.signal),
    h(
      "form",
      { class: "filters", "aria-label": "Filter jobs", on: { submit: (e) => e.preventDefault() } },
      field("Status", status),
      field("Kind", kind),
      h("div", { class: "filters-extra" }, autoRefresh),
    ),
    results,
  );
}

const HEALTH_TONES = { ok: "success", degraded: "warning", unavailable: "danger", not_configured: "neutral", unknown: "neutral" };

/**
 * System health (`GET /api/v1/admin/health`): overall status, each component and the
 * job queue depth by status. Component details are shown as plain key/value text.
 * @param {AbortSignal} signal
 * @returns {HTMLElement}
 */
function healthCard(signal) {
  const body = h("div", null, loadingBlock("Checking system health\u2026"));
  const load = async () => {
    try {
      const health = (await api.get("/api/v1/admin/health", { signal })) || {};
      const components = health.components && typeof health.components === "object" ? Object.entries(health.components) : [];
      const depth = health.queue && health.queue.depth && typeof health.queue.depth === "object" ? Object.entries(health.queue.depth) : [];
      mount(
        body,
        h(
          "p",
          { class: "health-summary" },
          "Overall: ",
          badge(humanize(health.status || "unknown"), HEALTH_TONES[health.status] || "neutral"),
          health.checked_at ? h("span", { class: "muted small" }, ` checked ${formatDateTime(health.checked_at)}`) : null,
        ),
        components.length > 0 &&
          h(
            "ul",
            { class: "health-list" },
            components.map(([name, component]) => {
              const state = component && typeof component === "object" ? component : { status: String(component) };
              const details =
                state.detail && typeof state.detail === "object"
                  ? Object.entries(state.detail)
                      .slice(0, 4)
                      .map(([key, value]) => `${humanize(key)}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
                      .join(" \u00b7 ")
                  : "";
              return h(
                "li",
                { class: "health-item" },
                h("span", { class: "cell-title" }, h("span", null, humanize(name)), details && h("span", { class: "muted small" }, details)),
                badge(humanize(state.status || "unknown"), HEALTH_TONES[state.status] || "neutral"),
              );
            }),
          ),
        depth.length > 0 &&
          h(
            "p",
            { class: "muted small" },
            `Job queue${health.queue.scope === "platform" ? " (all organisations)" : ""}: `,
            depth.map(([state, count]) => `${humanize(state)} ${formatNumber(count)}`).join(", "),
          ),
      );
    } catch (error) {
      mount(body, errorCallout(error, { retry: load }));
    }
  };
  load();
  return card({ title: "System health", actions: [button("Refresh", { small: true, variant: "ghost", onClick: load })], body });
}
