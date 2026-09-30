/**
 * Upcoming deadlines (contract expirations, renewals, invoice due dates) found in the
 * documents the user can read. Every row links to the page holding the evidence.
 *
 * @module views/deadlines
 */

import { api, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { daysUntil, fieldLabel, formatDate, pageRef } from "../format.js";
import { navigate, updateQuery } from "../router.js";
import { can, putHandoff } from "../session.js";
import {
  badge,
  button,
  checkbox,
  dataTable,
  emptyState,
  errorCallout,
  field,
  loadingBlock,
  pageHeader,
  select,
  toast,
  withBusy,
} from "../ui.js";
import { docTypeOptions, documentLink } from "./common.js";

const WINDOWS = [30, 60, 90, 180, 365];
/** The viewer's IANA time zone, so "days left" matches their calendar. */
const TIME_ZONE = (() => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
})();

/**
 * @param {number | null} days
 * @returns {HTMLElement}
 */
function daysBadge(days) {
  if (days === null) return badge("Unknown", "neutral");
  if (days < 0) return badge(`${Math.abs(days)} day${days === -1 ? "" : "s"} overdue`, "danger");
  if (days === 0) return badge("Today", "danger");
  if (days <= 14) return badge(`In ${days} day${days === 1 ? "" : "s"}`, "danger");
  if (days <= 45) return badge(`In ${days} days`, "warning");
  return badge(`In ${days} days`, "success");
}

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function deadlinesView(ctx) {
  const initialWindow = Number(ctx.query.get("within")) || 90;
  const within = select(
    WINDOWS.map((d) => ({ value: String(d), label: `Next ${d} days` })),
    { value: String(WINDOWS.includes(initialWindow) ? initialWindow : 90) },
  );
  const docType = select(docTypeOptions({ any: "All document types" }), { value: ctx.query.get("type") || "" });
  const overdue = checkbox("Include overdue (last 30 days)", { checked: ctx.query.get("overdue") !== "0" });
  const overdueDays = () => (overdue.querySelector("input").checked ? 30 : 0);
  const summary = h("div", { class: "stat-row" });
  const results = h("div", { "aria-live": "polite", "aria-busy": "false" });

  const load = async () => {
    updateQuery({ within: within.value === "90" ? null : within.value, type: docType.value || null, overdue: overdueDays() ? null : "0" });
    results.setAttribute("aria-busy", "true");
    mount(results, loadingBlock("Looking for deadlines\u2026"));
    try {
      const response = await api.get("/api/v1/intelligence/deadlines", {
        query: {
          within_days: within.value,
          overdue_days: overdueDays(),
          doc_type: docType.value || undefined,
          tz: TIME_ZONE,
        },
        signal: ctx.signal,
      });
      const rows = pageOf(response, ["deadlines", "rows"]).items.map((row) => ({
        ...row,
        _days: Number.isFinite(Number(row.days_left)) ? Number(row.days_left) : daysUntil(row.date || row.value_date),
      }));
      rows.sort((a, b) => (a._days ?? 1e9) - (b._days ?? 1e9));
      const overdue = rows.filter((r) => r._days !== null && r._days < 0).length;
      const soon = rows.filter((r) => r._days !== null && r._days >= 0 && r._days <= 30).length;
      mount(
        summary,
        stat("Overdue", overdue, overdue ? "danger" : "neutral"),
        stat("Within 30 days", soon, soon ? "warning" : "neutral"),
        stat("In this period", rows.length, "info"),
      );
      mount(
        results,
        dataTable({
          caption: "Upcoming deadlines",
          rows,
          empty: emptyState({
            title: "No deadlines in this period.",
            text: "Deadlines are detected in contracts and invoices you can read. Try a longer period.",
            icon: "calendar",
          }),
          rowClass: (r) => (r._days !== null && r._days < 0 ? "row-danger" : null),
          columns: [
            { label: "Date", primary: true, render: (r) => formatDate(r.date || r.value_date) },
            { label: "When", render: (r) => daysBadge(r._days) },
            { label: "Document", render: (r) => documentLink(r) },
            { label: "Deadline", render: (r) => fieldLabel(r.field) },
            { label: "Evidence", className: "col-evidence", render: (r) => (r.evidence ? h("q", { class: "evidence-quote" }, String(r.evidence)) : null) },
            {
              label: "Source",
              render: (r) =>
                r.document_id && Number(r.page) > 0
                  ? button(`Open ${pageRef(r.page)}`, {
                      small: true,
                      variant: "link",
                      onClick: () => {
                        putHandoff(`highlight:${r.document_id}`, { quote: String(r.evidence || ""), chunkId: r.chunk_id || "" });
                        navigate(`/documents/${r.document_id}`, { tab: "preview", page: Number(r.page) > 1 ? Number(r.page) : undefined });
                      },
                    })
                  : null,
            },
          ],
        }),
      );
    } catch (error) {
      if (!ctx.signal.aborted) mount(results, errorCallout(error, { retry: load }));
    } finally {
      results.setAttribute("aria-busy", "false");
    }
  };

  within.addEventListener("change", load);
  docType.addEventListener("change", load);
  overdue.querySelector("input").addEventListener("change", load);

  const exportButton =
    can("export:create") &&
    button("Export CSV", {
      icon: "download",
      onClick: (event) =>
        withBusy(event.currentTarget, async () => {
          try {
            await api.post("/api/v1/exports", {
              kind: "deadlines",
              format: "csv",
              params: { within_days: Number(within.value), overdue_days: overdueDays(), doc_type: docType.value || null, timezone: TIME_ZONE },
            });
            toast("Export started. Download it from the Exports page when it is ready.", "success");
          } catch (error) {
            toast(error && error.detail ? error.detail : "The export could not be created.", "danger");
          }
        }, "Exporting\u2026"),
    });

  load();
  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Deadlines",
      subtitle: "Expirations, renewals and payment due dates found in your documents.",
      actions: [exportButton].filter(Boolean),
    }),
    h("form", { class: "filters", "aria-label": "Deadline filters", on: { submit: (e) => e.preventDefault() } }, field("Period", within), field("Document type", docType), h("div", { class: "filters-extra" }, overdue)),
    summary,
    results,
  );
}

function stat(label, value, tone) {
  return h("div", { class: ["stat", `stat-${tone}`] }, h("span", { class: "stat-value" }, String(value)), h("span", { class: "stat-label" }, label));
}
