/**
 * Audit log viewer with filters and hash-chain verification.
 *
 * Event details are shown as formatted JSON text (never interpreted). The "Verify
 * integrity" action asks the server to recompute the HMAC hash chain and reports whether
 * any sealed record was altered or removed.
 *
 * @module views/audit
 */

import { api, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { formatDateTime, formatNumber, humanize, roleLabel, shortId } from "../format.js";
import { updateQuery } from "../router.js";
import { can } from "../session.js";
import {
  badge,
  button,
  callout,
  dataTable,
  debounce,
  descriptionList,
  emptyState,
  errorCallout,
  field,
  input,
  loadingBlock,
  loadMoreButton,
  pageHeader,
  select,
  statusBadge,
  withBusy,
} from "../ui.js";

const FILTER_KEYS = ["action", "actor", "resource_type", "resource_id", "outcome", "from", "to"];
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function auditView(ctx) {
  const filters = Object.fromEntries(FILTER_KEYS.map((k) => [k, ctx.query.get(k) || ""]));
  const results = h("div", { "aria-live": "polite" });
  const verifyArea = h("div");
  let cursor = null;
  let table = null;

  const columns = [
    { label: "Time", primary: true, render: (e) => h("time", { datetime: String(e.occurred_at || "") }, formatDateTime(e.occurred_at)) },
    { label: "Action", render: (e) => h("span", { class: "cell-title" }, h("span", null, humanize(String(e.action || "").replace(/\./g, " "))), h("code", { class: "muted small" }, String(e.action || ""))) },
    {
      label: "Actor",
      render: (e) =>
        e.actor_email || e.actor_user_id
          ? h("span", { class: "cell-title" }, h("span", null, String(e.actor_email || shortId(e.actor_user_id))), e.actor_role && h("span", { class: "muted small" }, roleLabel(e.actor_role)))
          : "System",
    },
    { label: "Resource", render: (e) => (e.resource_type ? `${humanize(e.resource_type)} ${e.resource_id ? shortId(e.resource_id) : ""}`.trim() : null) },
    { label: "Outcome", render: (e) => statusBadge(e.outcome) },
    { label: "Sealed", render: (e) => (e.seal_seq !== undefined && e.seal_seq !== null ? badge(`#${e.seal_seq}`, "success") : e.sealed === false ? badge("Pending", "neutral") : null) },
    {
      label: "Details",
      render: (e) => {
        const details = e.details && typeof e.details === "object" ? e.details : null;
        const extra = { request_id: e.request_id, ip_prefix: e.actor_ip_prefix, ...(details || {}) };
        const entries = Object.entries(extra).filter(([, v]) => v !== undefined && v !== null && v !== "");
        if (!entries.length) return null;
        return h("details", { class: "details details-inline" }, h("summary", null, "Show"), h("pre", { class: "json" }, JSON.stringify(Object.fromEntries(entries), null, 2)));
      },
    },
  ];

  const query = (after) => ({
    action: filters.action || undefined,
    actor: UUID_PATTERN.test(filters.actor) ? filters.actor : undefined,
    resource_type: filters.resource_type || undefined,
    resource_id: filters.resource_id || undefined,
    outcome: filters.outcome || undefined,
    from: filters.from ? new Date(`${filters.from}T00:00:00`).toISOString() : undefined,
    to: filters.to ? new Date(`${filters.to}T23:59:59`).toISOString() : undefined,
    cursor: after || undefined,
    limit: 50,
  });

  const load = async () => {
    updateQuery(filters);
    mount(results, loadingBlock("Loading audit events\u2026"));
    try {
      const page = pageOf(await api.get("/api/v1/audit/events", { query: query(null), signal: ctx.signal }), ["events"]);
      cursor = page.next;
      table = dataTable({
        caption: "Audit events",
        columns,
        rows: page.items,
        rowClass: (e) => (e.outcome === "denied" || e.outcome === "failure" ? "row-warning" : null),
        empty: emptyState({ title: "No events match these filters.", icon: "shield" }),
      });
      mount(
        results,
        table,
        cursor &&
          loadMoreButton(async () => {
            const more = pageOf(await api.get("/api/v1/audit/events", { query: query(cursor), signal: ctx.signal }), ["events"]);
            cursor = more.next;
            table.appendRows(more.items);
            return Boolean(cursor);
          }),
      );
    } catch (error) {
      if (!ctx.signal.aborted) mount(results, errorCallout(error, { retry: load }));
    }
  };

  const bindInput = (control, key) => {
    control.addEventListener(
      control.tagName === "SELECT" || control.type === "date" ? "change" : "input",
      debounce(() => {
        filters[key] = control.value.trim();
        load();
      }, control.tagName === "SELECT" ? 0 : 400),
    );
    return control;
  };

  const verify = button("Verify integrity", {
    icon: "shield",
    onClick: () =>
      withBusy(verify, async () => {
        mount(verifyArea, loadingBlock("Recomputing the hash chain\u2026"));
        try {
          const report = await api.post("/api/v1/audit/verify", {}, { timeoutMs: 300_000 });
          mount(verifyArea, verificationReport(report));
        } catch (error) {
          mount(verifyArea, errorCallout(error, { title: "Verification could not run" }));
        }
      }, "Verifying\u2026"),
  });

  load();
  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Audit log",
      subtitle: "Every security-relevant action, sealed in a tamper-evident hash chain.",
      actions: can("audit:verify") ? [verify] : [],
    }),
    verifyArea,
    h(
      "form",
      { class: "filters", role: "search", "aria-label": "Filter audit events", on: { submit: (e) => e.preventDefault() } },
      field("Action", bindInput(input({ value: filters.action, placeholder: "e.g. document.download", maxlength: 64 }), "action")),
      field(
        "Actor (user ID)",
        bindInput(input({ value: filters.actor, maxlength: 36, pattern: UUID_PATTERN.source, placeholder: "Full user ID" }), "actor"),
        { hint: "Paste the complete ID; partial IDs are ignored." },
      ),
      field("Resource type", bindInput(input({ value: filters.resource_type, placeholder: "e.g. document", maxlength: 32 }), "resource_type")),
      field("Resource ID", bindInput(input({ value: filters.resource_id, maxlength: 64 }), "resource_id")),
      field(
        "Outcome",
        bindInput(
          select(
            [
              { value: "", label: "Any outcome" },
              { value: "success", label: "Success" },
              { value: "denied", label: "Denied" },
              { value: "failure", label: "Failure" },
            ],
            { value: filters.outcome },
          ),
          "outcome",
        ),
      ),
      field("From", bindInput(input({ type: "date", value: filters.from }), "from")),
      field("To", bindInput(input({ type: "date", value: filters.to }), "to")),
    ),
    results,
  );
}

/**
 * Renders the verification report returned by `POST /api/v1/audit/verify`
 * (`{chain, valid, checked, head_seq, unsealed, first_bad_seq, reason, verified_at}`).
 * @param {any} report
 * @returns {HTMLElement}
 */
function verificationReport(report) {
  const data = report && typeof report === "object" ? report : {};
  const ok = data.valid === true;
  return callout(
    ok ? "success" : "danger",
    ok ? "The audit log is intact" : "Integrity problem detected",
    [
      h(
        "p",
        null,
        ok
          ? "Every sealed record matches its hash chain; nothing was altered or removed."
          : `The hash chain breaks${data.first_bad_seq !== null && data.first_bad_seq !== undefined ? ` at record #${data.first_bad_seq}` : ""}. Escalate to your security team.`,
      ),
      descriptionList([
        ["Chain", data.chain ? humanize(data.chain) : null],
        ["Records checked", Number.isFinite(Number(data.checked)) ? formatNumber(data.checked) : null],
        ["Latest sealed record", data.head_seq !== undefined && data.head_seq !== null ? `#${data.head_seq}` : null],
        ["Not yet sealed", Number.isFinite(Number(data.unsealed)) ? formatNumber(data.unsealed) : null],
        ["Reason", data.reason ? String(data.reason) : null],
        ["Verified at", data.verified_at ? formatDateTime(data.verified_at) : null],
      ]),
    ],
    { live: true },
  );
}
