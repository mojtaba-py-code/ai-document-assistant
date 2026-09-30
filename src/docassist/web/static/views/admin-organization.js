/**
 * Organisation settings (AI data ceiling, token budget, retention overrides) and monthly
 * AI usage.
 *
 * `GET /api/v1/organization` returns the tenant's own overrides (`settings`), the values in
 * force (`effective`) and the deployment limits (`deployment`). Overrides can only make the
 * policy stricter (the API enforces it); an empty field sends `null`, which removes the
 * override and falls back to the deployment value.
 *
 * @module views/admin-organization
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { classificationLabel, CLASSIFICATIONS, formatNumber, formatPercent, humanize } from "../format.js";
import { can, invalidateOrganization } from "../session.js";
import {
  button,
  callout,
  card,
  dataTable,
  descriptionList,
  errorCallout,
  field,
  formStatus,
  input,
  loadingBlock,
  pageHeader,
  select,
  statusBadge,
  toast,
  withBusy,
} from "../ui.js";

const RETENTION_FIELDS = [
  { key: "conversation_days", label: "Assistant conversations (days)", min: 1, max: 3650 },
  { key: "job_days", label: "Finished background jobs (days)", min: 1, max: 3650 },
  { key: "deleted_document_purge_days", label: "Deleted documents kept before purge (days)", min: 0, max: 3650 },
  { key: "llm_usage_days", label: "AI usage records (days, at least the default)", min: 30, max: 3650 },
  { key: "audit_days", label: "Audit log (days, at least the default)", min: 30, max: 36500 },
];

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function organizationView(ctx) {
  const settingsArea = h("div", null, loadingBlock("Loading organisation\u2026"));
  const usageArea = h("div");

  const loadOrganization = async () => {
    try {
      const data = await api.get("/api/v1/organization", { signal: ctx.signal });
      mount(settingsArea, organizationPanels(data, loadOrganization));
    } catch (error) {
      if (!ctx.signal.aborted) mount(settingsArea, errorCallout(error, { retry: loadOrganization }));
    }
  };

  const loadUsage = async () => {
    mount(usageArea, loadingBlock("Loading usage\u2026"));
    try {
      const usage = await api.get("/api/v1/usage", { signal: ctx.signal });
      mount(usageArea, usagePanel(usage));
    } catch (error) {
      if (!ctx.signal.aborted) mount(usageArea, card({ title: "AI usage this month", body: errorCallout(error, { retry: loadUsage }) }));
    }
  };

  if (can("org:read", "org:update")) loadOrganization();
  else settingsArea.replaceChildren();
  if (can("usage:read")) loadUsage();

  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Organisation", subtitle: "Settings that apply to everyone in your organisation." }),
    h("div", { class: "stack" }, settingsArea, usageArea),
  );
}

function numberOrEmpty(value) {
  return value === null || value === undefined ? "" : String(value);
}

function organizationPanels(data, reload) {
  const org = (data && data.organization) || data || {};
  const settings = (data && data.settings) || {};
  const effective = (data && data.effective) || {};
  const deployment = (data && data.deployment) || {};
  const llm = settings.llm || {};
  const retention = settings.retention || {};
  const deploymentRetention = deployment.retention || {};
  const editable = can("org:update");
  const deploymentRank = CLASSIFICATIONS.findIndex((c) => c.value === deployment.external_max_classification);

  const overview = card({
    title: "Overview",
    body: descriptionList([
      ["Name", org.name ? String(org.name) : null],
      ["Short name", org.slug ? String(org.slug) : null],
      ["Status", org.status ? statusBadge(org.status) : null],
      [
        "External AI may process",
        effective.external_max_classification ? `Documents up to ${classificationLabel(effective.external_max_classification)}` : null,
      ],
      [
        "Monthly AI token budget",
        effective.monthly_token_budget ? formatNumber(effective.monthly_token_budget) : effective.monthly_token_budget === null ? "Unlimited" : null,
      ],
    ]),
  });

  const status = formStatus();
  const ceilingOptions = [
    {
      value: "",
      label: deployment.external_max_classification
        ? `Deployment default (up to ${classificationLabel(deployment.external_max_classification)})`
        : "Deployment default",
    },
    ...CLASSIFICATIONS.filter((_, index) => deploymentRank < 0 || index <= deploymentRank).map((c) => ({ value: c.value, label: `Up to ${c.label}` })),
  ];
  const ceiling = select(ceilingOptions, { value: llm.external_max_classification || "", disabled: !editable });
  const budget = input({
    type: "number",
    min: "1",
    step: "1000",
    max: deployment.monthly_token_budget ? String(deployment.monthly_token_budget) : undefined,
    value: numberOrEmpty(llm.monthly_token_budget),
    placeholder: deployment.monthly_token_budget ? `Default: ${formatNumber(deployment.monthly_token_budget)}` : "Default: unlimited",
    disabled: !editable,
  });
  const retentionInputs = RETENTION_FIELDS.map((f) => ({
    ...f,
    control: input({
      type: "number",
      min: String(f.min),
      max: String(f.max),
      value: numberOrEmpty(retention[f.key]),
      placeholder: deploymentRetention[f.key] !== undefined ? `Default: ${deploymentRetention[f.key]}` : "Default",
      disabled: !editable,
    }),
  }));

  const save = button("Save settings", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    h("h3", { class: "subheading" }, "AI data protection"),
    field("Documents that may be sent to external AI services", ceiling, {
      hint: "More sensitive documents are answered by an in-house model or not at all. You can only make this stricter than the deployment setting.",
    }),
    field("Monthly AI token budget", budget, { hint: "Leave empty for the deployment default. Requests are refused once the budget is used up." }),
    h("h3", { class: "subheading" }, "Data retention"),
    h("p", { class: "muted small" }, "Leave a value empty to use the deployment default. Audit and usage records can only be kept longer, never shorter."),
    h("div", { class: "field-grid" }, retentionInputs.map((f) => field(f.label, f.control))),
    status.element,
    editable && h("div", { class: "form-actions" }, save),
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    const body = {
      llm: {
        external_max_classification: ceiling.value || null,
        monthly_token_budget: budget.value === "" ? null : Number(budget.value),
      },
      retention: Object.fromEntries(retentionInputs.map((f) => [f.key, f.control.value === "" ? null : Number(f.control.value)])),
    };
    await withBusy(save, async () => {
      try {
        await api.patch("/api/v1/organization/settings", body);
        invalidateOrganization();
        toast("Organisation settings saved.", "success");
        reload();
      } catch (error) {
        status.error(error);
      }
    }, "Saving\u2026");
  });

  return h(
    "div",
    { class: "stack" },
    overview,
    card({
      title: "Policies",
      description: editable ? undefined : "Only organisation administrators can change these settings.",
      body: form,
    }),
  );
}

function usagePanel(usage) {
  const data = usage && typeof usage === "object" ? usage : {};
  const totals = data.totals || {};
  const budget = data.budget || {};
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const used = Number(budget.used_tokens ?? totals.total_tokens);
  const limit = Number(budget.effective_limit);
  const remaining = Number(budget.remaining_tokens);
  const ratio = Number.isFinite(used) && Number.isFinite(limit) && limit > 0 ? Math.min(1, used / limit) : null;
  const cost = Number(totals.cost_usd);

  return card({
    title: `AI usage${data.month ? ` \u2014 ${data.month}` : " this month"}`,
    body: h(
      "div",
      { class: "stack" },
      h(
        "div",
        { class: "stat-row" },
        Number.isFinite(used) && stat("Tokens used", formatNumber(used)),
        Number.isFinite(limit) && limit > 0 ? stat("Monthly budget", formatNumber(limit)) : stat("Monthly budget", "Unlimited"),
        Number.isFinite(remaining) && budget.remaining_tokens !== null && stat("Remaining", formatNumber(remaining)),
        Number.isFinite(Number(totals.requests)) && stat("Requests", formatNumber(totals.requests)),
        Number.isFinite(cost) && stat("Estimated cost", `$${cost.toFixed(2)}`),
      ),
      ratio !== null &&
        h(
          "div",
          { class: "budget-bar" },
          h("label", { class: "field-label", for: "budget-progress" }, `Budget used: ${formatPercent(ratio)}`),
          h("progress", { id: "budget-progress", class: ["progress", ratio > 0.9 && "progress-danger"].filter(Boolean), max: "1", value: String(ratio) }),
        ),
      budget.exhausted === true
        ? callout("danger", "Budget used up", "AI requests are refused until next month or until the budget is raised.")
        : ratio !== null && ratio > 0.9 && callout("warning", null, "The monthly AI budget is almost used up. Requests will be refused when it runs out."),
      rows.length > 0 &&
        dataTable({
          caption: "Usage by model and task",
          rows,
          columns: [
            { label: "Model", primary: true, render: (r) => String(r.model || "") },
            { label: "Task", render: (r) => humanize(r.task) },
            { label: "Requests", render: (r) => formatNumber(r.requests) },
            { label: "Input tokens", render: (r) => formatNumber(r.input_tokens) },
            { label: "Output tokens", render: (r) => formatNumber(r.output_tokens) },
            { label: "Cost", render: (r) => (Number.isFinite(Number(r.cost_usd)) ? `$${Number(r.cost_usd).toFixed(2)}` : null) },
          ],
        }),
    ),
  });
}

function stat(label, value) {
  return h("div", { class: "stat stat-info" }, h("span", { class: "stat-value" }, String(value)), h("span", { class: "stat-label" }, label));
}

