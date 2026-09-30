/**
 * Platform operator view: organisations (tenants). Platform admins manage tenants but can
 * never read tenant documents; creating an organisation emails its first administrator a
 * link to set their own password.
 *
 * @module views/platform
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { formatDateTime, formatNumber } from "../format.js";
import { can } from "../session.js";
import {
  button,
  callout,
  confirmDialog,
  dataTable,
  emptyState,
  errorCallout,
  field,
  formStatus,
  input,
  loadingBlock,
  openDialog,
  pageHeader,
  statusBadge,
  toast,
  withBusy,
} from "../ui.js";
import { slugify } from "./admin-departments.js";

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function platformView(ctx) {
  const results = h("div", { "aria-live": "polite" });
  const canUpdate = can("org:update_any");

  const load = async () => {
    mount(results, loadingBlock("Loading organisations\u2026"));
    try {
      const items = pageOf(await api.get("/api/v1/platform/organizations", { query: { limit: 100 }, signal: ctx.signal }), ["organizations"]).items;
      mount(
        results,
        dataTable({
          caption: "Organisations",
          rows: items,
          empty: emptyState({ title: "No organisations yet.", icon: "globe" }),
          columns: [
            { label: "Name", primary: true, render: (o) => h("span", { class: "cell-title" }, h("span", null, String(o.name || "Organisation")), h("span", { class: "muted small" }, String(o.slug || ""))) },
            { label: "Status", render: (o) => statusBadge(o.status) },
            { label: "Users", render: (o) => (o.user_count !== undefined ? formatNumber(o.user_count) : null) },
            { label: "Created", render: (o) => formatDateTime(o.created_at) },
            {
              label: "Actions",
              render: (o) => {
                if (!canUpdate) return null;
                const suspended = o.status === "suspended";
                return button(suspended ? "Reactivate" : "Suspend", {
                  small: true,
                  variant: suspended ? "secondary" : "ghost",
                  ariaLabel: `${suspended ? "Reactivate" : "Suspend"} ${o.name}`,
                  onClick: async (event) => {
                    const control = event.currentTarget;
                    const ok = await confirmDialog({
                      title: suspended ? `Reactivate ${o.name}?` : `Suspend ${o.name}?`,
                      message: suspended ? "Its users can sign in again." : "All of its users are signed out immediately and cannot sign in until it is reactivated.",
                      confirmLabel: suspended ? "Reactivate" : "Suspend",
                      tone: suspended ? "primary" : "danger",
                    });
                    if (!ok) return;
                    await withBusy(control, async () => {
                      try {
                        await api.patch(apiPath("/api/v1/platform/organizations", o.id), { status: suspended ? "active" : "suspended" });
                        toast("Organisation updated.", "success");
                        load();
                      } catch (error) {
                        toast(error && error.detail ? error.detail : "The organisation could not be updated.", "danger");
                      }
                    });
                  },
                });
              },
            },
          ],
        }),
      );
    } catch (error) {
      if (!ctx.signal.aborted) mount(results, errorCallout(error, { retry: load }));
    }
  };

  load();
  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Organisations",
      subtitle: "Tenants on this platform. Platform administrators cannot read tenant documents.",
      actions: can("org:create") ? [button("New organisation", { variant: "primary", icon: "plus", onClick: () => openCreateDialog(load) })] : [],
    }),
    results,
  );
}

function openCreateDialog(onDone) {
  const status = formStatus();
  const name = input({ required: true, maxlength: 200 });
  const slug = input({ required: true, maxlength: 63, pattern: "[a-z0-9]+(-[a-z0-9]+)*" });
  const adminEmail = input({ type: "email", required: true, maxlength: 320 });
  const adminName = input({ required: true, maxlength: 200 });
  let slugTouched = false;
  slug.addEventListener("input", () => (slugTouched = true));
  name.addEventListener("input", () => {
    if (!slugTouched) slug.value = slugify(name.value);
  });
  const submit = button("Create organisation", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    h("div", { class: "field-row" }, field("Organisation name", name, { required: true }), field("Short name", slug, { hint: "Lowercase letters, numbers and dashes." })),
    h("div", { class: "field-row" }, field("First administrator's email", adminEmail, { required: true }), field("First administrator's name", adminName, { required: true })),
    callout("info", null, "The administrator receives an email with a link to choose their own password."),
    status.element,
    h("div", { class: "form-actions" }, submit),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/platform/organizations", {
          slug: slug.value.trim(),
          name: name.value.trim(),
          admin_email: adminEmail.value.trim(),
          admin_name: adminName.value.trim(),
        });
        toast("Organisation created. The administrator has been emailed.", "success");
        handle.close();
        onDone();
      } catch (error) {
        status.error(error);
      }
    }, "Creating\u2026");
  });
  const handle = openDialog({ title: "New organisation", size: "lg", content: form });
}
