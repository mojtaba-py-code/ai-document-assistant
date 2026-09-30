/**
 * Department administration (create, rename, describe, delete when unused).
 *
 * @module views/admin-departments
 */

import { api, apiPath } from "../api.js";
import { h, mount } from "../dom.js";
import { formatDate, formatNumber } from "../format.js";
import { can, getDepartments } from "../session.js";
import {
  button,
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
  textarea,
  toast,
  withBusy,
} from "../ui.js";

/**
 * Derives a URL-safe slug from a name ("Human Resources" -> "human-resources").
 * @param {string} name
 * @returns {string}
 */
export function slugify(name) {
  return String(name)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 63);
}

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function departmentsView(ctx) {
  const canManage = can("department:manage");
  const results = h("div", { "aria-live": "polite" });

  const load = async (refresh = false) => {
    mount(results, loadingBlock("Loading departments\u2026"));
    try {
      const departments = await getDepartments({ refresh });
      if (ctx.signal.aborted) return;
      mount(
        results,
        dataTable({
          caption: "Departments",
          rows: departments,
          empty: emptyState({
            title: "No departments yet.",
            text: "Departments group people so confidential documents can be shared with the right team.",
            icon: "building",
          }),
          columns: [
            { label: "Name", primary: true, render: (d) => String(d.name || "Department") },
            { label: "Short name", render: (d) => (d.slug ? String(d.slug) : null) },
            { label: "Description", render: (d) => (d.description ? String(d.description) : null) },
            { label: "Members", render: (d) => (d.member_count !== undefined ? formatNumber(d.member_count) : null) },
            { label: "Managers", render: (d) => (d.manager_count !== undefined ? formatNumber(d.manager_count) : null) },
            { label: "Created", render: (d) => formatDate(d.created_at) },
            {
              label: "Actions",
              render: (d) =>
                canManage
                  ? h(
                      "span",
                      { class: "btn-row" },
                      button("Edit", { small: true, variant: "ghost", ariaLabel: `Edit ${d.name}`, onClick: () => openDepartmentDialog(d, () => load(true)) }),
                      button("Delete", {
                        small: true,
                        variant: "ghost",
                        ariaLabel: `Delete ${d.name}`,
                        onClick: async () => {
                          const ok = await confirmDialog({
                            title: `Delete ${d.name}?`,
                            message: "Only departments with no documents can be deleted. Members lose this membership.",
                            confirmLabel: "Delete department",
                            tone: "danger",
                          });
                          if (!ok) return;
                          try {
                            await api.del(apiPath("/api/v1/departments", d.id));
                            toast("Department deleted.", "success");
                            load(true);
                          } catch (error) {
                            toast(error && error.detail ? error.detail : "The department could not be deleted.", "danger");
                          }
                        },
                      }),
                    )
                  : null,
            },
          ],
        }),
      );
    } catch (error) {
      mount(results, errorCallout(error, { retry: () => load(true) }));
    }
  };

  load(true);
  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Departments",
      subtitle: "Confidential documents are shared within their department.",
      actions: canManage ? [button("New department", { variant: "primary", icon: "plus", onClick: () => openDepartmentDialog(null, () => load(true)) })] : [],
    }),
    results,
  );
}

function openDepartmentDialog(existing, onDone) {
  const status = formStatus();
  const name = input({ required: true, maxlength: 120, value: existing ? String(existing.name || "") : "" });
  const slug = input({ maxlength: 63, pattern: "[a-z0-9]+(-[a-z0-9]+)*", value: existing ? String(existing.slug || "") : "" });
  const description = textarea({ maxlength: 500, rows: 3, value: existing ? String(existing.description || "") : "" });
  let slugTouched = Boolean(existing);
  slug.addEventListener("input", () => (slugTouched = true));
  name.addEventListener("input", () => {
    if (!slugTouched) slug.value = slugify(name.value);
  });
  const submit = button(existing ? "Save" : "Create department", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    field("Name", name, { required: true }),
    field("Short name", slug, { hint: "Lowercase letters, numbers and dashes. Leave empty to derive it from the name." }),
    field("Description", description),
    status.element,
    h("div", { class: "form-actions" }, submit),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        if (existing) {
          const patch = {};
          if (name.value.trim() !== existing.name) patch.name = name.value.trim();
          if (slug.value.trim() && slug.value.trim() !== existing.slug) patch.slug = slug.value.trim();
          if (description.value.trim() !== (existing.description || "")) patch.description = description.value.trim() || null;
          if (Object.keys(patch).length) await api.patch(apiPath("/api/v1/departments", existing.id), patch);
        } else {
          await api.post("/api/v1/departments", {
            name: name.value.trim(),
            slug: slug.value.trim() || undefined,
            description: description.value.trim() || undefined,
          });
        }
        toast(existing ? "Department updated." : "Department created.", "success");
        handle.close();
        onDone();
      } catch (error) {
        status.error(error);
      }
    }, "Saving\u2026");
  });
  const handle = openDialog({ title: existing ? `Edit ${existing.name}` : "New department", content: form });
}
