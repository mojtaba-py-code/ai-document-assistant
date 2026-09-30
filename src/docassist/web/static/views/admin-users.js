/**
 * User administration: list/filter users, invite new users (they set their own password
 * from an emailed link - no password ever passes through this UI), edit role, clearance,
 * status and departments, trigger password resets and revoke sessions.
 *
 * The API enforces the escalation rules (assignable roles, clearance ceiling, no self
 * changes, last-admin protection); the UI mirrors them so people are not offered choices
 * that will be refused.
 *
 * @module views/admin-users
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { CLASSIFICATIONS, classificationRank, relativeTime, roleLabel, TENANT_ROLES, formatDateTime } from "../format.js";
import { updateQuery } from "../router.js";
import { can, departmentName, getDepartments, hasRole, session } from "../session.js";
import {
  badge,
  button,
  callout,
  checkbox,
  classificationBadge,
  confirmDialog,
  dataTable,
  debounce,
  emptyState,
  errorCallout,
  field,
  formStatus,
  input,
  loadingBlock,
  loadMoreButton,
  openDialog,
  pageHeader,
  select,
  statusBadge,
  toast,
  withBusy,
} from "../ui.js";

/** Roles the current user may assign (mirrors ASSIGNABLE_ROLES on the server). */
function assignableRoles() {
  if (hasRole("organization_admin")) return TENANT_ROLES.filter((r) => r.value !== "platform_admin");
  return [];
}

/** Clearances up to the actor's own. */
function assignableClearances() {
  const ceiling = session.user ? classificationRank(session.user.clearance) : -1;
  return CLASSIFICATIONS.filter((_, index) => index <= ceiling).map((c) => ({ value: c.value, label: c.label }));
}

/**
 * Normalises a user's department memberships.
 * @param {any} user
 * @returns {{id: string, isManager: boolean}[]}
 */
function membershipsOf(user) {
  if (Array.isArray(user.departments)) {
    return user.departments
      .map((d) =>
        typeof d === "string"
          ? { id: d, isManager: false, name: "" }
          : { id: String(d.department_id || d.id || ""), isManager: Boolean(d.is_manager), name: String(d.department_name || d.name || "") },
      )
      .filter((d) => d.id);
  }
  const managed = Array.isArray(user.managed_department_ids) ? user.managed_department_ids : [];
  return (Array.isArray(user.department_ids) ? user.department_ids : []).map((id) => ({ id: String(id), isManager: managed.includes(id), name: "" }));
}

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function usersView(ctx) {
  const departments = await getDepartments();
  const canManage = can("user:manage");
  const filters = { q: ctx.query.get("q") || "", role: ctx.query.get("role") || "", status: ctx.query.get("status") || "" };
  const results = h("div", { "aria-live": "polite" });
  let cursor = null;
  let table = null;

  const columns = [
    {
      label: "Name",
      primary: true,
      render: (u) => h("div", { class: "cell-title" }, h("span", { class: "strong" }, String(u.full_name || u.email || "User")), h("span", { class: "muted small" }, String(u.email || ""))),
    },
    { label: "Role", render: (u) => roleLabel(u.role) },
    { label: "Clearance", render: (u) => classificationBadge(u.clearance) },
    { label: "Status", render: (u) => h("span", { class: "badge-row" }, statusBadge(u.status), u.locked === true && badge("Locked", "warning")) },
    {
      label: "Departments",
      render: (u) => {
        const list = membershipsOf(u);
        return list.length ? list.map((m) => `${m.name || departmentName(departments, m.id)}${m.isManager ? " (manager)" : ""}`).join(", ") : "None";
      },
    },
    { label: "Two-step", render: (u) => (u.mfa_enabled === true ? badge("On", "success") : u.mfa_enabled === false ? badge("Off", "neutral") : null) },
    { label: "Last sign-in", render: (u) => (u.last_login_at ? h("time", { title: formatDateTime(u.last_login_at) }, relativeTime(u.last_login_at)) : "Never") },
    {
      label: "Actions",
      render: (u) =>
        button(canManage ? "Manage" : "View", {
          small: true,
          variant: "ghost",
          ariaLabel: `${canManage ? "Manage" : "View"} ${u.full_name || u.email || "user"}`,
          onClick: () => openUserDialog(u, departments, canManage, load),
        }),
    },
  ];

  const fetchPage = async (after) =>
    pageOf(await api.get("/api/v1/users", { query: { ...filters, cursor: after, limit: 50 }, signal: ctx.signal }), ["users"]);

  const load = async () => {
    updateQuery(filters);
    mount(results, loadingBlock("Loading users\u2026"));
    try {
      const page = await fetchPage(null);
      cursor = page.next;
      table = dataTable({
        caption: "Users",
        columns,
        rows: page.items,
        rowClass: (u) => (u.status === "disabled" ? "row-muted" : null),
        empty: emptyState({ title: "No users match.", icon: "users" }),
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
      if (!ctx.signal.aborted) mount(results, errorCallout(error, { retry: load }));
    }
  };

  const search = input({ type: "search", value: filters.q, placeholder: "Name or email", maxlength: 200 });
  search.addEventListener("input", debounce(() => {
    filters.q = search.value.trim();
    load();
  }, 350));
  const role = select([{ value: "", label: "All roles" }, ...TENANT_ROLES], { value: filters.role });
  role.addEventListener("change", () => {
    filters.role = role.value;
    load();
  });
  const status = select(
    [
      { value: "", label: "Any status" },
      { value: "active", label: "Active" },
      { value: "disabled", label: "Disabled" },
    ],
    { value: filters.status },
  );
  status.addEventListener("change", () => {
    filters.status = status.value;
    load();
  });

  load();
  return h(
    "div",
    { class: "page" },
    pageHeader({
      title: "Users",
      subtitle: canManage ? "Invite people, set their role and clearance, and manage access." : "People in your organisation (read-only).",
      actions: canManage ? [button("Invite user", { variant: "primary", icon: "plus", onClick: () => openInviteDialog(departments, load) })] : [],
    }),
    h(
      "form",
      { class: "filters", role: "search", "aria-label": "Filter users", on: { submit: (e) => e.preventDefault() } },
      field("Search", search, { className: "filter-grow" }),
      field("Role", role),
      field("Status", status),
    ),
    results,
  );
}

function departmentChooser(departments, memberships) {
  const selected = new Map(memberships.map((m) => [m.id, m.isManager]));
  const rows = departments.map((d) => {
    const member = checkbox(String(d.name || "Department"), { value: String(d.id), checked: selected.has(String(d.id)) });
    const manager = checkbox("Manager", { value: String(d.id), checked: Boolean(selected.get(String(d.id))) });
    const memberInput = member.querySelector("input");
    const managerInput = manager.querySelector("input");
    const sync = () => {
      managerInput.disabled = !memberInput.checked;
      if (!memberInput.checked) managerInput.checked = false;
    };
    memberInput.addEventListener("change", sync);
    sync();
    return { element: h("div", { class: "dept-choice" }, member, manager), memberInput, managerInput, id: String(d.id) };
  });
  const element = departments.length
    ? h("fieldset", { class: "fieldset" }, h("legend", { class: "field-label" }, "Departments"), h("div", { class: "dept-choices" }, rows.map((r) => r.element)))
    : h("p", { class: "muted small" }, "No departments exist yet.");
  return {
    element,
    value: () => rows.filter((r) => r.memberInput.checked).map((r) => ({ department_id: r.id, is_manager: r.managerInput.checked })),
  };
}

function openInviteDialog(departments, onDone) {
  const status = formStatus();
  const email = input({ type: "email", required: true, maxlength: 320, autocomplete: "off" });
  const name = input({ required: true, maxlength: 200, autocomplete: "off" });
  const role = select(assignableRoles(), { value: "employee" });
  const clearance = select(assignableClearances(), { value: "INTERNAL" });
  const depts = departmentChooser(departments, []);
  const submit = button("Send invitation", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    h("div", { class: "field-row" }, field("Work email", email, { required: true }), field("Full name", name, { required: true })),
    h("div", { class: "field-row" }, field("Role", role), field("Clearance", clearance, { hint: "The most sensitive classification they may read." })),
    depts.element,
    callout("info", null, "The person receives an email with a link to choose their own password. You never see or set it."),
    status.element,
    h("div", { class: "form-actions" }, submit),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/users", {
          email: email.value.trim(),
          full_name: name.value.trim(),
          role: role.value,
          clearance: clearance.value,
          departments: depts.value(),
        });
        toast(`Invitation sent to ${email.value.trim()}.`, "success");
        handle.close();
        onDone();
      } catch (error) {
        status.error(error);
      }
    }, "Sending\u2026");
  });
  const handle = openDialog({ title: "Invite user", size: "lg", content: form });
}

function openUserDialog(user, departments, canManage, onDone) {
  const isSelf = session.user && user.id === session.user.id;
  const status = formStatus();
  const readOnly = !canManage;
  const name = input({ value: String(user.full_name || ""), maxlength: 200, disabled: readOnly });
  const roles = assignableRoles();
  const roleOptions = roles.some((r) => r.value === user.role) ? roles : [...roles, { value: user.role, label: roleLabel(user.role) }];
  const role = select(roleOptions, { value: user.role, disabled: readOnly || isSelf });
  const clearanceOptions = assignableClearances();
  const clearance = select(
    clearanceOptions.some((c) => c.value === user.clearance) ? clearanceOptions : [...clearanceOptions, { value: user.clearance, label: String(user.clearance) }],
    { value: user.clearance, disabled: readOnly || isSelf },
  );
  const userStatus = select(
    [
      { value: "active", label: "Active" },
      { value: "disabled", label: "Disabled (cannot sign in)" },
    ],
    { value: user.status || "active", disabled: readOnly || isSelf },
  );
  const depts = departmentChooser(departments, membershipsOf(user));
  if (readOnly) for (const control of depts.element.querySelectorAll("input")) control.disabled = true;

  const save = button("Save changes", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    isSelf && callout("info", null, "You cannot change your own role, clearance or status. Ask another administrator."),
    field("Full name", name),
    h("div", { class: "field-row" }, field("Role", role), field("Clearance", clearance)),
    field("Status", userStatus),
    depts.element,
    h("p", { class: "muted small" }, "Changing role, clearance, status or departments signs the person out so the change takes effect immediately."),
    status.element,
    canManage && h("div", { class: "form-actions" }, save),
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    const patch = {};
    if (name.value.trim() && name.value.trim() !== user.full_name) patch.full_name = name.value.trim();
    if (!isSelf) {
      if (role.value !== user.role) patch.role = role.value;
      if (clearance.value !== user.clearance) patch.clearance = clearance.value;
      if (userStatus.value !== (user.status || "active")) patch.status = userStatus.value;
    }
    const before = JSON.stringify(membershipsOf(user).map((m) => [m.id, m.isManager]).sort());
    const chosen = depts.value();
    if (JSON.stringify(chosen.map((m) => [m.department_id, m.is_manager]).sort()) !== before) patch.departments = chosen;
    if (!Object.keys(patch).length) {
      handle.close();
      return;
    }
    if (patch.status === "disabled") {
      const ok = await confirmDialog({
        title: "Disable this account?",
        message: `${user.full_name || user.email} will be signed out everywhere and cannot sign in until re-enabled.`,
        confirmLabel: "Disable account",
        tone: "danger",
      });
      if (!ok) return;
    }
    await withBusy(save, async () => {
      try {
        await api.patch(apiPath("/api/v1/users", user.id), patch);
        toast("User updated.", "success");
        handle.close();
        onDone();
      } catch (error) {
        status.error(error);
      }
    }, "Saving\u2026");
  });

  const actions = [];
  if (canManage && !isSelf) {
    const reset = button("Send password reset", {
      onClick: () =>
        withBusy(reset, async () => {
          try {
            await api.post(apiPath("/api/v1/users", user.id, "send-reset"));
            toast("Password reset email sent.", "success");
          } catch (error) {
            toast(error && error.detail ? error.detail : "Could not send the reset email.", "danger");
          }
        }),
    });
    const revoke = button("Sign out everywhere", {
      onClick: async () => {
        const ok = await confirmDialog({
          title: "Sign this person out everywhere?",
          message: "All of their active sessions end immediately. They can sign in again.",
          confirmLabel: "Sign out everywhere",
          tone: "danger",
        });
        if (!ok) return;
        try {
          const result = await api.post(apiPath("/api/v1/users", user.id, "revoke-sessions"));
          const count = result && Number.isFinite(Number(result.revoked_sessions)) ? Number(result.revoked_sessions) : null;
          toast(count === null ? "All sessions revoked." : `${count} session${count === 1 ? "" : "s"} revoked.`, "success");
        } catch (error) {
          toast(error && error.detail ? error.detail : "Could not revoke sessions.", "danger");
        }
      },
    });
    const resetMfa = button("Reset two-step verification", {
      onClick: async () => {
        const ok = await confirmDialog({
          title: "Reset two-step verification?",
          message:
            "Use this when the person lost their authenticator or did not set it up themselves. " +
            "Their sessions end, they are notified by email and can set it up again after signing in.",
          confirmLabel: "Reset",
          tone: "danger",
        });
        if (!ok) return;
        try {
          await api.post(apiPath("/api/v1/users", user.id, "reset-mfa"));
          toast("Two-step verification was reset.", "success");
        } catch (error) {
          toast(error && error.detail ? error.detail : "Could not reset two-step verification.", "danger");
        }
      },
    });
    actions.push(reset, revoke, resetMfa);
  }
  actions.push(button("Close", { onClick: () => handle.close() }));

  const handle = openDialog({
    title: String(user.full_name || user.email || "User"),
    description: `${user.email || ""}${user.created_at ? ` \u00b7 member since ${formatDateTime(user.created_at)}` : ""}`,
    size: "lg",
    content: form,
    actions,
  });
}
