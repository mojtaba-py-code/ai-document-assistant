/**
 * Document detail: status banners, preview, metadata (editable by managers), versions,
 * access grants, extracted data and summary.
 *
 * Route: `#/documents/:id?tab=preview&page=3&version=2`. A citation or search result can
 * pass the passage to highlight through the in-memory hand-off (`highlight:<id>`), keeping
 * document text out of the URL.
 *
 * @module views/document
 */

import { api, apiPath, download, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import {
  classificationLabel,
  classificationRank,
  CLASSIFICATIONS,
  displayValue,
  docTypeLabel,
  EMPTY,
  formatBytes,
  formatDate,
  formatDateTime,
  formatNumber,
  formatPercent,
  humanize,
  ingestionErrorText,
  roleLabel,
  shortId,
  TENANT_ROLES,
} from "../format.js";
import { navigate, updateQuery } from "../router.js";
import { can, departmentName, getDepartments, hasRole, session, takeHandoff } from "../session.js";
import {
  badge,
  button,
  callout,
  card,
  checkbox,
  classificationBadge,
  confirmDialog,
  dataTable,
  descriptionList,
  emptyState,
  errorCallout,
  field,
  formStatus,
  input,
  linkButton,
  loadingBlock,
  openDialog,
  pageHeader,
  poll,
  select,
  statusBadge,
  tabs,
  toast,
  withBusy,
} from "../ui.js";
import { classificationOptions, currentVersionNumber, currentVersionOf, departmentOptions, docTypeOptions, tagList, uploadableDepartments, versionsOf } from "./common.js";
import { fieldsPanel, openReportDialog, summaryPanel } from "./doc-intel.js";
import { previewPane } from "./preview.js";
import { openVersionDialog } from "./upload.js";

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function documentView(ctx) {
  const id = ctx.params.id;
  const root = h("div", { class: "page" });
  const departments = await getDepartments();
  const state = {
    tab: ctx.query.get("tab") || "",
    page: Number(ctx.query.get("page")) || 1,
    version: ctx.query.get("version") || "",
    highlight: takeHandoff(`highlight:${id}`) || null,
  };
  let lastStatus = null;

  const load = async () => {
    try {
      const doc = await api.get(apiPath("/api/v1/documents", id), { signal: ctx.signal });
      ctx.setTitle(String(doc.title || "Document"));
      render(doc);
    } catch (error) {
      if (ctx.signal.aborted) return;
      mount(
        root,
        pageHeader({ title: "Document", back: { href: "#/documents", label: "All documents" } }),
        errorCallout(error, { retry: load, title: error && error.status === 404 ? "Document not available" : undefined }),
      );
    }
  };

  const render = (doc) => {
    lastStatus = doc.status;
    const versions = versionsOf(doc);
    const canRead = doc.can_read !== false && doc.status === "ready";
    const canManage = doc.can_manage === true;
    const title = String(doc.title || "Untitled document");

    const openPage = (page, quote, chunkId) => {
      state.page = page;
      state.highlight = quote || chunkId ? { quote, chunkId } : null;
      updateQuery({ tab: "preview", page: page > 1 ? page : null });
      if (tabbed) {
        tabbed.rerender("preview");
        tabbed.select("preview");
      }
    };

    const actions = [];
    if (canRead) {
      actions.push(
        button("Download", {
          icon: "download",
          onClick: (event) => runDownload(event.currentTarget, apiPath("/api/v1/documents", id, "download"), title),
        }),
      );
      if (can("assistant:use")) {
        actions.push(linkButton("Ask about this document", `#/assistant?document=${encodeURIComponent(id)}`, { icon: "sparkle", variant: "primary" }));
      }
    }
    if (canManage && doc.status !== "deleted") {
      actions.push(
        button("New version", { icon: "upload", onClick: () => openVersionDialog({ documentId: id, title, onUploaded: () => load() }) }),
        button("Edit details", { icon: "edit", onClick: () => openEditDialog(doc, departments, () => load()) }),
        button("Delete", { icon: "trash", variant: "danger", onClick: () => deleteDocument(doc) }),
      );
    }

    const header = pageHeader({
      title,
      back: { href: "#/documents", label: "All documents" },
      meta: [
        classificationBadge(doc.classification),
        statusBadge(doc.status),
        badge(docTypeLabel(doc.doc_type), "neutral"),
        doc.legal_hold && badge("Legal hold", "accent"),
      ].filter(Boolean),
      actions,
    });

    const banners = statusBanners(doc, canManage, versions, () => load());

    const items = [];
    if (canRead) {
      items.push({
        id: "preview",
        label: "Preview",
        render: () =>
          previewPane({
            documentId: id,
            versions: versions
              .filter((v) => v.status === "indexed")
              .map((v) => ({ value: String(v.version_number), label: `Version ${v.version_number}` })),
            version: state.version,
            pageCount: pageCountOf(doc),
            page: state.page,
            highlight: state.highlight,
            signal: ctx.signal,
          }),
      });
    }
    items.push({ id: "details", label: "Details", render: () => detailsPanel(doc, departments, versions) });
    items.push({ id: "versions", label: `Versions (${versions.length || doc.version_count || 0})`, render: () => versionsPanel(doc, versions, canRead, canManage, () => load()) });
    if (canManage) items.push({ id: "access", label: "Access", render: () => accessPanel(doc, departments, ctx.signal) });
    if (canRead && can("intelligence:use")) {
      items.push({
        id: "data",
        label: "Extracted data",
        render: () =>
          h(
            "div",
            { class: "stack" },
            fieldsPanel({ documentId: id, canRun: true, signal: ctx.signal, openPage }),
            card({
              title: "Document report",
              description: "Key facts, deadlines and points to review in one page.",
              actions: [button("Open report", { small: true, onClick: () => openReportDialog(id, title) })],
            }),
          ),
      });
      items.push({ id: "summary", label: "Summary", render: () => summaryPanel({ documentId: id, signal: ctx.signal, openPage: (p) => openPage(p) }) });
    }

    const tabbed = tabs({
      label: "Document sections",
      items,
      selected: items.some((i) => i.id === state.tab) ? state.tab : items[0].id,
      onSelect: (tabId) => updateQuery({ tab: tabId === items[0].id ? null : tabId }),
    });

    mount(root, header, banners, tabbed);
  };

  const runDownload = async (control, path, fallbackName, query) => {
    await withBusy(control, async () => {
      try {
        await download(path, { fallbackName, query, signal: ctx.signal });
      } catch (error) {
        toast(error && error.detail ? error.detail : "The download failed.", "danger");
      }
    }, "Downloading\u2026");
  };

  const deleteDocument = async (doc) => {
    const ok = await confirmDialog({
      title: "Delete this document?",
      message: [
        h("p", null, `"${doc.title || "This document"}" will be removed from the library, search and the assistant immediately.`),
        h("p", { class: "muted" }, "The stored files are purged after the retention period. Documents on legal hold cannot be deleted."),
      ],
      confirmLabel: "Delete document",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await api.del(apiPath("/api/v1/documents", id));
      toast(`"${doc.title || "Document"}" was deleted.`, "success");
      navigate("/documents");
    } catch (error) {
      toast(error && error.detail ? error.detail : "The document could not be deleted.", "danger");
    }
  };

  // Keep the page current while the document is being processed.
  poll(
    ctx.signal,
    async () => {
      if (lastStatus !== "processing") return true;
      try {
        const fresh = await api.get(apiPath("/api/v1/documents", id), { signal: ctx.signal });
        if (fresh.status !== lastStatus) {
          render(fresh);
          toast(fresh.status === "ready" ? "The document is ready." : `Processing finished: ${humanize(fresh.status)}.`, fresh.status === "ready" ? "success" : "warning");
        }
      } catch {
        // retry on the next tick
      }
      return true;
    },
    4000,
  );

  mount(root, loadingBlock("Loading document\u2026"));
  await load();
  return root;
}

function pageCountOf(doc) {
  const current = currentVersionOf(doc);
  const value = Number((current && current.page_count) ?? doc.page_count);
  return value > 0 ? value : null;
}

function statusBanners(doc, canManage, versions, reload) {
  const nodes = [];
  const latest = versions[0] || {};
  const errorCode = doc.error_code || (doc.ingestion && doc.ingestion.error_code) || latest.error_code;
  if (doc.status === "processing") {
    nodes.push(
      callout("info", "Processing", "The document is being read, indexed and scanned. This page updates automatically when it is ready.", { live: true }),
    );
  } else if (doc.status === "failed") {
    nodes.push(callout("danger", "Processing failed", ingestionErrorText(errorCode)));
  } else if (doc.status === "quarantined") {
    // Findings are only included for managers, per version.
    const findings = versions.flatMap((v) => (Array.isArray(v.findings) ? v.findings : []));
    nodes.push(
      callout(
        "danger",
        "Held for security review",
        [
          h("p", null, "The security scan found potentially unsafe content, so this file is not opened, indexed or offered for download to readers."),
          canManage && findings.length > 0 &&
            h(
              "ul",
              { class: "findings" },
              findings.map((f) =>
                h(
                  "li",
                  null,
                  badge(humanize(f.severity || "finding"), f.severity === "critical" || f.severity === "high" ? "danger" : "warning"),
                  " ",
                  String(f.code || f),
                  f.detail ? ` \u2014 ${f.detail}` : "",
                ),
              ),
            ),
        ],
        {
          actions: canManage
            ? [
                button("Download for inspection", {
                  small: true,
                  variant: "danger",
                  onClick: async (event) => {
                    const control = event.currentTarget;
                    const ok = await confirmDialog({
                      title: "Download a quarantined file?",
                      message: "This file may contain malware or active content. Only open it in an isolated environment. The download is recorded in the audit log.",
                      confirmLabel: "I understand, download",
                      tone: "danger",
                    });
                    if (!ok) return;
                    await withBusy(control, async () => {
                      try {
                        await download(apiPath("/api/v1/documents", doc.id, "download"), { query: { acknowledge_risk: "true" }, fallbackName: "quarantined-file" });
                      } catch (error) {
                        toast(error && error.detail ? error.detail : "The download failed.", "danger");
                      }
                    });
                  },
                }),
              ]
            : [],
        },
      ),
    );
  } else if (doc.status === "ready" && latest.status && latest.status !== "indexed" && latest.is_current !== true) {
    if (latest.status === "failed") nodes.push(callout("warning", `Version ${latest.version_number} could not be processed`, ingestionErrorText(latest.error_code)));
    else if (latest.status === "processing" || latest.status === "uploaded") {
      nodes.push(callout("info", `Version ${latest.version_number} is being processed`, "The previous version stays in use until the new one is ready."));
    }
  }
  if (doc.can_read === false && doc.status === "ready") {
    nodes.push(
      callout("info", "Metadata only", "You can see this document's details, but its content requires additional access (for example a grant from the owner)."),
    );
  }
  if (
    canManage &&
    doc.suggested_classification &&
    classificationRank(doc.suggested_classification) > classificationRank(doc.classification)
  ) {
    const signals = Array.isArray(doc.sensitivity_signals) ? doc.sensitivity_signals : [];
    const apply = button(`Change to ${classificationLabel(doc.suggested_classification)}`, {
      small: true,
      variant: "primary",
      onClick: () =>
        withBusy(apply, async () => {
          try {
            await api.patch(apiPath("/api/v1/documents", doc.id), { classification: doc.suggested_classification });
            toast("Classification updated.", "success");
            reload();
          } catch (error) {
            toast(error && error.detail ? error.detail : "The classification could not be changed.", "danger");
          }
        }),
    });
    nodes.push(
      callout(
        "warning",
        `Suggested classification: ${classificationLabel(doc.suggested_classification)}`,
        `Sensitive information was detected${signals.length ? ` (${signals.map(humanize).join(", ")})` : ""}. The classification is never raised automatically; please review it.`,
        { actions: [apply] },
      ),
    );
  }
  return h("div", { class: "banners" }, nodes);
}

function detailsPanel(doc, departments, versions) {
  const current = currentVersionOf(doc) || versions[0] || {};
  const typeText =
    doc.doc_type_source === "auto" && doc.doc_type_confidence !== null && doc.doc_type_confidence !== undefined
      ? `${docTypeLabel(doc.doc_type)} (detected automatically, ${formatPercent(doc.doc_type_confidence)} confidence)`
      : docTypeLabel(doc.doc_type);
  const owner = doc.owner_name || doc.owner_email || (doc.owner_id && session.user && doc.owner_id === session.user.id ? "You" : shortId(doc.owner_id));
  const metadata = doc.metadata || doc.extracted_metadata || current.metadata || current.doc_metadata || null;
  const allowedRoles = Array.isArray(doc.allowed_roles) && doc.allowed_roles.length ? doc.allowed_roles.map(roleLabel).join(", ") : "No extra role limit";
  const classificationHelp = CLASSIFICATIONS.find((c) => c.value === String(doc.classification || "").toUpperCase());

  return h(
    "div",
    { class: "stack" },
    card({
      title: "About this document",
      body: descriptionList([
        ["Type", typeText],
        ["Classification", h("span", null, classificationBadge(doc.classification), classificationHelp ? h("span", { class: "muted small" }, ` ${classificationHelp.help}`) : null)],
        ["Department", departmentName(departments, doc.department_id) || "None"],
        ["Owner", owner],
        ["Tags", tagList(doc.tags) || "None"],
        ["Limited to roles", allowedRoles],
        ["Created", formatDateTime(doc.created_at)],
        ["Last updated", formatDateTime(doc.updated_at)],
        ["Retention until", doc.retention_until ? formatDate(doc.retention_until) : "Organisation default"],
        ["Legal hold", doc.legal_hold ? "Yes \u2014 cannot be deleted" : "No"],
      ]),
    }),
    card({
      title: "Current version",
      body: descriptionList([
        ["Version", current.version_number ? String(current.version_number) : EMPTY],
        ["File name", current.original_filename ? String(current.original_filename) : EMPTY],
        ["File type", current.detected_mime ? String(current.detected_mime) : EMPTY],
        ["Size", formatBytes(current.size_bytes)],
        ["Pages", formatNumber(current.page_count)],
        ["Passages indexed", formatNumber(current.chunk_count)],
        ["Language", current.language ? String(current.language) : null],
        ["Semantic search", current.semantic_indexed === false ? "Keyword search only (sensitivity policy)" : current.semantic_indexed ? "Enabled" : null],
        ["Needs text recognition", current.needs_ocr ? "Yes \u2014 some pages are scanned images" : null],
      ]),
    }),
    metadata && typeof metadata === "object" && Object.keys(metadata).length > 0 &&
      card({
        title: "File properties",
        description: "Read from the file itself; shown as plain text.",
        body: descriptionList(Object.entries(metadata).slice(0, 30).map(([key, value]) => [humanize(key), displayValue(value)])),
      }),
  );
}

function versionsPanel(doc, versions, canRead, canManage, reload) {
  const currentNumber = currentVersionNumber(doc);
  const table = dataTable({
    caption: "Version history",
    rows: versions,
    empty: emptyState({ title: "No versions recorded.", icon: "file" }),
    columns: [
      {
        label: "Version",
        primary: true,
        render: (v) => h("span", null, `v${v.version_number}`, Number(v.version_number) === currentNumber ? h("span", null, " ", badge("Current", "success")) : null),
      },
      { label: "File", render: (v) => String(v.original_filename || EMPTY) },
      { label: "Status", render: (v) => statusBadge(v.status) },
      { label: "Size", render: (v) => formatBytes(v.size_bytes) },
      { label: "Pages", render: (v) => formatNumber(v.page_count) },
      { label: "Uploaded", render: (v) => formatDateTime(v.created_at) },
      { label: "Note", render: (v) => (v.change_note ? String(v.change_note) : null) },
      {
        label: "Actions",
        render: (v) =>
          canRead && v.status !== "quarantined"
            ? button("Download", {
                small: true,
                variant: "ghost",
                icon: "download",
                ariaLabel: `Download version ${v.version_number}`,
                onClick: async (event) => {
                  const control = event.currentTarget;
                  await withBusy(control, async () => {
                    try {
                      await download(apiPath("/api/v1/documents", doc.id, "versions", v.version_number, "download"), {
                        fallbackName: String(v.original_filename || "document"),
                      });
                    } catch (error) {
                      toast(error && error.detail ? error.detail : "The download failed.", "danger");
                    }
                  });
                },
              })
            : null,
      },
    ],
  });
  const indexed = versions.filter((v) => v.status === "indexed");
  const actions = [];
  if (canManage) actions.push(button("Upload new version", { small: true, icon: "upload", onClick: () => openVersionDialog({ documentId: doc.id, title: String(doc.title || "document"), onUploaded: reload }) }));
  if (indexed.length >= 2 && can("intelligence:use") && canRead) {
    actions.push(
      linkButton("Compare versions", `#/compare?document=${encodeURIComponent(doc.id)}&from=${indexed[1].version_number}&to=${indexed[0].version_number}`, {
        small: true,
        icon: "compare",
      }),
    );
  }
  return card({ title: "Version history", actions, body: table });
}

function granteeText(grant, departments, users) {
  const type = grant.grantee_type;
  if (grant.grantee_label || grant.grantee_name) return String(grant.grantee_label || grant.grantee_name);
  if (type === "user") {
    const user = users.find((u) => u.id === grant.grantee_user_id);
    return user ? `${user.full_name || user.email} (${user.email || "user"})` : grant.grantee_email || `User ${shortId(grant.grantee_user_id)}`;
  }
  if (type === "department") return `Department: ${departmentName(departments, grant.grantee_department_id)}`;
  if (type === "role") return `Everyone with role: ${roleLabel(grant.grantee_role)}`;
  return humanize(type);
}

function accessPanel(doc, departments, signal) {
  const listArea = h("div", { "aria-live": "polite" }, loadingBlock("Loading access list\u2026"));
  const formArea = h("div");
  let users = [];

  const loadGrants = async () => {
    try {
      const response = await api.get(apiPath("/api/v1/documents", doc.id, "grants"), { signal });
      const grants = pageOf(response, ["grants"]).items;
      mount(
        listArea,
        dataTable({
          caption: "People and groups with explicit access",
          rows: grants,
          empty: emptyState({
            title: "No explicit grants.",
            text: "Access follows the classification rules. Add a grant to share with a specific person, department or role.",
            icon: "lock",
          }),
          columns: [
            { label: "Who", primary: true, render: (g) => granteeText(g, departments, users) },
            { label: "Access", render: (g) => badge(g.permission === "manage" ? "Can manage" : "Can read", g.permission === "manage" ? "accent" : "info") },
            {
              label: "Expires",
              render: (g) =>
                g.active === false
                  ? badge("Expired", "neutral")
                  : g.expires_at
                    ? formatDateTime(g.expires_at)
                    : "Never",
            },
            { label: "Granted", render: (g) => formatDateTime(g.created_at) },
            {
              label: "Actions",
              render: (g) =>
                button("Revoke", {
                  small: true,
                  variant: "ghost",
                  ariaLabel: `Revoke access for ${granteeText(g, departments, users)}`,
                  onClick: async () => {
                    const ok = await confirmDialog({
                      title: "Revoke access?",
                      message: `${granteeText(g, departments, users)} will lose this access immediately.`,
                      confirmLabel: "Revoke",
                      tone: "danger",
                    });
                    if (!ok) return;
                    try {
                      await api.del(apiPath("/api/v1/documents", doc.id, "grants", g.id));
                      toast("Access revoked.", "success");
                      loadGrants();
                    } catch (error) {
                      toast(error && error.detail ? error.detail : "Could not revoke access.", "danger");
                    }
                  },
                }),
            },
          ],
        }),
      );
    } catch (error) {
      if (!signal.aborted) mount(listArea, errorCallout(error, { retry: loadGrants }));
    }
  };

  const buildForm = () => {
    const status = formStatus();
    const type = select(
      [
        { value: "user", label: "A person" },
        { value: "department", label: "A department" },
        { value: "role", label: "Everyone with a role" },
      ],
      { name: "grantee_type" },
    );
    const who = h("div");
    const permission = select(
      [
        { value: "read", label: "Can read" },
        { value: "manage", label: "Can manage (edit, share, delete)" },
      ],
      { name: "permission" },
    );
    const expires = input({ type: "date", name: "expires_at", min: new Date(Date.now() + 86_400_000).toISOString().slice(0, 10) });
    let whoControl = null;
    const renderWho = () => {
      if (type.value === "user") {
        whoControl = users.length
          ? select([{ value: "", label: "Choose a person\u2026" }, ...users.map((u) => ({ value: String(u.id), label: `${u.full_name || u.email} \u2014 ${u.email || ""}` }))], { required: true })
          : input({ required: true, placeholder: "User ID", maxlength: 80, pattern: "[A-Za-z0-9_-]{1,80}" });
        mount(who, field("Person", whoControl, { hint: users.length ? undefined : "Ask an administrator for the user's ID." }));
      } else if (type.value === "department") {
        whoControl = select(departmentOptions(departments, "Choose a department\u2026"), { required: true });
        mount(who, field("Department", whoControl));
      } else {
        whoControl = select([{ value: "", label: "Choose a role\u2026" }, ...TENANT_ROLES.map((r) => ({ value: r.value, label: r.label }))], { required: true });
        mount(who, field("Role", whoControl));
      }
    };
    type.addEventListener("change", renderWho);
    renderWho();
    const submit = button("Add access", { type: "submit", variant: "primary" });
    const form = h(
      "form",
      { class: "form" },
      h("div", { class: "field-row" }, field("Share with", type), who),
      h("div", { class: "field-row" }, field("Access level", permission), field("Expires on", expires, { hint: "Optional. Access ends automatically." })),
      status.element,
      h("div", { class: "form-actions" }, submit),
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.clear();
      const body = { grantee_type: type.value, permission: permission.value };
      const value = whoControl ? whoControl.value.trim() : "";
      if (!value) return;
      if (type.value === "user") body.user_id = value;
      else if (type.value === "department") body.department_id = value;
      else body.role = value;
      if (expires.value) body.expires_at = new Date(`${expires.value}T23:59:59`).toISOString();
      await withBusy(submit, async () => {
        try {
          await api.post(apiPath("/api/v1/documents", doc.id, "grants"), body);
          status.success("Access added.");
          form.reset();
          renderWho();
          loadGrants();
        } catch (error) {
          status.error(error);
        }
      }, "Adding\u2026");
    });
    mount(formArea, form);
  };

  (async () => {
    if (can("user:read")) {
      try {
        users = pageOf(await api.get("/api/v1/users", { query: { limit: 200, status: "active" }, signal }), ["users"]).items;
      } catch {
        users = [];
      }
    }
    buildForm();
    loadGrants();
  })();

  return h(
    "div",
    { class: "stack" },
    card({
      title: "Who can access this document",
      description: `Besides these grants, access follows the ${classificationLabel(doc.classification)} classification rules. Restricted documents need an explicit grant.`,
      body: listArea,
    }),
    card({ title: "Share with more people", body: formArea }),
  );
}

function openEditDialog(doc, departments, onSaved) {
  const status = formStatus();
  const title = input({ name: "title", value: String(doc.title || ""), maxlength: 300, required: true });
  const docType = select(docTypeOptions(), { value: doc.doc_type || "other" });
  const classification = select(classificationOptions(), { value: String(doc.classification || "INTERNAL").toUpperCase() });
  const deptChoices = hasRole("organization_admin") ? departments : uploadableDepartments(departments);
  const withCurrent = doc.department_id && !deptChoices.some((d) => d.id === doc.department_id) ? [...deptChoices, departments.find((d) => d.id === doc.department_id) || { id: doc.department_id, name: "Current department" }] : deptChoices;
  const department = select(departmentOptions(withCurrent, "No department"), { value: doc.department_id || "" });
  const tagsInput = input({ value: Array.isArray(doc.tags) ? doc.tags.join(", ") : "", maxlength: 1400 });
  const retention = input({ type: "date", value: doc.retention_until ? String(doc.retention_until).slice(0, 10) : "" });
  const currentRoles = Array.isArray(doc.allowed_roles) ? doc.allowed_roles : [];
  const roleBoxes = TENANT_ROLES.map((r) => checkbox(r.label, { value: r.value, checked: currentRoles.includes(r.value) }));
  const legalHold = checkbox("Legal hold (prevents deletion)", { checked: Boolean(doc.legal_hold) });

  const form = h(
    "form",
    { class: "form" },
    field("Title", title),
    h("div", { class: "field-row" }, field("Document type", docType), field("Classification", classification, { hint: "You cannot choose a level above your own clearance." })),
    h("div", { class: "field-row" }, field("Department", department), field("Retention until", retention, { hint: "Leave empty to use the organisation default." })),
    field("Tags", tagsInput, { hint: "Separate tags with commas." }),
    h("fieldset", { class: "fieldset" }, h("legend", { class: "field-label" }, "Limit to roles"), h("div", { class: "check-grid" }, roleBoxes)),
    hasRole("organization_admin") && legalHold,
    status.element,
  );
  const save = button("Save changes", { type: "submit", variant: "primary" });
  form.appendChild(h("div", { class: "form-actions" }, save));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    const patch = {};
    if (title.value.trim() !== String(doc.title || "")) patch.title = title.value.trim();
    if (docType.value !== doc.doc_type) patch.doc_type = docType.value;
    if (classification.value !== String(doc.classification || "").toUpperCase()) patch.classification = classification.value;
    if ((department.value || null) !== (doc.department_id || null)) patch.department_id = department.value || null;
    const tags = tagsInput.value.split(",").map((t) => t.trim()).filter(Boolean);
    if (JSON.stringify(tags) !== JSON.stringify(Array.isArray(doc.tags) ? doc.tags : [])) patch.tags = tags;
    const roles = roleBoxes.map((b) => b.querySelector("input")).filter((i) => i.checked).map((i) => i.value);
    if (JSON.stringify(roles.slice().sort()) !== JSON.stringify(currentRoles.slice().sort())) patch.allowed_roles = roles;
    const retentionValue = retention.value || null;
    if (retentionValue !== (doc.retention_until ? String(doc.retention_until).slice(0, 10) : null)) patch.retention_until = retentionValue;
    if (hasRole("organization_admin")) {
      const hold = legalHold.querySelector("input").checked;
      if (hold !== Boolean(doc.legal_hold)) patch.legal_hold = hold;
    }
    if (!Object.keys(patch).length) {
      handle.close();
      return;
    }
    if (patch.classification && classificationRank(patch.classification) < classificationRank(doc.classification)) {
      const ok = await confirmDialog({
        title: "Lower the classification?",
        message: `More people may be able to read this document once it is ${classificationLabel(patch.classification)}.`,
        confirmLabel: "Lower classification",
        tone: "danger",
      });
      if (!ok) return;
    }
    await withBusy(save, async () => {
      try {
        await api.patch(apiPath("/api/v1/documents", doc.id), patch);
        toast("Changes saved.", "success");
        handle.close();
        onSaved();
      } catch (error) {
        status.error(error);
      }
    }, "Saving\u2026");
  });
  const handle = openDialog({ title: "Edit document details", size: "lg", content: form });
}
