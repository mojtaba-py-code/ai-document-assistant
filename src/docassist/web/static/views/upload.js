/**
 * Upload dialogs: a new document (file or, when enabled, a URL import) and a new version
 * of an existing document. Shows upload progress, duplicate detection and the security
 * scan outcome (quarantine) in plain language.
 *
 * @module views/upload
 */

import { api, apiPath, upload } from "../api.js";
import { h, mount, uid } from "../dom.js";
import { CLASSIFICATIONS, formatBytes, TENANT_ROLES } from "../format.js";
import {
  button,
  callout,
  checkbox,
  errorCallout,
  field,
  input,
  openDialog,
  select,
  tabs,
  textarea,
  toast,
} from "../ui.js";
import { classificationOptions, departmentOptions, docTypeOptions, uploadableDepartments } from "./common.js";

export const ACCEPTED_EXTENSIONS = ["pdf", "docx", "xlsx", "csv", "txt", "md"];
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/**
 * Validates a chosen file on the client (the server re-validates everything: magic bytes,
 * size, archive structure, active content, malware).
 * @param {File | undefined} file
 * @returns {string} an error message, or "" when acceptable
 */
export function checkFile(file) {
  if (!file) return "Choose a file to upload.";
  const extension = (file.name.split(".").pop() || "").toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return `This file type is not supported. Use one of: ${ACCEPTED_EXTENSIONS.join(", ").toUpperCase()}.`;
  }
  if (file.size === 0) return "The file is empty.";
  if (file.size > MAX_UPLOAD_BYTES) return `The file is larger than ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  return "";
}

function fileDropZone(fileInput) {
  const summary = h("p", { class: "dropzone-file", "aria-live": "polite" }, "No file chosen");
  const zone = h(
    "div",
    { class: "dropzone" },
    h("p", { class: "dropzone-title" }, "Drag a file here, or"),
    h("label", { for: fileInput.id, class: "btn btn-secondary" }, "Choose a file"),
    fileInput,
    h("p", { class: "field-hint" }, `PDF, Word (.docx), Excel (.xlsx), CSV, text or Markdown, up to ${formatBytes(MAX_UPLOAD_BYTES)}.`),
    summary,
  );
  const describe = () => {
    const file = fileInput.files && fileInput.files[0];
    summary.textContent = file ? `${file.name} (${formatBytes(file.size)})` : "No file chosen";
  };
  fileInput.addEventListener("change", describe);
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dropzone-active");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dropzone-active"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dropzone-active");
    if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
      fileInput.files = event.dataTransfer.files;
      describe();
      fileInput.dispatchEvent(new Event("change"));
    }
  });
  return zone;
}

function metadataFields(departments) {
  const classification = select(classificationOptions(), { name: "classification", value: "INTERNAL", id: uid("cls") });
  const classificationHelp = h("p", { class: "field-hint", id: uid("cls-help"), "aria-live": "polite" });
  classification.setAttribute("aria-describedby", classificationHelp.id);
  const describe = () => {
    const found = CLASSIFICATIONS.find((c) => c.value === classification.value);
    classificationHelp.textContent = found ? `Who can read it: ${found.help}` : "";
  };
  classification.addEventListener("change", describe);
  describe();

  const allowed = uploadableDepartments(departments);
  const department = select(departmentOptions(allowed, "No department"), { name: "department_id" });
  if (allowed.length === 1) department.value = String(allowed[0].id);

  const roles = h(
    "fieldset",
    { class: "fieldset" },
    h("legend", { class: "field-label" }, "Limit to roles (optional)"),
    h("p", { class: "field-hint" }, "Leave empty to use the classification rules only."),
    h("div", { class: "check-grid" }, TENANT_ROLES.map((r) => checkbox(r.label, { name: "allowed_roles", value: r.value, dataset: { multi: "true" } }))),
  );

  return {
    classification,
    department,
    nodes: [
      field("Title", input({ name: "title", maxlength: 300, placeholder: "Defaults to the file name" })),
      h(
        "div",
        { class: "field-row" },
        h(
          "div",
          { class: "field" },
          h("label", { class: "field-label", for: classification.id }, "Classification", h("span", { class: "req", "aria-hidden": "true" }, " *")),
          classification,
          classificationHelp,
        ),
        field("Department", department, { hint: "Confidential documents are shared with this department." }),
      ),
      h(
        "div",
        { class: "field-row" },
        field("Document type", select(docTypeOptions({ any: "Detect automatically" }), { name: "doc_type" })),
        field("Tags", input({ name: "tags", maxlength: 1400, placeholder: "e.g. supplier, 2026" }), { hint: "Separate tags with commas." }),
      ),
      h("details", { class: "details" }, h("summary", null, "Advanced access options"), roles),
    ],
  };
}

function collect(form) {
  const values = {};
  const roles = [];
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.name === "allowed_roles") {
      if (element.checked) roles.push(element.value);
    } else if (element.type !== "file") {
      values[element.name] = typeof element.value === "string" ? element.value.trim() : element.value;
    }
  }
  values.allowed_roles = roles;
  return values;
}

function resultMessage(result, fileName) {
  const status = result && typeof result.status === "string" ? result.status : "processing";
  const findings = result && Array.isArray(result.findings) ? result.findings : [];
  if (status === "quarantined") {
    return callout(
      "warning",
      "Upload held for security review",
      `"${fileName}" was stored securely but will not be opened or indexed because the security scan found potentially unsafe content${findings.length ? ` (${findings.map((f) => (typeof f === "string" ? f : f.code || "finding")).join(", ")})` : ""}. A document manager can review it.`,
      { live: true },
    );
  }
  return callout(
    "success",
    "Upload complete",
    `"${fileName}" is being processed. It becomes searchable and available to the assistant once processing finishes (usually within a minute).`,
    { live: true },
  );
}

/**
 * Opens the "Upload document" dialog.
 * @param {{departments: any[], onUploaded?: (result: any) => void}} options
 */
export function openUploadDialog(options) {
  const fileTab = () => {
    const status = h("div", { class: "form-status" });
    const fileInput = h("input", { type: "file", name: "file", id: uid("file"), class: "visually-hidden-file", accept: ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(",") });
    const meta = metadataFields(options.departments);
    const progress = h("progress", { class: "progress", max: "100", value: "0", hidden: true, "aria-label": "Upload progress" });
    const submit = button("Upload", { type: "submit", variant: "primary", icon: "upload" });
    const form = h("form", { class: "form" }, fileDropZone(fileInput), meta.nodes, progress, status, h("div", { class: "form-actions" }, submit));

    const send = async (allowDuplicate) => {
      const file = fileInput.files && fileInput.files[0];
      const problem = checkFile(file);
      if (problem) {
        mount(status, callout("danger", null, problem, { live: true }));
        return;
      }
      const values = collect(form);
      const data = new FormData();
      data.append("file", file, file.name);
      if (values.title) data.append("title", values.title);
      data.append("classification", values.classification);
      if (values.department_id) data.append("department_id", values.department_id);
      if (values.doc_type) data.append("doc_type", values.doc_type);
      if (values.tags) data.append("tags", values.tags);
      if (values.allowed_roles.length) data.append("allowed_roles", values.allowed_roles.join(","));
      if (allowDuplicate) data.append("allow_duplicate", "true");

      submit.disabled = true;
      handle.dialog.dataset.busy = "true";
      progress.hidden = false;
      progress.value = 0;
      mount(status, h("p", { class: "muted", role: "status" }, "Uploading and scanning\u2026"));
      try {
        const result = await upload("/api/v1/documents", data, {
          onProgress: (fraction) => {
            progress.value = Math.round(fraction * 100);
          },
        });
        progress.value = 100;
        mount(status, resultMessage(result, file.name));
        const docId = result && (result.document_id || result.id);
        const actions = [button("Upload another", { onClick: () => { form.reset(); fileInput.dispatchEvent(new Event("change")); progress.hidden = true; status.replaceChildren(); submit.disabled = false; } })];
        if (docId) {
          actions.push(h("a", { class: "btn btn-primary", href: `#/documents/${encodeURIComponent(String(docId))}`, on: { click: () => handle.close() } }, "Open document"));
        }
        status.appendChild(h("div", { class: "form-actions" }, actions));
        if (options.onUploaded) options.onUploaded(result);
        toast(`Uploaded "${file.name}".`, "success");
      } catch (error) {
        progress.hidden = true;
        submit.disabled = false;
        const duplicate = error && error.status === 409 ? error.problem && error.problem.duplicate_of : null;
        if (duplicate) {
          mount(
            status,
            callout("warning", "This file is already in the library", "An identical file already exists in a document you can see.", {
              live: true,
              actions: [
                h("a", { class: "btn btn-secondary btn-sm", href: `#/documents/${encodeURIComponent(String(duplicate))}`, on: { click: () => handle.close() } }, "Open the existing document"),
                button("Upload anyway", { small: true, onClick: () => send(true) }),
              ],
            }),
          );
        } else if (error && error.name === "AbortError") {
          status.replaceChildren();
        } else {
          mount(status, errorCallout(error, { title: "Upload failed" }));
        }
      } finally {
        delete handle.dialog.dataset.busy;
      }
    };
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send(false);
    });
    return form;
  };

  const urlTab = () => {
    const status = h("div", { class: "form-status" });
    const url = input({ type: "url", name: "url", required: true, maxlength: 2048, placeholder: "Address of the file on an approved site" });
    const meta = metadataFields(options.departments);
    const submit = button("Import", { type: "submit", variant: "primary" });
    const form = h(
      "form",
      { class: "form" },
      field("File address (URL)", url, { hint: "Only sites approved by your administrator can be imported from." }),
      meta.nodes,
      status,
      h("div", { class: "form-actions" }, submit),
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const values = collect(form);
      const body = {
        url: values.url,
        classification: values.classification,
        title: values.title || undefined,
        department_id: values.department_id || undefined,
        doc_type: values.doc_type || undefined,
        tags: values.tags ? values.tags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
        allowed_roles: values.allowed_roles.length ? values.allowed_roles : undefined,
      };
      submit.disabled = true;
      mount(status, h("p", { class: "muted", role: "status" }, "Importing\u2026"));
      try {
        const result = await api.post("/api/v1/documents/import-url", body, { timeoutMs: 120_000 });
        mount(status, resultMessage(result, values.url));
        if (options.onUploaded) options.onUploaded(result);
      } catch (error) {
        mount(status, errorCallout(error, { title: "Import failed" }));
      } finally {
        submit.disabled = false;
      }
    });
    return form;
  };

  const handle = openDialog({
    title: "Upload document",
    size: "lg",
    content: tabs({
      label: "Upload method",
      items: [
        { id: "file", label: "From your computer", render: fileTab },
        { id: "url", label: "From a web address", render: urlTab },
      ],
    }),
  });
}

/**
 * Opens the "Upload new version" dialog for a document the user manages.
 * @param {{documentId: string, title: string, onUploaded?: (result: any) => void}} options
 */
export function openVersionDialog(options) {
  const status = h("div", { class: "form-status" });
  const fileInput = h("input", { type: "file", name: "file", id: uid("file"), class: "visually-hidden-file", accept: ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(",") });
  const note = textarea({ name: "change_note", maxlength: 500, rows: 3, placeholder: "What changed in this version?" });
  const progress = h("progress", { class: "progress", max: "100", value: "0", hidden: true, "aria-label": "Upload progress" });
  const submit = button("Upload version", { type: "submit", variant: "primary", icon: "upload" });
  const form = h(
    "form",
    { class: "form" },
    fileDropZone(fileInput),
    field("Change note", note, { hint: "Optional. Shown in the version history." }),
    callout("info", null, "The current version stays available until the new one has been processed."),
    progress,
    status,
    h("div", { class: "form-actions" }, submit),
  );
  const send = async (allowDuplicate) => {
    const file = fileInput.files && fileInput.files[0];
    const problem = checkFile(file);
    if (problem) {
      mount(status, callout("danger", null, problem, { live: true }));
      return;
    }
    const data = new FormData();
    data.append("file", file, file.name);
    if (note.value.trim()) data.append("change_note", note.value.trim());
    if (allowDuplicate) data.append("allow_duplicate", "true");
    submit.disabled = true;
    progress.hidden = false;
    handle.dialog.dataset.busy = "true";
    try {
      const result = await upload(apiPath("/api/v1/documents", options.documentId, "versions"), data, {
        onProgress: (fraction) => {
          progress.value = Math.round(fraction * 100);
        },
      });
      toast(`New version of "${options.title}" uploaded.`, "success");
      if (options.onUploaded) options.onUploaded(result);
      handle.close();
    } catch (error) {
      progress.hidden = true;
      submit.disabled = false;
      const duplicate = error && error.status === 409 ? error.problem && error.problem.duplicate_of : null;
      if (duplicate) {
        mount(
          status,
          callout("warning", "This file was uploaded before", "An identical file already exists in a document you can see.", {
            live: true,
            actions: [button("Upload anyway", { small: true, onClick: () => send(true) })],
          }),
        );
      } else {
        mount(status, errorCallout(error, { title: "Upload failed" }));
      }
    } finally {
      delete handle.dialog.dataset.busy;
    }
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    send(false);
  });
  const handle = openDialog({ title: `New version of "${options.title}"`, size: "md", content: form });
}
