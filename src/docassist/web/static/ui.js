/**
 * Reusable, accessible UI components built on {@link module:dom}.
 *
 * Conventions: components return DOM nodes; every visible string is inserted as text;
 * dialogs use the native `<dialog>` element (focus trapping and Escape handling for free)
 * and restore focus to the opener; asynchronous status goes to an `aria-live` region.
 *
 * @module ui
 */

import { append, h, mount, uid } from "./dom.js";
import {
  classificationLabel,
  EMPTY,
  statusLabel,
} from "./format.js";

/* ------------------------------------------------------------------ primitives */

/**
 * Decorative icon (CSS mask over a same-origin SVG); hidden from assistive technology.
 * @param {string} name
 * @returns {HTMLElement}
 */
export function icon(name) {
  return h("span", { class: ["icon", `icon-${name}`], "aria-hidden": "true" });
}

/**
 * @param {string} label
 * @param {object} [options]
 * @param {"primary"|"secondary"|"danger"|"ghost"|"link"} [options.variant]
 * @param {string} [options.icon]
 * @param {(event: Event) => void} [options.onClick]
 * @param {"button"|"submit"|"reset"} [options.type]
 * @param {boolean} [options.small]
 * @param {boolean} [options.disabled]
 * @param {string} [options.ariaLabel] accessible name when it differs from the label
 * @param {string} [options.title]
 * @param {string|string[]} [options.className]
 * @returns {HTMLButtonElement}
 */
export function button(label, options = {}) {
  const { variant = "secondary", onClick, type = "button", small, disabled, ariaLabel, title, className } = options;
  return h(
    "button",
    {
      type,
      class: ["btn", `btn-${variant}`, small && "btn-sm", ...(Array.isArray(className) ? className : [className])].filter(
        Boolean,
      ),
      disabled: Boolean(disabled),
      "aria-label": ariaLabel,
      title,
      on: onClick ? { click: onClick } : undefined,
    },
    options.icon && icon(options.icon),
    h("span", { class: "btn-label" }, label),
  );
}

/**
 * An anchor styled as a button, for navigation.
 * @param {string} label
 * @param {string} target hash href such as "#/documents"
 * @param {{variant?: string, icon?: string, small?: boolean}} [options]
 * @returns {HTMLAnchorElement}
 */
export function linkButton(label, target, options = {}) {
  const { variant = "secondary", small } = options;
  return h(
    "a",
    { href: target, class: ["btn", `btn-${variant}`, small && "btn-sm"].filter(Boolean) },
    options.icon && icon(options.icon),
    h("span", { class: "btn-label" }, label),
  );
}

/**
 * Wraps a form control with its label, optional hint and an error slot. The control gets
 * an id (if missing) and `aria-describedby` for the hint.
 * @param {string} label
 * @param {HTMLElement} control
 * @param {{hint?: string, required?: boolean, className?: string, inline?: boolean}} [options]
 * @returns {HTMLElement}
 */
export function field(label, control, options = {}) {
  if (!control.id) control.id = uid("field");
  const hintId = options.hint ? uid("hint") : null;
  if (hintId) control.setAttribute("aria-describedby", hintId);
  if (options.required) control.required = true;
  return h(
    "div",
    { class: ["field", options.inline && "field-inline", options.className].filter(Boolean) },
    h("label", { for: control.id, class: "field-label" }, label, options.required && h("span", { class: "req", "aria-hidden": "true" }, " *")),
    control,
    hintId && h("p", { id: hintId, class: "field-hint" }, options.hint),
  );
}

/**
 * @param {Record<string, any>} [props] attributes for the input
 * @returns {HTMLInputElement}
 */
export function input(props = {}) {
  return h("input", { type: "text", class: "input", ...props });
}

/**
 * @param {Record<string, any>} [props]
 * @returns {HTMLTextAreaElement}
 */
export function textarea(props = {}) {
  return h("textarea", { class: "input textarea", rows: 4, ...props });
}

/**
 * @param {{value: string, label: string}[]} options
 * @param {Record<string, any>} [props] attributes; `value` selects an option
 * @returns {HTMLSelectElement}
 */
export function select(options, props = {}) {
  const { value, ...rest } = props;
  const element = h(
    "select",
    { class: "input select", ...rest },
    options.map((option) => h("option", { value: option.value }, option.label)),
  );
  if (value !== undefined && value !== null) element.value = String(value);
  return element;
}

/**
 * A checkbox with its label (the label wraps the input).
 * @param {string} label
 * @param {Record<string, any>} [props]
 * @returns {HTMLLabelElement}
 */
export function checkbox(label, props = {}) {
  const box = h("input", { type: "checkbox", class: "checkbox", ...props });
  return h("label", { class: "check" }, box, h("span", null, label));
}

/**
 * Radio-button group rendered as a segmented control.
 * @param {string} legend accessible group name
 * @param {{value: string, label: string}[]} options
 * @param {{name?: string, value?: string, onChange?: (value: string) => void}} [config]
 * @returns {HTMLFieldSetElement}
 */
export function segmented(legend, options, config = {}) {
  const name = config.name || uid("seg");
  return h(
    "fieldset",
    { class: "segmented" },
    h("legend", { class: "visually-hidden" }, legend),
    options.map((option) => {
      const id = uid("opt");
      return h(
        "span",
        { class: "segment" },
        h("input", {
          type: "radio",
          id,
          name,
          value: option.value,
          checked: option.value === config.value,
          on: { change: () => config.onChange && config.onChange(option.value) },
        }),
        h("label", { for: id }, option.label),
      );
    }),
  );
}

/* ------------------------------------------------------------------ badges */

/**
 * @param {string} text
 * @param {string} [tone] neutral | info | success | warning | danger | accent
 * @returns {HTMLElement}
 */
export function badge(text, tone = "neutral") {
  return h("span", { class: ["badge", `badge-${tone}`] }, text);
}

const CLASSIFICATION_TONES = { PUBLIC: "success", INTERNAL: "info", CONFIDENTIAL: "warning", RESTRICTED: "danger" };
const STATUS_TONES = {
  ready: "success",
  indexed: "success",
  succeeded: "success",
  success: "success",
  active: "success",
  processing: "info",
  uploaded: "info",
  queued: "info",
  running: "info",
  pending: "info",
  failed: "danger",
  dead: "danger",
  failure: "danger",
  denied: "warning",
  quarantined: "danger",
  deleted: "neutral",
  cancelled: "neutral",
  expired: "neutral",
  disabled: "neutral",
  suspended: "warning",
};

/**
 * Classification badge with an accessible prefix.
 * @param {string} value
 * @returns {HTMLElement}
 */
export function classificationBadge(value) {
  const key = String(value || "").toUpperCase();
  const element = badge(classificationLabel(key), CLASSIFICATION_TONES[key] || "neutral");
  element.classList.add("badge-classification");
  element.prepend(h("span", { class: "visually-hidden" }, "Classification: "));
  return element;
}

/**
 * Status badge; "processing"-like states get an animated dot.
 * @param {string} value
 * @returns {HTMLElement}
 */
export function statusBadge(value) {
  const key = String(value || "").toLowerCase();
  const tone = STATUS_TONES[key] || "neutral";
  const element = badge(statusLabel(key), tone);
  if (["processing", "queued", "running", "pending", "uploaded"].includes(key)) {
    element.prepend(h("span", { class: "pulse", "aria-hidden": "true" }));
  }
  return element;
}

/* ------------------------------------------------------------------ feedback */

/**
 * Human-readable text for an error (ApiError or anything else).
 * @param {any} error
 * @returns {string}
 */
export function errorMessage(error) {
  if (!error) return "Something went wrong.";
  if (error.name === "AbortError") return "The request was cancelled.";
  const status = typeof error.status === "number" ? error.status : null;
  const detail = typeof error.detail === "string" && error.detail ? error.detail : "";
  if (status === 0) return detail || "Cannot reach the server.";
  if (status === 429) {
    const wait = error.retryAfter ? ` Please wait about ${error.retryAfter} seconds.` : " Please wait a moment.";
    return `Too many requests.${wait}`;
  }
  if (status === 413) return detail || "The file or request is too large.";
  if (status === 404) return detail && detail !== "Not Found" ? detail : "Not found, or you do not have access to it.";
  if (status === 403) return detail || "You do not have permission to do that.";
  if (status === 422 && error.errors && error.errors.length) {
    const parts = error.errors.slice(0, 5).map((e) => {
      const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
      return loc ? `${loc}: ${e.msg}` : String(e.msg || "invalid");
    });
    return `${detail || "Please check your input."} ${parts.join("; ")}`;
  }
  if (status && status >= 500) {
    return detail && error.code !== "internal_error"
      ? detail
      : "Something went wrong on our side. Please try again in a moment.";
  }
  if (detail) return detail;
  return typeof error.message === "string" && error.message ? error.message : "Something went wrong.";
}

/**
 * Inline message box.
 * @param {"info"|"success"|"warning"|"danger"} tone
 * @param {string | null} title
 * @param {unknown} [body] text or nodes
 * @param {{actions?: Node[], live?: boolean}} [options]
 * @returns {HTMLElement}
 */
export function callout(tone, title, body, options = {}) {
  const role = options.live ? (tone === "danger" ? "alert" : "status") : undefined;
  return h(
    "div",
    { class: ["callout", `callout-${tone}`], role },
    icon(tone === "success" ? "check" : tone === "info" ? "info" : "alert"),
    h(
      "div",
      { class: "callout-body" },
      title && h("p", { class: "callout-title" }, title),
      body !== undefined && body !== null && h("div", { class: "callout-text" }, body),
      options.actions && options.actions.length > 0 && h("div", { class: "callout-actions" }, options.actions),
    ),
  );
}

/**
 * Error callout for a failed request with an optional retry action.
 * @param {any} error
 * @param {{retry?: () => void, title?: string}} [options]
 * @returns {HTMLElement}
 */
export function errorCallout(error, options = {}) {
  const reference = error && error.requestId ? h("span", { class: "muted small" }, ` Reference: ${error.requestId}`) : null;
  const actions = options.retry ? [button("Try again", { onClick: options.retry, small: true })] : [];
  return callout("danger", options.title || "Something went wrong", [errorMessage(error), reference], {
    actions,
    live: true,
  });
}

/**
 * A slot that shows form-level errors/success messages (announced to screen readers).
 * @returns {{element: HTMLElement, error: (e: any) => void, success: (msg: string) => void, info: (msg: string) => void, clear: () => void}}
 */
export function formStatus() {
  const element = h("div", { class: "form-status" });
  return {
    element,
    error(err) {
      mount(element, errorCallout(err, { title: "That did not work" }));
    },
    success(message) {
      mount(element, callout("success", null, message, { live: true }));
    },
    info(message) {
      mount(element, callout("info", null, message, { live: true }));
    },
    clear() {
      element.replaceChildren();
    },
  };
}

/**
 * @param {string} [label]
 * @returns {HTMLElement}
 */
export function spinner(label = "Loading") {
  return h("span", { class: "spinner", role: "status" }, h("span", { class: "visually-hidden" }, label));
}

/**
 * @param {string} [label]
 * @returns {HTMLElement}
 */
export function loadingBlock(label = "Loading\u2026") {
  return h("div", { class: "loading-block" }, h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { role: "status" }, label));
}

/**
 * @param {{title: string, text?: string, action?: Node, icon?: string}} options
 * @returns {HTMLElement}
 */
export function emptyState(options) {
  return h(
    "div",
    { class: "empty" },
    icon(options.icon || "inbox"),
    h("p", { class: "empty-title" }, options.title),
    options.text && h("p", { class: "empty-text" }, options.text),
    options.action,
  );
}

let toastRoot = null;

/**
 * Shows a transient notification (polite live region; errors use role=alert).
 * @param {string} message
 * @param {"info"|"success"|"warning"|"danger"} [tone]
 */
export function toast(message, tone = "info") {
  toastRoot = toastRoot || document.getElementById("toasts");
  if (!toastRoot) return;
  const close = () => item.remove();
  const item = h(
    "div",
    { class: ["toast", `toast-${tone}`], role: tone === "danger" ? "alert" : "status" },
    icon(tone === "success" ? "check" : tone === "danger" || tone === "warning" ? "alert" : "info"),
    h("span", { class: "toast-text" }, message),
    h("button", { type: "button", class: "toast-close", "aria-label": "Dismiss notification", on: { click: close } }, "\u00d7"),
  );
  toastRoot.appendChild(item);
  setTimeout(close, tone === "danger" ? 10_000 : 6_000);
}

/**
 * Announces a message to screen-reader users without moving focus.
 * @param {string} message
 */
export function announce(message) {
  const region = document.getElementById("live-region");
  if (!region) return;
  region.textContent = "";
  setTimeout(() => {
    region.textContent = message;
  }, 50);
}

/* ------------------------------------------------------------------ layout */

/**
 * Page header with the view's `<h1>` (focus target after navigation).
 * @param {{title: string, subtitle?: string, actions?: Node[], back?: {href: string, label: string}, meta?: Node[]}} options
 * @returns {HTMLElement}
 */
export function pageHeader(options) {
  return h(
    "header",
    { class: "page-header" },
    options.back && h("a", { class: "back-link", href: options.back.href }, icon("arrow-left"), options.back.label),
    h(
      "div",
      { class: "page-header-row" },
      h(
        "div",
        { class: "page-heading" },
        h("h1", { class: "page-title", tabindex: "-1" }, options.title),
        options.meta && options.meta.length > 0 && h("div", { class: "page-meta" }, options.meta),
        options.subtitle && h("p", { class: "page-subtitle" }, options.subtitle),
      ),
      options.actions && options.actions.length > 0 && h("div", { class: "page-actions" }, options.actions),
    ),
  );
}

/**
 * A titled panel.
 * @param {{title?: string, level?: number, actions?: Node[], body?: unknown, className?: string, description?: string}} options
 * @returns {HTMLElement}
 */
export function card(options) {
  const headingTag = `h${options.level || 2}`;
  return h(
    "section",
    { class: ["card", options.className].filter(Boolean) },
    (options.title || options.actions) &&
      h(
        "div",
        { class: "card-header" },
        h("div", null, options.title && h(headingTag, { class: "card-title" }, options.title), options.description && h("p", { class: "card-description" }, options.description)),
        options.actions && h("div", { class: "card-actions" }, options.actions),
      ),
    options.body !== undefined && h("div", { class: "card-body" }, options.body),
  );
}

/**
 * Definition list; entries whose value is null/undefined/"" are skipped.
 * @param {[string, unknown][]} entries
 * @returns {HTMLDListElement}
 */
export function descriptionList(entries) {
  return h(
    "dl",
    { class: "dl" },
    entries
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([term, value]) => h("div", { class: "dl-row" }, h("dt", null, term), h("dd", null, value instanceof Node ? value : String(value)))),
  );
}

/**
 * @typedef {object} Column
 * @property {string} label header text
 * @property {(row: any) => unknown} render cell content (text or nodes)
 * @property {string} [className]
 * @property {boolean} [primary] rendered as the row header (`th scope=row`)
 */

/**
 * Responsive data table (collapses into labelled cards on narrow screens).
 * @param {{caption: string, columns: Column[], rows: any[], empty?: Node, rowClass?: (row: any) => string | null}} options
 * @returns {HTMLElement}
 */
export function dataTable(options) {
  if (!options.rows.length && options.empty) return options.empty;
  const body = h("tbody");
  appendRows(body, options.columns, options.rows, options.rowClass);
  const table = h(
    "table",
    { class: "table" },
    h("caption", { class: "visually-hidden" }, options.caption),
    h("thead", null, h("tr", null, options.columns.map((c) => h("th", { scope: "col", class: c.className }, c.label)))),
    body,
  );
  const wrapper = h("div", { class: "table-wrap" }, table);
  wrapper.appendRows = (rows) => appendRows(body, options.columns, rows, options.rowClass);
  return wrapper;
}

function appendRows(body, columns, rows, rowClass) {
  for (const row of rows) {
    const cls = rowClass ? rowClass(row) : null;
    body.appendChild(
      h(
        "tr",
        { class: cls || undefined },
        columns.map((column) => {
          let content;
          try {
            content = column.render(row);
          } catch {
            content = EMPTY;
          }
          const isEmpty = content === null || content === undefined || content === "";
          return h(
            column.primary ? "th" : "td",
            { class: column.className, scope: column.primary ? "row" : undefined, dataset: { label: column.label } },
            isEmpty ? EMPTY : content,
          );
        }),
      ),
    );
  }
}

/**
 * Accessible tabs (roving tabindex, arrow keys, Home/End). Panels render lazily.
 * @param {{label: string, items: {id: string, label: string, render: () => Node}[], selected?: string, onSelect?: (id: string) => void}} options
 * @returns {HTMLElement & {select: (id: string) => void}}
 */
export function tabs(options) {
  const base = uid("tabs");
  const list = h("div", { class: "tablist", role: "tablist", "aria-label": options.label });
  const panels = h("div", { class: "tabpanels" });
  const buttons = new Map();
  const rendered = new Map();
  const container = h("div", { class: "tabs" }, list, panels);

  const select = (id, focus = false) => {
    if (!buttons.has(id)) id = options.items[0].id;
    for (const [key, tab] of buttons) {
      const active = key === id;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (focus && active) tab.focus();
    }
    for (const [key, panel] of rendered) panel.hidden = key !== id;
    if (!rendered.has(id)) {
      const item = options.items.find((i) => i.id === id);
      const panel = h("div", {
        class: "tabpanel",
        role: "tabpanel",
        id: `${base}-panel-${id}`,
        "aria-labelledby": `${base}-tab-${id}`,
        tabindex: "0",
      });
      try {
        append(panel, item.render());
      } catch (error) {
        append(panel, errorCallout(error));
      }
      rendered.set(id, panel);
      panels.appendChild(panel);
    }
    if (options.onSelect) options.onSelect(id);
  };

  options.items.forEach((item, index) => {
    const tab = h(
      "button",
      {
        type: "button",
        role: "tab",
        class: "tab",
        id: `${base}-tab-${item.id}`,
        "aria-controls": `${base}-panel-${item.id}`,
        on: {
          click: () => select(item.id),
          keydown: (event) => {
            const ids = options.items.map((i) => i.id);
            let next = null;
            if (event.key === "ArrowRight") next = ids[(index + 1) % ids.length];
            else if (event.key === "ArrowLeft") next = ids[(index - 1 + ids.length) % ids.length];
            else if (event.key === "Home") next = ids[0];
            else if (event.key === "End") next = ids[ids.length - 1];
            if (next) {
              event.preventDefault();
              select(next, true);
            }
          },
        },
      },
      item.label,
    );
    buttons.set(item.id, tab);
    list.appendChild(tab);
  });
  container.select = (id) => select(id, false);
  container.rerender = (id) => {
    const panel = rendered.get(id);
    if (panel) {
      panel.remove();
      rendered.delete(id);
    }
  };
  select(options.selected || options.items[0].id);
  return container;
}

/* ------------------------------------------------------------------ dialogs */

/**
 * Opens a modal dialog. Returns helpers to close it. Focus returns to the element that
 * was focused before opening.
 * @param {{title: string, content: unknown, actions?: Node[], size?: "sm"|"md"|"lg", onClose?: () => void, description?: string}} options
 * @returns {{dialog: HTMLDialogElement, close: () => void, body: HTMLElement, footer: HTMLElement}}
 */
export function openDialog(options) {
  const titleId = uid("dlg-title");
  const opener = document.activeElement;
  const body = h("div", { class: "dialog-body" }, options.content);
  const footer = h("div", { class: "dialog-footer" }, options.actions || []);
  const dialog = h(
    "dialog",
    { class: ["dialog", `dialog-${options.size || "md"}`], "aria-labelledby": titleId },
    h(
      "div",
      { class: "dialog-header" },
      h("h2", { id: titleId, class: "dialog-title" }, options.title),
      h("button", { type: "button", class: "icon-btn", "aria-label": "Close dialog", on: { click: () => close() } }, icon("close")),
    ),
    options.description && h("p", { class: "dialog-description" }, options.description),
    body,
    footer,
  );
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    if (dialog.open) dialog.close();
    dialog.remove();
    if (options.onClose) options.onClose();
    if (opener && typeof opener.focus === "function" && document.contains(opener)) opener.focus();
  };
  dialog.addEventListener("close", close);
  dialog.addEventListener("cancel", (event) => {
    if (dialog.dataset.busy === "true") event.preventDefault();
  });
  document.body.appendChild(dialog);
  dialog.showModal();
  // Start keyboard users in the first form control rather than on the close button.
  const firstControl = body.querySelector("input:not([type=hidden]):not([disabled]), select:not([disabled]), textarea:not([disabled])");
  if (firstControl) firstControl.focus();
  return { dialog, close, body, footer };
}

/**
 * Confirmation dialog.
 * @param {{title: string, message: unknown, confirmLabel?: string, tone?: "danger"|"primary"}} options
 * @returns {Promise<boolean>}
 */
export function confirmDialog(options) {
  return new Promise((resolve) => {
    let result = false;
    const confirmButton = button(options.confirmLabel || "Confirm", {
      variant: options.tone === "danger" ? "danger" : "primary",
      onClick: () => {
        result = true;
        handle.close();
      },
    });
    const handle = openDialog({
      title: options.title,
      size: "sm",
      content: h("div", { class: "confirm-text" }, options.message),
      actions: [button("Cancel", { onClick: () => handle.close() }), confirmButton],
      onClose: () => resolve(result),
    });
    confirmButton.focus();
  });
}

/* ------------------------------------------------------------------ helpers */

/**
 * Runs an async action while a button shows a busy state (prevents double submits).
 * @template T
 * @param {HTMLButtonElement | null} control
 * @param {() => Promise<T>} action
 * @param {string} [busyLabel]
 * @returns {Promise<T>}
 */
export async function withBusy(control, action, busyLabel) {
  if (!control) return action();
  const labelNode = control.querySelector(".btn-label");
  const original = labelNode ? labelNode.textContent : null;
  control.disabled = true;
  control.setAttribute("aria-busy", "true");
  if (labelNode && busyLabel) labelNode.textContent = busyLabel;
  try {
    return await action();
  } finally {
    control.disabled = false;
    control.removeAttribute("aria-busy");
    if (labelNode && original !== null) labelNode.textContent = original;
  }
}

/**
 * Button that copies text to the clipboard.
 * @param {() => string} getText
 * @param {string} [label]
 * @returns {HTMLButtonElement}
 */
export function copyButton(getText, label = "Copy") {
  const control = button(label, {
    small: true,
    icon: "copy",
    onClick: async () => {
      try {
        await navigator.clipboard.writeText(getText());
        toast("Copied to clipboard.", "success");
      } catch {
        toast("Copy failed. Select the text and copy it manually.", "warning");
      }
    },
  });
  return control;
}

/**
 * Repeatedly runs `task` every `intervalMs` until `signal` aborts or `task` returns false.
 * The interval grows gently (up to 3x) to limit load on long-running jobs.
 * @param {AbortSignal} signal
 * @param {() => Promise<boolean | void>} task
 * @param {number} [intervalMs]
 */
export function poll(signal, task, intervalMs = 3000) {
  let delay = intervalMs;
  let timer = 0;
  const tick = async () => {
    if (signal.aborted) return;
    let keepGoing = true;
    try {
      keepGoing = (await task()) !== false;
    } catch {
      keepGoing = true;
    }
    if (!keepGoing || signal.aborted) return;
    delay = Math.min(delay * 1.25, intervalMs * 3);
    timer = setTimeout(tick, delay);
  };
  signal.addEventListener("abort", () => clearTimeout(timer), { once: true });
  timer = setTimeout(tick, delay);
}

/**
 * Debounces a function.
 * @param {Function} fn
 * @param {number} [wait]
 * @returns {Function}
 */
export function debounce(fn, wait = 300) {
  let timer = 0;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

/**
 * Reads a form's named controls into a plain object (checkboxes \u2192 boolean, multi-selects
 * and repeated names \u2192 arrays, empty strings trimmed to "").
 * @param {HTMLFormElement} form
 * @returns {Record<string, any>}
 */
export function formValues(form) {
  const values = {};
  for (const element of form.elements) {
    if (!element.name || element.disabled) continue;
    if (element.type === "checkbox") {
      if (element.dataset.multi === "true") {
        values[element.name] = values[element.name] || [];
        if (element.checked) values[element.name].push(element.value);
      } else {
        values[element.name] = element.checked;
      }
    } else if (element.type === "radio") {
      if (element.checked) values[element.name] = element.value;
    } else if (element.type === "file") {
      values[element.name] = element.files;
    } else if (element.tagName === "SELECT" && element.multiple) {
      values[element.name] = [...element.selectedOptions].map((o) => o.value);
    } else {
      values[element.name] = typeof element.value === "string" ? element.value.trim() : element.value;
    }
  }
  return values;
}

/**
 * "Load more" button that calls `loader` and hides itself when there is nothing left.
 * @param {() => Promise<boolean>} loader resolves true when more pages remain
 * @returns {HTMLElement}
 */
export function loadMoreButton(loader) {
  const control = button("Load more", {
    onClick: async () => {
      try {
        const more = await withBusy(control, loader, "Loading\u2026");
        if (!more) control.hidden = true;
      } catch (error) {
        toast(errorMessage(error), "danger");
      }
    },
  });
  return h("div", { class: "load-more" }, control);
}
