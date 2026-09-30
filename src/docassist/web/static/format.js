/**
 * Formatting helpers and human-readable vocabularies.
 *
 * Everything here is defensive: unknown values fall back to a humanised version of the raw
 * string, and missing values render as an em dash, so a new backend field or enum member
 * never breaks a screen.
 *
 * @module format
 */

export const EMPTY = "\u2014";

/** Classifications ordered from least to most sensitive. */
export const CLASSIFICATIONS = [
  { value: "PUBLIC", label: "Public", help: "Anyone in your organisation; safe to share outside it." },
  { value: "INTERNAL", label: "Internal", help: "Everyone in your organisation." },
  { value: "CONFIDENTIAL", label: "Confidential", help: "Members of the document's department (and explicit grants)." },
  { value: "RESTRICTED", label: "Restricted", help: "Only the owner and people explicitly granted access." },
];

export const DOC_TYPES = [
  { value: "contract", label: "Contract" },
  { value: "invoice", label: "Invoice" },
  { value: "policy", label: "Policy" },
  { value: "hr", label: "HR" },
  { value: "technical", label: "Technical" },
  { value: "financial", label: "Financial" },
  { value: "legal", label: "Legal" },
  { value: "report", label: "Report" },
  { value: "other", label: "Other" },
];

export const DOCUMENT_STATUSES = [
  { value: "processing", label: "Processing" },
  { value: "ready", label: "Ready" },
  { value: "failed", label: "Failed" },
  { value: "quarantined", label: "Quarantined" },
];

export const ROLES = [
  { value: "organization_admin", label: "Organisation admin" },
  { value: "department_manager", label: "Department manager" },
  { value: "employee", label: "Employee" },
  { value: "auditor", label: "Auditor" },
  { value: "platform_admin", label: "Platform admin" },
];

/** Roles an organisation admin may assign (mirrors the backend ASSIGNABLE_ROLES). */
export const TENANT_ROLES = ROLES.filter((r) => r.value !== "platform_admin");

export const JOB_STATUSES = [
  { value: "queued", label: "Queued" },
  { value: "running", label: "Running" },
  { value: "succeeded", label: "Succeeded" },
  { value: "failed", label: "Failed (will retry)" },
  { value: "dead", label: "Dead" },
  { value: "cancelled", label: "Cancelled" },
];

const STATUS_LABELS = {
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
  quarantined: "Quarantined",
  deleted: "Deleted",
  uploaded: "Uploaded",
  indexed: "Indexed",
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  dead: "Dead",
  cancelled: "Cancelled",
  pending: "Preparing",
  expired: "Expired",
  active: "Active",
  disabled: "Disabled",
  suspended: "Suspended",
  success: "Success",
  denied: "Denied",
  failure: "Failure",
};

const FIELD_LABELS = {
  effective_date: "Effective date",
  expiration_date: "Expiration date",
  termination_date: "Termination date",
  renewal_date: "Renewal date",
  due_date: "Due date",
  invoice_date: "Invoice date",
  issue_date: "Issue date",
  payment_terms: "Payment terms",
  payment_deadline_days: "Payment deadline (days)",
  amount: "Amount",
  total: "Total",
  total_value: "Total value",
  subtotal: "Subtotal",
  tax: "Tax",
  currency: "Currency",
  parties: "Parties",
  party: "Party",
  invoice_number: "Invoice number",
  vendor: "Vendor",
  customer: "Customer",
  renewal_terms: "Renewal terms",
  late_penalty: "Late payment penalty",
  governing_law: "Governing law",
  termination_notice_days: "Termination notice (days)",
};

/** Plain-language explanations of ingestion error codes. */
const INGESTION_ERRORS = {
  parse_error: "The file could not be read. It may be damaged or saved in an unusual format.",
  parse_timeout: "Reading the file took too long. Try splitting it into smaller documents.",
  unsupported: "This file format is not supported.",
  empty_document: "No text could be found in this file. Scanned documents may need OCR.",
  too_large: "The file is too large to process.",
  ocr_unavailable: "The file looks scanned and text recognition (OCR) is not available.",
};

/**
 * Replaces underscores/dashes with spaces and capitalises the first letter.
 * @param {unknown} value
 * @returns {string}
 */
export function humanize(value) {
  if (value === null || value === undefined || value === "") return EMPTY;
  const text = String(value).replace(/[_-]+/g, " ").trim();
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : EMPTY;
}

function labelFrom(list, value) {
  const found = list.find((item) => item.value === value);
  return found ? found.label : humanize(value);
}

export const classificationLabel = (value) => labelFrom(CLASSIFICATIONS, String(value || "").toUpperCase());
export const docTypeLabel = (value) => labelFrom(DOC_TYPES, value);
export const roleLabel = (value) => labelFrom(ROLES, value);
export const statusLabel = (value) => STATUS_LABELS[value] || humanize(value);
export const fieldLabel = (value) => FIELD_LABELS[value] || humanize(value);

/**
 * @param {string} code
 * @returns {string}
 */
export function ingestionErrorText(code) {
  if (!code) return "Processing failed.";
  return INGESTION_ERRORS[code] || `Processing failed (${humanize(code)}).`;
}

/**
 * Rank of a classification (unknown \u2192 -1).
 * @param {string} value
 * @returns {number}
 */
export function classificationRank(value) {
  return CLASSIFICATIONS.findIndex((c) => c.value === String(value || "").toUpperCase());
}

/**
 * @param {unknown} value
 * @returns {Date | null}
 */
export function toDate(value) {
  if (value === null || value === undefined || value === "") return null;
  const date = value instanceof Date ? value : new Date(String(value));
  return Number.isNaN(date.getTime()) ? null : date;
}

const dateFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });
const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});
const relativeFormat = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
const numberFormat = new Intl.NumberFormat();

/**
 * Formats a calendar date. Date-only ISO strings (YYYY-MM-DD) are shown as-is in the local
 * calendar (not shifted by the UTC offset).
 * @param {unknown} value
 * @returns {string}
 */
export function formatDate(value) {
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [y, m, d] = value.split("-").map(Number);
    return dateFormat.format(new Date(y, m - 1, d));
  }
  const date = toDate(value);
  return date ? dateFormat.format(date) : EMPTY;
}

/** @param {unknown} value @returns {string} */
export function formatDateTime(value) {
  const date = toDate(value);
  return date ? dateTimeFormat.format(date) : EMPTY;
}

/**
 * "3 minutes ago", "in 2 days".
 * @param {unknown} value
 * @returns {string}
 */
export function relativeTime(value) {
  const date = toDate(value);
  if (!date) return EMPTY;
  const seconds = Math.round((date.getTime() - Date.now()) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 45) return relativeFormat.format(0, "second");
  if (abs < 2700) return relativeFormat.format(Math.round(seconds / 60), "minute");
  if (abs < 79_200) return relativeFormat.format(Math.round(seconds / 3600), "hour");
  if (abs < 2_592_000) return relativeFormat.format(Math.round(seconds / 86_400), "day");
  return formatDate(date);
}

/** @param {unknown} value @returns {string} */
export function formatNumber(value) {
  const number = Number(value);
  return value === null || value === undefined || value === "" || !Number.isFinite(number)
    ? EMPTY
    : numberFormat.format(number);
}

/**
 * Human-readable byte size.
 * @param {unknown} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return EMPTY;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

/**
 * Formats a 0..1 fraction as a percentage.
 * @param {unknown} fraction
 * @returns {string}
 */
export function formatPercent(fraction) {
  const value = Number(fraction);
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : EMPTY;
}

/**
 * Page reference like "p. 4" or "pp. 4\u20136".
 * @param {unknown} start
 * @param {unknown} [end]
 * @returns {string}
 */
export function pageRef(start, end) {
  const a = Number(start);
  const b = Number(end);
  if (!Number.isFinite(a) || a <= 0) return "";
  if (Number.isFinite(b) && b > a) return `pp. ${a}\u2013${b}`;
  return `p. ${a}`;
}

/**
 * Coerces a value to display text (objects are summarised as JSON, never interpreted).
 * @param {unknown} value
 * @returns {string}
 */
export function displayValue(value) {
  if (value === null || value === undefined || value === "") return EMPTY;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return EMPTY;
    }
  }
  return String(value);
}

/**
 * Shortens an identifier for display ("0190c2a1\u2026").
 * @param {unknown} id
 * @returns {string}
 */
export function shortId(id) {
  const text = String(id ?? "");
  return text.length > 12 ? `${text.slice(0, 8)}\u2026` : text || EMPTY;
}

/**
 * Days between today (local) and a date; negative when in the past.
 * @param {unknown} value
 * @returns {number | null}
 */
export function daysUntil(value) {
  let date;
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [y, m, d] = value.split("-").map(Number);
    date = new Date(y, m - 1, d);
  } else {
    date = toDate(value);
  }
  if (!date) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  date.setHours(0, 0, 0, 0);
  return Math.round((date.getTime() - today.getTime()) / 86_400_000);
}
