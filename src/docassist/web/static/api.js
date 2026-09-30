/**
 * API client.
 *
 * Session model:
 * - The short-lived access token lives ONLY in this module's memory (never in Web Storage,
 *   cookies or the URL), so an injected script cannot harvest a long-lived credential from
 *   storage and a reload starts clean.
 * - The refresh token is an `HttpOnly; SameSite=Strict` cookie set by the server. It is
 *   exchanged via `POST /api/v1/auth/refresh` with `credentials: "same-origin"` plus the
 *   `X-CSRF-Protection: 1` header that a cross-site form cannot send.
 * - Refresh is single-flight: concurrent 401s share one refresh request, and the Web Locks
 *   API serialises refreshes across tabs so two tabs never present the same (rotated)
 *   refresh token - the server treats token reuse as theft and revokes the session.
 * - Errors are RFC 9457 problem documents; {@link ApiError} carries the safe fields.
 *
 * @module api
 */

import { h } from "./dom.js";

const API_PREFIX = "/api/";
const REFRESH_PATH = "/api/v1/auth/refresh";
const CSRF_HEADER = "X-CSRF-Protection";
const DEFAULT_TIMEOUT_MS = 30_000;
const REFRESH_LEAD_MS = 60_000;

let accessToken = null;
let accessExpiresAt = 0;
let refreshTimer = 0;
let refreshInFlight = null;
const expiredListeners = new Set();

/** Error raised for every failed API call (network failures use status 0). */
export class ApiError extends Error {
  /**
   * @param {number} status HTTP status (0 when the server could not be reached)
   * @param {Record<string, any>} [problem] parsed problem+json body
   * @param {Headers} [headers]
   */
  constructor(status, problem = {}, headers = undefined) {
    const detail = typeof problem.detail === "string" ? problem.detail : "";
    super(detail || `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.code = typeof problem.code === "string" ? problem.code : "";
    this.title = typeof problem.title === "string" ? problem.title : "";
    this.detail = detail;
    this.requestId = typeof problem.request_id === "string" ? problem.request_id : "";
    this.errors = Array.isArray(problem.errors) ? problem.errors : [];
    this.problem = problem;
    const retry = headers ? Number.parseInt(headers.get("Retry-After") || "", 10) : Number.NaN;
    this.retryAfter = Number.isFinite(retry) ? retry : null;
  }
}

/**
 * Registers a callback invoked when the session can no longer be refreshed.
 * @param {() => void} listener
 * @returns {() => void} unsubscribe
 */
export function onSessionExpired(listener) {
  expiredListeners.add(listener);
  return () => expiredListeners.delete(listener);
}

function emitExpired() {
  for (const listener of expiredListeners) {
    try {
      listener();
    } catch {
      // A failing listener must not break the others.
    }
  }
}

/** @returns {boolean} whether an access token is currently held in memory */
export function isAuthenticated() {
  return accessToken !== null;
}

/**
 * Stores the token pair returned by login / MFA / refresh (access token only; the refresh
 * token stays in its HttpOnly cookie) and schedules a proactive refresh.
 * @param {{access_token?: string, expires_at?: string}} tokens
 */
export function acceptTokens(tokens) {
  if (!tokens || typeof tokens.access_token !== "string" || !tokens.access_token) {
    throw new ApiError(0, { detail: "The server returned an unexpected sign-in response." });
  }
  accessToken = tokens.access_token;
  const parsed = Date.parse(tokens.expires_at || "");
  accessExpiresAt = Number.isFinite(parsed) ? parsed : Date.now() + 5 * 60_000;
  scheduleRefresh();
}

/** Forgets the in-memory access token (the caller handles server-side logout). */
export function clearTokens() {
  accessToken = null;
  accessExpiresAt = 0;
  if (refreshTimer) {
    clearTimeout(refreshTimer);
    refreshTimer = 0;
  }
}

function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  const delay = Math.max(5_000, accessExpiresAt - Date.now() - REFRESH_LEAD_MS);
  refreshTimer = setTimeout(() => {
    refreshTimer = 0;
    refreshSession().then((ok) => {
      if (!ok) emitExpired();
    });
  }, delay);
}

async function withCrossTabLock(task) {
  const locks = globalThis.navigator && navigator.locks;
  if (locks && typeof locks.request === "function") {
    return locks.request("docassist-token-refresh", task);
  }
  return task();
}

async function performRefresh() {
  let response;
  try {
    response = await fetch(REFRESH_PATH, {
      method: "POST",
      credentials: "same-origin",
      cache: "no-store",
      headers: { Accept: "application/json", [CSRF_HEADER]: "1" },
    });
  } catch {
    return false;
  }
  if (!response.ok) {
    clearTokens();
    return false;
  }
  try {
    acceptTokens(await response.json());
    return true;
  } catch {
    clearTokens();
    return false;
  }
}

/**
 * Exchanges the refresh cookie for a new access token. Concurrent callers share one
 * request (single-flight) and other tabs wait on a Web Lock.
 * @returns {Promise<boolean>} true when a fresh access token is now held
 */
export function refreshSession() {
  if (!refreshInFlight) {
    refreshInFlight = withCrossTabLock(performRefresh).finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

/**
 * Builds an API path from segments, percent-encoding each dynamic segment so an id taken
 * from the URL hash can never traverse to another endpoint.
 * @param {string} base e.g. "/api/v1/documents"
 * @param {...(string|number)} segments
 * @returns {string}
 */
export function apiPath(base, ...segments) {
  return [base.replace(/\/+$/, ""), ...segments.map((s) => encodeURIComponent(String(s)))].join("/");
}

function buildUrl(path, query) {
  if (!path.startsWith(API_PREFIX)) throw new Error("API paths must start with /api/");
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item !== undefined && item !== null && item !== "") params.append(key, String(item));
      }
    } else {
      params.append(key, String(value));
    }
  }
  const encoded = params.toString();
  return encoded ? `${path}?${encoded}` : path;
}

function linkedSignal(external, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("Timed out", "TimeoutError")), timeoutMs);
  const onAbort = () => controller.abort(external.reason);
  if (external) {
    if (external.aborted) controller.abort(external.reason);
    else external.addEventListener("abort", onAbort, { once: true });
  }
  return {
    signal: controller.signal,
    done() {
      clearTimeout(timer);
      if (external) external.removeEventListener("abort", onAbort);
    },
  };
}

async function parseBody(response) {
  if (response.status === 204) return null;
  const type = response.headers.get("Content-Type") || "";
  if (type.includes("json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  return response.text();
}

/**
 * Performs an API request.
 *
 * @param {string} method
 * @param {string} path must start with `/api/`
 * @param {object} [options]
 * @param {Record<string, any>} [options.query] query parameters (arrays repeat the key)
 * @param {unknown} [options.body] JSON body
 * @param {AbortSignal} [options.signal]
 * @param {number} [options.timeoutMs]
 * @param {boolean} [options.raw] resolve with the `Response` instead of the parsed body
 * @param {boolean} [options.auth] attach the bearer token and refresh on 401 (default true)
 * @returns {Promise<any>}
 */
export async function request(method, path, options = {}) {
  const { query, body, signal, timeoutMs = DEFAULT_TIMEOUT_MS, raw = false, auth = true } = options;
  const url = buildUrl(path, query);
  const attempt = async () => {
    const headers = { Accept: raw ? "*/*" : "application/json", [CSRF_HEADER]: "1" };
    if (auth && accessToken) headers.Authorization = `Bearer ${accessToken}`;
    const init = { method, headers, credentials: "same-origin", cache: "no-store" };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const linked = linkedSignal(signal, timeoutMs);
    init.signal = linked.signal;
    try {
      return await fetch(url, init);
    } catch (error) {
      if (signal && signal.aborted) throw error;
      const timedOut = error && error.name === "TimeoutError";
      throw new ApiError(0, {
        code: timedOut ? "timeout" : "network_error",
        detail: timedOut
          ? "The server took too long to respond. Please try again."
          : "Cannot reach the server. Check your connection and try again.",
      });
    } finally {
      linked.done();
    }
  };

  let response = await attempt();
  if (response.status === 401 && auth) {
    const refreshed = await refreshSession();
    if (!refreshed) {
      emitExpired();
      throw new ApiError(401, await parseBody(response).catch(() => ({})), response.headers);
    }
    response = await attempt();
    if (response.status === 401) emitExpired();
  }
  if (!response.ok) {
    const problem = await parseBody(response);
    throw new ApiError(response.status, problem && typeof problem === "object" ? problem : {}, response.headers);
  }
  if (raw) return response;
  return parseBody(response);
}

/** Convenience wrappers. */
export const api = {
  get: (path, options) => request("GET", path, options),
  post: (path, body, options = {}) => request("POST", path, { ...options, body }),
  patch: (path, body, options = {}) => request("PATCH", path, { ...options, body }),
  put: (path, body, options = {}) => request("PUT", path, { ...options, body }),
  del: (path, options) => request("DELETE", path, options),
};

/**
 * Uploads multipart form data with progress events (XMLHttpRequest, because fetch has no
 * upload progress). Retries once after a token refresh on 401.
 *
 * @param {string} path
 * @param {FormData} form
 * @param {{onProgress?: (fraction: number) => void, signal?: AbortSignal}} [options]
 * @returns {Promise<any>} parsed JSON response
 */
export function upload(path, form, options = {}) {
  const url = buildUrl(path);
  const send = () =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);
      xhr.responseType = "text";
      xhr.setRequestHeader("Accept", "application/json");
      xhr.setRequestHeader(CSRF_HEADER, "1");
      if (accessToken) xhr.setRequestHeader("Authorization", `Bearer ${accessToken}`);
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && options.onProgress) options.onProgress(event.loaded / event.total);
      });
      xhr.addEventListener("load", () => {
        let parsed = null;
        try {
          parsed = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          parsed = null;
        }
        resolve({ status: xhr.status, body: parsed, retryAfter: xhr.getResponseHeader("Retry-After") });
      });
      xhr.addEventListener("error", () =>
        reject(new ApiError(0, { code: "network_error", detail: "The upload failed. Check your connection." })),
      );
      xhr.addEventListener("abort", () => reject(new DOMException("Upload cancelled", "AbortError")));
      if (options.signal) {
        if (options.signal.aborted) {
          reject(new DOMException("Upload cancelled", "AbortError"));
          return;
        }
        options.signal.addEventListener("abort", () => xhr.abort(), { once: true });
      }
      xhr.send(form);
    });

  const toError = (result) => {
    const headers = new Headers();
    if (result.retryAfter) headers.set("Retry-After", result.retryAfter);
    return new ApiError(result.status, result.body && typeof result.body === "object" ? result.body : {}, headers);
  };

  return (async () => {
    let result = await send();
    if (result.status === 401) {
      if (!(await refreshSession())) {
        emitExpired();
        throw toError(result);
      }
      result = await send();
    }
    if (result.status < 200 || result.status >= 300) throw toError(result);
    return result.body;
  })();
}

/**
 * Extracts a display filename from a Content-Disposition header (RFC 6266 / 5987).
 * @param {string | null} header
 * @returns {string}
 */
export function filenameFromDisposition(header) {
  if (!header) return "";
  const extended = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(header);
  if (extended) {
    try {
      return decodeURIComponent(extended[1].trim());
    } catch {
      // fall through to the ASCII fallback
    }
  }
  const plain = /filename\s*=\s*"([^"]*)"/i.exec(header) || /filename\s*=\s*([^;]+)/i.exec(header);
  return plain ? plain[1].trim() : "";
}

function safeFilename(name) {
  const cleaned = String(name || "")
    .replace(/[\u0000-\u001f\u007f<>:"/\\|?*]+/g, "_")
    .replace(/^[.\s]+/, "")
    .slice(0, 150);
  return cleaned || "download";
}

/**
 * Downloads an authenticated resource: fetches it with the bearer token, then hands the
 * bytes to the browser as a `blob:` URL with the `download` attribute (the file is saved,
 * never rendered by this page).
 * @param {string} path
 * @param {{query?: Record<string, any>, fallbackName?: string, signal?: AbortSignal}} [options]
 */
export async function download(path, options = {}) {
  const response = await request("GET", path, {
    query: options.query,
    raw: true,
    signal: options.signal,
    timeoutMs: 300_000,
  });
  const blob = await response.blob();
  const name = safeFilename(filenameFromDisposition(response.headers.get("Content-Disposition")) || options.fallbackName);
  const url = URL.createObjectURL(blob);
  const anchor = h("a", { href: url, download: name, class: "visually-hidden" }, name);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/**
 * Normalises list responses: accepts a bare array or an object holding the list under a
 * common key, plus an optional cursor.
 * @param {unknown} response
 * @param {string[]} [keys] preferred list keys
 * @returns {{items: any[], next: string | null, total: number | null}}
 */
export function pageOf(response, keys = []) {
  if (Array.isArray(response)) return { items: response, next: null, total: null };
  if (!response || typeof response !== "object") return { items: [], next: null, total: null };
  const candidates = [...keys, "items", "results", "data"];
  let items = [];
  for (const key of candidates) {
    if (Array.isArray(response[key])) {
      items = response[key];
      break;
    }
  }
  const nextValue = response.next_cursor ?? response.cursor ?? response.next ?? null;
  const next = typeof nextValue === "string" && nextValue ? nextValue : null;
  const total = typeof response.total === "number" ? response.total : null;
  return { items, next, total };
}
