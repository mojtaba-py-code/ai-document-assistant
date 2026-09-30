/**
 * Signed-in user state, permission checks and small per-session caches.
 *
 * Permission checks here only decide what the UI *offers*; the API enforces every rule
 * again (RBAC + document ACL), so hiding a button is never a security boundary.
 *
 * @module session
 */

import { api, clearTokens, pageOf } from "./api.js";

/**
 * @typedef {object} Me
 * @property {string} id
 * @property {string} email
 * @property {string | null} organization_id
 * @property {string} role
 * @property {string} clearance
 * @property {string[]} permissions
 * @property {string[]} department_ids
 * @property {string[]} managed_department_ids
 */

/** @type {{user: Me | null}} */
export const session = { user: null };

let departmentsPromise = null;
let organizationPromise = null;
const handoff = new Map();

/**
 * Loads `/auth/me` into the session.
 * @returns {Promise<Me>}
 */
export async function loadCurrentUser() {
  const me = await api.get("/api/v1/auth/me");
  session.user = {
    ...me,
    permissions: Array.isArray(me && me.permissions) ? me.permissions : [],
    department_ids: Array.isArray(me && me.department_ids) ? me.department_ids : [],
    managed_department_ids: Array.isArray(me && me.managed_department_ids) ? me.managed_department_ids : [],
  };
  return session.user;
}

/** Clears all client-side session state (tokens and caches). */
export function resetSession() {
  session.user = null;
  departmentsPromise = null;
  organizationPromise = null;
  handoff.clear();
  clearTokens();
}

/**
 * True when the user holds ANY of the given permissions.
 * @param {...string} permissions
 * @returns {boolean}
 */
export function can(...permissions) {
  const held = session.user ? session.user.permissions : [];
  return permissions.some((p) => held.includes(p));
}

/**
 * @param {string} role
 * @returns {boolean}
 */
export function hasRole(role) {
  return Boolean(session.user && session.user.role === role);
}

/**
 * Departments of the organisation (cached for the session; `refresh` forces a reload).
 * Resolves to an empty list when the user may not list departments.
 * @param {{refresh?: boolean}} [options]
 * @returns {Promise<any[]>}
 */
export function getDepartments(options = {}) {
  if (!session.user || !session.user.organization_id) return Promise.resolve([]);
  if (!departmentsPromise || options.refresh) {
    departmentsPromise = api
      .get("/api/v1/departments", { query: { limit: 200 } })
      .then((response) => pageOf(response, ["departments"]).items)
      .catch(() => {
        departmentsPromise = null;
        return [];
      });
  }
  return departmentsPromise;
}

/**
 * Name of a department id, using the cached list.
 * @param {any[]} departments
 * @param {string | null | undefined} id
 * @returns {string}
 */
export function departmentName(departments, id) {
  if (!id) return "";
  const found = departments.find((d) => d && d.id === id);
  return found && found.name ? String(found.name) : "Department";
}

/**
 * The caller's organisation (name, settings) or null when not permitted.
 * @returns {Promise<any | null>}
 */
export function getOrganization() {
  if (!session.user || !session.user.organization_id) return Promise.resolve(null);
  if (!organizationPromise) {
    organizationPromise = api.get("/api/v1/organization").catch(() => {
      organizationPromise = null;
      return null;
    });
  }
  return organizationPromise;
}

/**
 * Display name of the organisation from `GET /api/v1/organization` (the tenant record is
 * nested under `organization`).
 * @param {any} data
 * @returns {string}
 */
export function organizationName(data) {
  const org = data && typeof data === "object" ? data.organization || data : null;
  return org && typeof org.name === "string" ? org.name : "";
}

/** Forgets the cached organisation (after an admin edits it). */
export function invalidateOrganization() {
  organizationPromise = null;
}

/**
 * One-shot in-memory hand-off between views (e.g. the quote to highlight when a citation
 * opens the preview). Keeps document text out of the URL and browser history.
 * @param {string} key
 * @param {unknown} value
 */
export function putHandoff(key, value) {
  handoff.set(key, value);
}

/**
 * @param {string} key
 * @returns {any}
 */
export function takeHandoff(key) {
  const value = handoff.get(key);
  handoff.delete(key);
  return value;
}
