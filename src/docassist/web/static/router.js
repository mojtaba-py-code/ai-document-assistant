/**
 * Minimal hash router.
 *
 * Routes look like `#/documents/:id?tab=preview`. Path parameters are percent-decoded and
 * validated against a conservative pattern, so a crafted link cannot smuggle `/`, `..` or
 * query syntax into API paths built from them. Each navigation gets a fresh AbortSignal
 * that is aborted when the user leaves the view (cancels fetches and polling).
 *
 * @module router
 */

const PARAM_PATTERN = /^[A-Za-z0-9_-]{1,80}$/;

/**
 * @typedef {object} RouteMatch
 * @property {object} route the route definition
 * @property {Record<string, string>} params
 * @property {URLSearchParams} query
 * @property {string} path
 */

/**
 * @typedef {object} RouteDefinition
 * @property {string} path e.g. "/documents/:id"
 * @property {() => Promise<{default: Function}>} load lazy view module
 * @property {boolean} [public] reachable without a session
 * @property {string|string[]} [perm] required permission (any of)
 * @property {string} [nav] navigation key highlighted for this route
 */

/** @type {{def: RouteDefinition, parts: string[]}[]} */
const table = [];

/**
 * Registers routes (first match wins).
 * @param {RouteDefinition[]} definitions
 */
export function defineRoutes(definitions) {
  for (const def of definitions) {
    table.push({ def, parts: def.path.split("/").filter(Boolean) });
  }
}

/**
 * Parses the current location hash.
 * @param {string} [hash]
 * @returns {{path: string, query: URLSearchParams}}
 */
export function parseHash(hash = window.location.hash) {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const [pathPart, queryPart = ""] = raw.split("?", 2);
  const path = pathPart.startsWith("/") ? pathPart : `/${pathPart}`;
  return { path: path.replace(/\/+$/, "") || "/", query: new URLSearchParams(queryPart) };
}

/**
 * Finds the route for a path.
 * @param {string} path
 * @param {URLSearchParams} query
 * @returns {RouteMatch | null}
 */
export function match(path, query) {
  const segments = path.split("/").filter(Boolean);
  for (const { def, parts } of table) {
    if (parts.length !== segments.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i];
      let segment;
      try {
        segment = decodeURIComponent(segments[i]);
      } catch {
        ok = false;
        break;
      }
      if (part.startsWith(":")) {
        if (!PARAM_PATTERN.test(segment)) {
          ok = false;
          break;
        }
        params[part.slice(1)] = segment;
      } else if (part !== segment) {
        ok = false;
        break;
      }
    }
    if (ok) return { route: def, params, query, path };
  }
  return null;
}

/**
 * Builds a hash href from a path and optional query object.
 * @param {string} path
 * @param {Record<string, any>} [query]
 * @returns {string}
 */
export function href(path, query) {
  const params = new URLSearchParams();
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    }
  }
  const qs = params.toString();
  return `#${path}${qs ? `?${qs}` : ""}`;
}

/**
 * Navigates to a route (renders the matching view).
 * @param {string} path
 * @param {Record<string, any>} [query]
 * @param {{replace?: boolean}} [options]
 */
export function navigate(path, query, options = {}) {
  const target = href(path, query);
  if (options.replace) {
    window.history.replaceState(null, "", target);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else if (window.location.hash === target) {
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = target;
  }
}

/**
 * Updates the query string of the current route WITHOUT re-rendering (tab/page state).
 * @param {Record<string, any>} changes values of null/"" remove the key
 */
export function updateQuery(changes) {
  const { path, query } = parseHash();
  for (const [key, value] of Object.entries(changes)) {
    if (value === undefined || value === null || value === "") query.delete(key);
    else query.set(key, String(value));
  }
  const qs = query.toString();
  window.history.replaceState(null, "", `#${path}${qs ? `?${qs}` : ""}`);
}

/**
 * Starts listening for hash changes.
 * @param {(match: RouteMatch | null, path: string, query: URLSearchParams) => void} onRoute
 */
export function startRouter(onRoute) {
  const handle = () => {
    const { path, query } = parseHash();
    onRoute(match(path, query), path, query);
  };
  window.addEventListener("hashchange", handle);
  handle();
}
