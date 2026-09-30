/**
 * Application bootstrap: restores the session from the refresh cookie, wires the router
 * to lazily loaded views and handles sign-in / sign-out / session expiry.
 *
 * View contract: every module in `views/` default-exports `async (ctx) => Node` where
 * `ctx` is a {@link ViewContext}. Views render immediately (with loading placeholders) and
 * must stop background work when `ctx.signal` aborts (the user navigated away).
 *
 * @module app
 */

import { acceptTokens, api, onSessionExpired, refreshSession } from "./api.js";
import { h, mount } from "./dom.js";
import { defineRoutes, navigate, parseHash, startRouter } from "./router.js";
import { createShell, homePath } from "./shell.js";
import { can, loadCurrentUser, putHandoff, resetSession, session } from "./session.js";
import { announce, emptyState, errorCallout, linkButton, loadingBlock, pageHeader } from "./ui.js";

const APP_NAME = "AI Document Assistant";

/**
 * @typedef {object} ViewContext
 * @property {Record<string, string>} params validated path parameters
 * @property {URLSearchParams} query
 * @property {AbortSignal} signal aborted when the view is left
 * @property {(title: string) => void} setTitle updates the document title
 * @property {(tokens: object) => Promise<void>} completeSignIn stores tokens, loads the user, redirects
 */

defineRoutes([
  { path: "/login", public: true, guestOnly: true, title: "Sign in", load: () => import("./views/login.js") },
  { path: "/forgot-password", public: true, title: "Forgot password", load: () => import("./views/forgot-password.js") },
  { path: "/reset-password", public: true, title: "Choose a new password", load: () => import("./views/reset-password.js") },
  { path: "/documents", nav: "documents", perm: ["document:read"], title: "Documents", load: () => import("./views/library.js") },
  { path: "/documents/:id", nav: "documents", perm: ["document:read"], title: "Document", load: () => import("./views/document.js") },
  { path: "/search", nav: "search", perm: ["search:use"], title: "Search", load: () => import("./views/search.js") },
  { path: "/assistant", nav: "assistant", perm: ["assistant:use"], title: "AI assistant", load: () => import("./views/assistant.js") },
  {
    path: "/assistant/:conversationId",
    nav: "assistant",
    perm: ["assistant:use"],
    title: "AI assistant",
    load: () => import("./views/assistant.js"),
  },
  { path: "/deadlines", nav: "deadlines", perm: ["intelligence:use"], title: "Deadlines", load: () => import("./views/deadlines.js") },
  { path: "/compare", nav: "compare", perm: ["intelligence:use"], title: "Compare", load: () => import("./views/compare.js") },
  { path: "/exports", nav: "exports", perm: ["export:create"], title: "Exports", load: () => import("./views/exports.js") },
  { path: "/admin/users", nav: "users", perm: ["user:read"], title: "Users", load: () => import("./views/admin-users.js") },
  {
    path: "/admin/departments",
    nav: "departments",
    perm: ["department:manage", "user:read"],
    title: "Departments",
    load: () => import("./views/admin-departments.js"),
  },
  {
    path: "/admin/organization",
    nav: "organization",
    perm: ["org:update", "usage:read", "org:read"],
    title: "Organisation",
    load: () => import("./views/admin-organization.js"),
  },
  { path: "/admin/jobs", nav: "jobs", perm: ["jobs:read", "jobs:manage"], title: "Background jobs", load: () => import("./views/admin-jobs.js") },
  { path: "/audit", nav: "audit", perm: ["audit:read"], title: "Audit log", load: () => import("./views/audit.js") },
  { path: "/profile", nav: "profile", title: "Profile & security", load: () => import("./views/profile.js") },
  {
    path: "/platform/organizations",
    nav: "organizations",
    perm: ["org:read_any"],
    title: "Organisations",
    load: () => import("./views/platform.js"),
  },
]);

const root = document.getElementById("app");
let shell = null;
let authFrame = null;
let controller = null;
let renderSeq = 0;
let returnTo = null;
const channel = typeof BroadcastChannel === "function" ? new BroadcastChannel("docassist-session") : null;

function setTitle(title) {
  document.title = title ? `${title} \u00b7 ${APP_NAME}` : APP_NAME;
}

function ensureShell() {
  if (!shell) {
    shell = createShell({ onLogout: (everywhere) => signOut(everywhere) });
    authFrame = null;
    mount(root, shell.element);
  }
  return shell.main;
}

function ensureAuthFrame() {
  if (!authFrame) {
    shell = null;
    const main = h("main", { id: "main", class: "auth-main", tabindex: "-1" });
    authFrame = h(
      "div",
      { class: "auth-layout" },
      h(
        "div",
        { class: "auth-brand" },
        h("span", { class: "brand-mark brand-mark-lg", "aria-hidden": "true" }),
        h("span", { class: "auth-brand-name" }, APP_NAME),
      ),
      main,
      h("p", { class: "auth-footer" }, "Protected workspace. Activity is recorded in a tamper-evident audit log."),
    );
    authFrame.main = main;
    mount(root, authFrame);
  }
  return authFrame.main;
}

function notFoundView() {
  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Page not found" }),
    emptyState({
      title: "We could not find that page.",
      text: "The link may be outdated, or the page may have moved.",
      icon: "compass",
      action: linkButton("Go to the start page", "#/", { variant: "primary" }),
    }),
  );
}

function forbiddenView() {
  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "No access" }),
    emptyState({
      title: "Your role does not include this area.",
      text: "If you need access, ask your organisation administrator.",
      icon: "lock",
      action: linkButton("Go to the start page", "#/", { variant: "primary" }),
    }),
  );
}

async function renderRoute(matched, path, query) {
  if (controller) controller.abort();
  controller = new AbortController();
  const seq = ++renderSeq;
  const signal = controller.signal;

  if (path === "/" || path === "") {
    if (!session.user) navigate("/login", undefined, { replace: true });
    else navigate(homePath(), undefined, { replace: true });
    return;
  }
  if (matched && !matched.route.public && !session.user) {
    const qs = query.toString();
    returnTo = `${path}${qs ? `?${qs}` : ""}`;
    navigate("/login", undefined, { replace: true });
    return;
  }
  if (matched && matched.route.guestOnly && session.user) {
    navigate(homePath(), undefined, { replace: true });
    return;
  }

  const isPublic = Boolean(matched && matched.route.public) || !session.user;
  const target = isPublic ? ensureAuthFrame() : ensureShell();
  if (!isPublic) shell.setActive(matched ? matched.route.nav : undefined);
  mount(target, loadingBlock());

  let node;
  let title = matched ? matched.route.title : "Page not found";
  let viewTitle = null;
  // A view may name the page more precisely (e.g. the document's title); that wins over
  // the route's generic title, but only while this navigation is still current.
  const setViewTitle = (text) => {
    if (seq !== renderSeq) return;
    viewTitle = String(text || "");
    setTitle(viewTitle);
  };
  if (!matched) {
    node = notFoundView();
  } else if (matched.route.perm && !can(...matched.route.perm)) {
    node = forbiddenView();
    title = "No access";
  } else {
    try {
      const module = await matched.route.load();
      if (seq !== renderSeq) return;
      node = await module.default({
        params: matched.params,
        query,
        signal,
        setTitle: setViewTitle,
        completeSignIn,
      });
    } catch (error) {
      if (signal.aborted || seq !== renderSeq) return;
      node = h("div", { class: "page" }, pageHeader({ title }), errorCallout(error, { retry: () => navigate(path, Object.fromEntries(query)) }));
    }
  }
  if (seq !== renderSeq) return;
  mount(target, node);
  if (!viewTitle) setTitle(title);
  const heading = target.querySelector("h1");
  if (heading) heading.focus({ preventScroll: false });
  announce(`${viewTitle || title} page loaded`);
}

/**
 * Called by the login view after a successful sign-in (password or MFA step).
 * @param {{access_token: string, expires_at?: string}} tokens
 */
async function completeSignIn(tokens) {
  acceptTokens(tokens);
  await loadCurrentUser();
  shell = null;
  const destination = returnTo;
  returnTo = null;
  if (destination) {
    const { path, query } = parseHash(`#${destination}`);
    navigate(path, Object.fromEntries(query), { replace: true });
  } else {
    navigate(homePath(), undefined, { replace: true });
  }
}

/**
 * Signs out (this device or all devices), clears local state and returns to the sign-in page.
 * @param {boolean} everywhere
 */
async function signOut(everywhere) {
  try {
    await api.post("/api/v1/auth/logout", { everywhere }, { timeoutMs: 10_000 });
  } catch {
    // The local session is discarded regardless; the server session expires on its own.
  }
  endLocalSession(everywhere ? "You have been signed out on all devices." : "You have been signed out.");
  if (channel) channel.postMessage({ type: "signed-out" });
}

/**
 * Drops client state and shows the sign-in page with a notice.
 * @param {string} notice
 */
export function endLocalSession(notice) {
  resetSession();
  shell = null;
  putHandoff("login-notice", notice);
  navigate("/login", undefined, { replace: true });
}

function handleExpired() {
  if (!session.user) return;
  const { path, query } = parseHash();
  const qs = query.toString();
  returnTo = `${path}${qs ? `?${qs}` : ""}`;
  resetSession();
  shell = null;
  putHandoff("login-notice", "Your session has expired. Please sign in again.");
  navigate("/login", undefined, { replace: true });
}

async function boot() {
  const skip = document.getElementById("skip-link");
  if (skip) {
    skip.addEventListener("click", (event) => {
      event.preventDefault();
      const main = document.getElementById("main");
      if (main) main.focus();
    });
  }
  onSessionExpired(handleExpired);
  if (channel) {
    channel.addEventListener("message", (event) => {
      if (event.data && event.data.type === "signed-out" && session.user) {
        endLocalSession("You signed out in another tab.");
      }
    });
  }
  // Views (profile: password change, sign out everywhere) end the session through this event.
  window.addEventListener("app:signed-out", (event) => {
    endLocalSession(event.detail && event.detail.notice ? String(event.detail.notice) : "Please sign in again.");
    if (channel) channel.postMessage({ type: "signed-out" });
  });
  if (await refreshSession()) {
    try {
      await loadCurrentUser();
    } catch {
      resetSession();
    }
  }
  startRouter(renderRoute);
}

boot();
