/**
 * Application chrome: top bar, permission-aware navigation and the `<main>` region.
 *
 * @module shell
 */

import { h, mount } from "./dom.js";
import { roleLabel } from "./format.js";
import { can, getOrganization, hasRole, organizationName, session } from "./session.js";
import { icon } from "./ui.js";

/**
 * @typedef {object} NavItem
 * @property {string} key
 * @property {string} label
 * @property {string} href
 * @property {string} icon
 * @property {() => boolean} visible
 */

/** @type {{group: string, items: NavItem[]}[]} */
const NAVIGATION = [
  {
    group: "Workspace",
    items: [
      { key: "documents", label: "Documents", href: "#/documents", icon: "folder", visible: () => can("document:read") },
      { key: "search", label: "Search", href: "#/search", icon: "search", visible: () => can("search:use") },
      { key: "assistant", label: "AI assistant", href: "#/assistant", icon: "sparkle", visible: () => can("assistant:use") },
      { key: "deadlines", label: "Deadlines", href: "#/deadlines", icon: "calendar", visible: () => can("intelligence:use") },
      { key: "compare", label: "Compare", href: "#/compare", icon: "compare", visible: () => can("intelligence:use") },
      { key: "exports", label: "Exports", href: "#/exports", icon: "download", visible: () => can("export:create") },
    ],
  },
  {
    group: "Administration",
    items: [
      { key: "users", label: "Users", href: "#/admin/users", icon: "users", visible: () => can("user:read") },
      {
        key: "departments",
        label: "Departments",
        href: "#/admin/departments",
        icon: "building",
        visible: () => can("department:manage", "user:read"),
      },
      {
        key: "organization",
        label: "Organisation",
        href: "#/admin/organization",
        icon: "settings",
        visible: () => can("org:update", "usage:read"),
      },
      {
        key: "jobs",
        label: "Background jobs",
        href: "#/admin/jobs",
        icon: "jobs",
        visible: () => can("jobs:manage") || (can("jobs:read") && can("user:read")),
      },
      { key: "audit", label: "Audit log", href: "#/audit", icon: "shield", visible: () => can("audit:read") },
    ],
  },
  {
    group: "Platform",
    items: [
      {
        key: "organizations",
        label: "Organisations",
        href: "#/platform/organizations",
        icon: "globe",
        visible: () => can("org:read_any"),
      },
    ],
  },
];

/**
 * The default landing route for the signed-in user.
 * @returns {string} a path such as "/documents"
 */
export function homePath() {
  if (can("document:read")) return "/documents";
  if (hasRole("platform_admin") || can("org:read_any")) return "/platform/organizations";
  if (can("audit:read")) return "/audit";
  return "/profile";
}

/**
 * Builds the application shell.
 * @param {{onLogout: (everywhere: boolean) => void}} handlers
 * @returns {{element: HTMLElement, main: HTMLElement, setActive: (key: string | undefined) => void}}
 */
export function createShell(handlers) {
  const main = h("main", { id: "main", class: "main", tabindex: "-1" });
  const links = new Map();

  const nav = h(
    "nav",
    { class: "sidenav", id: "sidenav", "aria-label": "Main navigation" },
    NAVIGATION.map((section) => {
      const visible = section.items.filter((item) => item.visible());
      if (!visible.length) return null;
      const headingId = `nav-${section.group.toLowerCase()}`;
      return h(
        "div",
        { class: "nav-section" },
        h("p", { class: "nav-heading", id: headingId }, section.group),
        h(
          "ul",
          { class: "nav-list", "aria-labelledby": headingId },
          visible.map((item) => {
            const link = h(
              "a",
              { href: item.href, class: "nav-link", on: { click: () => root.classList.remove("nav-open") } },
              icon(item.icon),
              h("span", null, item.label),
            );
            links.set(item.key, link);
            return h("li", null, link);
          }),
        ),
      );
    }),
    h(
      "div",
      { class: "nav-footer" },
      h("p", { class: "nav-note" }, "AI answers can be wrong. Always check the cited sources."),
    ),
  );

  const menuToggle = h(
    "button",
    {
      type: "button",
      class: "icon-btn menu-toggle",
      "aria-controls": "sidenav",
      "aria-expanded": false,
      "aria-label": "Open navigation menu",
      on: {
        click: () => {
          const open = !root.classList.contains("nav-open");
          root.classList.toggle("nav-open", open);
          menuToggle.setAttribute("aria-expanded", String(open));
          menuToggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
        },
      },
    },
    icon("menu"),
  );

  const orgName = h("span", { class: "org-name" });
  getOrganization().then((org) => {
    orgName.textContent = organizationName(org);
  });

  const user = session.user || {};
  const userMenu = createUserMenu(user, handlers);

  const topbar = h(
    "header",
    { class: "topbar" },
    menuToggle,
    h(
      "a",
      { href: "#/", class: "brand", "aria-label": "AI Document Assistant home" },
      h("span", { class: "brand-mark", "aria-hidden": "true" }),
      h("span", { class: "brand-name" }, "Document Assistant"),
    ),
    orgName,
    h("div", { class: "topbar-spacer" }),
    userMenu,
  );

  const root = h("div", { class: "shell" }, topbar, h("div", { class: "shell-body" }, nav, main));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && root.classList.contains("nav-open")) {
      root.classList.remove("nav-open");
      menuToggle.setAttribute("aria-expanded", "false");
      menuToggle.focus();
    }
  });

  return {
    element: root,
    main,
    setActive(key) {
      for (const [itemKey, link] of links) {
        if (itemKey === key) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      }
    },
  };
}

function createUserMenu(user, handlers) {
  const menuId = "user-menu";
  const displayName = String(user.full_name || user.email || "");
  const initials = displayName.slice(0, 1).toUpperCase() || "?";
  const menu = h("div", { class: "menu", id: menuId, hidden: true });
  const trigger = h(
    "button",
    {
      type: "button",
      class: "user-trigger",
      "aria-haspopup": "true",
      "aria-expanded": false,
      "aria-controls": menuId,
    },
    h("span", { class: "avatar", "aria-hidden": "true" }, initials),
    h(
      "span",
      { class: "user-trigger-text" },
      h("span", { class: "user-email", title: user.email || "" }, displayName),
      h("span", { class: "user-role" }, roleLabel(user.role)),
    ),
    icon("chevron-down"),
  );

  const close = (focusTrigger) => {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
    if (focusTrigger) trigger.focus();
  };
  const open = () => {
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    const first = menu.querySelector("a, button");
    if (first) first.focus();
  };

  mount(
    menu,
    h("a", { href: "#/profile", class: "menu-item", on: { click: () => close(false) } }, icon("user"), "Profile & security"),
    h(
      "button",
      { type: "button", class: "menu-item", on: { click: () => { close(false); handlers.onLogout(false); } } },
      icon("logout"),
      "Sign out",
    ),
    h(
      "button",
      { type: "button", class: "menu-item", on: { click: () => { close(false); handlers.onLogout(true); } } },
      icon("devices"),
      "Sign out on all devices",
    ),
  );

  trigger.addEventListener("click", () => (menu.hidden ? open() : close(false)));
  const wrapper = h("div", { class: "user-menu" }, trigger, menu);
  wrapper.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !menu.hidden) {
      event.stopPropagation();
      close(true);
    }
    if ((event.key === "ArrowDown" || event.key === "ArrowUp") && !menu.hidden) {
      const items = [...menu.querySelectorAll(".menu-item")];
      const index = items.indexOf(document.activeElement);
      const next = event.key === "ArrowDown" ? (index + 1) % items.length : (index - 1 + items.length) % items.length;
      event.preventDefault();
      items[next].focus();
    }
  });
  document.addEventListener("click", (event) => {
    if (!menu.hidden && !wrapper.contains(event.target)) close(false);
  });
  return wrapper;
}
