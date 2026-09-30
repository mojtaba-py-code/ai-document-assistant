/**
 * Sign-in: email + password, then (when enrolled) the TOTP step.
 * The MFA challenge is kept in memory only and the password field is cleared as soon as
 * it has been sent.
 *
 * @module views/login
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { takeHandoff } from "../session.js";
import { button, callout, field, formStatus, input, withBusy } from "../ui.js";

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function loginView(ctx) {
  const container = h("section", { class: "auth-card", "aria-labelledby": "auth-title" });
  const notice = takeHandoff("login-notice");
  showPasswordStep(container, ctx, notice ? String(notice) : "");
  return container;
}

function passwordToggle(passwordInput) {
  const toggle = h(
    "button",
    {
      type: "button",
      class: "input-addon",
      "aria-pressed": false,
      "aria-label": "Show password",
      on: {
        click: () => {
          const show = passwordInput.type === "password";
          passwordInput.type = show ? "text" : "password";
          toggle.setAttribute("aria-pressed", String(show));
          toggle.setAttribute("aria-label", show ? "Hide password" : "Show password");
          toggle.textContent = show ? "Hide" : "Show";
        },
      },
    },
    "Show",
  );
  return toggle;
}

function showPasswordStep(container, ctx, notice) {
  const status = formStatus();
  const email = input({ type: "email", name: "email", autocomplete: "username", required: true, maxlength: 320, spellcheck: "false" });
  const password = input({ type: "password", name: "password", autocomplete: "current-password", required: true, maxlength: 1024 });
  const submit = button("Sign in", { type: "submit", variant: "primary", className: "btn-block" });

  const form = h(
    "form",
    { class: "form", novalidate: false },
    field("Work email", email),
    h("div", { class: "field" },
      h("label", { for: "login-password", class: "field-label" }, "Password"),
      h("div", { class: "input-group" }, password, passwordToggle(password)),
    ),
    status.element,
    submit,
  );
  password.id = "login-password";

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        const result = await api.post(
          "/api/v1/auth/login",
          { email: email.value.trim(), password: password.value },
          { auth: false },
        );
        password.value = "";
        if (result && result.mfa_required && result.mfa_challenge) {
          showMfaStep(container, ctx, String(result.mfa_challenge));
          return;
        }
        await ctx.completeSignIn(result);
      } catch (error) {
        password.value = "";
        status.error(error);
        password.focus();
      }
    }, "Signing in\u2026");
  });

  mount(
    container,
    h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Sign in"),
    h("p", { class: "auth-lead" }, "Use your work account to access your organisation's documents."),
    notice && callout("info", null, notice, { live: true }),
    form,
    h("p", { class: "auth-links" }, h("a", { href: "#/forgot-password" }, "Forgot your password?")),
  );
  email.focus();
}

function showMfaStep(container, ctx, challenge) {
  const status = formStatus();
  const code = input({
    type: "text",
    name: "code",
    inputmode: "numeric",
    autocomplete: "one-time-code",
    pattern: "[0-9 ]{6,10}",
    maxlength: 10,
    required: true,
    class: "input input-code",
  });
  const submit = button("Verify and sign in", { type: "submit", variant: "primary", className: "btn-block" });
  const form = h(
    "form",
    { class: "form" },
    field("6-digit code", code, { hint: "Open your authenticator app and enter the current code for AI Document Assistant." }),
    status.element,
    submit,
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        const tokens = await api.post(
          "/api/v1/auth/mfa/verify",
          { challenge, code: code.value.replace(/\s+/g, "") },
          { auth: false },
        );
        await ctx.completeSignIn(tokens);
      } catch (error) {
        code.value = "";
        status.error(error);
        code.focus();
      }
    }, "Verifying\u2026");
  });

  mount(
    container,
    h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Two-step verification"),
    h("p", { class: "auth-lead" }, "Your account is protected with a second factor."),
    form,
    h(
      "p",
      { class: "auth-links" },
      button("Use a different account", { variant: "link", onClick: () => showPasswordStep(container, ctx, "") }),
    ),
  );
  code.focus();
}
