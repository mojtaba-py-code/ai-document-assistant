/**
 * Password reset from the emailed link (`#/reset-password?token=...`).
 *
 * The token is read once, kept in memory and immediately removed from the address bar and
 * session history (`history.replaceState`) so it does not linger where it could be copied.
 *
 * @module views/reset-password
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { button, callout, field, formStatus, input, withBusy } from "../ui.js";

export const PASSWORD_HINT = "At least 12 characters. Avoid common passwords and your name or email address.";

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function resetPasswordView(ctx) {
  const token = ctx.query.get("token") || "";
  if (token) window.history.replaceState(null, "", "#/reset-password");

  const container = h("section", { class: "auth-card", "aria-labelledby": "auth-title" });
  if (!token) {
    mount(
      container,
      h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Reset link missing"),
      callout("warning", null, "Open the link from your password reset email, or request a new one."),
      h("p", { class: "auth-links" }, h("a", { href: "#/forgot-password" }, "Request a new link")),
    );
    return container;
  }

  const status = formStatus();
  const password = input({ type: "password", name: "new_password", autocomplete: "new-password", required: true, minlength: 12, maxlength: 256 });
  const confirm = input({ type: "password", name: "confirm", autocomplete: "new-password", required: true, maxlength: 256 });
  const submit = button("Set new password", { type: "submit", variant: "primary", className: "btn-block" });
  const form = h(
    "form",
    { class: "form" },
    field("New password", password, { hint: PASSWORD_HINT }),
    field("Confirm new password", confirm),
    status.element,
    submit,
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    if (password.value !== confirm.value) {
      confirm.setCustomValidity("The passwords do not match.");
      confirm.reportValidity();
      return;
    }
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/auth/password/reset", { token, new_password: password.value }, { auth: false });
        password.value = "";
        confirm.value = "";
        mount(
          container,
          h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Password updated"),
          callout("success", null, "Your password has been changed and all your other sessions were signed out.", { live: true }),
          h("p", { class: "auth-links" }, h("a", { href: "#/login", class: "btn btn-primary" }, "Sign in")),
        );
        const heading = container.querySelector("h1");
        if (heading) heading.focus();
      } catch (error) {
        status.error(error);
      }
    }, "Saving\u2026");
  });
  confirm.addEventListener("input", () => confirm.setCustomValidity(""));

  mount(
    container,
    h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Choose a new password"),
    form,
    h("p", { class: "auth-links" }, h("a", { href: "#/login" }, "Back to sign in")),
  );
  return container;
}
