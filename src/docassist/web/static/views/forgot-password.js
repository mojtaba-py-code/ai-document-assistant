/**
 * "Forgot password": always shows the same confirmation, whether or not the account exists
 * (the API does not reveal it either).
 *
 * @module views/forgot-password
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { button, callout, field, formStatus, input, withBusy } from "../ui.js";

/**
 * @returns {Promise<HTMLElement>}
 */
export default async function forgotPasswordView() {
  const container = h("section", { class: "auth-card", "aria-labelledby": "auth-title" });
  const status = formStatus();
  const email = input({ type: "email", name: "email", autocomplete: "username", required: true, maxlength: 320 });
  const submit = button("Send reset link", { type: "submit", variant: "primary", className: "btn-block" });
  const form = h("form", { class: "form" }, field("Work email", email), status.element, submit);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/auth/password/forgot", { email: email.value.trim() }, { auth: false });
        mount(
          container,
          h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Check your email"),
          callout(
            "success",
            null,
            "If an account exists for that address, we have sent a link to reset the password. The link expires soon, so use it promptly.",
            { live: true },
          ),
          h("p", { class: "auth-links" }, h("a", { href: "#/login" }, "Back to sign in")),
        );
        const heading = container.querySelector("h1");
        if (heading) heading.focus();
      } catch (error) {
        status.error(error);
      }
    }, "Sending\u2026");
  });

  mount(
    container,
    h("h1", { id: "auth-title", class: "auth-title", tabindex: "-1" }, "Reset your password"),
    h("p", { class: "auth-lead" }, "Enter your work email and we will send you a link to choose a new password."),
    form,
    h("p", { class: "auth-links" }, h("a", { href: "#/login" }, "Back to sign in")),
  );
  return container;
}
