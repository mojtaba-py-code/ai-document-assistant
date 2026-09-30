/**
 * Profile & security: account facts, password change, two-step verification (TOTP)
 * enrolment/disable and "sign out on all devices".
 *
 * MFA enrolment shows the setup key and the `otpauth://` URI as text (no QR image and no
 * third-party QR service - the secret never leaves this page). The secret is dropped from
 * the DOM as soon as enrolment is confirmed or cancelled.
 *
 * @module views/profile
 */

import { api } from "../api.js";
import { h, mount } from "../dom.js";
import { classificationLabel, roleLabel } from "../format.js";
import { departmentName, getDepartments, getOrganization, organizationName, session } from "../session.js";
import {
  button,
  callout,
  card,
  classificationBadge,
  confirmDialog,
  copyButton,
  descriptionList,
  field,
  formStatus,
  input,
  pageHeader,
  withBusy,
} from "../ui.js";
import { PASSWORD_HINT } from "./reset-password.js";

function signedOut(notice) {
  window.dispatchEvent(new CustomEvent("app:signed-out", { detail: { notice } }));
}

/**
 * Groups a base32 secret in blocks of four for easier manual entry.
 * @param {string} secret
 * @returns {string}
 */
export function groupSecret(secret) {
  return String(secret).replace(/\s+/g, "").replace(/(.{4})/g, "$1 ").trim();
}

/**
 * @returns {Promise<HTMLElement>}
 */
export default async function profileView() {
  const user = session.user || {};
  const [departments, org] = await Promise.all([getDepartments(), getOrganization()]);
  const memberships = (user.department_ids || []).map(
    (id) => `${departmentName(departments, id)}${(user.managed_department_ids || []).includes(id) ? " (manager)" : ""}`,
  );

  return h(
    "div",
    { class: "page" },
    pageHeader({ title: "Profile & security", subtitle: "Your account, password and sign-in protection." }),
    h(
      "div",
      { class: "grid-2" },
      card({
        title: "Your account",
        body: descriptionList([
          ["Email", String(user.email || "")],
          ["Organisation", organizationName(org) || (user.organization_id ? "Your organisation" : "Platform operator")],
          ["Role", roleLabel(user.role)],
          ["Clearance", h("span", null, classificationBadge(user.clearance), h("span", { class: "muted small" }, ` You can read documents up to ${classificationLabel(user.clearance)}.`))],
          ["Departments", memberships.length ? memberships.join(", ") : "None"],
        ]),
      }),
      mfaCard(),
      passwordCard(),
      sessionsCard(),
    ),
  );
}

function passwordCard() {
  const status = formStatus();
  const current = input({ type: "password", autocomplete: "current-password", required: true, maxlength: 1024 });
  const next = input({ type: "password", autocomplete: "new-password", required: true, minlength: 12, maxlength: 256 });
  const confirm = input({ type: "password", autocomplete: "new-password", required: true, maxlength: 256 });
  const submit = button("Change password", { type: "submit", variant: "primary" });
  const form = h(
    "form",
    { class: "form" },
    field("Current password", current),
    field("New password", next, { hint: PASSWORD_HINT }),
    field("Confirm new password", confirm),
    status.element,
    h("div", { class: "form-actions" }, submit),
  );
  confirm.addEventListener("input", () => confirm.setCustomValidity(""));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.clear();
    if (next.value !== confirm.value) {
      confirm.setCustomValidity("The passwords do not match.");
      confirm.reportValidity();
      return;
    }
    await withBusy(submit, async () => {
      try {
        await api.post("/api/v1/auth/password/change", { current_password: current.value, new_password: next.value });
        current.value = "";
        next.value = "";
        confirm.value = "";
        signedOut("Your password was changed. Please sign in with the new password.");
      } catch (error) {
        current.value = "";
        status.error(error);
      }
    }, "Changing\u2026");
  });
  return card({ title: "Password", description: "Changing it signs you out everywhere, including this browser.", body: form });
}

function mfaCard() {
  const body = h("div");
  const knownState = typeof (session.user || {}).mfa_enabled === "boolean" ? session.user.mfa_enabled : null;

  const showIntro = () => {
    // Enrolment requires the current password, so a stolen session alone cannot bind an
    // attacker's authenticator to this account.
    const password = input({ type: "password", autocomplete: "current-password", required: true, maxlength: 1024 });
    const status = formStatus();
    const start = button("Set up two-step verification", {
      variant: "primary",
      onClick: () =>
        withBusy(start, async () => {
          if (!password.value) {
            status.error(new Error("Enter your current password to continue."));
            password.focus();
            return;
          }
          try {
            const enrollment = await api.post("/api/v1/auth/mfa/enroll", { password: password.value });
            password.value = "";
            showEnrollment(enrollment);
          } catch (error) {
            password.value = "";
            status.error(error);
          }
        }, "Starting\u2026"),
    });
    mount(
      body,
      intro(),
      knownState !== true && field("Current password", password),
      status.element,
      h("div", { class: "form-actions" }, knownState !== true && start, knownState !== false && disableToggle()),
    );
  };

  const intro = () =>
    h(
      "p",
      null,
      knownState === true
        ? "Two-step verification is on. You enter a code from your authenticator app when you sign in."
        : "Protect your account with a one-time code from an authenticator app (for example Microsoft Authenticator, Google Authenticator or 1Password).",
    );

  const disableToggle = () =>
    button("Turn off two-step verification", {
      variant: "ghost",
      onClick: () => showDisable(),
    });

  const showEnrollment = (enrollment) => {
    const secret = String((enrollment && enrollment.secret) || "");
    const uri = String((enrollment && enrollment.provisioning_uri) || "");
    const status = formStatus();
    const code = input({ inputmode: "numeric", autocomplete: "one-time-code", pattern: "[0-9 ]{6,10}", maxlength: 10, required: true, class: "input input-code" });
    const confirmButton = button("Confirm and turn on", { type: "submit", variant: "primary" });
    const uriBox = h("textarea", { class: "input textarea mono", rows: 3, readonly: true, "aria-label": "Setup link (otpauth URI)" });
    uriBox.value = uri;
    const form = h(
      "form",
      { class: "form" },
      h(
        "ol",
        { class: "steps-list" },
        h(
          "li",
          null,
          h("p", null, "In your authenticator app, choose to add an account and enter this setup key:"),
          h("p", { class: "secret-key mono", "aria-label": "Setup key" }, groupSecret(secret)),
          copyButton(() => secret, "Copy setup key"),
        ),
        h(
          "li",
          null,
          h("p", null, "Or, if your app accepts a setup link, copy this link into it:"),
          uriBox,
          copyButton(() => uri, "Copy setup link"),
        ),
        h("li", null, field("Enter the 6-digit code the app now shows", code)),
      ),
      callout("warning", null, "Keep this key private. Anyone who has it can generate your sign-in codes."),
      status.element,
      h("div", { class: "form-actions" }, button("Cancel", { onClick: showIntro }), confirmButton),
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.clear();
      await withBusy(confirmButton, async () => {
        try {
          await api.post("/api/v1/auth/mfa/confirm", { code: code.value.replace(/\s+/g, "") });
          if (session.user) session.user.mfa_enabled = true;
          mount(body, callout("success", "Two-step verification is on", "From now on you will be asked for a code when you sign in.", { live: true }));
        } catch (error) {
          code.value = "";
          status.error(error);
        }
      }, "Confirming\u2026");
    });
    mount(body, form);
    code.focus();
  };

  const showDisable = () => {
    const status = formStatus();
    const password = input({ type: "password", autocomplete: "current-password", required: true, maxlength: 1024 });
    const code = input({ inputmode: "numeric", autocomplete: "one-time-code", maxlength: 10, required: true, class: "input input-code" });
    const submit = button("Turn off", { type: "submit", variant: "danger" });
    const form = h(
      "form",
      { class: "form" },
      callout("warning", null, "Without two-step verification, your password alone protects your account."),
      field("Password", password),
      field("Current code from your app", code),
      status.element,
      h("div", { class: "form-actions" }, button("Cancel", { onClick: showIntro }), submit),
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.clear();
      await withBusy(submit, async () => {
        try {
          await api.post("/api/v1/auth/mfa/disable", { password: password.value, code: code.value.replace(/\s+/g, "") });
          password.value = "";
          if (session.user) session.user.mfa_enabled = false;
          mount(body, callout("success", "Two-step verification is off", "You can turn it on again at any time.", { live: true }));
        } catch (error) {
          password.value = "";
          code.value = "";
          status.error(error);
        }
      }, "Turning off\u2026");
    });
    mount(body, form);
    password.focus();
  };

  showIntro();
  return card({ title: "Two-step verification", body });
}

function sessionsCard() {
  const signOutAll = button("Sign out on all devices", {
    variant: "danger",
    onClick: async () => {
      const ok = await confirmDialog({
        title: "Sign out on all devices?",
        message: "Every session, including this one, ends immediately. Use this if you lost a device or suspect someone else used your account.",
        confirmLabel: "Sign out everywhere",
        tone: "danger",
      });
      if (!ok) return;
      await withBusy(signOutAll, async () => {
        try {
          await api.post("/api/v1/auth/logout", { everywhere: true });
        } catch {
          // Signed out locally regardless.
        }
        signedOut("You have been signed out on all devices.");
      });
    },
  });
  return card({
    title: "Sessions",
    body: h("div", null, h("p", null, "Signed in on a shared or lost device? End every session at once."), h("div", { class: "form-actions" }, signOutAll)),
  });
}
