/**
 * AI assistant: conversation list, question box and the answer thread.
 *
 * Routes: `#/assistant` (new conversation), `#/assistant/:conversationId` (history) and
 * `?document=<id>` to restrict retrieval to one document. Conversations are private to the
 * user (enforced by the API with row-level security).
 *
 * @module views/assistant
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, mount } from "../dom.js";
import { classificationLabel, CLASSIFICATIONS, formatDateTime, relativeTime } from "../format.js";
import { navigate } from "../router.js";
import { can } from "../session.js";
import {
  announce,
  button,
  callout,
  checkbox,
  confirmDialog,
  emptyState,
  errorCallout,
  loadingBlock,
  pageHeader,
  textarea,
  toast,
} from "../ui.js";
import { answerBlock } from "./answer.js";
import { fetchDocument } from "./common.js";

const MAX_QUESTION = 2000;
const EXAMPLES = [
  "Which contracts expire in the next 90 days?",
  "What are the payment terms in our supplier agreements?",
  "Summarise the annual leave policy.",
];

/**
 * @param {import("../app.js").ViewContext} ctx
 * @returns {Promise<HTMLElement>}
 */
export default async function assistantView(ctx) {
  let conversationId = ctx.params.conversationId || null;
  const scopeId = ctx.query.get("document") || "";
  const scopeDoc = scopeId ? await fetchDocument(scopeId, ctx.signal) : null;
  let pending = null;

  const conversationList = h("div", { class: "conversation-list", "aria-live": "polite" });
  const thread = h("ol", { class: "thread", "aria-label": "Conversation" });
  const threadWrap = h("div", { class: "thread-wrap" }, thread);
  const question = textarea({
    name: "question",
    rows: 3,
    maxlength: MAX_QUESTION,
    required: true,
    placeholder: scopeDoc ? `Ask about \u201c${scopeDoc.title || "this document"}\u201d\u2026` : "Ask a question about your documents\u2026",
    class: "input textarea composer-input",
  });
  question.id = "assistant-question";
  const counter = h("span", { class: "muted small", "aria-live": "off" }, `0 / ${MAX_QUESTION}`);
  const oldVersions = checkbox("Include older versions");
  const agentToggle = can("assistant:agent") ? checkbox("Research mode (multi-step, slower)") : null;
  const policyPromise = api.get("/api/v1/assistant/policy", { signal: ctx.signal }).catch(() => null);
  policyPromise.then((policy) => {
    // Research mode needs a tool-capable model; hide the option when none is configured.
    if (agentToggle && policy && policy.agent_available === false) agentToggle.hidden = true;
  });
  const send = button("Ask", { type: "submit", variant: "primary", icon: "send" });
  const cancel = button("Stop", { variant: "ghost", onClick: () => pending && pending.abort() });
  cancel.hidden = true;
  const statusLine = h("p", { class: "composer-status", role: "status" });

  question.addEventListener("input", () => {
    counter.textContent = `${question.value.length} / ${MAX_QUESTION}`;
  });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  const composer = h(
    "form",
    { class: "composer" },
    h("label", { for: "assistant-question", class: "field-label" }, "Your question"),
    question,
    h(
      "div",
      { class: "composer-bar" },
      h("div", { class: "composer-options" }, oldVersions, agentToggle),
      h("div", { class: "composer-actions" }, counter, cancel, send),
    ),
    h("p", { class: "field-hint" }, "Press Enter to send, Shift+Enter for a new line. Answers use only documents you are allowed to read."),
    statusLine,
  );

  const scrollToEnd = () => {
    const last = thread.lastElementChild;
    if (last) last.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  const exchange = (text, answerNode) =>
    h(
      "li",
      { class: "exchange" },
      h("div", { class: "question-bubble" }, h("span", { class: "visually-hidden" }, "You asked: "), h("p", null, text)),
      answerNode,
    );

  const showIntro = () => {
    mount(
      thread,
      h(
        "li",
        { class: "thread-intro" },
        emptyState({
          title: scopeDoc ? `Ask about \u201c${scopeDoc.title || "this document"}\u201d` : "Ask about your documents",
          text: "Every answer shows the exact passages it is based on. If the documents do not contain the answer, the assistant says so.",
          icon: "sparkle",
        }),
        !scopeDoc &&
          h(
            "div",
            { class: "examples" },
            h("p", { class: "muted small" }, "Try:"),
            EXAMPLES.map((example) =>
              button(example, {
                small: true,
                variant: "ghost",
                onClick: () => {
                  question.value = example;
                  question.dispatchEvent(new Event("input"));
                  question.focus();
                },
              }),
            ),
          ),
      ),
    );
  };

  const loadConversations = async () => {
    try {
      const page = pageOf(await api.get("/api/v1/assistant/conversations", { query: { limit: 50 }, signal: ctx.signal }), ["conversations"]);
      mount(
        conversationList,
        page.items.length
          ? h(
              "ul",
              { class: "conv-items" },
              page.items.map((conv) => {
                const id = String(conv.id || "");
                const active = id === conversationId;
                return h(
                  "li",
                  { class: ["conv-item", active && "conv-active"].filter(Boolean) },
                  h(
                    "a",
                    { href: `#/assistant/${encodeURIComponent(id)}`, class: "conv-link", "aria-current": active ? "page" : undefined },
                    h("span", { class: "conv-title" }, String(conv.title || "Untitled conversation")),
                    h("span", { class: "conv-date" }, relativeTime(conv.updated_at || conv.created_at)),
                  ),
                  h(
                    "button",
                    {
                      type: "button",
                      class: "icon-btn icon-btn-sm",
                      "aria-label": `Delete conversation \u201c${conv.title || "Untitled"}\u201d`,
                      on: { click: () => deleteConversation(conv) },
                    },
                    h("span", { class: "icon icon-trash", "aria-hidden": "true" }),
                  ),
                );
              }),
            )
          : h("p", { class: "muted small conv-empty" }, "Your conversations appear here."),
      );
    } catch (error) {
      if (!ctx.signal.aborted) mount(conversationList, h("p", { class: "muted small" }, "Conversations could not be loaded."));
    }
  };

  const deleteConversation = async (conv) => {
    const ok = await confirmDialog({
      title: "Delete this conversation?",
      message: `\u201c${conv.title || "Untitled conversation"}\u201d and its answers will be removed.`,
      confirmLabel: "Delete",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await api.del(apiPath("/api/v1/assistant/conversations", conv.id));
      toast("Conversation deleted.", "success");
      if (String(conv.id) === conversationId) navigate("/assistant");
      else loadConversations();
    } catch (error) {
      toast(error && error.detail ? error.detail : "The conversation could not be deleted.", "danger");
    }
  };

  const loadHistory = async () => {
    mount(thread, h("li", null, loadingBlock("Loading conversation\u2026")));
    try {
      const conversation = await api.get(apiPath("/api/v1/assistant/conversations", conversationId), { signal: ctx.signal });
      const messages = pageOf(conversation, ["messages"]).items;
      if (conversation && conversation.title) ctx.setTitle(String(conversation.title));
      thread.replaceChildren();
      let lastQuestion = null;
      for (const message of messages) {
        if (message.role === "user") {
          if (lastQuestion !== null) thread.appendChild(exchange(lastQuestion, null));
          lastQuestion = String(message.content || "");
        } else {
          thread.appendChild(exchange(lastQuestion ?? "", answerBlock(message)));
          lastQuestion = null;
        }
      }
      if (lastQuestion !== null) thread.appendChild(exchange(lastQuestion, null));
      if (!messages.length) showIntro();
      scrollToEnd();
    } catch (error) {
      if (ctx.signal.aborted) return;
      mount(thread, h("li", null, errorCallout(error, { retry: loadHistory, title: "Conversation not available" })));
    }
  };

  composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text || pending) return;
    if (thread.querySelector(".thread-intro")) thread.replaceChildren();
    const slot = h("div", { class: "answer-pending" }, loadingBlock("Searching your documents and drafting an answer\u2026"));
    const item = exchange(text, slot);
    thread.appendChild(item);
    scrollToEnd();
    question.value = "";
    counter.textContent = `0 / ${MAX_QUESTION}`;
    send.disabled = true;
    cancel.hidden = false;
    statusLine.textContent = "Working on your answer\u2026";
    pending = new AbortController();
    const useAgent = agentToggle && agentToggle.querySelector("input").checked;
    try {
      let result;
      if (useAgent) {
        result = await api.post("/api/v1/assistant/agent", { task: text }, { timeoutMs: 300_000, signal: pending.signal });
      } else {
        const body = { question: text };
        if (conversationId) body.conversation_id = conversationId;
        if (scopeDoc && scopeDoc.id) body.filters = { document_ids: [String(scopeDoc.id)] };
        if (oldVersions.querySelector("input").checked) body.include_old_versions = true;
        result = await api.post("/api/v1/assistant/ask", body, { timeoutMs: 180_000, signal: pending.signal });
      }
      mount(slot, answerBlock(result));
      slot.classList.remove("answer-pending");
      statusLine.textContent = "";
      announce("The answer is ready. The source evidence is listed after it.");
      if (result && result.conversation_id && !conversationId) {
        conversationId = String(result.conversation_id);
        if (/^[A-Za-z0-9_-]{1,80}$/.test(conversationId)) {
          const suffix = scopeId ? `?document=${encodeURIComponent(scopeId)}` : "";
          window.history.replaceState(null, "", `#/assistant/${encodeURIComponent(conversationId)}${suffix}`);
        }
      }
      loadConversations();
    } catch (error) {
      statusLine.textContent = "";
      if (error && error.name === "AbortError") {
        mount(slot, callout("info", null, "Stopped. Ask again whenever you are ready."));
      } else {
        mount(slot, errorCallout(error, { title: "No answer this time" }));
        question.value = text;
        counter.textContent = `${text.length} / ${MAX_QUESTION}`;
      }
    } finally {
      pending = null;
      send.disabled = false;
      cancel.hidden = true;
      question.focus();
    }
  });

  ctx.signal.addEventListener("abort", () => pending && pending.abort(), { once: true });

  const scopeBar =
    scopeDoc &&
    h(
      "div",
      { class: "scope-bar" },
      h("span", null, "Only searching: "),
      h("a", { href: `#/documents/${encodeURIComponent(String(scopeDoc.id || scopeId))}` }, String(scopeDoc.title || "Document")),
      h("a", { href: conversationId ? `#/assistant/${encodeURIComponent(conversationId)}` : "#/assistant", class: "scope-clear" }, "Search all documents instead"),
    );
  const scopeMissing = scopeId && !scopeDoc && callout("warning", null, "The selected document is not available, so all your documents will be used.");

  if (conversationId) loadHistory();
  else showIntro();
  loadConversations();

  return h(
    "div",
    { class: "page page-assistant" },
    pageHeader({
      title: "AI assistant",
      subtitle: "Ask questions in plain language. Answers are generated by AI from your documents and always list their sources.",
      actions: [button("New conversation", { icon: "plus", onClick: () => navigate("/assistant") })],
    }),
    h(
      "div",
      { class: "assistant-layout" },
      h(
        "aside",
        { class: "assistant-sidebar", "aria-label": "Your conversations" },
        h("h2", { class: "sidebar-title" }, "Conversations"),
        conversationList,
        policyPanel(policyPromise),
      ),
      h("section", { class: "assistant-main", "aria-label": "Chat" }, scopeBar, scopeMissing, threadWrap, composer),
    ),
  );
}

/**
 * "How your data is handled" panel, filled from `GET /api/v1/assistant/policy`.
 * @param {Promise<any>} policyPromise
 * @returns {HTMLElement}
 */
function policyPanel(policyPromise) {
  const body = h("div", { class: "policy-body" }, h("p", { class: "muted small" }, "Loading\u2026"));
  policyPromise.then((policy) => {
    mount(body, policy ? renderPolicy(policy) : h("p", { class: "muted small" }, "The data policy could not be loaded."));
  });
  return h("details", { class: "details policy" }, h("summary", null, "How your data is handled"), body);
}

/**
 * Describes where text of one classification is processed.
 * @param {any} route `{classification, allowed, provider, model, external}`
 * @returns {string}
 */
function describeRoute(route) {
  if (!route || typeof route !== "object" || route.allowed === false || (!route.provider && !route.model)) {
    return "Not sent to any AI model";
  }
  const where = route.external === true ? "external AI service" : route.external === false ? "in-house model" : "";
  return [String(route.provider || ""), route.model && `(${route.model})`, where && `\u2014 ${where}`].filter(Boolean).join(" ");
}

function renderPolicy(policy) {
  const routes = Array.isArray(policy.routes) ? policy.routes : [];
  const rows = CLASSIFICATIONS.map((c) => [c.label, routes.find((r) => r && String(r.classification).toUpperCase() === c.value)])
    .filter(([, route]) => route)
    .map(([label, route]) => [label, describeRoute(route)]);
  return h(
    "div",
    null,
    policy.external_max_classification &&
      h(
        "p",
        { class: "small" },
        `Text classified up to ${classificationLabel(policy.external_max_classification)} may be processed by an external AI service; more sensitive text stays with an in-house model or is not sent at all.`,
      ),
    rows.length
      ? h("dl", { class: "dl dl-compact" }, rows.map(([term, value]) => h("div", { class: "dl-row" }, h("dt", null, term), h("dd", null, value))))
      : h("p", { class: "muted small" }, "No policy details available."),
    policy.pseudonymize_pii_for_external === true &&
      h("p", { class: "small" }, "Personal data (names, emails, account numbers) is replaced with placeholders before text is sent to an external service."),
    h("p", { class: "muted small" }, `Checked ${formatDateTime(new Date())}`),
  );
}
