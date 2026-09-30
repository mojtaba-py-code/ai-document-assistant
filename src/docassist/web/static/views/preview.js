/**
 * Text preview of a document (`GET /api/v1/documents/{id}/content`), one page at a time.
 *
 * The preview shows the text extracted during ingestion - never the original file - and
 * renders it strictly as text. Passages flagged by the prompt-injection scanner are marked
 * so readers know why the assistant may have ignored them. A quote can be highlighted
 * (used when a citation opens the preview).
 *
 * @module views/preview
 */

import { api, apiPath, pageOf } from "../api.js";
import { h, highlightPassage, mount } from "../dom.js";
import { updateQuery } from "../router.js";
import { badge, button, callout, emptyState, errorCallout, input, loadingBlock, select } from "../ui.js";

/**
 * @param {object} options
 * @param {string} options.documentId
 * @param {{value: string, label: string}[]} [options.versions] selectable versions (value = number)
 * @param {string | null} [options.version] initially selected version number ("" = current)
 * @param {number | null} [options.pageCount]
 * @param {number} [options.page] initial page (1-based)
 * @param {{chunkId?: string, quote?: string} | null} [options.highlight]
 * @param {AbortSignal} options.signal
 * @returns {HTMLElement}
 */
export function previewPane(options) {
  let page = Math.max(1, Number(options.page) || 1);
  let version = options.version || "";
  let pageCount = Number(options.pageCount) > 0 ? Number(options.pageCount) : null;
  let highlight = options.highlight || null;
  let loadSeq = 0;

  const body = h("div", { class: "preview-body", "aria-live": "polite", "aria-busy": "false" });
  const pageInput = input({ type: "number", min: "1", value: String(page), class: "input input-page", "aria-label": "Page number" });
  const pageTotal = h("span", { class: "muted" });
  const prev = button("Previous", { small: true, icon: "chevron-left", onClick: () => go(page - 1) });
  const next = button("Next", { small: true, icon: "chevron-right", onClick: () => go(page + 1) });

  const versionControl =
    options.versions && options.versions.length > 1
      ? select([{ value: "", label: "Current version" }, ...options.versions], {
          value: version,
          "aria-label": "Version to preview",
          class: "input select select-sm",
          on: {
            change: (event) => {
              version = event.target.value;
              page = 1;
              updateQuery({ version: version || null, page: null });
              load();
            },
          },
        })
      : null;

  const updateControls = () => {
    pageInput.value = String(page);
    if (pageCount) pageInput.max = String(pageCount);
    pageTotal.textContent = pageCount ? `of ${pageCount}` : "";
    prev.disabled = page <= 1;
    next.disabled = pageCount !== null && page >= pageCount;
  };

  const go = (target) => {
    const wanted = Math.max(1, Math.floor(Number(target) || 1));
    if (pageCount && wanted > pageCount) return;
    page = wanted;
    updateQuery({ page: page > 1 ? page : null });
    load();
  };

  pageInput.addEventListener("change", () => go(pageInput.value));

  const renderChunk = (chunk) => {
    const text = String(chunk.content ?? chunk.text ?? "");
    const chunkId = chunk.chunk_id || chunk.id || "";
    const flags = Array.isArray(chunk.injection_flags) ? chunk.injection_flags : [];
    const flagged = chunk.suspicious === true || chunk.flagged === true || flags.length > 0;
    let nodes = null;
    let isTarget = false;
    // The content API does not expose chunk ids, so the quote itself locates the passage.
    if (highlight && highlight.quote && (!highlight.chunkId || !chunkId || highlight.chunkId === chunkId)) {
      nodes = highlightPassage(text, highlight.quote);
      isTarget = Boolean(nodes);
    }
    if (!isTarget && highlight && highlight.chunkId && highlight.chunkId === chunkId) isTarget = true;
    const section = chunk.section || (Array.isArray(chunk.heading_path) ? chunk.heading_path.join(" \u203a ") : "");
    return h(
      "article",
      { class: ["passage", isTarget && "passage-target", flagged && "passage-flagged"].filter(Boolean), tabindex: isTarget ? "-1" : undefined },
      section && h("p", { class: "passage-section" }, String(section)),
      flagged &&
        h(
          "p",
          { class: "passage-flag" },
          badge("Instruction-like text", "warning"),
          " This passage contains text that looks like instructions to an AI. The assistant treats it as data and may exclude it.",
        ),
      h("p", { class: "passage-text" }, nodes || text),
    );
  };

  async function load(cursor = null, appendTo = null) {
    const seq = ++loadSeq;
    updateControls();
    if (!appendTo) {
      body.setAttribute("aria-busy", "true");
      mount(body, loadingBlock("Loading page\u2026"));
    }
    try {
      const response = await api.get(apiPath("/api/v1/documents", options.documentId, "content"), {
        query: { page, version: version || undefined, cursor: cursor || undefined },
        signal: options.signal,
      });
      if (seq !== loadSeq) return;
      const result = pageOf(response, ["chunks", "passages"]);
      const reportedPages = Number(response && (response.page_count ?? response.total_pages));
      if (reportedPages > 0) pageCount = reportedPages;
      updateControls();
      const passages = result.items.map(renderChunk);
      const more =
        result.next &&
        button("Show more of this page", {
          small: true,
          onClick: (event) => {
            event.currentTarget.remove();
            load(result.next, container);
          },
        });
      const container = appendTo || h("div", { class: "passages" });
      container.append(...passages);
      if (more) container.appendChild(more);
      if (!appendTo) {
        mount(
          body,
          passages.length
            ? container
            : emptyState({ title: "No text on this page.", text: "The page may contain only images or tables that could not be read.", icon: "file" }),
        );
      }
      const target = body.querySelector(".passage-target");
      if (target) {
        target.scrollIntoView({ block: "center" });
        target.focus({ preventScroll: true });
        highlight = null;
      }
    } catch (error) {
      if (options.signal.aborted || seq !== loadSeq) return;
      mount(body, errorCallout(error, { retry: () => load() }));
    } finally {
      if (seq === loadSeq) body.setAttribute("aria-busy", "false");
    }
  }

  load();

  return h(
    "div",
    { class: "preview" },
    h(
      "div",
      { class: "preview-toolbar", role: "group", "aria-label": "Page navigation" },
      prev,
      h("span", { class: "page-indicator" }, h("span", { class: "muted" }, "Page"), pageInput, pageTotal),
      next,
      versionControl,
    ),
    callout("info", null, "This is the text extracted for search and the assistant. Download the original file for the exact layout."),
    body,
  );
}

