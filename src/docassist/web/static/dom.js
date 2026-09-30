/**
 * Safe DOM construction.
 *
 * The page runs under `require-trusted-types-for 'script'; trusted-types 'none'`, so no
 * string is ever parsed as HTML. Every element is created with `document.createElement`,
 * text goes in through `textContent` / text nodes, and attributes through `setAttribute`
 * after an allow-list check (no event-handler attributes, no inline styles, and URL-bearing
 * attributes only accept same-origin paths, fragments or `blob:` object URLs).
 *
 * @module dom
 */

const URL_ATTRIBUTES = new Set(["href", "src", "action", "formaction", "poster", "cite", "data", "xlink:href"]);
const FORBIDDEN_ATTRIBUTES = new Set(["style", "srcdoc", "srcset"]);
const ATTRIBUTE_NAME = /^[a-z][a-z0-9-]*$/;

let idCounter = 0;

/**
 * Returns a document-unique id with the given prefix (for label/aria wiring).
 * @param {string} [prefix]
 * @returns {string}
 */
export function uid(prefix = "id") {
  idCounter += 1;
  return `${prefix}-${idCounter}`;
}

/**
 * True for URLs the UI may place in `href`/`src`: same-origin absolute paths, in-page
 * fragments and `blob:` object URLs created by this page. Scheme-relative (`//host`) and
 * every other scheme (script and data URLs included) are rejected.
 * @param {unknown} value
 * @returns {boolean}
 */
export function isSafeUrl(value) {
  const url = String(value);
  if (url.startsWith("#")) return true;
  if (url.startsWith("blob:")) return true;
  return url.startsWith("/") && !url.startsWith("//") && !url.startsWith("/\\");
}

/**
 * Sets one attribute after validating its name and (for URL attributes) its value.
 * @param {Element} element
 * @param {string} name
 * @param {string | number | boolean} value
 */
export function setAttr(element, name, value) {
  const lower = name.toLowerCase();
  if (!ATTRIBUTE_NAME.test(lower) || lower.startsWith("on") || FORBIDDEN_ATTRIBUTES.has(lower)) {
    throw new Error(`Refusing unsafe attribute "${name}"`);
  }
  if (URL_ATTRIBUTES.has(lower) && !isSafeUrl(value)) {
    throw new Error(`Refusing unsafe URL in "${name}"`);
  }
  element.setAttribute(lower, value === true ? "" : String(value));
}

function addClasses(element, value) {
  const list = Array.isArray(value) ? value : String(value).split(/\s+/);
  for (const name of list) {
    if (name) element.classList.add(name);
  }
}

/**
 * Appends children, flattening arrays. Strings and numbers become text nodes (never HTML);
 * `null`, `undefined` and booleans are skipped so callers can write `cond && node`.
 * @param {Node} parent
 * @param {...unknown} children
 * @returns {Node}
 */
export function append(parent, ...children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === true) continue;
    if (child instanceof Node) {
      parent.appendChild(child);
    } else {
      parent.appendChild(document.createTextNode(String(child)));
    }
  }
  return parent;
}

/**
 * Creates an element.
 *
 * `props` keys: `class` (string or array), `text` (textContent), `on` (event \u2192 listener map),
 * `dataset`, `value`/`checked`/`selected` (set as properties), `aria-*` booleans are
 * stringified; everything else goes through {@link setAttr}. `false`/`null`/`undefined`
 * values are skipped.
 *
 * @param {string} tag
 * @param {Record<string, any> | null} [props]
 * @param {...unknown} children
 * @returns {HTMLElement}
 */
export function h(tag, props, ...children) {
  const element = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (key.startsWith("aria-") && typeof value === "boolean") {
        setAttr(element, key, String(value));
        continue;
      }
      if (value === undefined || value === null || value === false) continue;
      switch (key) {
        case "class":
          addClasses(element, value);
          break;
        case "text":
          element.textContent = String(value);
          break;
        case "on":
          for (const [event, listener] of Object.entries(value)) {
            if (typeof listener === "function") element.addEventListener(event, listener);
          }
          break;
        case "dataset":
          for (const [name, data] of Object.entries(value)) {
            if (data !== undefined && data !== null) element.dataset[name] = String(data);
          }
          break;
        case "value":
          element.value = String(value);
          break;
        case "checked":
        case "selected":
          element[key] = Boolean(value);
          break;
        default:
          setAttr(element, key, value);
      }
    }
  }
  append(element, ...children);
  return element;
}

/**
 * Replaces all children of `parent`.
 * @param {Element} parent
 * @param {...unknown} children
 * @returns {Element}
 */
export function mount(parent, ...children) {
  parent.replaceChildren();
  append(parent, ...children);
  return parent;
}

/**
 * Splits `text` into text nodes and `<mark>` elements for every case-insensitive
 * occurrence of any of `terms` (no HTML parsing involved).
 * @param {string} text
 * @param {string[]} terms
 * @returns {Node[]}
 */
export function highlightTerms(text, terms) {
  const source = String(text ?? "");
  const cleaned = [...new Set(terms.map((t) => String(t).trim().toLowerCase()).filter((t) => t.length >= 2))];
  if (!cleaned.length) return [document.createTextNode(source)];
  const lower = source.toLowerCase();
  const nodes = [];
  let index = 0;
  while (index < source.length) {
    let best = -1;
    let bestLength = 0;
    for (const term of cleaned) {
      const found = lower.indexOf(term, index);
      if (found !== -1 && (best === -1 || found < best || (found === best && term.length > bestLength))) {
        best = found;
        bestLength = term.length;
      }
    }
    if (best === -1) break;
    if (best > index) nodes.push(document.createTextNode(source.slice(index, best)));
    nodes.push(h("mark", null, source.slice(best, best + bestLength)));
    index = best + bestLength;
  }
  if (index < source.length) nodes.push(document.createTextNode(source.slice(index)));
  return nodes;
}

/**
 * Like {@link highlightTerms} for one exact passage, tolerant to whitespace differences.
 * Returns `null` when the passage does not occur in `text`.
 * @param {string} text
 * @param {string} passage
 * @returns {Node[] | null}
 */
export function highlightPassage(text, passage) {
  const source = String(text ?? "");
  const wanted = String(passage ?? "").trim();
  if (wanted.length < 3) return null;
  const pattern = wanted
    .split(/\s+/)
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("\\s+");
  let match;
  try {
    match = new RegExp(pattern, "i").exec(source);
  } catch {
    return null;
  }
  if (!match) return null;
  return [
    document.createTextNode(source.slice(0, match.index)),
    h("mark", { class: "passage-mark" }, match[0]),
    document.createTextNode(source.slice(match.index + match[0].length)),
  ];
}
