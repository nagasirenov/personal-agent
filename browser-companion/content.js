"use strict";

const MAX_PAGE_TEXT = 20_000;
const MAX_SELECTION = 4_000;
const MAX_HEADINGS = 24;
const MAX_LINKS = 20;

function visible(element) {
  return Boolean(element && element.getClientRects().length);
}

function textOf(element, limit) {
  return (element?.innerText || element?.textContent || "").replace(/\s+/g, " ").trim().slice(0, limit);
}

function collectPage() {
  const pageText = (document.body?.innerText || "").replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").slice(0, MAX_PAGE_TEXT);
  const selection = window.getSelection()?.toString() || "";
  const headings = Array.from(document.querySelectorAll("h1, h2, h3"))
    .filter(visible)
    .map((item) => textOf(item, 160))
    .filter(Boolean)
    .slice(0, MAX_HEADINGS);
  const links = Array.from(document.querySelectorAll("a[href]"))
    .filter(visible)
    .slice(0, 200)
    .map((item) => ({ text: textOf(item, 120), url: item.href }))
    .filter((item) => item.text || item.url)
    .slice(0, MAX_LINKS);
  return {
    url: location.href,
    title: document.title || "",
    page_text: pageText,
    selected_text: selection.slice(0, MAX_SELECTION),
    dom_summary: {
      headings,
      links,
      link_count: document.querySelectorAll("a[href]").length,
      button_count: document.querySelectorAll("button, [role='button']").length,
      form_count: document.forms.length,
      text_length: pageText.length,
    },
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "personal_agent_capture") {
    sendResponse(collectPage());
  }
});

let selectionTimer;
document.addEventListener("selectionchange", () => {
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(() => {
    chrome.runtime.sendMessage({ type: "personal_agent_selection_changed" }).catch(() => {});
  }, 500);
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") chrome.runtime.sendMessage({ type: "personal_agent_dom_changed" }).catch(() => {});
});

let mutationTimer;
let lastMutationNotice = 0;
const mutationObserver = new MutationObserver(() => {
  const now = Date.now();
  if (now - lastMutationNotice < 10_000) return;
  clearTimeout(mutationTimer);
  mutationTimer = setTimeout(() => {
    lastMutationNotice = Date.now();
    chrome.runtime.sendMessage({ type: "personal_agent_dom_changed" }).catch(() => {});
  }, 1500);
});

if (document.body) mutationObserver.observe(document.body, { childList: true, subtree: true, characterData: true });
