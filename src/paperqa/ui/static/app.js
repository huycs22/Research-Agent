const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  documents: [],
  conversations: [],
  activeConversation: null,
  activeRun: null,
  polling: null,
  config: null,
  mode: localStorage.getItem("paperqa-prompt-mode") || "auto",
};

const panelLayout = {
  breakpoint: 820,
  mainMin: 360,
  sidebar: { min: 180, max: 420, default: 244, storage: "paperqa-sidebar-width" },
  inspector: { min: 260, max: 600, default: 330, storage: "paperqa-inspector-width" },
  settings: { min: 170, max: 520, default: 340, contentMin: 140, storage: "paperqa-settings-height" },
};

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function savedPanelWidth(name) {
  const config = panelLayout[name];
  const stored = Number(localStorage.getItem(config.storage));
  return Number.isFinite(stored) && stored > 0 ? clamp(stored, config.min, config.max) : config.default;
}

function currentPanelWidth(name) {
  const value = Number.parseFloat(getComputedStyle($(".app-shell")).getPropertyValue(`--${name}-size`));
  return Number.isFinite(value) ? value : panelLayout[name].default;
}

function applyPanelWidth(name, width, { persist = false } = {}) {
  const config = panelLayout[name];
  const bounded = Math.round(clamp(width, config.min, config.max));
  $(".app-shell").style.setProperty(`--${name}-size`, `${bounded}px`);
  const handle = $(`#${name}-resizer`);
  handle.setAttribute("aria-valuenow", String(bounded));
  handle.setAttribute("aria-valuetext", `${bounded} pixels`);
  if (persist) localStorage.setItem(config.storage, String(bounded));
  return bounded;
}

function fitPanelWidths() {
  if (window.innerWidth <= panelLayout.breakpoint) return;
  let sidebar = clamp(currentPanelWidth("sidebar"), panelLayout.sidebar.min, panelLayout.sidebar.max);
  let inspector = clamp(currentPanelWidth("inspector"), panelLayout.inspector.min, panelLayout.inspector.max);
  const shell = $(".app-shell");
  const sidebarVisible = !shell.classList.contains("sidebar-collapsed");
  const inspectorVisible = !shell.classList.contains("inspector-collapsed");
  const available = Math.max(0, window.innerWidth - panelLayout.mainMin);
  let overflow = (sidebarVisible ? sidebar : 0) + (inspectorVisible ? inspector : 0) - available;
  if (overflow > 0 && inspectorVisible) {
    const reduction = Math.min(overflow, inspector - panelLayout.inspector.min);
    inspector -= reduction;
    overflow -= reduction;
  }
  if (overflow > 0 && sidebarVisible) sidebar -= Math.min(overflow, sidebar - panelLayout.sidebar.min);
  applyPanelWidth("sidebar", sidebar);
  applyPanelWidth("inspector", inspector);
}

function settingsMaximumHeight() {
  const inspector = $("#inspector");
  const reserved = 69 + 38 + panelLayout.settings.contentMin + 9;
  return Math.max(panelLayout.settings.min, Math.min(panelLayout.settings.max, inspector.clientHeight - reserved));
}

function currentSettingsHeight() {
  const value = Number.parseFloat(getComputedStyle($("#inspector")).getPropertyValue("--settings-size"));
  return Number.isFinite(value) ? value : panelLayout.settings.default;
}

function applySettingsHeight(height, { persist = false } = {}) {
  const config = panelLayout.settings;
  const maximum = settingsMaximumHeight();
  const bounded = Math.round(clamp(height, config.min, maximum));
  $("#inspector").style.setProperty("--settings-size", `${bounded}px`);
  const handle = $("#settings-resizer");
  handle.setAttribute("aria-valuemax", String(Math.round(maximum)));
  handle.setAttribute("aria-valuenow", String(bounded));
  handle.setAttribute("aria-valuetext", `${bounded} pixels high`);
  if (persist) localStorage.setItem(config.storage, String(bounded));
  return bounded;
}

function resizeSettingsFromKeyboard(event) {
  const config = panelLayout.settings;
  const step = event.shiftKey ? 32 : 12;
  let next = currentSettingsHeight();
  if (event.key === "Home") next = config.min;
  else if (event.key === "End") next = settingsMaximumHeight();
  else if (event.key === "0") next = config.default;
  else if (event.key === "ArrowUp") next += step;
  else if (event.key === "ArrowDown") next -= step;
  else return;
  event.preventDefault();
  applySettingsHeight(next, { persist: true });
}

function beginSettingsResize(event) {
  if (event.button !== 0) return;
  event.preventDefault();
  const handle = $("#settings-resizer");
  const startY = event.clientY;
  const startHeight = currentSettingsHeight();
  handle.classList.add("active");
  document.body.classList.add("settings-resizing");

  const move = (moveEvent) => applySettingsHeight(startHeight + startY - moveEvent.clientY);
  const finish = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    window.removeEventListener("pointercancel", finish);
    handle.classList.remove("active");
    document.body.classList.remove("settings-resizing");
    applySettingsHeight(currentSettingsHeight(), { persist: true });
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish);
  window.addEventListener("pointercancel", finish);
}

function resizePanelFromKeyboard(event, name) {
  const config = panelLayout[name];
  const step = event.shiftKey ? 32 : 12;
  let next = currentPanelWidth(name);
  if (event.key === "Home") next = config.min;
  else if (event.key === "End") next = config.max;
  else if (event.key === "0") next = config.default;
  else if (event.key === "ArrowLeft") next += name === "sidebar" ? -step : step;
  else if (event.key === "ArrowRight") next += name === "sidebar" ? step : -step;
  else return;
  event.preventDefault();
  applyPanelWidth(name, next, { persist: true });
  fitPanelWidths();
}

function beginPanelResize(event, name) {
  if (event.button !== 0 || window.innerWidth <= panelLayout.breakpoint) return;
  event.preventDefault();
  const shell = $(".app-shell");
  const handle = $(`#${name}-resizer`);
  const startX = event.clientX;
  const startWidth = currentPanelWidth(name);
  shell.classList.add("resizing");
  handle.classList.add("active");
  document.body.classList.add("panel-resizing");

  const move = (moveEvent) => {
    const delta = name === "sidebar" ? moveEvent.clientX - startX : startX - moveEvent.clientX;
    applyPanelWidth(name, startWidth + delta);
    fitPanelWidths();
  };
  const finish = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    window.removeEventListener("pointercancel", finish);
    shell.classList.remove("resizing");
    handle.classList.remove("active");
    document.body.classList.remove("panel-resizing");
    applyPanelWidth(name, currentPanelWidth(name), { persist: true });
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish);
  window.addEventListener("pointercancel", finish);
}

function initializePanelLayout() {
  applyPanelWidth("sidebar", savedPanelWidth("sidebar"));
  applyPanelWidth("inspector", savedPanelWidth("inspector"));
  const storedSettingsHeight = Number(localStorage.getItem(panelLayout.settings.storage));
  applySettingsHeight(
    Number.isFinite(storedSettingsHeight) && storedSettingsHeight > 0
      ? storedSettingsHeight
      : panelLayout.settings.default,
  );
  fitPanelWidths();
  ["sidebar", "inspector"].forEach((name) => {
    const handle = $(`#${name}-resizer`);
    handle.addEventListener("pointerdown", (event) => beginPanelResize(event, name));
    handle.addEventListener("keydown", (event) => resizePanelFromKeyboard(event, name));
    handle.addEventListener("dblclick", () => {
      applyPanelWidth(name, panelLayout[name].default, { persist: true });
      fitPanelWidths();
    });
  });
  const settingsHandle = $("#settings-resizer");
  settingsHandle.addEventListener("pointerdown", beginSettingsResize);
  settingsHandle.addEventListener("keydown", resizeSettingsFromKeyboard);
  settingsHandle.addEventListener("dblclick", () => {
    applySettingsHeight(panelLayout.settings.default, { persist: true });
  });
  let resizeFrame;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      fitPanelWidths();
      applySettingsHeight(currentSettingsHeight());
      updateBackdrop();
    });
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function citationMatchesSource(label, source) {
  const citation = String(label).replace(/^\(|\)$/g, "").trim().toLowerCase();
  if (/^src-[a-f0-9]{24}$/.test(citation)) return citation === source.id;
  return [source.name, source.docname].filter(Boolean).some((name) => {
    const candidate = String(name).trim().toLowerCase();
    return citation === candidate || citation.startsWith(`${candidate} `) || candidate.startsWith(`${citation} `);
  });
}

function shortCitationLabel(source = {}) {
  const full = String(source.name || source.docname || "Source").trim();
  if (!/\bMAKES-QA(?=[_\s]|$)/i.test(full)) return full;
  const pageFromName = full.match(/\bpages?\s+([\d-]+)\s*$/i)?.[1] || "";
  const page = String(source.page || pageFromName).trim();
  const pagePrefix = page.includes("-") ? "pp." : "p.";
  return `MAKES-QA${page ? ` · ${pagePrefix} ${page}` : ""}`;
}

function inlineMarkdown(value = "", sources = []) {
  const codeSpans = [];
  let text = String(value).replace(/`([^`\n]+)`/g, (_, code) => {
    const marker = `\u0000CODE${codeSpans.length}\u0000`;
    codeSpans.push(`<code>${escapeHtml(code)}</code>`);
    return marker;
  });
  text = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/gi, (_, label, url) =>
      `<a href="${url.replaceAll("&amp;", "&")}" target="_blank" rel="noopener noreferrer">${label}</a>`)
    .replace(/\(([^()]*?(?:pages?|et al\.|\d{4})[^()]*)\)/gi, (match, label) =>
      !/src-[a-f0-9]{24}/.test(label) && sources.some((source) => citationMatchesSource(label, source))
        ? `<button type="button" class="citation-pill" title="Show cited source" aria-label="Show source ${label}">(${label})</button>`
        : match);

  text = text.replace(/\(?(src-[a-f0-9]{24})\)?/g, (match, id) => {
    const source = sources.find((item) => item.id === id);
    if (!source) return match;
    const label = shortCitationLabel(source);
    const full = source.name || source.docname || label;
    return `<button type="button" class="citation-pill" data-source-id="${id}" title="${escapeHtml(full)}" aria-label="Show supporting passage from ${escapeHtml(full)}">(${escapeHtml(label)})</button>`;
  });
  return text.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeSpans[Number(index)]);
}

function tableCells(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  const cells = tableCells(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function startsMarkdownBlock(lines, index) {
  const line = lines[index] || "";
  return /^\s*```/.test(line)
    || /^#{1,4}\s+/.test(line)
    || /^\s*(?:[-*+] |\d+[.)] |>\s?|(?:-{3,}|\*{3,})\s*$)/.test(line)
    || (line.includes("|") && isTableDivider(lines[index + 1] || ""));
}

function markdown(value = "", sources = []) {
  const lines = String(value).trim().replaceAll("\r\n", "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    if (!lines[index].trim()) { index += 1; continue; }
    const fence = lines[index].match(/^\s*```([\w+-]*)\s*$/);
    if (fence) {
      const language = fence[1];
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        code.push(lines[index]); index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(`<pre${language ? ` data-language="${escapeHtml(language)}"` : ""}><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = lines[index].match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      blocks.push(`<h${level}>${inlineMarkdown(heading[2], sources)}</h${level}>`);
      index += 1;
      continue;
    }
    if (lines[index].includes("|") && isTableDivider(lines[index + 1] || "")) {
      const headers = tableCells(lines[index]);
      const dividers = tableCells(lines[index + 1]);
      const alignments = dividers.map((cell) => cell.startsWith(":") && cell.endsWith(":") ? "center" : cell.endsWith(":") ? "right" : "left");
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
        rows.push(tableCells(lines[index])); index += 1;
      }
      const header = headers.map((cell, cellIndex) => `<th scope="col" class="align-${alignments[cellIndex] || "left"}">${inlineMarkdown(cell, sources)}</th>`).join("");
      const body = rows.map((row) => `<tr>${headers.map((_, cellIndex) => `<td class="align-${alignments[cellIndex] || "left"}">${inlineMarkdown(row[cellIndex] || "", sources)}</td>`).join("")}</tr>`).join("");
      blocks.push(`<div class="markdown-table-wrap" role="region" aria-label="Scrollable comparison table" tabindex="0"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`);
      continue;
    }
    const list = lines[index].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
    if (list) {
      const ordered = /^\d/.test(list[1]);
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(`<li>${inlineMarkdown(item[2], sources)}</li>`); index += 1;
      }
      blocks.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (/^\s*>/.test(lines[index])) {
      const quoted = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) {
        quoted.push(lines[index].replace(/^\s*>\s?/, "")); index += 1;
      }
      blocks.push(`<blockquote>${inlineMarkdown(quoted.join("\n"), sources).replaceAll("\n", "<br>")}</blockquote>`);
      continue;
    }
    if (/^\s*(?:-{3,}|\*{3,})\s*$/.test(lines[index])) {
      blocks.push("<hr>"); index += 1; continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !startsMarkdownBlock(lines, index)) {
      paragraph.push(lines[index]); index += 1;
    }
    if (paragraph.length) blocks.push(`<p>${paragraph.map((line) => inlineMarkdown(line, sources)).join("<br>")}</p>`);
    else index += 1;
  }
  return blocks.join("");
}

function toast(message, type = "") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  $("#toasts").append(node);
  setTimeout(() => node.remove(), 4300);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function setProvider(dot, label, configured) {
  dot.classList.toggle("online", configured);
  dot.classList.toggle("warning", !configured);
  label.textContent = configured ? "Key set" : "Key missing";
}

async function loadConfig() {
  state.config = await api("/api/config");
  const workspacePath = state.config.workspace_path || "Local workspace";
  const workspaceName = workspacePath.split(/[\\/]/).filter(Boolean).pop() || "local";
  $("#workspace-breadcrumb").textContent = `Workspace: ${workspaceName}`;
  $("#workspace-badge").lastChild.textContent = ` Workspace: ${workspaceName}`;
  $("#workspace-breadcrumb").title = workspacePath;
  $("#workspace-badge").title = `Active workspace: ${workspacePath}`;
  setProvider($("#cloudflare-dot"), $("#cloudflare-label"), state.config.cloudflare_configured);
  $("#model").innerHTML = state.config.model_options.map((option) => `
    <option value="${escapeHtml(option.value)}" ${option.available ? "" : "disabled"}>
      ${escapeHtml(option.label)}${option.available ? "" : " · setup required"}
    </option>
  `).join("");
  $("#model").value = state.config.default_model;
  $("#reasoning-effort").value = state.config.default_reasoning_effort;
  $("#embedding").innerHTML = state.config.embedding_options.map((option) => `
    <option value="${escapeHtml(option.value)}" ${option.available ? "" : "disabled"}>
      ${escapeHtml(option.label)}${option.available ? "" : " · setup required"}
    </option>
  `).join("");
  $("#embedding").value = state.config.default_embedding;
  updateEmbeddingHelp();
  updateModelHelp();
  updateWebSearchAvailability();
}

function updateModelHelp() {
  const option = state.config?.model_options?.find((item) => item.value === $("#model").value);
  $("#model-help").textContent = option?.description || "Choose the agent model provider.";
}

function updateWebSearchAvailability() {
  const option = state.config?.model_options?.find((item) => item.value === $("#model").value);
  const enabled = Boolean(option?.available);
  const input = $("#web-search");
  input.disabled = !enabled;
  input.checked = enabled;
  $("#web-search-help").textContent = enabled
    ? "Searches public arXiv and Crossref records; opened sources are cited"
    : "Unavailable until the selected language model is configured";
}

function updateEmbeddingHelp() {
  const option = state.config?.embedding_options?.find((item) => item.value === $("#embedding").value);
  $("#embedding-help").textContent = option?.description || "Choose how passages are retrieved.";
}

async function loadDocuments() {
  const payload = await api("/api/documents");
  state.documents = payload.documents;
  renderDocuments();
}

function renderDocuments() {
  const list = $("#documents");
  const count = state.documents.length;
  $("#document-count").textContent = count;
  $("#document-total").textContent = `${count} ${count === 1 ? "file" : "files"}`;
  $("#library-summary").textContent = count ? `${count} ${count === 1 ? "paper" : "papers"} in library` : "No papers in library";
  if (!count) {
    list.innerHTML = '<div class="documents-empty">No documents added yet.</div>';
    return;
  }
  list.innerHTML = state.documents.map((doc) => `
    <div class="document">
      <div class="file-icon">${escapeHtml(doc.type)}</div>
      <a href="/api/documents/${encodeURIComponent(doc.name)}" target="_blank" title="Open ${escapeHtml(doc.name)}">
        <strong>${escapeHtml(doc.name)}</strong><small>${formatBytes(doc.size)}</small>
      </a>
      <button class="delete-document" data-document="${escapeHtml(doc.name)}" title="Remove document">×</button>
    </div>
  `).join("");
  $$(".delete-document", list).forEach((button) => button.addEventListener("click", () => deleteDocument(button.dataset.document)));
}

async function uploadFiles(files) {
  for (const file of files) {
    try {
      toast(`Adding ${file.name}…`);
      await api("/api/documents", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Filename": file.name },
        body: file,
      });
      toast(`${file.name} added`, "success");
    } catch (error) {
      toast(error.message, "error");
    }
  }
  await loadDocuments();
}

async function deleteDocument(name) {
  if (!window.confirm(`Remove ${name} from the research library?`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`${name} removed`);
    await loadDocuments();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function addDemo() {
  const button = $("#demo-button");
  button.disabled = true;
  button.querySelector("strong").textContent = "Downloading demo paper…";
  try {
    await api("/api/demo", { method: "POST" });
    await loadDocuments();
    toast("PaperQA2 demo paper added", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.querySelector("strong").textContent = "Add the PaperQA2 demo paper";
  }
}

function switchTab(name) {
  $$(".inspector-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  $("#inspector").classList.add("open");
  $(".app-shell").classList.remove("inspector-collapsed", "mobile-sidebar-open");
  fitPanelWidths();
  updateBackdrop();
}

function updateBackdrop() {
  const mobile = window.matchMedia("(max-width: 820px)").matches;
  const shell = $(".app-shell");
  const panelOpen = shell.classList.contains("mobile-sidebar-open") || $("#inspector").classList.contains("open");
  $("#panel-backdrop").classList.toggle("visible", mobile && panelOpen);
}

function closeInspector() {
  $("#inspector").classList.remove("open");
  if (!window.matchMedia("(max-width: 820px)").matches) {
    $(".app-shell").classList.add("inspector-collapsed");
  }
  updateBackdrop();
}

function toggleInspector() {
  const shell = $(".app-shell");
  if (window.matchMedia("(max-width: 820px)").matches) {
    $("#inspector").classList.toggle("open");
  } else {
    shell.classList.toggle("inspector-collapsed");
  }
  fitPanelWidths();
  updateBackdrop();
}

function toggleSidebar() {
  const shell = $(".app-shell");
  if (window.matchMedia("(max-width: 820px)").matches) {
    shell.classList.toggle("mobile-sidebar-open");
  } else {
    shell.classList.toggle("sidebar-collapsed");
  }
  fitPanelWidths();
  updateBackdrop();
}

function openSidebar() {
  const shell = $(".app-shell");
  if (window.matchMedia("(max-width: 820px)").matches) {
    shell.classList.add("mobile-sidebar-open");
  } else {
    shell.classList.remove("sidebar-collapsed");
  }
  fitPanelWidths();
  updateBackdrop();
}

function closeMobilePanels() {
  $(".app-shell").classList.remove("mobile-sidebar-open");
  $("#inspector").classList.remove("open");
  updateBackdrop();
}

function setMode(mode) {
  if (!["auto", "research", "chat"].includes(mode)) return;
  state.mode = mode;
  localStorage.setItem("paperqa-prompt-mode", mode);
  const labels = { auto: "Auto", research: "Research", chat: "Chat" };
  $("#mode-label").textContent = labels[mode];
  $$("#mode-menu button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#mode-menu").hidden = true;
  $("#mode-button").setAttribute("aria-expanded", "false");
  $("#question").placeholder = mode === "chat" ? "Message PaperQA…" : "Ask a question about your papers…";
}

function appendUserMessage(question) {
  $("#welcome").classList.add("hidden");
  $("#messages").insertAdjacentHTML("beforeend", `
    <div class="message user-message"><div class="user-bubble">${escapeHtml(question).replaceAll("\n", "<br>")}</div></div>
  `);
}

function appendRunCard() {
  $("#messages").insertAdjacentHTML("beforeend", `
    <div class="message assistant-message" id="active-message">
      <div class="assistant-avatar">P</div>
      <div class="assistant-content">
        <div class="assistant-meta"><strong>PaperQA</strong><span>Working</span></div>
        <div class="run-card"><div class="run-card-title"><span class="spinner"></span><span id="run-title">Preparing research…</span></div><p id="run-detail">Starting the agent workspace</p></div>
      </div>
    </div>
  `);
  scrollConversation();
}

function updateRunCard(job) {
  const last = job.events[job.events.length - 1];
  if (!last) return;
  const title = $("#run-title");
  const detail = $("#run-detail");
  if (title) title.textContent = last.title;
  if (detail) detail.textContent = `${last.detail || "PaperQA is working"} · ${usageLabel(job.usage, job.budget)}`;
  renderTrace(job.events, "Current response · running");
}

function renderTrace(events = [], scope = "Latest response") {
  $("#trace-scope").textContent = scope;
  $("#trace-empty").style.display = events.length ? "none" : "flex";
  $("#trace").innerHTML = events.map((event) => `
    <div class="trace-event ${event.status}">
      <div class="trace-node">${event.status === "complete" ? "✓" : event.status === "error" ? "!" : "·"}</div>
      <div class="trace-copy"><strong>${escapeHtml(event.title)}</strong>${event.timestamp ? `<small>${new Date(event.timestamp * 1000).toLocaleTimeString()}</small>` : ""}${event.detail ? `<span>${escapeHtml(event.detail)}</span>` : ""}</div>
    </div>
  `).join("");
}

function renderSources(sources = [], focus = "", scope = "Latest response", isConversation = false) {
  $("#sources-scope").textContent = scope;
  $("#sources-empty strong").textContent = isConversation ? "No library sources" : "No sources for this response";
  $("#sources-empty p").textContent = isConversation
    ? "This reply did not inspect library sources. Ask a research question in Auto or Research mode to obtain cited evidence."
    : "No paper passage was attached to this response.";
  $("#sources-empty").style.display = sources.length ? "none" : "flex";
  const cited = sources.filter((source) => source.used);
  const visible = cited.length ? cited : sources;
  const focused = focus.replace(/^\(|\)$/g, "").trim();
  const matches = (source) => focused && citationMatchesSource(focused, source);
  const focusedSources = focused ? sources.filter(matches) : [];
  const ordered = [...focusedSources, ...visible.filter((source) => !focusedSources.includes(source))];
  const extra = sources.length - cited.length;
  $("#sources").innerHTML = `${sources.length ? `<p class="sources-note">${cited.length ? `${cited.length} cited ${cited.length === 1 ? "passage" : "passages"}${extra ? ` · ${extra} other passages inspected` : ""}` : "No passage cited · inspected passages shown"}</p>` : ""}${ordered.map((source) => `
    <article class="source-card ${matches(source) ? "source-focused" : ""}">
      <div class="source-head"><strong>${escapeHtml(source.name || source.docname)}</strong><span class="score">${source.score == null ? "Excerpt" : `${escapeHtml(source.score)}/10`}</span></div>
      <p>${escapeHtml(source.summary || "Evidence passage used by the agent.")}</p>
      <p class="source-citation">${escapeHtml(source.citation || source.docname)}</p>
      ${source.filename && state.documents.some((doc) => doc.name === source.filename) ? `<a class="source-open" href="/api/documents/${encodeURIComponent(source.filename)}${/^\d+$/.test(source.page || "") ? `#page=${source.page}` : ""}" target="_blank" rel="noopener">Open paper ↗</a>` : /^https?:\/\//i.test(source.url || "") ? `<a class="source-open" href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">Open web source ↗</a>` : ""}
    </article>
  `).join("")}`;
}

async function showRunDetails(button, tab, focus = "") {
  const runId = button.closest("[data-run]")?.dataset.run;
  if (!runId || !state.activeConversation) return toast("Run details are unavailable for this response.", "error");
  try {
    const payload = await api(`/api/conversations/${state.activeConversation}`);
    const responses = payload.conversation.messages.filter((item) => item.role === "assistant");
    const index = responses.findIndex((item) => item.run_id === runId);
    if (index < 0) return toast("Saved details for this response were not found.", "error");
    const message = responses[index];
    const scope = `Response ${index + 1} of ${responses.length}`;
    renderTrace(message.events || [], scope);
    renderSources(message.result?.sources || [], focus, scope, message.result?.route === "conversation");
    switchTab(tab);
  } catch (error) {
    toast(error.message, "error");
  }
}

function usageLabel(usage = {}, budget = {}) {
  if (!usage || usage.tokens == null) return "Usage unavailable for this saved response";
  const total = Number(usage.tokens || 0);
  const estimated = Number(usage.estimated_tokens || 0);
  const reported = Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0);
  const tokenDetail = estimated
    ? `${total.toLocaleString()} tracked tokens (${reported.toLocaleString()} provider-reported + ${estimated.toLocaleString()} reserved estimate)`
    : `${total.toLocaleString()} provider-reported tokens`;
  return `${tokenDetail} · ${usage.model_calls || 0} model calls · ${usage.tool_calls || 0} tools · ${Math.round(usage.seconds || 0)}s active`;
}

function pausedMarkup(message) {
  const id = message.run_id || message.id;
  return `<div class="message assistant-message" data-run="${escapeHtml(id)}">
    <div class="assistant-avatar">P</div><div class="assistant-content">
    <div class="assistant-meta"><strong>PaperQA</strong><span>${escapeHtml(message.status || "paused")}</span></div>
    <div class="run-card"><strong>${message.pending_question ? "Clarification needed" : message.status === "failed" ? "Research failed" : message.status === "stopped" ? "Research stopped" : "Research paused"}</strong>
    <p>${escapeHtml(message.pending_question || message.error || "Saved work is available to resume.")}</p>
    ${message.retry_at ? `<p>Provider retry time: ${new Date(message.retry_at * 1000).toLocaleString()}</p>` : ""}
    <p>${usageLabel(message.usage, message.budget)}</p>
    ${message.pending_question ? '<textarea class="clarification-input" aria-label="Clarification answer" maxlength="4000" placeholder="Your answer…"></textarea>' : ""}
    <button class="answer-action resume-run" data-resume="${escapeHtml(id)}">Resume saved work</button>
    <button class="answer-action" data-inspect="trace">View trace</button></div></div></div>`;
}

async function resumeRun(button) {
  const id = button.dataset.resume;
  const body = {};
  const input = button.closest(".assistant-message").querySelector(".clarification-input");
  if (input) { body.answer = input.value.trim(); if (!body.answer) return input.focus(); }
  button.disabled = true;
  try {
    await api(`/api/runs/${id}/resume`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    button.closest(".assistant-message").remove();
    appendRunCard(); state.activeRun = id; setRunning(true); pollRun();
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

function assistantMessageMarkup(message) {
  if (["paused", "stopped", "failed"].includes(message.status) && (message.run_id || message.id)) return pausedMarkup(message);
  const result = message.result;
  if (!result) {
    return `
      <div class="message assistant-message">
        <div class="assistant-avatar">P</div>
        <div class="assistant-content">
          <div class="assistant-meta"><strong>PaperQA</strong><span>Stopped</span></div>
          <div class="run-card error-card"><div class="run-card-title"><span>!</span><span>Research run stopped</span></div><p>${escapeHtml(message.error || message.content || "An unexpected error occurred.")}</p></div>
        </div>
      </div>`;
  }
  const tools = result.tools?.length ? [...new Set(result.tools)].join(" → ") : "agent complete";
  const events = message.events || [];
  const conversational = result.route === "conversation";
  const resultStatus = result.status && result.status !== "success" ? result.status : "";
  return `
    <div class="message assistant-message ${resultStatus ? "result-warning" : ""}" data-run="${escapeHtml(message.run_id || "")}">
      <div class="assistant-avatar">P</div>
      <div class="assistant-content">
        <div class="assistant-meta"><strong>PaperQA</strong><span>${escapeHtml(message.model || "PaperQA")}</span></div>
        <div class="answer">${markdown(result.answer || message.content || "", result.sources || [])}</div>
        <div class="answer-footer">
          <span>${conversational ? "Conversation" : `${result.reference_count || 0} ${(result.reference_count || 0) === 1 ? "source" : "sources"}`}</span>
          ${!conversational && result.embedding === "sparse" && message.embedding === "sparse" ? '<span>Sparse retrieval</span>' : ""}
          <span>${events.length} agent stages</span>
          <span>${escapeHtml(tools)}</span>
          ${resultStatus ? `<span class="result-status">${escapeHtml(resultStatus)}</span>` : ""}
          <span>${usageLabel(result.usage || message.usage, message.budget)}</span>
          ${result.cost ? `<span>$${Number(result.cost).toFixed(4)}</span>` : ""}
          <button type="button" class="answer-action" data-inspect="trace">View trace</button>
          ${result.sources?.length ? '<button type="button" class="answer-action" data-inspect="sources">View sources</button>' : ""}
        </div>
      </div>
    </div>`;
}

function appendAnswer(job) {
  const result = job.result;
  const active = $("#active-message");
  if (!active || !result) return;
  active.outerHTML = assistantMessageMarkup({
    content: result?.answer || "",
    usage: job.usage,
    budget: job.budget,
    model: job.model,
    embedding: job.embedding,
    run_id: job.id,
    events: job.events,
    result,
  });
  const scope = `Response ${$$(".assistant-message", $("#messages")).length}`;
  renderTrace(job.events, scope);
  renderSources(result.sources || [], "", scope, result.route === "conversation");
  if (result.route !== "conversation") switchTab("sources");
  scrollConversation();
}

function appendError(job) {
  const active = $("#active-message");
  if (!active) return;
  active.querySelector(".assistant-meta span").textContent = "Stopped";
  const card = active.querySelector(".run-card");
  card.classList.add("error-card");
  card.innerHTML = `<div class="run-card-title"><span>!</span><span>Research run stopped</span></div><p>${escapeHtml(job.error || "An unexpected error occurred.")}</p>`;
  toast(job.error || "Research run stopped", "error");
}

function setRunning(running) {
  $("#send-button").disabled = running;
  $("#stop-button").hidden = !running;
  $("#question").disabled = running;
}

async function startRun() {
  const input = $("#question");
  const question = input.value.trim();
  const isNewConversation = !state.activeConversation;
  if (!question) return input.focus();
  appendUserMessage(question);
  appendRunCard();
  input.value = "";
  resizeTextarea();
  setRunning(true);
  switchTab("trace");
  const payload = {
    request_id: crypto.randomUUID(),
    question,
    conversation_id: state.activeConversation,
    model: $("#model").value,
    reasoning_effort: $("#reasoning-effort").value,
    embedding: $("#embedding").value,
    evidence_k: Number($("#evidence-k").value),
    answer_max_sources: Number($("#max-sources").value),
    web_search: $("#web-search").checked,
    mode: state.mode,
  };
  try {
    const response = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeRun = response.run.id;
    state.activeConversation = response.run.conversation_id;
    localStorage.setItem("paperqa-active-conversation", state.activeConversation);
    if (isNewConversation) {
      $("#conversation-title").textContent = question.length > 64 ? `${question.slice(0, 61)}…` : question;
    }
    await loadHistory();
    pollRun();
  } catch (error) {
    appendError({ error: error.message });
    setRunning(false);
  }
}

async function pollRun() {
  clearTimeout(state.polling);
  if (!state.activeRun) return;
  try {
    const job = await api(`/api/runs/${state.activeRun}`);
    updateRunCard(job);
    if (["paused", "stopped", "failed"].includes(job.status)) {
      const active = $("#active-message");
      if (active) active.outerHTML = pausedMarkup(job);
      state.activeRun = null; setRunning(false); await loadHistory(); return;
    }
    if (job.status === "complete" || (job.status === "unsure" && job.result)) {
      appendAnswer(job);
      if (job.status !== "complete") toast(`Run finished with status: ${job.status}`, "error");
      state.activeRun = null;
      setRunning(false);
      await loadHistory();
      return;
    }
    if (["failed", "error"].includes(job.status) || (job.status === "unsure" && !job.result)) {
      appendError(job);
      state.activeRun = null;
      setRunning(false);
      await loadHistory();
      return;
    }
    state.polling = setTimeout(pollRun, 900);
  } catch (error) {
    toast(error.message, "error");
    state.polling = setTimeout(pollRun, 1800);
  }
}

async function loadHistory() {
  const payload = await api("/api/conversations");
  state.conversations = payload.conversations;
  const list = $("#history-list");
  if (!state.conversations.length) {
    list.innerHTML = '<div class="history-empty">Your conversations will appear here.</div>';
    return;
  }
  list.innerHTML = state.conversations.map((conversation) => `
    <div class="history-row">
      <button class="history-item ${conversation.id === state.activeConversation ? "active" : ""}" data-conversation="${conversation.id}">
        <strong>${escapeHtml(conversation.title)}</strong>
        <small><span class="history-status ${["failed", "error", "unsure", "interrupted"].includes(conversation.status) ? conversation.status : ""}">${escapeHtml(conversation.status)}</span> · ${conversation.turn_count} ${conversation.turn_count === 1 ? "turn" : "turns"} · ${new Date(conversation.updated_at * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}</small>
      </button>
      <button class="delete-conversation" data-delete-conversation="${conversation.id}" title="Delete conversation" aria-label="Delete ${escapeHtml(conversation.title)}">×</button>
    </div>
  `).join("");
  $$(".history-item", list).forEach((button) => button.addEventListener("click", () => showConversation(button.dataset.conversation)));
  $$(".delete-conversation", list).forEach((button) => button.addEventListener("click", () => deleteConversation(button.dataset.deleteConversation)));
}

async function deleteConversation(id) {
  const conversation = state.conversations.find((item) => item.id === id);
  if (!conversation || !window.confirm(`Delete “${conversation.title}”? This cannot be undone.`)) return;
  try {
    await api(`/api/conversations/${id}`, { method: "DELETE" });
    if (state.activeConversation === id) resetConversation();
    await loadHistory();
    toast("Conversation deleted", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function showConversation(id, { openInspector = true } = {}) {
  if (state.activeRun && id !== state.activeConversation) {
    return toast("Wait for the active run to finish before switching conversations.");
  }
  try {
    const payload = await api(`/api/conversations/${id}`);
    const conversation = payload.conversation;
    state.activeConversation = conversation.id;
    localStorage.setItem("paperqa-active-conversation", conversation.id);
    $("#conversation-title").textContent = conversation.title;
    $("#welcome").classList.add("hidden");
    $("#messages").innerHTML = conversation.messages.map((message) => {
      if (message.role === "user") {
        return `<div class="message user-message"><div class="user-bubble">${escapeHtml(message.content).replaceAll("\n", "<br>")}</div></div>`;
      }
      return assistantMessageMarkup(message);
    }).join("");

    const runs = await api("/api/runs");
    const activeJob = runs.runs.find((job) => job.conversation_id === id && ["queued", "running"].includes(job.status));
    if (activeJob) {
      const previous = document.querySelector(`[data-run="${activeJob.id}"]`);
      if (previous) previous.remove();
      appendRunCard(); state.activeRun = activeJob.id; setRunning(true); pollRun();
    }
    const responses = conversation.messages.filter((message) => message.role === "assistant");
    const latestAssistant = responses.at(-1);
    const scope = responses.length ? `Response ${responses.length} of ${responses.length}` : "No response selected";
    renderTrace(latestAssistant?.events || [], scope);
    renderSources(latestAssistant?.result?.sources || [], "", scope, latestAssistant?.result?.route === "conversation");
    if (openInspector && latestAssistant && latestAssistant.result?.route !== "conversation") switchTab(latestAssistant.result ? "sources" : "trace");
    await loadHistory();
    scrollConversation();
    $(".app-shell").classList.remove("mobile-sidebar-open");
    updateBackdrop();
    $("#question").focus({ preventScroll: true });
  } catch (error) {
    toast(error.message, "error");
  }
}

function resetConversation() {
  if (state.activeRun) return toast("Wait for the active run to finish.");
  state.activeConversation = null;
  localStorage.removeItem("paperqa-active-conversation");
  $("#messages").innerHTML = "";
  $("#welcome").classList.remove("hidden");
  $("#conversation-title").textContent = "New research";
  renderTrace([], "No response selected");
  renderSources([], "", "No response selected");
  loadHistory();
  $("#question").focus({ preventScroll: true });
}

function scrollConversation() {
  const view = $("#conversation");
  requestAnimationFrame(() => { view.scrollTop = view.scrollHeight; });
}

function resizeTextarea() {
  const area = $("#question");
  area.style.height = "auto";
  area.style.height = `${Math.min(area.scrollHeight, 160)}px`;
}

function bindEvents() {
  $("#messages").addEventListener("click", (event) => {
    const citation = event.target.closest(".citation-pill");
    if (citation) return showRunDetails(citation, "sources", citation.dataset.sourceId || citation.textContent || "");
    const resume = event.target.closest(".resume-run");
    if (resume) return resumeRun(resume);
    const action = event.target.closest(".answer-action");
    if (action) showRunDetails(action, action.dataset.inspect);
  });
  $("#send-button").addEventListener("click", startRun);
  $("#stop-button").addEventListener("click", async () => {
    if (state.activeRun) { try { await api(`/api/runs/${state.activeRun}/stop`, {method: "POST"}); } catch (error) { toast(error.message, "error"); } }
  });
  $("#question").addEventListener("input", resizeTextarea);
  $("#question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      startRun();
    }
  });
  $$(".starter").forEach((button) => button.addEventListener("click", () => {
    $("#question").value = button.dataset.prompt;
    resizeTextarea();
    $("#question").focus();
  }));
  $("#new-task").addEventListener("click", () => { resetConversation(); closeMobilePanels(); });
  $("#refresh-history").addEventListener("click", loadHistory);
  $("#library-nav").addEventListener("click", () => switchTab("library"));
  $("#agent-nav").addEventListener("click", () => { closeMobilePanels(); $("#question").focus({ preventScroll: true }); });
  $("#settings-toggle").addEventListener("click", toggleInspector);
  $("#close-inspector").addEventListener("click", closeInspector);
  $("#sidebar-collapse").addEventListener("click", toggleSidebar);
  $("#mobile-menu").addEventListener("click", openSidebar);
  $("#panel-backdrop").addEventListener("click", closeMobilePanels);
  $$(".inspector-tabs button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  $("#browse-button").addEventListener("click", () => $("#file-input").click());
  $("#attach-button").addEventListener("click", () => { switchTab("library"); $("#file-input").click(); });
  $("#file-input").addEventListener("change", async (event) => { await uploadFiles(event.target.files); event.target.value = ""; });
  $("#demo-button").addEventListener("click", addDemo);
  $("#embedding").addEventListener("change", updateEmbeddingHelp);
  $("#model").addEventListener("change", () => {
    updateModelHelp();
    updateWebSearchAvailability();
  });
  $("#mode-button").addEventListener("click", (event) => {
    event.stopPropagation();
    const menu = $("#mode-menu");
    menu.hidden = !menu.hidden;
    $("#mode-button").setAttribute("aria-expanded", String(!menu.hidden));
  });
  $$("#mode-menu button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".mode-picker")) {
      $("#mode-menu").hidden = true;
      $("#mode-button").setAttribute("aria-expanded", "false");
    }
  });
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      resetConversation();
    }
  });
}

async function init() {
  initializePanelLayout();
  bindEvents();
  setMode(state.mode);
  try {
    await Promise.all([loadConfig(), loadDocuments(), loadHistory()]);
    const rememberedConversation = localStorage.getItem("paperqa-active-conversation");
    if (
      rememberedConversation &&
      state.conversations.some((conversation) => conversation.id === rememberedConversation)
    ) {
      await showConversation(rememberedConversation, { openInspector: false });
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

init();
