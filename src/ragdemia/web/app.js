/* RAGDemia web app — vanilla JS, no build step.
 *
 * Talks to the FastAPI backend:
 *   GET  /health            model info, defaults, status
 *   GET  /documents         library
 *   POST /documents         upload (multipart)       DELETE /documents/{id}
 *   POST /ask/stream        Server-Sent Events: meta → token* → done → closed_book?
 */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const STOP = new Set("about above after again also among an and any are because been before being between both but by can could did does doing during each from have having here how into its itself just more most other over same should some such than that their them then there these they this those through under until very what when where which while who whom why will with would your".split(" "));

const els = {
  sidebar: $("#sidebar"), scrim: $("#scrim"), docList: $("#docList"), docCount: $("#docCount"),
  docSearch: $("#docSearch"), docSearchWrap: $("#docSearchWrap"), fileInput: $("#fileInput"),
  thread: $("#thread"), empty: $("#empty"), suggestions: $("#suggestions"), input: $("#input"),
  send: $("#send"), composer: $("#composer"), status: $("#status"), statusText: $("#statusText"),
  convTitle: $("#convTitle"), models: $("#models"), evidence: $("#evidence"),
  evidenceBody: $("#evidenceBody"), evidenceLabel: $("#evidenceLabel"), dropOverlay: $("#dropOverlay"),
  topK: $("#topK"), topKOut: $("#topKOut"), compare: $("#compare"), modeGroup: $("#modeGroup"),
  toasts: $("#toasts"), confirmModal: $("#confirmModal"),
};

const state = {
  turns: [],            // {question, answer?, closedBook?, error?, stopped?}
  busy: false,
  controller: null,
  docs: [],
  health: null,
  settings: { mode: "hybrid", topK: 5, compare: false },
  active: null,         // {turn, n} shown in the evidence panel
};

/* ------------------------------------------------------------------ storage */
const store = {
  get(key, fallback) { try { const v = localStorage.getItem("ragdemia." + key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem("ragdemia." + key, JSON.stringify(value)); } catch (e) { /* private mode */ } },
};

/* ------------------------------------------------------------------ identity */
// A random id for this browser's private workspace (used by the public demo; harmless locally).
function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
    (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16));
}
const workspaceId = store.get("workspace", null) || newId();
store.set("workspace", workspaceId);
let adminToken = "";
try { adminToken = sessionStorage.getItem("ragdemia.admin") || ""; } catch (e) { /* ignore */ }
function authHeaders(extra = {}) {
  const h = { "X-Workspace": workspaceId, ...extra };
  if (adminToken) h["X-Admin-Token"] = adminToken;
  return h;
}

/* ------------------------------------------------------------------ helpers */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const icon = {
  check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  alert: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.5"/><circle cx="12" cy="12" r="9"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/></svg>',
  quote: '<svg viewBox="0 0 24 24"><path d="M7 7h4v4c0 3-1.5 5-4 6M14 7h4v4c0 3-1.5 5-4 6"/></svg>',
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 011-1h10"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>',
  lock: '<svg viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 018 0v3"/></svg>',
  chevron: '<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>',
  spinner: '<svg viewBox="0 0 24 24"><path d="M12 3a9 9 0 019 9"/></svg>',
  book: '<svg viewBox="0 0 24 24"><path d="M4 5a2 2 0 012-2h13v16H6a2 2 0 00-2 2zM4 5v16M8 7h7"/></svg>',
  compare: '<svg viewBox="0 0 24 24"><path d="M8 4v16M16 4v16M4 8h8M12 16h8"/></svg>',
  hash: '<svg viewBox="0 0 24 24"><path d="M9 4L7 20M17 4l-2 16M4 9h16M3 15h16"/></svg>',
  shield: '<svg viewBox="0 0 24 24"><path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/></svg>',
  logo: '<svg viewBox="0 0 32 32"><rect x="3" y="5" width="11" height="22" rx="2.5" fill="var(--accent)" opacity=".9"/><rect x="16" y="5" width="11" height="22" rx="2.5" fill="var(--accent)" opacity=".45"/><path d="M21.5 10.5l1.1 2.4 2.4 1.1-2.4 1.1-1.1 2.4-1.1-2.4-2.4-1.1 2.4-1.1z" fill="#fff"/></svg>',
};

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: authHeaders(opts.headers || {}) });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

function toast(kind, title, sub = "", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `${kind === "busy" ? icon.spinner : kind === "err" ? icon.alert : icon.check}<div><strong>${esc(title)}</strong>${sub ? `<small>${esc(sub)}</small>` : ""}</div>`;
  els.toasts.appendChild(el);
  const close = () => el.remove();
  if (ms) setTimeout(close, ms);
  return close;
}

function confirmDialog(title, text) {
  return new Promise((resolve) => {
    $("#confirmTitle").textContent = title;
    $("#confirmText").textContent = text;
    els.confirmModal.returnValue = "";
    els.confirmModal.addEventListener("close", () => resolve(els.confirmModal.returnValue === "ok"), { once: true });
    els.confirmModal.showModal();
  });
}

/* ------------------------------------------------------------------ status & models */
function setStatus(kind, text) {
  els.status.className = `status ${kind}`;
  els.statusText.textContent = text;
  els.status.title = text;
}

async function loadHealth() {
  try {
    const h = await api("/health");
    state.health = h;
    setStatus("ok", `Ready · ${h.n_documents} document${h.n_documents === 1 ? "" : "s"}`);
    const rows = [["LLM", h.llm], ["Embeddings", h.embedding_model], ["Reranker", h.reranker || "off"]];
    $("#composerHint").textContent = h.llm.startsWith("openai:")
      ? `Enter to send · Shift + Enter for a new line · Answers are written by an open model (${h.llm.slice(7)}) from the retrieved passages; check important facts against the sources.`
      : "Enter to send · Shift + Enter for a new line · Answers come from a local model; check important facts against the sources.";
    $("#publicBadge").hidden = !h.public;
    $("#publicBadge").textContent = h.is_admin ? "Admin" : "Public demo";
    $("#publicBadge").classList.toggle("badge-admin", !!h.is_admin);
    $("#adminBtn").hidden = !(h.public && h.admin_enabled);
    $("#adminBtn").textContent = h.is_admin ? "Sign out" : "Admin";
    $("#dropzoneText").innerHTML = !h.public || h.can_edit_shared
      ? `<strong>Add PDFs</strong> ${h.public ? "to the shared library" : "or drop them anywhere"}`
      : `<strong>Add your own PDFs</strong> · private to you${h.workspace ? `, up to ${h.workspace.max_docs} · ${h.workspace.max_mb} MB` : ""}`;
    els.models.innerHTML = rows.map(([k, v]) => `<div class="model"><span>${k}</span><code title="${esc(v)}">${esc(v)}</code></div>`).join("");
    if (!store.get("settings", null)) {
      state.settings.mode = h.defaults.mode;
      state.settings.topK = h.defaults.top_k;
      applySettings();
    }
  } catch (e) {
    setStatus("err", "Backend unreachable");
  }
}

/* ------------------------------------------------------------------ library */
function renderDocs() {
  const q = els.docSearch.value.trim().toLowerCase();
  const docs = state.docs.filter((d) => !q || d.title.toLowerCase().includes(q) || d.doc_id.toLowerCase().includes(q));
  els.docCount.textContent = state.docs.length;
  els.docSearchWrap.hidden = state.docs.length < 6;
  $("#emptyLibrary").hidden = state.docs.length > 0 || (state.health?.public && !state.health?.can_edit_shared);
  els.suggestions.hidden = state.docs.length === 0;
  if (!state.docs.length) {
    els.docList.innerHTML = `<li class="doc-meta" style="padding:6px 8px">No documents yet. Add a PDF to get started.</li>`;
    return;
  }
  const docItem = (d) => `
    <li class="doc${d._new ? " is-new" : ""}${d.scope === "workspace" ? " mine" : ""}" title="${esc(d.doc_id)}">
      <div class="doc-icon">PDF</div>
      <div>
        <div class="doc-title">${esc(d.title)}</div>
        <div class="doc-meta">${d.num_pages} page${d.num_pages === 1 ? "" : "s"} · ${d.n_chunks} chunk${d.n_chunks === 1 ? "" : "s"}</div>
      </div>
      ${d.deletable
        ? `<button class="icon-btn danger" data-delete="${esc(d.doc_id)}" aria-label="Remove ${esc(d.title)}" title="Remove from library">${icon.trash}</button>`
        : `<span class="lock" title="${state.health?.public ? "Shared library — only the admin can change it" : "Added on disk — manage the file directly"}">${icon.lock}</span>`}
    </li>`;
  const mine = docs.filter((d) => d.scope === "workspace");
  const shared = docs.filter((d) => d.scope !== "workspace");
  const ttl = state.health?.workspace?.ttl_hours;
  let html = "";
  if (state.health?.public || mine.length) {
    html += `<li class="doc-group">Your uploads <small>private${ttl ? ` · kept ${ttl} h` : ""}</small></li>`;
    html += mine.map(docItem).join("") || `<li class="doc-meta" style="padding:2px 8px 6px">Upload a PDF to ask about it here.</li>`;
    html += `<li class="doc-group">Shared library</li>`;
  }
  html += shared.map(docItem).join("");
  els.docList.innerHTML = docs.length || state.health?.public ? html : `<li class="doc-meta" style="padding:6px 8px">No matches.</li>`;
}

async function loadDocs(newIds = []) {
  try {
    state.docs = (await api("/documents")).map((d) => ({ ...d, _new: newIds.includes(d.doc_id) }));
    renderDocs();
  } catch (e) {
    els.docList.innerHTML = `<li class="doc-meta" style="padding:6px 8px">Could not load the library.</li>`;
  }
}

async function upload(fileList) {
  const files = [...fileList].filter((f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
  if (!files.length) { toast("err", "Only PDF files can be added"); return; }
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const done = toast("busy", `Indexing ${files.length} file${files.length > 1 ? "s" : ""}…`, "Parsing, chunking and embedding", 0);
  setStatus("busy", "Indexing…");
  try {
    const r = await api("/documents", { method: "POST", body: fd });
    done();
    const n = r.added.length + r.updated.length;
    toast("ok", r.scope === "workspace"
      ? `Added ${n} document${n === 1 ? "" : "s"} to your private workspace`
      : `Added ${n} document${n === 1 ? "" : "s"}${state.health?.public ? " to the shared library" : ""}`,
      `${r.n_embedded} chunks embedded in ${r.seconds.toFixed(1)}s — searchable now`);
    r.failures.forEach((f) => toast("err", "Skipped a file", f.error, 8000));
    await loadDocs([...r.added, ...r.updated]);
  } catch (e) {
    done();
    toast("err", "Upload failed", e.message, 8000);
  } finally {
    loadHealth();
  }
}

async function removeDoc(docId) {
  const doc = state.docs.find((d) => d.doc_id === docId);
  if (!doc) return;
  const ok = await confirmDialog("Remove document?", doc.scope === "workspace"
    ? `“${doc.title}” will be removed from your private workspace.`
    : `“${doc.title}” will be deleted from the shared library and the index. Answers will no longer cite it.`);
  if (!ok) return;
  try {
    await api(`/documents/${docId.split("/").map(encodeURIComponent).join("/")}`, { method: "DELETE" });
    toast("ok", "Document removed", doc.title);
    await loadDocs();
    loadHealth();
  } catch (e) {
    toast("err", "Could not remove document", e.message, 8000);
  }
}

/* ------------------------------------------------------------------ markdown + citations */
function inline(s, nSources) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (_, group) => expandGroup(group).map((n) => {
      const ok = n >= 1 && n <= nSources;
      return `<button class="cite${ok ? "" : " invalid"}" data-n="${n}" ${ok ? "" : 'title="No source with this number was provided"'}>${n}</button>`;
    }).join(""));
}

function expandGroup(group) {
  const out = [];
  group.split(/\s*,\s*/).forEach((part) => {
    const m = part.match(/^(\d+)\s*[–-]\s*(\d+)$/);
    if (m) { const a = +m[1], b = +m[2]; for (let i = Math.min(a, b); i <= Math.max(a, b) && i - a < 20; i++) out.push(i); }
    else if (/^\d+$/.test(part.trim())) out.push(+part);
  });
  return out;
}

function renderMarkdown(text, nSources) {
  const lines = esc(text).split("\n");
  const html = [];
  let para = [], list = null;
  const flushPara = () => { if (para.length) html.push(`<p>${inline(para.join("<br>"), nSources)}</p>`); para = []; };
  const flushList = () => { if (list) html.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i, nSources)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (const raw of lines) {
    const line = raw.trim();
    const ul = line.match(/^[-*•]\s+(.*)$/), ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
    } else if (!line) { flushPara(); flushList(); }
    else { flushList(); para.push(line); }
  }
  flushPara(); flushList();
  return html.join("") || "<p></p>";
}

/* ------------------------------------------------------------------ thread rendering */
function turnEl(i) { return els.thread.querySelector(`.turn[data-turn="${i}"]`); }

function pct(x) { return `${Math.round(100 * x)}%`; }

function renderVerify(t, idx) {
  const a = t.answer, c = a.citations;
  if (c.abstained) {
    return `<div class="notice">${icon.shield}<div><strong>Not found in your library.</strong> RAGDemia searched ${a.sources.length} passages and declined rather than guess.</div></div>`;
  }
  const g = c.groundedness;
  const gClass = g >= 0.8 ? "good" : g >= 0.5 ? "mid" : "bad";
  const total = (a.timings.retrieval_s || 0) + (a.timings.generation_s || 0) + (a.timings.rewrite_s || 0);
  const chips = [
    `<span class="chip ${gClass}" title="Share of answer sentences whose content words are found in a retrieved passage">${icon.shield}${pct(g)} grounded</span>`,
    `<span class="chip" title="Sources the answer cites">${icon.quote}${c.cited.length + c.auto_attributed.length} cited</span>`,
  ];
  if (c.auto_attributed.length) chips.push(`<span class="chip mid" title="Supported sentences the model forgot to cite; attributed automatically">${c.auto_attributed.length} auto-attributed</span>`);
  if (c.invalid.length) chips.push(`<span class="chip bad" title="Citations to sources that were never provided">${c.invalid.length} invalid citation${c.invalid.length > 1 ? "s" : ""}</span>`);
  if (total) chips.push(`<span class="chip" title="Retrieval + generation time">${icon.clock}${total.toFixed(1)}s</span>`);
  chips.push(`<button class="chip chip-btn" data-copy="${idx}" title="Copy answer">${icon.copy}Copy</button>`);
  return `<div class="verify">${chips.join("")}</div>`;
}

function renderSources(t, i) {
  const a = t.answer, c = a.citations || { cited: [], auto_attributed: [] };
  const used = (s) => c.cited.includes(s.n) || c.auto_attributed.includes(s.n);
  // Cited sources first; the rest stay one click away (verification needs them, reading doesn't).
  const primary = a.citations && a.sources.some(used) && !t.showAll ? a.sources.filter(used) : a.sources;
  const hidden = a.sources.length - primary.length;
  return `<div class="sources">${primary.map((s) => {
    const tag = c.cited.includes(s.n) ? `<span class="src-tag model">cited</span>` : c.auto_attributed.includes(s.n) ? `<span class="src-tag auto">auto</span>` : "";
    const pages = s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
    const leaf = s.section.split(" > ").pop();
    return `<button class="src${used(s) || !a.citations ? "" : " uncited"}" data-n="${s.n}" title="${esc(s.citation)}">
      <span class="src-top"><span class="src-n">${s.n}</span>${pages}${tag}</span>
      <span class="src-title">${esc(s.doc_title)}</span>
      <span class="src-sec">${esc(leaf)}</span>
    </button>`;
  }).join("")}${hidden ? `<button class="sources-toggle" data-more="${i}">+${hidden} more retrieved</button>` : ""}</div>`;
}

function renderTurn(t, i) {
  const el = document.createElement("div");
  el.className = "turn";
  el.dataset.turn = i;
  el.innerHTML = `
    <div class="q"><div class="q-bubble">${esc(t.question)}</div></div>
    <div class="a"><div class="a-avatar">${icon.logo}</div><div class="a-body"></div></div>`;
  fillAnswer(el, t, i);
  return el;
}

function fillAnswer(el, t, i) {
  const body = el.querySelector(".a-body");
  const a = t.answer;
  let html = "";
  if (a?.search_query) html += `<div class="rewrite" title="Your follow-up was rewritten into a standalone search query">${icon.search}<span>Searched for “${esc(a.search_query)}”</span></div>`;
  if (t.error) {
    html += `<div class="notice error">${icon.alert}<div>${esc(t.error)}</div></div>`;
  } else if (a) {
    html += `<div class="answer">${renderMarkdown(a.text, a.sources.length)}</div>`;
    if (t.stopped) html += `<div class="verify"><span class="chip mid">Stopped — answer is incomplete and was not verified</span></div>`;
    else if (a.citations) html += renderVerify(t, i);
    if (a.sources.length) html += renderSources(t, i);
    if (t.closedBook !== undefined) {
      html += `<div class="compare"><div class="compare-head">${icon.compare}Same model, no retrieval</div>
        <div class="compare-body">${t.closedBook === null ? '<span class="skeleton" style="display:block;height:44px;border-radius:8px"></span>' : renderMarkdown(t.closedBook, 0)}</div></div>`;
    }
  }
  body.innerHTML = html;
}

function renderThread() {
  els.thread.querySelectorAll(".turn").forEach((n) => n.remove());
  els.empty.hidden = state.turns.length > 0;
  state.turns.forEach((t, i) => els.thread.appendChild(renderTurn(t, i)));
  els.convTitle.textContent = state.turns[0]?.question || "New conversation";
}

function scrollToBottom(force = false) {
  const th = els.thread;
  if (force || th.scrollHeight - th.scrollTop - th.clientHeight < 160) th.scrollTop = th.scrollHeight;
}

function persist() {
  store.set("conversation", state.turns.filter((t) => t.answer && !t.error).slice(-20));
}

/* ------------------------------------------------------------------ asking (SSE) */
async function* sse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      raw.split("\n").forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      });
      yield { event, data: data ? JSON.parse(data) : {} };
    }
  }
}

function setBusy(busy) {
  state.busy = busy;
  els.send.classList.toggle("stop", busy);
  els.send.setAttribute("aria-label", busy ? "Stop generating" : "Send");
  els.send.disabled = !busy && !els.input.value.trim();
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  closeEvidence();
  const history = state.turns.filter((t) => t.answer && !t.error && !t.stopped).slice(-3)
    .map((t) => ({ question: t.question, answer: t.answer.text }));
  const t = { question };
  state.turns.push(t);
  const i = state.turns.length - 1;
  els.empty.hidden = true;
  els.convTitle.textContent = state.turns[0].question;
  const el = renderTurn(t, i);
  els.thread.appendChild(el);
  const body = el.querySelector(".a-body");
  const nDocs = state.health?.n_documents ?? "your";
  body.innerHTML = `<div class="progress">
      ${history.length ? `<div class="step" data-step="rewrite"><span class="spinner"></span>Understanding your follow-up</div>` : ""}
      <div class="step" data-step="search"><span class="spinner"></span>Searching ${nDocs} documents</div>
    </div>`;
  scrollToBottom(true);

  setBusy(true);
  setStatus("busy", "Answering…");
  state.controller = new AbortController();
  let text = "", meta = null, frame = 0;
  const paint = () => {
    frame = 0;
    t.answer = { text, sources: meta.sources, search_query: meta.search_query, citations: null, timings: {} };
    fillAnswer(el, t, i);
    el.querySelector(".answer")?.classList.add("streaming");
    scrollToBottom();
  };
  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question, history, top_k: state.settings.topK, mode: state.settings.mode, compare: state.settings.compare }),
      signal: state.controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    for await (const { event, data } of sse(res)) {
      if (event === "meta") {
        meta = data;
        body.querySelectorAll(".step").forEach((s) => s.classList.add("done"));
        body.querySelector(".progress").insertAdjacentHTML("beforeend",
          `<div class="step"><span class="spinner"></span>Reading ${data.sources.length} passages and writing a cited answer</div>`);
      } else if (event === "token") {
        text += data.t;
        if (!frame) frame = requestAnimationFrame(paint);
      } else if (event === "done") {
        if (frame) cancelAnimationFrame(frame);
        t.answer = { ...data, text: data.answer };
        if (state.settings.compare) t.closedBook = null;
        fillAnswer(el, t, i);
        scrollToBottom();
      } else if (event === "closed_book") {
        t.closedBook = data.text;
        fillAnswer(el, t, i);
        scrollToBottom();
      } else if (event === "error") {
        throw new Error(data.message);
      }
    }
    setStatus("ok", `Ready · ${state.health?.n_documents ?? 0} documents`);
  } catch (e) {
    if (frame) cancelAnimationFrame(frame);
    if (e.name === "AbortError") {
      t.stopped = true;
      if (meta) { t.answer = { text: text || "…", sources: meta.sources, search_query: meta.search_query, citations: null, timings: {} }; }
      else t.error = "Stopped before any sources were retrieved.";
    } else {
      t.error = e.message;
    }
    fillAnswer(el, t, i);
    setStatus(e.name === "AbortError" ? "ok" : "err", e.name === "AbortError" ? "Stopped" : "Error");
  } finally {
    state.controller = null;
    setBusy(false);
    persist();
  }
}

/* ------------------------------------------------------------------ evidence panel */
function contentWords(s) {
  return (s.toLowerCase().match(/[a-z0-9][a-z0-9'-]{3,}/g) || []).filter((w) => !STOP.has(w));
}

function openEvidence(turnIdx, n) {
  const t = state.turns[turnIdx];
  const a = t?.answer;
  const s = a?.sources?.[n - 1];
  if (!s) return;
  state.active = { turn: turnIdx, n };
  if (!turnEl(turnIdx)?.querySelector(`.src[data-n="${n}"]`)) { t.showAll = true; fillAnswer(turnEl(turnIdx), t, turnIdx); }
  const c = a.citations;
  const claims = (c?.sentences || []).filter((x) => x.cited.includes(n) || x.best_source === n);
  const words = new Set(claims.flatMap((x) => contentWords(x.sentence)));
  let body = esc(s.text);
  if (words.size) {
    const re = new RegExp(`\\b(${[...words].map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
    body = body.replace(re, "<mark>$1</mark>");
  }
  const pages = s.page_start === s.page_end ? `Page ${s.page_start}` : `Pages ${s.page_start}–${s.page_end}`;
  const role = c?.cited.includes(n) ? `<span class="chip good">${icon.check}Cited by the model</span>`
    : c?.auto_attributed.includes(n) ? `<span class="chip mid">Auto-attributed</span>`
    : `<span class="chip">Retrieved, not cited</span>`;
  const sc = s.scores || {};
  const diag = [
    ["Dense rank", sc.dense_rank ?? "—"],
    ["BM25 rank", Object.entries(sc).find(([k]) => /^bm25(_\d+)?_rank$/.test(k))?.[1] ?? "—"],
    [sc.rerank_score !== undefined ? "Rerank score" : "Score", sc.rerank_score !== undefined ? sc.rerank_score.toFixed(3) : s.score.toFixed(3)],
  ];
  els.evidenceLabel.textContent = `Source ${n} of ${a.sources.length}`;
  els.evidenceBody.innerHTML = `
    <div class="ev-title">${esc(s.doc_title)}</div>
    <div class="crumbs">${s.section.split(" > ").map(esc).join(icon.chevron)}</div>
    <div class="ev-badges"><span class="chip">${icon.book}${pages}</span>${role}</div>
    <div class="ev-text">${body}</div>
    ${claims.length ? `<div class="ev-claims"><h4>Answer sentences supported by this passage</h4>${claims.map((x) =>
      `<div class="claim"><span>${esc(x.sentence.replace(/\s*\[[\d,\s–-]+\]/g, ""))}</span><span class="claim-score" style="color:var(${x.support >= 0.6 ? "--ok" : "--warn"})" title="Share of the sentence's content words found in this passage">${pct(x.support)}</span></div>`).join("")}</div>` : ""}
    <div class="ev-diag"><h4>Retrieval diagnostics</h4><div class="diag">${diag.map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("")}</div></div>
    <div class="ev-path">${esc(s.doc_id)} · ${esc(s.chunk_id)}</div>`;
  els.evidence.hidden = false;
  els.evidenceBody.scrollTop = 0;
  document.querySelectorAll(".cite.active, .src.active").forEach((x) => x.classList.remove("active"));
  turnEl(turnIdx)?.querySelectorAll(`.cite[data-n="${n}"], .src[data-n="${n}"]`).forEach((x) => x.classList.add("active"));
}

function closeEvidence() {
  els.evidence.hidden = true;
  state.active = null;
  document.querySelectorAll(".cite.active, .src.active").forEach((x) => x.classList.remove("active"));
}

function stepEvidence(delta) {
  if (!state.active) return;
  const a = state.turns[state.active.turn].answer;
  const n = ((state.active.n - 1 + delta + a.sources.length) % a.sources.length) + 1;
  openEvidence(state.active.turn, n);
}

/* ------------------------------------------------------------------ settings */
function applySettings() {
  const s = state.settings;
  els.modeGroup.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === s.mode)));
  els.topK.value = s.topK;
  els.topKOut.textContent = s.topK;
  els.compare.checked = s.compare;
  $("#settingsSum").textContent = `${{ hybrid: "Hybrid", dense: "Dense", bm25: "BM25" }[s.mode]} · top ${s.topK}${s.compare ? " · compare" : ""}`;
}
function saveSettings() { store.set("settings", state.settings); }

/* ------------------------------------------------------------------ suggestions */
const SUGGESTIONS = [
  { title: "Explain a concept", sub: "Then follow up with “how is it trained?”", q: "What is byte-pair encoding?" },
  { title: "Compare two papers", sub: "Combines evidence from several documents", q: "How do DPR and HyDE differ in the supervision they need?" },
  { title: "Find an exact number", sub: "Answers quote figures with their page", q: "What batch size did DPR use for in-batch negative training?" },
  { title: "Test its honesty", sub: "Ask something your documents don't cover", q: "What accuracy does Self-RAG achieve on the MMLU benchmark?" },
];
function renderSuggestions() {
  els.suggestions.innerHTML = SUGGESTIONS.map((s, i) =>
    `<button class="suggestion" data-suggest="${i}">${[icon.book, icon.compare, icon.hash, icon.shield][i]}<div><strong>${esc(s.title)}</strong><span>${esc(s.q)}</span></div></button>`).join("");
}

/* ------------------------------------------------------------------ events */
function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  if (!state.busy) els.send.disabled = !els.input.value.trim();
}

function wire() {
  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) { state.controller?.abort(); return; }
    const q = els.input.value;
    els.input.value = "";
    autosize();
    ask(q);
  });
  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.composer.requestSubmit(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.evidence.hidden) closeEvidence();
    if (e.key === "/" && document.activeElement !== els.input && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); els.input.focus(); }
    if (!els.evidence.hidden && (e.key === "ArrowRight" || e.key === "ArrowLeft") && document.activeElement !== els.input) stepEvidence(e.key === "ArrowRight" ? 1 : -1);
  });

  els.thread.addEventListener("click", async (e) => {
    const more = e.target.closest("[data-more]");
    if (more) { const k = +more.dataset.more; state.turns[k].showAll = true; fillAnswer(turnEl(k), state.turns[k], k); return; }
    const sug = e.target.closest("[data-suggest]");
    if (sug) { ask(SUGGESTIONS[+sug.dataset.suggest].q); return; }
    const copy = e.target.closest("[data-copy]");
    if (copy) {
      const a = state.turns[+copy.dataset.copy].answer;
      const refs = a.sources.filter((s) => a.citations.cited.includes(s.n) || a.citations.auto_attributed.includes(s.n)).map((s) => `[${s.n}] ${s.citation}`).join("\n");
      try { await navigator.clipboard.writeText(`${a.text}\n\nSources:\n${refs}`); toast("ok", "Answer copied with sources", "", 2000); } catch (err) { toast("err", "Copy failed"); }
      return;
    }
    const hit = e.target.closest(".cite:not(.invalid), .src");
    if (hit) {
      const turn = +hit.closest(".turn").dataset.turn;
      const n = +hit.dataset.n;
      if (state.active && state.active.turn === turn && state.active.n === n) closeEvidence();
      else openEvidence(turn, n);
    }
  });

  $("#closeEvidence").addEventListener("click", closeEvidence);
  $("#prevSrc").addEventListener("click", () => stepEvidence(-1));
  $("#nextSrc").addEventListener("click", () => stepEvidence(1));

  $("#newChat").addEventListener("click", () => {
    if (state.busy) state.controller?.abort();
    state.turns = [];
    persist();
    closeEvidence();
    renderThread();
    closeSidebar();
    els.input.focus();
  });

  // library
  $("#uploadBtn").addEventListener("click", () => els.fileInput.click());
  $("#dropzone").addEventListener("click", () => els.fileInput.click());
  $("#emptyUpload").addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => { if (els.fileInput.files.length) upload(els.fileInput.files); els.fileInput.value = ""; });
  els.docList.addEventListener("click", (e) => { const b = e.target.closest("[data-delete]"); if (b) removeDoc(b.dataset.delete); });
  els.docSearch.addEventListener("input", renderDocs);
  let dragDepth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { dragDepth++; els.dropOverlay.hidden = false; } });
  window.addEventListener("dragleave", (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; els.dropOverlay.hidden = true; } });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    els.dropOverlay.hidden = true;
    upload(e.dataTransfer.files);
  });

  // settings
  els.modeGroup.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (!b) return;
    state.settings.mode = b.dataset.mode; applySettings(); saveSettings();
  });
  els.topK.addEventListener("input", () => { state.settings.topK = +els.topK.value; applySettings(); saveSettings(); });
  els.compare.addEventListener("change", () => { state.settings.compare = els.compare.checked; applySettings(); saveSettings(); });
  $("#themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("ragdemia.theme", document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
  });

  // admin sign-in (public demo only)
  $("#adminBtn").addEventListener("click", async () => {
    if (adminToken) {
      adminToken = "";
      try { sessionStorage.removeItem("ragdemia.admin"); } catch (e) { /* ignore */ }
      toast("ok", "Signed out of admin", "", 2500);
      await loadHealth(); loadDocs();
      return;
    }
    $("#adminError").hidden = true;
    $("#adminPassword").value = "";
    $("#adminModal").showModal();
  });
  $("#adminModal").addEventListener("close", async () => {
    if ($("#adminModal").returnValue !== "ok") return;
    const candidate = $("#adminPassword").value;
    const res = await fetch("/health", { headers: { "X-Admin-Token": candidate } }).then((r) => r.json()).catch(() => ({}));
    if (res.is_admin) {
      adminToken = candidate;
      try { sessionStorage.setItem("ragdemia.admin", candidate); } catch (e) { /* ignore */ }
      toast("ok", "Signed in as admin", "Uploads now go to the shared library", 3500);
      await loadHealth(); loadDocs();
    } else {
      $("#adminError").hidden = false;
      $("#adminModal").showModal();
    }
  });

  // mobile sidebar
  $("#openSidebar").addEventListener("click", () => { els.sidebar.classList.add("open"); els.scrim.hidden = false; });
  $("#closeSidebar").addEventListener("click", closeSidebar);
  els.scrim.addEventListener("click", closeSidebar);
}
function closeSidebar() { els.sidebar.classList.remove("open"); els.scrim.hidden = true; }

/* ------------------------------------------------------------------ boot */
function boot() {
  state.settings = { ...state.settings, ...store.get("settings", {}) };
  applySettings();
  renderSuggestions();
  els.docList.innerHTML = '<li class="doc-skeleton skeleton"></li>'.repeat(4);
  state.turns = store.get("conversation", []);
  renderThread();
  scrollToBottom(true);
  wire();
  autosize();
  loadHealth().then(loadDocs);
  els.input.focus();
}
boot();
