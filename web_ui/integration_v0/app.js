"use strict";

const API = "/api/v0";
const CONTEXT_VERSION = "integration_v0.conversation.v1";

/** @typedef {{ turnId: string, question: string, imageFile: File | null, status: "loading" | "success" | "error", payload?: object, error?: string }} Turn */

const state = {
  conversationId: null,
  /** @type {Turn[]} */
  turns: [],
  isBusy: false,
  hasAvailableModel: false,
  lockedSubject: null,
  lockedModel: null,
};

const elements = {
  form: document.getElementById("answer-form"),
  subject: document.getElementById("subject"),
  model: document.getElementById("model"),
  subjectControl: document.getElementById("subject-control"),
  modelControl: document.getElementById("model-control"),
  lockNotice: document.getElementById("lock-notice"),
  modelNote: document.getElementById("model-note"),
  subjectNav: document.getElementById("subject-nav"),
  headerContext: document.getElementById("header-context"),
  image: document.getElementById("image"),
  fileName: document.getElementById("file-name"),
  question: document.getElementById("question"),
  submit: document.getElementById("submit"),
  service: document.getElementById("service-status"),
  empty: document.getElementById("empty-state"),
  messages: document.getElementById("messages"),
  chatScroll: document.getElementById("chat-scroll"),
  newChat: document.getElementById("new-chat"),
  menu: document.getElementById("menu-button"),
  sidebar: document.getElementById("sidebar"),
  turnTemplate: document.getElementById("turn-template"),
};

function setText(element, value) {
  element.textContent = value;
}

function isConversationActive() {
  return state.turns.length > 0;
}

function reviewLabel(subject) {
  return Object.entries(subject.review_status_counts || {})
    .filter(([, count]) => count)
    .map(([name, count]) => `${count} ${name.replaceAll("_", " ")}`)
    .join(" · ");
}

function updateModelNote() {
  const option = elements.model.selectedOptions[0];
  setText(elements.modelNote, option?.dataset.description || "Only allowlisted models are shown.");
}

function updateSubjectState() {
  const option = elements.subject.selectedOptions[0];
  setText(
    elements.headerContext,
    option ? `${option.dataset.name || option.textContent} · ${option.value}` : "Subject-scoped assistant",
  );
  for (const button of elements.subjectNav.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.subject === elements.subject.value);
  }
}

function flashLockNotice() {
  elements.lockNotice.hidden = false;
  elements.lockNotice.classList.add("lock-notice--flash");
  window.setTimeout(() => elements.lockNotice.classList.remove("lock-notice--flash"), 1800);
}

function setConversationLock(locked) {
  elements.subject.disabled = locked || !state.hasAvailableModel;
  elements.model.disabled = locked || !state.hasAvailableModel;
  elements.subjectControl.classList.toggle("compact-control--locked", locked);
  elements.modelControl.classList.toggle("compact-control--locked", locked);
  elements.lockNotice.hidden = !locked;
  for (const button of elements.subjectNav.querySelectorAll("button")) {
    button.classList.toggle("subject-nav__button--locked", locked);
  }
}

function setBusy(busy) {
  state.isBusy = busy;
  elements.submit.disabled = busy || !state.hasAvailableModel;
  elements.submit.textContent = busy ? "…" : "↑";
  elements.question.disabled = busy;
  elements.image.disabled = busy;
}

function scrollToBottom() {
  elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
}

function renderMeta(metaElement, payload) {
  metaElement.replaceChildren();
  const imageStatus = payload.image?.status ? payload.image.status.replaceAll("_", " ") : "none";
  for (const [term, value] of [
    ["Subject", `${payload.subject} · ${payload.module}`],
    ["Model", payload.model || "runtime default"],
    ["Time", `${payload.timing_ms} ms`],
    ["Image", imageStatus],
  ]) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    setText(dt, term);
    setText(dd, value);
    metaElement.append(dt, dd);
  }
  if (payload.conversation?.turn_number != null) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    setText(dt, "Turn");
    setText(dd, String(payload.conversation.turn_number));
    metaElement.append(dt, dd);
  }
  metaElement.hidden = false;
}

function renderDetails(detailsElement, payload) {
  const sourcesList = detailsElement.querySelector(".source-list");
  const sourcesCount = detailsElement.querySelector(".sources-count");
  const traceList = detailsElement.querySelector(".trace-list");
  const warningsList = detailsElement.querySelector(".warning-list");

  sourcesList.replaceChildren();
  traceList.replaceChildren();
  warningsList.replaceChildren();

  const sources = payload.sources || [];
  setText(sourcesCount, `${sources.length} source${sources.length === 1 ? "" : "s"}`);
  for (const source of sources) {
    const item = document.createElement("li");
    setText(item, source.note_path || "Unknown source");
    sourcesList.append(item);
  }
  for (const entry of payload.trace || []) {
    const item = document.createElement("span");
    item.className = "trace-chip";
    setText(item, `${entry.stage}: ${entry.status}`);
    traceList.append(item);
  }
  for (const warning of payload.review?.warnings || []) {
    const item = document.createElement("li");
    setText(item, warning.replaceAll("_", " "));
    warningsList.append(item);
  }
  detailsElement.hidden = false;
}

function renderVision(visionCard, payload) {
  const vision = payload.vision;
  const visionStatus = visionCard.querySelector(".vision-status");
  const visionJson = visionCard.querySelector(".vision-json");
  if (!vision) {
    visionCard.hidden = true;
    return;
  }
  setText(visionStatus, vision.status.replaceAll("_", " "));
  setText(visionJson, JSON.stringify(vision, null, 2));
  visionCard.hidden = false;
}

function renderTurnAssistant(turnElement, turn) {
  const content = turnElement.querySelector(".assistant-content");
  const statusEl = content.querySelector(".response-status");
  const titleEl = content.querySelector(".answer-title");
  const bodyEl = content.querySelector(".answer-body");
  const metaEl = content.querySelector(".answer-meta");
  const detailsEl = content.querySelector(".details-grid");
  const visionCard = content.querySelector(".vision-card");
  const errorEl = content.querySelector(".turn-error");
  const errorMessage = content.querySelector(".error-message");

  metaEl.hidden = true;
  detailsEl.hidden = true;
  visionCard.hidden = true;
  errorEl.hidden = true;

  if (turn.status === "loading") {
    setText(statusEl, "Searching");
    setText(titleEl, "Retrieving subject-grounded context");
    setText(bodyEl, "The selected subject index is being searched.");
    return;
  }

  if (turn.status === "error") {
    setText(statusEl, "Unable to answer");
    setText(titleEl, "The request did not complete");
    setText(bodyEl, "");
    setText(errorMessage, turn.error || "Unexpected request failure.");
    errorEl.hidden = false;
    return;
  }

  const payload = turn.payload;
  if (!payload) return;

  setText(statusEl, "Preview answer");
  setText(titleEl, `${payload.subject} answer`);
  setText(bodyEl, payload.answer || "No answer was returned.");
  renderMeta(metaEl, payload);
  renderDetails(detailsEl, payload);
  renderVision(visionCard, payload);
}

function createTurnElement(turn) {
  const fragment = elements.turnTemplate.content.cloneNode(true);
  const turnElement = fragment.querySelector(".turn");
  turnElement.dataset.turnId = turn.turnId;
  setText(turnElement.querySelector(".user-bubble"), turn.question);

  const retryButton = turnElement.querySelector(".retry-button");
  retryButton.addEventListener("click", () => {
    void retryTurn(turn.turnId);
  });

  renderTurnAssistant(turnElement, turn);
  return turnElement;
}

function findTurnElement(turnId) {
  return elements.messages.querySelector(`.turn[data-turn-id="${turnId}"]`);
}

function findTurn(turnId) {
  return state.turns.find((turn) => turn.turnId === turnId);
}

function showThread() {
  elements.empty.hidden = true;
  elements.messages.hidden = false;
}

function resetComposerFields() {
  elements.question.value = "";
  elements.image.value = "";
  setText(elements.fileName, "No image selected");
}

function buildFormData(turn) {
  const formData = new FormData();
  formData.append("subject", elements.subject.value);
  formData.append("model", elements.model.value);
  formData.append("question", turn.question);
  formData.append("turn_id", turn.turnId);
  formData.append("context_version", CONTEXT_VERSION);
  if (state.conversationId) {
    formData.append("conversation_id", state.conversationId);
  }
  if (turn.imageFile) {
    formData.append("image", turn.imageFile);
  }
  return formData;
}

async function submitTurn(turn, { appendUserBubble }) {
  if (state.isBusy) return;

  turn.status = "loading";
  turn.error = undefined;
  turn.payload = undefined;

  let turnElement = findTurnElement(turn.turnId);
  if (appendUserBubble) {
    showThread();
    lockConversationContext();
    turnElement = createTurnElement(turn);
    elements.messages.append(turnElement);
    scrollToBottom();
  } else if (turnElement) {
    renderTurnAssistant(turnElement, turn);
  }

  setBusy(true);
  try {
    const response = await fetch(`${API}/answer`, { method: "POST", body: buildFormData(turn) });
    const responseText = await response.text();
    let payload;
    try {
      payload = JSON.parse(responseText);
    } catch {
      throw new Error(`The service returned ${response.status}: ${responseText || "empty response"}`);
    }
    if (!response.ok) {
      const detail = payload.detail;
      throw new Error(typeof detail === "string" ? detail : detail?.message || "Request failed.");
    }

    if (payload.conversation?.id && !state.conversationId) {
      state.conversationId = payload.conversation.id;
    }

    turn.status = "success";
    turn.payload = payload;
    turn.error = undefined;
    if (turnElement) renderTurnAssistant(turnElement, turn);
    scrollToBottom();
  } catch (error) {
    turn.status = "error";
    turn.error = error instanceof Error ? error.message : "Unexpected request failure.";
    if (turnElement) renderTurnAssistant(turnElement, turn);
    scrollToBottom();
  } finally {
    setBusy(false);
  }
}

async function retryTurn(turnId) {
  const turn = findTurn(turnId);
  if (!turn || state.isBusy) return;
  await submitTurn(turn, { appendUserBubble: false });
}

function lockConversationContext() {
  if (!state.lockedSubject) {
    state.lockedSubject = elements.subject.value;
    state.lockedModel = elements.model.value;
  }
  setConversationLock(true);
}

function resetChat() {
  state.conversationId = null;
  state.turns = [];
  state.isBusy = false;
  state.lockedSubject = null;
  state.lockedModel = null;
  elements.messages.replaceChildren();
  elements.form.reset();
  elements.empty.hidden = false;
  elements.messages.hidden = true;
  setText(elements.fileName, "No image selected");
  setConversationLock(false);
  setBusy(false);
  elements.question.disabled = false;
  elements.image.disabled = false;
  elements.submit.disabled = !state.hasAvailableModel;
  elements.question.focus();
}

async function loadCatalog() {
  const [healthResponse, subjectResponse, modelResponse] = await Promise.all([
    fetch(`${API}/health`),
    fetch(`${API}/subjects`),
    fetch(`${API}/models`),
  ]);
  if (!healthResponse.ok || !subjectResponse.ok || !modelResponse.ok) {
    throw new Error("The preview service is unavailable.");
  }

  const health = await healthResponse.json();
  const catalog = await subjectResponse.json();
  const modelCatalog = await modelResponse.json();

  elements.subject.replaceChildren();
  elements.subjectNav.replaceChildren();
  for (const subject of catalog.subjects) {
    const option = document.createElement("option");
    option.value = subject.subject;
    option.textContent = `${subject.subject} · ${subject.name}`;
    option.dataset.name = subject.name;
    option.disabled = !subject.preview_ready;
    elements.subject.append(option);

    const button = document.createElement("button");
    button.type = "button";
    button.dataset.subject = subject.subject;
    button.disabled = !subject.preview_ready;
    const codeSpan = document.createElement("span");
    setText(codeSpan, subject.subject);
    const reviewSpan = document.createElement("small");
    setText(reviewSpan, reviewLabel(subject) || "review state unavailable");
    button.append(codeSpan, reviewSpan);
    button.addEventListener("click", () => {
      if (isConversationActive()) {
        flashLockNotice();
        return;
      }
      elements.subject.value = subject.subject;
      updateSubjectState();
    });
    elements.subjectNav.append(button);
  }

  elements.model.replaceChildren();
  for (const model of modelCatalog.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} · ${model.tier}`;
    option.dataset.description = model.description;
    option.disabled = !model.available || model.availability === "unknown";
    option.selected = model.id === modelCatalog.default_model && !option.disabled;
    elements.model.append(option);
  }

  state.hasAvailableModel = modelCatalog.models.some(
    (model) => model.available && model.availability !== "unknown",
  );
  setConversationLock(isConversationActive());
  elements.submit.disabled = !state.hasAvailableModel || state.isBusy;
  updateSubjectState();
  updateModelNote();
  if (!state.hasAvailableModel) {
    setText(elements.modelNote, "The local model service is unavailable or could not be checked.");
  }

  setText(elements.service, `${health.ready_subject_count}/${health.subject_count} indexes ready`);
  document.querySelector(".service-line").classList.add("service-line--ready");
}

elements.image.addEventListener("change", () => {
  const file = elements.image.files?.[0];
  setText(elements.fileName, file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KiB` : "No image selected");
});

elements.subject.addEventListener("change", () => {
  if (isConversationActive()) {
    flashLockNotice();
    if (state.lockedSubject) elements.subject.value = state.lockedSubject;
    return;
  }
  updateSubjectState();
});

elements.model.addEventListener("change", () => {
  if (isConversationActive()) {
    flashLockNotice();
    if (state.lockedModel) elements.model.value = state.lockedModel;
    return;
  }
  updateModelNote();
});

elements.newChat.addEventListener("click", resetChat);
elements.menu.addEventListener("click", () => elements.sidebar.classList.toggle("sidebar--open"));

for (const suggestion of document.querySelectorAll("[data-question]")) {
  suggestion.addEventListener("click", () => {
    elements.question.value = suggestion.dataset.question || "";
    elements.question.focus();
  });
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isBusy) return;

  const question = elements.question.value.trim();
  if (!question) return;

  const imageFile = elements.image.files?.[0] || null;
  const turn = {
    turnId: crypto.randomUUID(),
    question,
    imageFile,
    status: "loading",
  };

  state.turns.push(turn);
  resetComposerFields();
  await submitTurn(turn, { appendUserBubble: true });
});

loadCatalog().catch((error) => {
  setText(elements.service, error instanceof Error ? error.message : "Service unavailable");
  document.querySelector(".service-line").classList.add("service-line--error");
});
