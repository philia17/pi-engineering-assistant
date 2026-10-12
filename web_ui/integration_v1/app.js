"use strict";

const API = "/api/v1";
const CONTEXT_VERSION = "integration_v1.conversation.v1";
const ANSWER_REQUEST_TIMEOUT_MS = 180000;
const ANSWER_REQUEST_TIMEOUT_SECONDS = Math.round(ANSWER_REQUEST_TIMEOUT_MS / 1000);
const CONVERSATION_RESET_TIMEOUT_MS = 12000;
const CONVERSATION_RESET_TIMEOUT_SECONDS = Math.round(CONVERSATION_RESET_TIMEOUT_MS / 1000);
const EXAMPLE_ROTATION_KEY = "integration_v1.example_questions.v1";
const EXAMPLE_QUESTIONS = [
  {
    subject: "DIES",
    label: "RAII and exceptions",
    question: "@DIES How does a scope-bound C++ object release its resources?",
  },
  {
    subject: "EECC",
    label: "Converter turns ratio",
    question: "@EECC What inputs are needed to choose an ideal converter turns ratio?",
  },
  {
    subject: "EM",
    label: "Complex number",
    question: "@EM Convert the complex number 5 + 12j to polar form.",
  },
  {
    subject: "EPC",
    label: "Schmitt-trigger hysteresis",
    question: "@EPC Explain the role of two thresholds in a Schmitt trigger.",
  },
  {
    subject: "IS",
    label: "Avoiding aliasing",
    question: "@IS Explain why sampling frequency matters when recording a waveform.",
  },
  {
    subject: "MMT",
    label: "Variance",
    question: "@MMT How do population variance and sample variance differ?",
  },
  {
    subject: "PE",
    label: "RC time constant",
    question: "@PE Find the time constant for a 3.3 kOhm resistor and a 47 uF capacitor.",
  },
  {
    subject: "SEP",
    label: "Pointers",
    question: "@SEP What is the difference between a pointer and the value it references?",
  },
];

// Client-only opaque ID helper for conversation/turn ids. Every current
// browser exposes crypto.randomUUID; slightly older browsers may not, so we
// fall back to crypto.getRandomValues (universally available) with the same
// UUIDv4 shape. Both paths use cryptographically secure entropy and the ids
// stay opaque and non-persistent. We never fall back to Math.random, which
// would weaken the auth-enabled session binding.
function makeOpaqueId() {
  const cryptoObject = (typeof crypto !== "undefined" && crypto) || null;
  if (cryptoObject && typeof cryptoObject.randomUUID === "function") {
    return cryptoObject.randomUUID();
  }
  if (cryptoObject && typeof cryptoObject.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    cryptoObject.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  throw new Error("Secure random id generation is unavailable in this browser.");
}

/** @typedef {{ turnId: string, question: string, imageFile: File | null, status: "loading" | "success" | "error", createdAt?: number, completedAt?: number, payload?: object, error?: string }} Turn */

const state = {
  conversationId: null,
  /** @type {Turn[]} */
  turns: [],
  isBusy: false,
  resettingConversation: false,
  conversationResetAmbiguous: false,
  hasAvailableModel: false,
  catalogRequestId: 0,
  catalogLoading: true,
  // Local Recent Chats archive bookkeeping. archiveChatId is a client-only
  // opaque id that is distinct from the backend conversation_id and is the
  // only conversation identity persisted by the archive.
  archiveChatId: null,
  archiveCreatedAt: null,
  restoredArchiveId: null,
  renamingArchiveId: null,
  subjectMentionIndex: -1,
  subjectMentionOptions: [],
  subjectMentionActive: false,
  mathSelectionStart: 0,
  mathSelectionEnd: 0,
  quizPayload: null,
  quizBusy: false,
  quizScopesRequestId: 0,
  quizTopics: [],
  activity: "ask",
  quizInitialized: false,
};

let thinkingIntervalId = null;

const elements = {
  form: document.getElementById("answer-form"),
  composerShell: document.getElementById("answer-form"),
  askActivity: document.getElementById("ask"),
  quizActivity: document.getElementById("quiz-activity"),
  askTab: document.getElementById("ask-tab"),
  quizTab: document.getElementById("quiz-tab"),
  subject: document.getElementById("subject"),
  mode: document.getElementById("mode"),
  model: document.getElementById("model"),
  subjectControl: document.getElementById("subject-control"),
  subjectMentionMenu: document.getElementById("subject-mention-menu"),
  formulaButton: document.getElementById("formula-button"),
  mathPreview: document.getElementById("math-preview"),
  mathPreviewBody: document.getElementById("math-preview-body"),
  mathDialog: document.getElementById("math-dialog"),
  mathField: document.getElementById("math-field"),
  mathDisplayMode: document.getElementById("math-display-mode"),
  mathDialogError: document.getElementById("math-dialog-error"),
  mathDialogClose: document.getElementById("math-dialog-close"),
  mathDialogCancel: document.getElementById("math-dialog-cancel"),
  mathDialogInsert: document.getElementById("math-dialog-insert"),
  modelControl: document.getElementById("model-control"),
  modeControl: document.getElementById("mode-control"),
  reviewBadge: document.getElementById("review-badge"),
  lockNotice: document.getElementById("lock-notice"),
  modeNotice: document.getElementById("mode-notice"),
  modelNote: document.getElementById("model-note"),
  headerContext: document.getElementById("header-context"),
  image: document.getElementById("image"),
  fileName: document.getElementById("file-name"),
  composerPreview: document.getElementById("composer-preview"),
  composerPreviewImage: document.getElementById("composer-preview-image"),
  composerPreviewRemove: document.getElementById("composer-preview-remove"),
  gallery: document.getElementById("circuit-gallery"),
  galleryGrid: document.getElementById("circuit-gallery-grid"),
  galleryHint: document.getElementById("circuit-gallery-hint"),
  localNoteGallery: document.getElementById("local-note-gallery"),
  localNoteGalleryGrid: document.getElementById("local-note-gallery-grid"),
  localNoteGalleryHint: document.getElementById("local-note-gallery-hint"),
  question: document.getElementById("question"),
  submit: document.getElementById("submit"),
  service: document.getElementById("service-status"),
  empty: document.getElementById("empty-state"),
  messages: document.getElementById("messages"),
  chatScroll: document.getElementById("chat-scroll"),
  newChat: document.getElementById("new-chat"),
  menu: document.getElementById("menu-button"),
  sidebar: document.getElementById("sidebar"),
  sidebarClose: document.getElementById("sidebar-close"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  sidebarToggleIcon: document.getElementById("sidebar-toggle-icon"),
  sidebarToggleLabel: document.getElementById("sidebar-toggle-label"),
  recentChatsList: document.getElementById("recent-chats-list"),
  recentChatsEmpty: document.getElementById("recent-chats-empty"),
  themeToggle: document.getElementById("theme-toggle"),
  modeBadge: document.getElementById("mode-badge"),
  appShell: document.getElementById("app-shell"),
  authGate: document.getElementById("auth-gate"),
  authForm: document.getElementById("auth-form"),
  accessKey: document.getElementById("access-key"),
  authError: document.getElementById("auth-gate-error"),
  sessionExpiry: document.getElementById("session-expiry"),
  authSubmit: document.getElementById("auth-submit"),
  logoutButton: document.getElementById("logout-button"),
  turnTemplate: document.getElementById("turn-template"),
  quizSetup: document.getElementById("quiz-setup"),
  quizSubject: document.getElementById("quiz-subject"),
  quizTopic: document.getElementById("quiz-topic"),
  quizKnowledgePoint: document.getElementById("quiz-knowledge-point"),
  quizQuestionCount: document.getElementById("quiz-question-count"),
  quizStart: document.getElementById("quiz-start"),
  quizLoading: document.getElementById("quiz-loading"),
  quizLoadingLabel: document.getElementById("quiz-loading-label"),
  quizLoadingDetail: document.getElementById("quiz-loading-detail"),
  quizCard: document.getElementById("quiz-card"),
  quizProgress: document.getElementById("quiz-progress"),
  quizProgressTrack: document.getElementById("quiz-progress-track"),
  quizProgressFill: document.getElementById("quiz-progress-fill"),
  quizPrompt: document.getElementById("quiz-prompt"),
  quizOptions: document.getElementById("quiz-options"),
  quizSubmit: document.getElementById("quiz-submit"),
  quizNext: document.getElementById("quiz-next"),
  quizFinish: document.getElementById("quiz-finish"),
  quizExit: document.getElementById("quiz-exit"),
  quizFeedback: document.getElementById("quiz-feedback"),
  quizSummary: document.getElementById("quiz-summary"),
  quizError: document.getElementById("quiz-error"),
};

function setText(element, value) {
  if (!element) return;
  element.textContent = value;
}

function formatDurationSeconds(milliseconds) {
  if (milliseconds === null || milliseconds === undefined || milliseconds === "") {
    return "n/a";
  }
  const value = Number(milliseconds);
  return Number.isFinite(value) ? `${(value / 1000).toFixed(3)} s` : "n/a";
}

// AUTH_GATE_START
const AUTH_ME_ENDPOINT = "/auth/me";
const AUTH_SESSION_ENDPOINT = "/auth/session";
const AUTH_LOGOUT_ENDPOINT = "/auth/logout";
const AUTH_CONVERSATION_RECOVER_ENDPOINT = "/auth/conversation/recover";
const AUTH_DISABLED_MESSAGE = "Demo auth is not enabled.";
const AUTH_SAME_ORIGIN = { credentials: "same-origin" };

class SessionExpiredError extends Error {
  constructor(message) {
    super(message);
    this.name = "SessionExpiredError";
  }
}

class ConversationMismatchError extends Error {
  constructor(message) {
    super(message);
    this.name = "ConversationMismatchError";
  }
}

const authState = {
  status: "unknown",
  checking: false,
  submitting: false,
  expiresAt: null,
};

function updateSessionExpiry(payload) {
  const expiresAt = Date.parse(payload?.expires_at || "");
  authState.expiresAt = Number.isFinite(expiresAt) ? expiresAt : null;
  if (!elements.sessionExpiry) return;
  elements.sessionExpiry.hidden = authState.expiresAt === null;
  setText(
    elements.sessionExpiry,
    authState.expiresAt === null
      ? ""
      : `Session valid until ${new Date(expiresAt).toLocaleString([], {
          month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        })}. Sign in again when it ends.`,
  );
}

function setAuthGateVisible(gate, visible) {
  if (gate) gate.hidden = !visible;
}

function setAuthError(errorElement, message) {
  if (!errorElement) return;
  if (message) {
    setText(errorElement, message);
    errorElement.hidden = false;
  } else {
    setText(errorElement, "");
    errorElement.hidden = true;
  }
}

function setLogoutVisible(logoutButton, visible) {
  if (logoutButton) logoutButton.hidden = !visible;
}

function setAuthPageStatus(status) {
  if (elements.appShell) elements.appShell.dataset.authStatus = status;
}

function authBlocksComposer() {
  return authState.status === "gated" || authState.status === "config_error";
}

function syncComposerAuthState() {
  const currentState =
    typeof state === "undefined"
      ? {
          isBusy: false,
          resettingConversation: false,
          conversationResetAmbiguous: false,
          hasAvailableModel: true,
        }
      : state;
  const authBlocked = authBlocksComposer();
  const resetBlocked =
    currentState.resettingConversation || currentState.conversationResetAmbiguous;
  const blocked = authBlocked || resetBlocked;
  if (elements.question) elements.question.disabled = blocked || currentState.isBusy;
  if (elements.image) elements.image.disabled = blocked || currentState.isBusy;
  if (elements.formulaButton) elements.formulaButton.disabled = blocked || currentState.isBusy;
  if (elements.submit) {
    elements.submit.disabled = blocked || currentState.isBusy || !currentState.hasAvailableModel;
  }
  elements.composerShell?.classList.toggle("composer-shell--auth-blocked", authBlocked);
  elements.composerShell?.classList.toggle("composer-shell--reset-blocked", resetBlocked);
}

function authDisabledPayload(payload) {
  return Boolean(payload && payload.detail && payload.detail.message === AUTH_DISABLED_MESSAGE);
}

function applyAuthStatus(status) {
  authState.status = status;
  if (status !== "authed") updateSessionExpiry(null);
  if (status === "authed") {
    setAuthPageStatus("ready");
    setAuthGateVisible(elements.authGate, false);
    setLogoutVisible(elements.logoutButton, true);
    syncComposerAuthState();
    return;
  }
  if (status === "gated" || status === "config_error") {
    setAuthPageStatus(status);
    setAuthGateVisible(elements.authGate, true);
    setLogoutVisible(elements.logoutButton, false);
    syncComposerAuthState();
    return;
  }
  // Auth is off or the status check failed: keep the existing preview usable.
  setAuthPageStatus("ready");
  setAuthGateVisible(elements.authGate, false);
  setLogoutVisible(elements.logoutButton, false);
  syncComposerAuthState();
}

async function checkAuthStatus() {
  if (authState.checking) return authState.status;
  authState.checking = true;
  try {
    const response = await fetch(AUTH_ME_ENDPOINT, AUTH_SAME_ORIGIN);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (response.status === 200) {
      setAuthError(elements.authError, "");
      updateSessionExpiry(payload);
      applyAuthStatus("authed");
      return "authed";
    }
    if (response.status === 503) {
      applyAuthStatus("config_error");
      setAuthError(
        elements.authError,
        "The demo access layer is not configured on this server. Contact the operator to enable it.",
      );
      return "config_error";
    }
    if (response.status === 401 && !authDisabledPayload(payload)) {
      applyAuthStatus("gated");
      setAuthError(elements.authError, "");
      return "gated";
    }
    // 401 with the disabled message, or any other reachable/non-200 response:
    // the opt-in auth layer is off, so the preview stays open.
    applyAuthStatus("off");
    setAuthError(elements.authError, "");
    return "off";
  } catch {
    applyAuthStatus("off");
    setAuthError(elements.authError, "");
    return "off";
  } finally {
    authState.checking = false;
  }
}

async function submitAccessKey(event) {
  if (event) event.preventDefault();
  if (authState.submitting) return;
  const accessKey = elements.accessKey ? elements.accessKey.value : "";
  if (elements.accessKey) elements.accessKey.value = "";
  if (!accessKey) {
    setAuthError(elements.authError, "Enter the invite/access key for this preview.");
    if (elements.accessKey) elements.accessKey.focus();
    return;
  }
  authState.submitting = true;
  if (elements.authSubmit) elements.authSubmit.disabled = true;
  setAuthError(elements.authError, "");
  try {
    const formData = new FormData();
    formData.append("access_key", accessKey);
    const response = await fetch(AUTH_SESSION_ENDPOINT, {
      method: "POST",
      body: formData,
      credentials: "same-origin",
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (response.status === 200) {
      setAuthError(elements.authError, "");
      showQuizError("");
      updateSessionExpiry(payload);
      applyAuthStatus("authed");
      void loadCircuitGallery();
      void loadLocalNoteGallery();
      void ensureQuizInitialized();
      if (elements.question && state.activity === "ask") elements.question.focus();
      return;
    }
    if (response.status === 503) {
      applyAuthStatus("config_error");
      setAuthError(
        elements.authError,
        "The demo access layer is not configured on this server. Contact the operator to enable it.",
      );
      return;
    }
    if (
      response.status === 404 ||
      (payload && payload.detail && payload.detail.code === "auth_disabled")
    ) {
      // Auth was turned off after the page loaded; the preview stays open.
      applyAuthStatus("off");
      setAuthError(elements.authError, "");
      return;
    }
    if (response.status === 401) {
      applyAuthStatus("gated");
      setAuthError(
        elements.authError,
        "That access key was not accepted. Check the invite key and try again.",
      );
      return;
    }
    if (
      response.status === 429 ||
      (payload && payload.detail && payload.detail.code === "too_many_attempts")
    ) {
      applyAuthStatus("gated");
      setAuthError(
        elements.authError,
        "Too many sign-in attempts. Please wait a minute and try again.",
      );
      return;
    }
    applyAuthStatus("gated");
    setAuthError(elements.authError, "The sign-in request failed. Please try again.");
  } catch {
    applyAuthStatus("gated");
    setAuthError(
      elements.authError,
      "Could not reach the sign-in endpoint. Check the connection and try again.",
    );
  } finally {
    authState.submitting = false;
    if (elements.authSubmit) elements.authSubmit.disabled = false;
    if (elements.accessKey) elements.accessKey.focus();
  }
}

async function logoutSession() {
  const resetPending =
    typeof state !== "undefined" && state.resettingConversation;
  const quizPending =
    typeof state !== "undefined" && state.quizBusy;
  if (
    authState.status !== "authed" ||
    authState.submitting ||
    resetPending ||
    quizPending
  ) {
    return;
  }
  authState.submitting = true;
  if (elements.logoutButton) elements.logoutButton.disabled = true;
  try {
    const response = await fetch(AUTH_LOGOUT_ENDPOINT, {
      method: "POST",
      credentials: "same-origin",
    });
    const success = response.ok || (response.status >= 200 && response.status < 300);
    if (!success && response.status !== 401) {
      throw new Error("Logout could not be confirmed. Please try again.");
    }
  } catch (error) {
    setAuthError(
      elements.authError,
      error instanceof Error ? error.message : "Logout could not be confirmed. Please try again.",
    );
    return;
  } finally {
    authState.submitting = false;
    if (typeof syncConversationResetUi === "function") {
      syncConversationResetUi();
    } else if (elements.logoutButton) {
      elements.logoutButton.disabled = false;
    }
  }
  resetChatLocal();
  applyAuthStatus("gated");
  setAuthError(elements.authError, "You have been logged out. Enter the access key again to continue.");
  if (elements.accessKey) elements.accessKey.focus();
}

function handleExpiredSession(message) {
  if (authState.status === "gated" || authState.status === "config_error") return;
  const quizIncomplete =
    typeof state !== "undefined" && state.quizPayload?.status === "in_progress";
  resetChatLocal();
  applyAuthStatus("gated");
  setAuthError(
    elements.authError,
    (message || "Your session is no longer valid. Sign in again to continue.") +
      (quizIncomplete ? " Your Quiz was interrupted and was not completed." : ""),
  );
  if (elements.accessKey) elements.accessKey.focus();
}
// AUTH_GATE_END

// MATH_RENDERER_START
const KATEX_RENDER_OPTIONS = Object.freeze({
  throwOnError: false,
  strict: "error",
  trust: false,
  maxSize: 50,
  maxExpand: 1000,
  output: "htmlAndMathml",
});

function isEscapedCharacter(source, index) {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && source[cursor] === "\\"; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function findNextMathDelimiter(source, start) {
  const delimiters = [
    { delimiter: "$$", closingDelimiter: "$$", display: true },
    { delimiter: "\\[", closingDelimiter: "\\]", display: true },
    { delimiter: "\\(", closingDelimiter: "\\)", display: false },
    { delimiter: "$", closingDelimiter: "$", display: false },
  ];
  for (let index = start; index < source.length; index += 1) {
    if (isEscapedCharacter(source, index)) continue;
    for (const candidate of delimiters) {
      if (!source.startsWith(candidate.delimiter, index)) continue;
      if (candidate.delimiter === "$" && source[index + 1] === "$") continue;
      if (
        candidate.delimiter === "$" &&
        /[0-9]/.test(source[index + 1] || "") &&
        !isDigitStartedDollarMath(source, index)
      ) {
        continue;
      }
      return { index, ...candidate };
    }
  }
  return null;
}

function isDigitStartedDollarMath(source, index) {
  const closer = findClosingMathDelimiter(source, "$", index + 1);
  if (closer < 0) return false;
  const fragment = source.slice(index + 1, closer);
  return fragment.length > 0 && !/\s/.test(fragment);
}

function findClosingMathDelimiter(source, delimiter, start) {
  for (let index = start; index <= source.length - delimiter.length; index += 1) {
    if (!source.startsWith(delimiter, index) || isEscapedCharacter(source, index)) continue;
    if (delimiter === "$" && (source[index - 1] === "$" || source[index + 1] === "$")) continue;
    return index;
  }
  return -1;
}

function tokenizeAnswerMath(answer) {
  const source = String(answer);
  const tokens = [];
  let cursor = 0;

  while (cursor < source.length) {
    const opener = findNextMathDelimiter(source, cursor);
    if (!opener) {
      if (cursor < source.length) tokens.push({ type: "text", value: source.slice(cursor) });
      break;
    }
    const mathStart = opener.index + opener.delimiter.length;
    const closer = findClosingMathDelimiter(source, opener.closingDelimiter, mathStart);
    if (closer < 0) {
      tokens.push({ type: "text", value: source.slice(cursor) });
      break;
    }
    if (opener.index > cursor) tokens.push({ type: "text", value: source.slice(cursor, opener.index) });
    const raw = source.slice(opener.index, closer + opener.closingDelimiter.length);
    const latex = source.slice(mathStart, closer);
    if (latex) {
      tokens.push({ type: "math", latex, raw, display: opener.display });
    } else {
      tokens.push({ type: "text", value: raw });
    }
    cursor = closer + opener.closingDelimiter.length;
  }
  return tokens;
}

function normalizeMarkdownText(text) {
  let value = String(text);
  value = value.replace(/^\s*#+\s+/gm, "");
  value = value.replace(/^\s*>\s?/gm, "");
  value = value.replace(/^\s*[-+*]\s+/gm, "• ");
  value = value.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1");
  value = value.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
  value = value.replaceAll(String.fromCharCode(96), "");
  value = value.replace(/~~([^~\n]+)~~/g, "$1");
  const openingBoundary = "(^|[\\s([{>])";
  const closingBoundary = "(?=$|[\\s.,!?;:)\\]}>])";
  const boundedEmphasis = (marker, contentPattern) =>
    new RegExp(`${openingBoundary}${marker}(${contentPattern})${marker}${closingBoundary}`, "gm");
  value = value.replace(boundedEmphasis("\\*\\*", "[^\\s*\\n](?:[^*\\n]*?[^\\s*\\n])?"), "$1$2");
  value = value.replace(boundedEmphasis("\\*", "[^\\s*\\n](?:[^*\\n]*?[^\\s*\\n])?"), "$1$2");
  value = value.replace(boundedEmphasis("__", "[^\\s_\\n](?:[^_\\n]*?[^\\s_\\n])?"), "$1$2");
  value = value.replace(boundedEmphasis("_", "[^\\s_\\n](?:[^_\\n]*?[^\\s_\\n])?"), "$1$2");
  return value;
}

function mathRenderOptions(displayMode, throwOnError) {
  return { ...KATEX_RENDER_OPTIONS, displayMode, throwOnError };
}

function createMathElement(token) {
  const renderer = globalThis.katex;
  if (!renderer || typeof renderer.render !== "function" || typeof renderer.renderToString !== "function") {
    return document.createTextNode(token.raw);
  }
  try {
    renderer.renderToString(token.latex, mathRenderOptions(token.display, true));
    const mathElement = document.createElement(token.display ? "div" : "span");
    mathElement.className = `math-fragment${token.display ? " math-fragment--display" : ""}`;
    mathElement.dataset.latex = token.raw;
    mathElement.setAttribute("role", "math");
    mathElement.setAttribute("aria-label", token.raw);
    renderer.render(token.latex, mathElement, mathRenderOptions(token.display, false));
    return mathElement;
  } catch {
    return document.createTextNode(token.raw);
  }
}

function renderAnswerBody(bodyElement, answer) {
  bodyElement.replaceChildren();
  for (const token of tokenizeAnswerMath(answer)) {
    bodyElement.append(
      token.type === "text"
        ? document.createTextNode(normalizeMarkdownText(token.value))
        : createMathElement(token),
    );
  }
}
// MATH_RENDERER_END

function configureMathField() {
  const Mathfield = globalThis.MathfieldElement;
  if (!Mathfield || !elements.mathField) return;
  Mathfield.fontsDirectory = "/static/integration-v1/vendor/mathlive-0.110.0/fonts";
  Mathfield.soundsDirectory = null;
  Mathfield.keypressVibration = false;
  elements.mathField.menuItems = elements.mathField.menuItems.filter(
    (item) => !["variant", "color", "background-color"].includes(item.id),
  );
  elements.mathField.macros = {
    ...elements.mathField.macros,
    htmlData: { args: 2, def: "#2" },
    href: { args: 2, def: "#2" },
    class: { args: 2, def: "#2" },
    cssId: { args: 2, def: "#2" },
    htmlId: { args: 2, def: "#2" },
    htmlStyle: { args: 2, def: "#2" },
  };
}

function mathSourceForSelection(node) {
  const element = node instanceof Element ? node : node?.parentElement;
  return element?.closest(".math-fragment[data-latex]") || null;
}

document.addEventListener("copy", (event) => {
  const selection = document.getSelection();
  if (!selection || selection.isCollapsed || !event.clipboardData) return;
  const startMath = mathSourceForSelection(selection.anchorNode);
  const endMath = mathSourceForSelection(selection.focusNode);
  if (!startMath || startMath !== endMath || !startMath.dataset.latex) return;
  event.clipboardData.setData("text/plain", startMath.dataset.latex);
  event.preventDefault();
});

function isConversationActive() {
  return state.turns.length > 0;
}

function resetConversationState() {
  state.conversationId = null;
  state.turns = [];
  state.isBusy = false;
  state.conversationResetAmbiguous = false;
  state.archiveChatId = null;
  state.archiveCreatedAt = null;
  state.restoredArchiveId = null;
  state.renamingArchiveId = null;
  state.subjectMentionIndex = -1;
  state.subjectMentionOptions = [];
  state.quizPayload = null;
  state.quizBusy = false;
  state.quizScopesRequestId += 1;
  state.quizTopics = [];
}

function updateModelNote() {
  const option = elements.model.selectedOptions[0];
  setText(elements.modelNote, option?.dataset.description || "Only allowlisted models are shown.");
}

function updateSubjectState() {
  // Subject is selected per question through the @ mention flow. Keep the
  // header neutral because one conversation may contain multiple subjects.
  setText(elements.headerContext, "");
}

function setConversationLock(locked) {
  // Subject and model are request-scoped controls and remain editable after a turn.
  const quizLocked = state.quizBusy || state.quizPayload?.status === "in_progress";
  elements.subject.disabled = state.catalogLoading || !state.hasAvailableModel;
  elements.mode.disabled = locked || state.catalogLoading;
  elements.model.disabled = state.catalogLoading || !state.hasAvailableModel;
  elements.quizSubject.disabled = state.catalogLoading || quizLocked;
  elements.quizTopic.disabled = quizLocked || state.quizTopics.length === 0;
  elements.quizKnowledgePoint.disabled = quizLocked || !elements.quizTopic.value;
  elements.quizQuestionCount.disabled = quizLocked;
  elements.subjectControl.classList.remove("compact-control--locked");
  elements.modeControl.classList.remove("compact-control--locked");
  elements.modelControl.classList.remove("compact-control--locked");
  elements.lockNotice.hidden = !locked && !quizLocked;
  if (quizLocked) {
    setText(
      elements.lockNotice,
      "Quiz subject, topic, and question count stay fixed; Ask remains available.",
    );
  } else if (locked) {
    setText(elements.lockNotice, "Subject and model can change between questions.");
  }
}

function setUnavailableModelOption() {
  elements.model.replaceChildren();
  const option = document.createElement("option");
  option.value = "";
  option.textContent = "No runtime model available";
  option.selected = true;
  elements.model.append(option);
}

function setBusy(busy) {
  state.isBusy = busy;
  if (elements.submit) elements.submit.textContent = busy ? "…" : "↑";
  syncComposerAuthState();
}

function scrollToBottom() {
  elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
}

function sourceNoteNames(sources) {
  const names = [];
  const seen = new Set();
  for (const source of Array.isArray(sources) ? sources : []) {
    const path = String(source?.note_path || "Unknown source").trim();
    const name = path.split(/[\\/]/).pop() || "Unknown source";
    if (seen.has(name)) continue;
    seen.add(name);
    names.push(name);
  }
  return names;
}

function appendMetaRow(metaElement, term, value) {
  if (!value) return false;
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  setText(dt, term);
  setText(dd, value);
  metaElement.append(dt, dd);
  return true;
}

// TOOLCHAIN_START
function isDeterministicAnswer(payload) {
  if (payload.route === "rag_fallback") return false;
  if (payload.answer_engine?.kind === "deterministic") return true;
  return sanitizeArchiveTrace(payload.trace).some(
    (entry) =>
      typeof entry === "object" &&
      entry.name === "deterministic_calculator" &&
      entry.output_summary &&
      entry.output_summary !== "error",
  );
}

function renderToolchain(card, payload) {
  if (!card) return;
  card.replaceChildren();
  const route = truncateString(payload.route, RECENT_CHATS_MAX_FIELD);
  const trace = sanitizeArchiveTrace(payload.trace);
  card.hidden = !route && trace.length === 0;
  if (card.hidden) return;

  const summary = document.createElement("summary");
  const title = document.createElement("span");
  title.textContent = "Toolchain";
  summary.append(title);
  if (route) {
    const routeLabel = document.createElement("span");
    routeLabel.className = "toolchain-route";
    routeLabel.textContent = route;
    summary.append(routeLabel);
  }
  card.append(summary);

  if (isDeterministicAnswer(payload)) {
    const engine = document.createElement("p");
    engine.className = "toolchain-engine";
    engine.textContent = "Answer computed by a deterministic tool; selected model shown for context.";
    card.append(engine);
  }
  if (trace.length === 0) {
    const note = document.createElement("p");
    note.className = "toolchain-empty";
    note.textContent = "No step trace was returned for this answer.";
    card.append(note);
    return;
  }

  const steps = document.createElement("ol");
  steps.className = "toolchain-steps";
  for (const entry of trace) {
    const item = document.createElement("li");
    if (typeof entry === "string") {
      item.textContent = entry;
      steps.append(item);
      continue;
    }
    const heading = document.createElement("div");
    heading.className = "toolchain-step-heading";
    const name = document.createElement("strong");
    const label = entry.name || entry.stage || entry.tool;
    name.textContent = [label, entry.tool && entry.tool !== label ? entry.tool : ""]
      .filter(Boolean).join(" · ");
    heading.append(name);
    const detail = [entry.status];
    if (entry.duration_ms !== undefined) detail.push(`${entry.duration_ms} ms`);
    if (detail.filter(Boolean).length) {
      const stateLabel = document.createElement("span");
      stateLabel.textContent = detail.filter(Boolean).join(" · ");
      heading.append(stateLabel);
    }
    item.append(heading);
    if (entry.input_summary || entry.output_summary) {
      const values = document.createElement("dl");
      values.className = "toolchain-step-values";
      if (entry.output_summary) appendMetaRow(values, "Output", entry.output_summary);
      else appendMetaRow(values, "Input", entry.input_summary);
      item.append(values);
    }
    steps.append(item);
  }
  card.append(steps);
}
// TOOLCHAIN_END

function renderCalculatorSteps(card, payload) {
  if (!card) return;
  card.replaceChildren();
  const steps = payload.status === "ok" ? sanitizeCalculatorSteps(payload.calculator_steps) : [];
  card.hidden = steps.length === 0;
  if (card.hidden) return;
  const summary = document.createElement("summary");
  summary.textContent = "Worked solution";
  card.append(summary);
  const list = document.createElement("ol");
  list.className = "worked-solution-steps";
  for (const step of steps) {
    const item = document.createElement("li");
    renderAnswerBody(item, step);
    list.append(item);
  }
  card.append(list);
}

function renderMeta(metaElement, payload) {
  metaElement.replaceChildren();
  let hasRows = false;
  const imageStatus = payload.image?.status ? humanizeToken(payload.image.status) : "";
  const sources = sourceNoteNames(payload.sources);
  const subject = [payload.subject, payload.module].filter(Boolean).join(" · ");
  const timing =
    payload.timing_ms !== null &&
    payload.timing_ms !== undefined &&
    payload.timing_ms !== "" &&
    Number.isFinite(Number(payload.timing_ms))
      ? formatDurationSeconds(payload.timing_ms)
      : "";
  for (const [term, value] of [
    ["Subject", subject],
    [payload.status === "needs_input" || isDeterministicAnswer(payload) ? "Selected model" : "Model", payload.model_requested || payload.model],
    ["Time", timing],
    ["Sources", sources.join(" · ")],
  ]) {
    hasRows = appendMetaRow(metaElement, term, value) || hasRows;
  }
  if (imageStatus && !["none", "not provided", "not requested"].includes(imageStatus)) {
    hasRows = appendMetaRow(metaElement, "Image", imageStatus) || hasRows;
  }
  if (payload.conversation?.turn_number != null) {
    hasRows =
      appendMetaRow(metaElement, "Turn", String(payload.conversation.turn_number)) || hasRows;
  }
  metaElement.hidden = !hasRows;
}

function visionField(value) {
  return value === null || value === undefined || value === "" ? "n/a" : String(value);
}

function renderVisionSummary(summaryElement, vision) {
  summaryElement.replaceChildren();
  const provider =
    vision.provider?.served_model_provider || vision.provider?.kind || "n/a";
  const latency = formatDurationSeconds(vision.vision_latency_ms);
  const status = String(vision.status ?? "unknown").replaceAll("_", " ");
  for (const [term, value] of [
    ["Provider", provider],
    ["Model", visionField(vision.provider?.model)],
    ["Revision", visionField(vision.provider?.revision)],
    ["Status", status],
    ["Vision latency", latency],
    ["Retrieval augmented", vision.retrieval_augmented ? "yes" : "no"],
  ]) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    setText(dt, term);
    setText(dd, value);
    summaryElement.append(dt, dd);
  }
}

// UI_POLISH_START
const THEME_STORAGE_KEY = "integration_v1.theme";

function storedTheme() {
  try {
    const value = window.localStorage?.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

function systemTheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  const resolved = theme === "light" || theme === "dark" ? theme : systemTheme();
  document.documentElement.dataset.theme = resolved;
  elements.themeToggle?.setAttribute("aria-pressed", String(resolved === "dark"));
  elements.themeToggle?.setAttribute(
    "aria-label",
    resolved === "dark" ? "Switch to light theme" : "Switch to dark theme",
  );
}

function initTheme() {
  applyTheme(storedTheme() || null);
  elements.themeToggle?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try {
      window.localStorage?.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Theme still applies for this session when storage is unavailable.
    }
    applyTheme(next);
  });
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if (!storedTheme()) applyTheme(null);
  });
}

function humanizeToken(value) {
  return String(value ?? "").replaceAll("_", " ").trim() || "unknown";
}

function vlmText(value) {
  return value === null || value === undefined || value === "" ? "n/a" : String(value);
}

function renderVlmStrip(stripElement, vision) {
  stripElement.replaceChildren();
  if (!vision || vision.status === "not_requested") {
    stripElement.hidden = true;
    return;
  }
  const provider = vision.provider?.served_model_provider || vision.provider?.kind || "n/a";
  const model = vlmText(vision.provider?.model);
  let revision = vlmText(vision.provider?.revision);
  if (revision.length > 12) revision = `${revision.slice(0, 10)}…`;
  const status = humanizeToken(vision.status);
  const latency = formatDurationSeconds(vision.vision_latency_ms);
  const augmented = vision.retrieval_augmented ? "yes" : "no";
  const label = document.createElement("span");
  label.className = "vlm-strip__label";
  setText(label, "VLM");
  stripElement.append(label);
  for (const [term, value] of [
    ["provider", provider],
    ["model", model],
    ["revision", revision],
    ["status", status],
    ["latency", latency],
    ["retrieval-augmented", augmented],
  ]) {
    const item = document.createElement("span");
    item.className = "vlm-strip__item";
    const termElement = document.createElement("span");
    termElement.className = "vlm-strip__term";
    setText(termElement, term);
    const valueElement = document.createElement("span");
    valueElement.className = "vlm-strip__value";
    setText(valueElement, value);
    item.append(termElement, valueElement);
    stripElement.append(item);
  }
  stripElement.classList.toggle(
    "vlm-strip--error",
    vision.status === "vision_unavailable" || vision.status === "error",
  );
  stripElement.hidden = false;
}

function buildSkeletonLines() {
  const lines = [];
  for (const width of [100, 96, 88, 62]) {
    const line = document.createElement("div");
    line.className = "skeleton-line";
    line.style.width = `${width}%`;
    lines.push(line);
  }
  return lines;
}

function stopThinkingClock() {
  if (thinkingIntervalId !== null) {
    window.clearInterval(thinkingIntervalId);
    thinkingIntervalId = null;
  }
}

function startThinkingClock(turn, turnElement) {
  stopThinkingClock();
  const statusElement = turnElement?.querySelector(".response-status");
  if (!statusElement) return;
  const startedAt = turn.thinkingStartedAt || Date.now();
  const update = () => {
    const elapsedSeconds = Math.floor(Math.max(0, Date.now() - startedAt) / 1000);
    setText(statusElement, `Thinking · ${elapsedSeconds}s`);
  };
  update();
  thinkingIntervalId = window.setInterval(update, 250);
}

function renderAbstainNote(noteElement, payload) {
  if (payload.status === "needs_input") {
    setText(noteElement, "Please provide the missing information to continue.");
    noteElement.hidden = false;
    return;
  }
  const reason =
    payload.status === "needs_subject"
      ? "subject needed"
      : humanizeToken(payload.abstain_reason || "no grounded answer");
  setText(noteElement, `The assistant abstained: ${reason}.`);
  noteElement.hidden = false;
}
// UI_POLISH_END

// RECENT_CHATS_ARCHIVE_START
// Client-only Recent Chats archive. This is a bounded, versioned transcript
// of completed local UI turns. It never persists the backend conversation_id,
// auth material, cookies, tokens, or raw File/Blob/image data. The stored
// schema never carries the backend conversation binding, and it never
// claims to restore a server conversation: opening an entry renders the saved
// transcript locally, and any continuation starts a fresh server conversation.
// Rows expose local pin/unpin and delete actions that persist immediately.
// Unpinned chats older than 48 hours are dropped automatically on read;
// pinning is an explicit keep choice and exempts a chat from that TTL.
const RECENT_CHATS_STORAGE_KEY = "integration_v1.recent_chats.v1";
const RECENT_CHATS_VERSION = 1;
const RECENT_CHATS_MAX_CHATS = 20;
const RECENT_CHATS_MAX_TURNS = 20;
const RECENT_CHATS_MAX_TITLE = 90;
const RECENT_CHATS_MAX_QUESTION = 600;
const RECENT_CHATS_MAX_ANSWER = 5000;
const RECENT_CHATS_MAX_ERROR = 600;
const RECENT_CHATS_MAX_FIELD = 120;
const RECENT_CHATS_MAX_SOURCES = 6;
const RECENT_CHATS_MAX_TRACE = 10;
const RECENT_CHATS_MAX_TRACE_SUMMARY = 360;
const RECENT_CHATS_MAX_CALCULATOR_STEPS = 12;
const RECENT_CHATS_MAX_CALCULATOR_STEP = 1200;
const RECENT_CHATS_MAX_WARNINGS = 6;
const RECENT_CHATS_TOTAL_BUDGET = 60000;
const RECENT_CHATS_TTL_MS = 48 * 60 * 60 * 1000;

function truncateString(value, maxLength) {
  const text = String(value ?? "").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 1))}…`;
}

function sanitizeArchiveSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.slice(0, RECENT_CHATS_MAX_SOURCES).map((source) => ({
    note_path: truncateString(source?.note_path, RECENT_CHATS_MAX_FIELD),
    heading: truncateString(source?.heading, RECENT_CHATS_MAX_FIELD),
  }));
}

function sanitizeArchiveTrace(trace) {
  if (!Array.isArray(trace)) return [];
  return trace
    .slice(0, RECENT_CHATS_MAX_TRACE)
    .map((entry) => {
      if (typeof entry === "string") return truncateString(entry, RECENT_CHATS_MAX_TRACE_SUMMARY);
      if (!entry || typeof entry !== "object") return null;
      const safe = {};
      for (const field of ["stage", "status", "tool", "name"]) {
        if (typeof entry[field] === "string" && entry[field].trim()) {
          safe[field] = truncateString(entry[field], RECENT_CHATS_MAX_FIELD);
        }
      }
      for (const field of ["input_summary", "output_summary"]) {
        if (typeof entry[field] === "string" && entry[field].trim()) {
          safe[field] = truncateString(entry[field], RECENT_CHATS_MAX_TRACE_SUMMARY);
        }
      }
      if (typeof entry.duration_ms === "number" && Number.isFinite(entry.duration_ms) && entry.duration_ms >= 0) {
        safe.duration_ms = entry.duration_ms;
      }
      return Object.keys(safe).length ? safe : null;
    }).filter(Boolean);
}

function sanitizeCalculatorSteps(steps) {
  if (!Array.isArray(steps)) return [];
  return steps
    .filter((step) => typeof step === "string" && step.trim())
    .slice(0, RECENT_CHATS_MAX_CALCULATOR_STEPS)
    .map((step) => truncateString(step, RECENT_CHATS_MAX_CALCULATOR_STEP));
}

function sanitizeArchiveWarnings(warnings) {
  if (!Array.isArray(warnings)) return [];
  return warnings.slice(0, RECENT_CHATS_MAX_WARNINGS).map((warning) => truncateString(warning, RECENT_CHATS_MAX_FIELD));
}

function sanitizeArchiveVision(vision) {
  if (!vision || vision.status === "not_requested") return null;
  return {
    status: truncateString(vision.status, 40),
    retrieval_augmented: Boolean(vision.retrieval_augmented),
  };
}

function sanitizeArchiveTurn(turn) {
  if (!turn || turn.status === "loading") return null;
  const safe = {
    turnId: truncateString(turn.turnId, 64),
    question: truncateString(turn.question, RECENT_CHATS_MAX_QUESTION),
    status: turn.status === "success" ? "success" : "error",
    hadImage: Boolean(turn.imageFile),
    subject: truncateString(turn.subject, 80),
    subjectLabel: truncateString(turn.subjectLabel, 100),
    mode: truncateString(turn.mode, 40),
    model: truncateString(turn.model, 80),
    modelLabel: truncateString(turn.modelLabel, 100),
    createdAt: Number.isFinite(Number(turn.createdAt)) ? Number(turn.createdAt) : null,
    completedAt: Number.isFinite(Number(turn.completedAt)) ? Number(turn.completedAt) : null,
  };
  if (turn.status === "error") {
    safe.error = truncateString(turn.error, RECENT_CHATS_MAX_ERROR);
    return safe;
  }
  const payload = turn.payload;
  if (payload && typeof payload === "object") {
    safe.payloadStatus = truncateString(payload.status, 40);
    safe.badge = truncateString(payload.review?.badge || payload.mode || "unified", 40);
    safe.subject = truncateString(turn.subject || payload.subject, 80);
    safe.module = truncateString(payload.module, 80);
    safe.mode = truncateString(turn.mode || payload.mode, 40);
    safe.model = truncateString(turn.model || payload.model, 80);
    safe.answer = truncateString(payload.answer, RECENT_CHATS_MAX_ANSWER);
    safe.timingMs =
      typeof payload.timing_ms === "number" && Number.isFinite(payload.timing_ms)
        ? payload.timing_ms
        : null;
    safe.sources = sanitizeArchiveSources(payload.sources);
    safe.route = truncateString(payload.route, RECENT_CHATS_MAX_FIELD);
    safe.modelRequested = truncateString(payload.model_requested, 80);
    safe.answerEngine = payload.answer_engine?.kind === "deterministic" ? { kind: "deterministic" } : null;
    safe.trace = sanitizeArchiveTrace(payload.trace);
    safe.calculatorSteps = payload.status === "ok" ? sanitizeCalculatorSteps(payload.calculator_steps) : [];
    safe.warnings = sanitizeArchiveWarnings(payload.review?.warnings);
    safe.vision = sanitizeArchiveVision(payload.vision);
  }
  return safe;
}

function archiveTitleFromTurns(turns) {
  for (const turn of turns) {
    if (turn && turn.question) return truncateString(turn.question, RECENT_CHATS_MAX_TITLE);
  }
  return "Untitled chat";
}

function archiveSelectionLabel(select) {
  return select?.selectedOptions?.[0]?.textContent?.trim() || "";
}

function ensureArchiveChatId() {
  if (!state.archiveChatId) {
    state.archiveChatId = makeOpaqueId();
    state.archiveCreatedAt = Date.now();
  }
  return state.archiveChatId;
}

function enforceArchiveChatBudget(chat) {
  while (chat.turns.length > 1 && JSON.stringify(chat).length > RECENT_CHATS_TOTAL_BUDGET) {
    chat.turns.shift();
  }
  if (chat.turns.length > 0 && chat.customTitle !== true) {
    chat.title = archiveTitleFromTurns(chat.turns);
  }
  return chat;
}

function buildArchiveChat() {
  const completedTurns = state.turns.filter(
    (turn) => turn.status === "success" || turn.status === "error",
  );
  if (completedTurns.length === 0) return null;
  const now = Date.now();
  const chat = {
    id: ensureArchiveChatId(),
    title: archiveTitleFromTurns(completedTurns),
    customTitle: false,
    pinned: false,
    createdAt: state.archiveCreatedAt || now,
    updatedAt: now,
    subject: truncateString(elements.subject?.value || "", 40),
    subjectLabel: truncateString(archiveSelectionLabel(elements.subject), 80),
    mode: truncateString(elements.mode?.value || "preview", 40),
    modeLabel: truncateString(archiveSelectionLabel(elements.mode), 80),
    model: truncateString(elements.model?.value || "", 80),
    modelLabel: truncateString(archiveSelectionLabel(elements.model), 80),
    turns: completedTurns.slice(0, RECENT_CHATS_MAX_TURNS).map(sanitizeArchiveTurn).filter(Boolean),
  };
  return enforceArchiveChatBudget(chat);
}

function normalizeArchivedTurn(turn) {
  if (!turn || typeof turn !== "object") return null;
  const status = turn.status === "error" ? "error" : "success";
  const safe = {
    turnId: truncateString(turn.turnId, 64),
    question: truncateString(turn.question, RECENT_CHATS_MAX_QUESTION),
    status,
    hadImage: Boolean(turn.hadImage),
    createdAt: Number.isFinite(Number(turn.createdAt)) ? Number(turn.createdAt) : null,
    completedAt: Number.isFinite(Number(turn.completedAt)) ? Number(turn.completedAt) : null,
  };
  if (status === "error") {
    safe.error = truncateString(turn.error, RECENT_CHATS_MAX_ERROR);
    return safe;
  }
  safe.payloadStatus = truncateString(turn.payloadStatus, 40);
  safe.badge = truncateString(turn.badge, 40);
  safe.subject = truncateString(turn.subject, 80);
  safe.subjectLabel = truncateString(turn.subjectLabel, 100);
  safe.module = truncateString(turn.module, 80);
  safe.mode = truncateString(turn.mode, 40);
  safe.model = truncateString(turn.model, 80);
  safe.modelLabel = truncateString(turn.modelLabel, 100);
  safe.answer = truncateString(turn.answer, RECENT_CHATS_MAX_ANSWER);
  safe.timingMs = Number.isFinite(Number(turn.timingMs)) ? Number(turn.timingMs) : null;
  safe.sources = sanitizeArchiveSources(turn.sources);
  safe.route = truncateString(turn.route, RECENT_CHATS_MAX_FIELD);
  safe.modelRequested = truncateString(turn.modelRequested, 80);
  safe.answerEngine = turn.answerEngine?.kind === "deterministic" ? { kind: "deterministic" } : null;
  safe.trace = sanitizeArchiveTrace(turn.trace);
  safe.calculatorSteps = safe.payloadStatus === "ok" ? sanitizeCalculatorSteps(turn.calculatorSteps) : [];
  safe.warnings = sanitizeArchiveWarnings(turn.warnings);
  safe.vision = sanitizeArchiveVision(turn.vision);
  return safe;
}

function normalizeArchiveChat(chat) {
  if (!chat || typeof chat !== "object") return null;
  const id = typeof chat.id === "string" ? chat.id.slice(0, 64) : "";
  if (!id) return null;
  const turns = Array.isArray(chat.turns)
    ? chat.turns.slice(0, RECENT_CHATS_MAX_TURNS).map(normalizeArchivedTurn).filter(Boolean)
    : [];
  if (turns.length === 0) return null;
  const safe = {
    id,
    title: truncateString(chat.title, RECENT_CHATS_MAX_TITLE),
    customTitle: chat.customTitle === true,
    createdAt: Number(chat.createdAt) || 0,
    updatedAt: Number(chat.updatedAt) || 0,
    pinned: chat.pinned === true,
    subject: truncateString(chat.subject, 40),
    subjectLabel: truncateString(chat.subjectLabel, 80),
    mode: truncateString(chat.mode, 40),
    modeLabel: truncateString(chat.modeLabel, 80),
    model: truncateString(chat.model, 80),
    modelLabel: truncateString(chat.modelLabel, 80),
    turns,
  };
  return enforceArchiveChatBudget(safe);
}

function normalizeArchiveChats(chats) {
  if (!Array.isArray(chats)) return [];
  return chats.map(normalizeArchiveChat).filter(Boolean);
}

function applyArchiveTtl(chats) {
  if (!Array.isArray(chats)) return [];
  const cutoff = Date.now() - RECENT_CHATS_TTL_MS;
  return chats.filter(
    (chat) => chat.pinned === true || (Number(chat.updatedAt) || 0) >= cutoff,
  );
}

function orderArchiveChats(chats) {
  const ordered = Array.isArray(chats) ? [...chats] : [];
  ordered.sort((a, b) => {
    const aPinned = a?.pinned === true ? 1 : 0;
    const bPinned = b?.pinned === true ? 1 : 0;
    if (aPinned !== bPinned) return bPinned - aPinned;
    return (Number(b?.updatedAt) || 0) - (Number(a?.updatedAt) || 0);
  });
  return ordered.slice(0, RECENT_CHATS_MAX_CHATS);
}

function readArchiveStore() {
  try {
    const raw = window.localStorage?.getItem(RECENT_CHATS_STORAGE_KEY);
    if (!raw) return { ok: true, chats: [] };
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.version !== RECENT_CHATS_VERSION) return { ok: true, chats: [] };
    const normalized = normalizeArchiveChats(parsed.chats);
    const chats = orderArchiveChats(applyArchiveTtl(normalized));
    if (chats.length !== normalized.length) writeArchiveStore(chats);
    return { ok: true, chats };
  } catch {
    // Corrupt or unavailable storage: fail soft and keep the app usable.
    try {
      window.localStorage?.removeItem(RECENT_CHATS_STORAGE_KEY);
    } catch {
      // Storage remains unavailable; the archive is simply not shown.
    }
    return { ok: false, chats: [] };
  }
}

function writeArchiveStore(chats) {
  const payload = { version: RECENT_CHATS_VERSION, chats };
  try {
    window.localStorage?.setItem(RECENT_CHATS_STORAGE_KEY, JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

function archiveCurrentChat() {
  const chat = buildArchiveChat();
  if (!chat) return;
  const stored = readArchiveStore();
  const chats = stored.ok ? stored.chats.filter((entry) => entry && entry.id !== chat.id) : [];
  const previous = stored.ok ? stored.chats.find((entry) => entry && entry.id === chat.id) : null;
  if (previous) {
    if (previous.pinned === true) chat.pinned = true;
    if (previous.customTitle === true && previous.title) {
      chat.title = previous.title;
      chat.customTitle = true;
    }
  }
  chats.unshift(chat);
  while (chats.length > RECENT_CHATS_MAX_CHATS) chats.pop();
  writeArchiveStore(chats);
  renderRecentChats();
}

function findArchivedChat(chatId) {
  if (!chatId) return null;
  const stored = readArchiveStore();
  return stored.chats.find((chat) => chat.id === chatId) || null;
}

function deleteArchivedChat(chatId) {
  const stored = readArchiveStore();
  if (!stored.ok || !chatId) return;
  const remaining = stored.chats.filter((chat) => chat.id !== chatId);
  if (remaining.length === stored.chats.length) return;
  writeArchiveStore(remaining);
  renderRecentChats();
}

function toggleArchivePin(chatId) {
  const stored = readArchiveStore();
  if (!stored.ok || !chatId) return;
  const chats = stored.chats.map((chat) =>
    chat.id === chatId ? { ...chat, pinned: !chat.pinned } : chat,
  );
  writeArchiveStore(chats);
  renderRecentChats();
}

function renameArchivedChat(chatId, nextTitle) {
  const stored = readArchiveStore();
  if (!stored.ok || !chatId) return false;
  const current = stored.chats.find((chat) => chat.id === chatId);
  if (!current) return false;
  const trimmed = String(nextTitle ?? "").trim();
  if (!trimmed) return false;
  const title = truncateString(trimmed, RECENT_CHATS_MAX_TITLE);
  const chats = stored.chats.map((chat) =>
    chat.id === chatId ? { ...chat, title, customTitle: true } : chat,
  );
  if (!writeArchiveStore(chats)) return false;
  state.renamingArchiveId = null;
  renderRecentChats();
  return true;
}

function startRenameArchivedChat(chatId) {
  if (!chatId) return;
  state.renamingArchiveId = chatId;
  renderRecentChats();
  const input = elements.recentChatsList?.querySelector?.("input[data-archive-id]");
  input?.focus?.();
  input?.select?.();
}

function cancelRenameArchivedChat() {
  state.renamingArchiveId = null;
  renderRecentChats();
}

function formatArchiveTime(timestamp) {
  return formatChatTimestamp(timestamp) || "saved chat";
}

function formatChatTimestamp(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value) || value <= 0) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setTurnTimestamp(element, timestamp) {
  const value = Number(timestamp);
  const label = formatChatTimestamp(value);
  if (!element) return;
  setText(element, label || "time unavailable");
  if (label && Number.isFinite(value)) element.dateTime = new Date(value).toISOString();
}

function renderRecentChats() {
  if (!elements.recentChatsList || !elements.recentChatsEmpty) return;
  const stored = readArchiveStore();
  elements.recentChatsList.replaceChildren();
  elements.recentChatsEmpty.hidden = stored.chats.length > 0;
  for (const chat of stored.chats) {
    const item = document.createElement("li");
    item.className = "recent-chat-row";
    if (chat.pinned === true) item.classList.add("recent-chat-row--pinned");
    if (state.renamingArchiveId === chat.id) {
      const editor = document.createElement("div");
      editor.className = "recent-chat__rename-editor";
      const input = document.createElement("input");
      input.type = "text";
      input.value = chat.title || "Untitled chat";
      input.maxLength = RECENT_CHATS_MAX_TITLE;
      input.dataset.archiveId = chat.id;
      input.setAttribute("aria-label", `Rename saved chat: ${chat.title || "Untitled chat"}`);
      const saveButton = document.createElement("button");
      saveButton.type = "button";
      saveButton.className = "recent-chat__action";
      saveButton.setAttribute("aria-label", `Save renamed chat: ${chat.title || "Untitled chat"}`);
      setText(saveButton, "Save");
      saveButton.addEventListener("click", (event) => {
        event.stopPropagation();
        event.preventDefault();
        renameArchivedChat(chat.id, input.value);
      });
      const cancelButton = document.createElement("button");
      cancelButton.type = "button";
      cancelButton.className = "recent-chat__action";
      cancelButton.setAttribute("aria-label", `Cancel rename for saved chat: ${chat.title || "Untitled chat"}`);
      setText(cancelButton, "Cancel");
      cancelButton.addEventListener("click", (event) => {
        event.stopPropagation();
        event.preventDefault();
        cancelRenameArchivedChat();
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") renameArchivedChat(chat.id, input.value);
        if (event.key === "Escape") cancelRenameArchivedChat();
      });
      editor.append(input, saveButton, cancelButton);
      item.append(editor);
      elements.recentChatsList.append(item);
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-chat";
    button.dataset.archiveId = chat.id;
    button.setAttribute("aria-label", `Open saved chat: ${chat.title || "Untitled chat"}`);
    if (chat.id === state.restoredArchiveId) button.classList.add("recent-chat--active");
    const title = document.createElement("span");
    title.className = "recent-chat__title";
    setText(title, chat.title || "Untitled chat");
    const meta = document.createElement("span");
    meta.className = "recent-chat__meta";
    setText(meta, `${chat.subjectLabel || chat.subject || "No subject"} · ${formatArchiveTime(chat.updatedAt)}`);
    button.append(title, meta);
    button.addEventListener("click", () => previewArchivedChat(chat.id));

    const actions = document.createElement("span");
    actions.className = "recent-chat__actions";
    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "recent-chat__action recent-chat__rename";
    renameButton.dataset.archiveId = chat.id;
    renameButton.setAttribute("aria-label", `Rename saved chat: ${chat.title || "Untitled chat"}`);
    setText(renameButton, "Rename");
    renameButton.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      startRenameArchivedChat(chat.id);
    });
    const pinButton = document.createElement("button");
    pinButton.type = "button";
    pinButton.className = "recent-chat__action recent-chat__pin";
    pinButton.dataset.archiveId = chat.id;
    pinButton.setAttribute("aria-pressed", chat.pinned === true ? "true" : "false");
    pinButton.setAttribute(
      "aria-label",
      chat.pinned === true
        ? `Unpin saved chat: ${chat.title || "Untitled chat"}`
        : `Pin saved chat: ${chat.title || "Untitled chat"}`,
    );
    setText(pinButton, chat.pinned === true ? "Unpin" : "Pin");
    pinButton.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      toggleArchivePin(chat.id);
    });
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "recent-chat__action recent-chat__delete";
    deleteButton.dataset.archiveId = chat.id;
    deleteButton.setAttribute("aria-label", `Delete saved chat: ${chat.title || "Untitled chat"}`);
    setText(deleteButton, "Delete");
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      deleteArchivedChat(chat.id);
    });
    actions.append(renameButton, pinButton, deleteButton);
    item.append(button, actions);
    elements.recentChatsList.append(item);
  }
}

function archivedTurnPayload(turn) {
  const payload = { status: turn.payloadStatus || (turn.status === "error" ? "error" : "ok") };
  if (turn.status === "error") return payload;
  payload.answer = turn.answer || "No answer was returned.";
  payload.subject = turn.subject;
  payload.module = turn.module;
  payload.model = turn.model;
  payload.timing_ms = turn.timingMs;
  payload.mode = turn.mode;
  payload.review = { badge: turn.badge, warnings: turn.warnings || [] };
  payload.sources = turn.sources || [];
  payload.route = turn.route;
  payload.model_requested = turn.modelRequested;
  payload.answer_engine = turn.answerEngine;
  payload.trace = turn.trace || [];
  payload.calculator_steps = turn.calculatorSteps || [];
  payload.vision = turn.vision || { status: "not_requested" };
  if (turn.hadImage) payload.image = { status: "not_archived" };
  return payload;
}

function createArchivedTurnElement(turn, fallbackTimestamp) {
  const fragment = elements.turnTemplate.content.cloneNode(true);
  const turnElement = fragment.querySelector(".turn");
  turnElement.dataset.turnId = turn.turnId;
  turnElement.classList.add("turn--archived");
  const userBubble = turnElement.querySelector(".user-bubble");
  userBubble.replaceChildren();
  if (turn.hadImage) {
    const note = document.createElement("span");
    note.className = "archived-image-note";
    setText(note, "Attached image — not saved in the local preview");
    userBubble.append(note);
  }
  const userQuestion = document.createElement("span");
  userQuestion.className = "user-question";
  renderAnswerBody(userQuestion, turn.question);
  userBubble.append(userQuestion);
  const retryButton = turnElement.querySelector(".retry-button");
  if (retryButton) retryButton.remove();
  renderTurnAssistant(turnElement, {
    turnId: turn.turnId,
    status: turn.status,
    error: turn.error,
    createdAt: turn.createdAt || fallbackTimestamp,
    completedAt: turn.completedAt || turn.createdAt || fallbackTimestamp,
    payload: archivedTurnPayload(turn),
  });
  setTurnTimestamp(
    turnElement.querySelector(".user-turn-time"),
    turn.createdAt || fallbackTimestamp,
  );
  return turnElement;
}

function renderArchivedChat(chat) {
  elements.messages.replaceChildren();
  const banner = document.createElement("div");
  banner.className = "archived-banner";
  const label = chat.subjectLabel || chat.subject || "saved chat";
  const model = chat.modelLabel || chat.model || "model";
  setText(
    banner,
    `Local transcript — ${label}, ${model}. New questions start a fresh server conversation.`,
  );
  elements.messages.append(banner);
  for (const turn of chat.turns) {
    elements.messages.append(createArchivedTurnElement(turn, chat.updatedAt));
  }
  elements.empty.hidden = true;
  elements.messages.hidden = false;
  scrollToBottom();
}

function setArchivePreviewMode(active) {
  void active;
  setConversationLock(isConversationActive());
  syncComposerAuthState();
  elements.lockNotice.hidden = !isConversationActive();
  elements.composerShell?.classList.remove("composer-shell--readonly");
}

function setSelectValueIfAvailable(select, value) {
  if (!select || !value) return false;
  const option = Array.from(select.options || []).find(
    (candidate) => candidate.value === value && !candidate.disabled,
  );
  if (!option) return false;
  select.value = value;
  return true;
}

function applyArchivedContext(chat) {
  if (!chat) return;
  if (chat.mode && elements.mode.value !== chat.mode && setSelectValueIfAvailable(elements.mode, chat.mode)) {
    state.catalogLoading = true;
    void loadCatalog()
      .then(() => applyArchivedContext({ ...chat, mode: elements.mode.value }))
      .catch(() => undefined);
    return;
  }
  setSelectValueIfAvailable(elements.subject, chat.subject);
  setSelectValueIfAvailable(elements.model, chat.model);
  updateSubjectState();
  updateModelNote();
}

function previewArchivedChat(chatId) {
  if (state.isBusy || state.resettingConversation) return;
  const chat = findArchivedChat(chatId);
  if (!chat) return;
  state.restoredArchiveId = chat.id;
  state.conversationId = null;
  state.turns = [];
  renderArchivedChat(chat);
  applyArchivedContext(chat);
  setArchivePreviewMode(false);
  renderRecentChats();
}
// RECENT_CHATS_ARCHIVE_END

// CIRCUIT_GALLERY_START
const CIRCUIT_GALLERY_ENDPOINT = "/api/v1/demo/circuit-gallery";
const CIRCUIT_GALLERY_PREVIEW_ONLY_NOTE =
  "View only — upload PNG/JPEG/WebP for VLM";
const LOCAL_NOTE_GALLERY_ENDPOINT = "/api/v1/demo/local-note-gallery";
const INTERNAL_ASSET_BADGE = "Internal Engineering Preview Asset";
const INTERNAL_SOURCE_LABEL = "locally supplied reference";
const INTERNAL_REDISTRIBUTION_LABEL = "public redistribution not established";

const objectUrlRegistry = new Set();

function galleryDisplayName(asset) {
  const description = String(asset.description || "").trim();
  if (description) return description;
  return String(asset.filename || asset.asset_id || "Circuit image");
}

function galleryThumbnailSource(asset, endpoint = CIRCUIT_GALLERY_ENDPOINT) {
  return `${endpoint}/assets/${encodeURIComponent(asset.filename)}`;
}

function galleryAttributionText(asset) {
  return [asset.license, asset.attribution_author].filter(Boolean).join(" · ");
}

function createGalleryCard(asset) {
  const card = document.createElement("article");
  card.className = "circuit-card";
  card.dataset.assetId = asset.asset_id;
  card.dataset.filename = asset.filename;

  const thumb = document.createElement("img");
  thumb.className = "circuit-card__thumb";
  thumb.alt = `${galleryDisplayName(asset)} preview`;
  thumb.loading = "lazy";
  thumb.src = galleryThumbnailSource(asset);
  card.append(thumb);

  const body = document.createElement("div");
  body.className = "circuit-card__body";
  const title = document.createElement("h3");
  title.className = "circuit-card__title";
  setText(title, galleryDisplayName(asset));
  body.append(title);

  const meta = document.createElement("p");
  meta.className = "circuit-card__meta";
  setText(meta, `${asset.filename} · ${asset.mime_type}`);
  body.append(meta);

  const source = document.createElement("p");
  source.className = "circuit-card__source";
  const sourceLabel = document.createElement("span");
  setText(sourceLabel, "Source: ");
  const sourceLink = document.createElement("a");
  sourceLink.href = asset.source_page_url;
  sourceLink.target = "_blank";
  sourceLink.rel = "noreferrer";
  setText(sourceLink, "Original source");
  source.append(sourceLabel, sourceLink);
  body.append(source);

  const license = document.createElement("p");
  license.className = "circuit-card__license";
  const licenseLabel = document.createElement("span");
  setText(licenseLabel, "License: ");
  const licenseLink = document.createElement("a");
  licenseLink.href = asset.license_url;
  licenseLink.target = "_blank";
  licenseLink.rel = "noreferrer";
  setText(licenseLink, asset.license);
  license.append(licenseLabel, licenseLink);
  body.append(license);

  const attribution = document.createElement("p");
  attribution.className = "circuit-card__attribution";
  setText(attribution, galleryAttributionText(asset));
  body.append(attribution);

  const actions = document.createElement("div");
  actions.className = "circuit-card__actions";
  if (asset.mime_type === "image/png") {
    const useButton = document.createElement("button");
    useButton.type = "button";
    useButton.className = "circuit-card__use";
    setText(useButton, "Use in question");
    useButton.addEventListener("click", () => {
      void useGalleryAsset(asset).catch((error) => {
        setText(
          elements.galleryHint,
          error instanceof Error ? error.message : "Could not load the circuit image.",
        );
        elements.galleryHint.hidden = false;
      });
    });
    actions.append(useButton);
  } else {
    const note = document.createElement("span");
    note.className = "circuit-card__view-only";
    setText(note, CIRCUIT_GALLERY_PREVIEW_ONLY_NOTE);
    actions.append(note);
  }
  body.append(actions);
  card.append(body);
  return card;
}

function internalClaimLines(asset) {
  return [
    `Source: ${asset.source_class_label || INTERNAL_SOURCE_LABEL}`,
    `Public redistribution: ${asset.public_redistribution || INTERNAL_REDISTRIBUTION_LABEL}`,
    `Benchmark evidence: ${asset.formal_benchmark_v1_evidence === true ? "yes" : "no"}`,
    `Course visual accuracy claim: ${asset.course_visual_accuracy_claim === true ? "yes" : "no"}`,
    `SHA-256: ${asset.sha256}`,
  ];
}

function createLocalNoteCard(asset) {
  const card = document.createElement("article");
  card.className = "circuit-card local-note-card";
  card.dataset.assetId = asset.asset_id;
  card.dataset.filename = asset.filename;

  const thumb = document.createElement("img");
  thumb.className = "circuit-card__thumb";
  thumb.alt = `${galleryDisplayName(asset)} preview`;
  thumb.loading = "lazy";
  thumb.src = galleryThumbnailSource(asset, LOCAL_NOTE_GALLERY_ENDPOINT);
  card.append(thumb);

  const body = document.createElement("div");
  body.className = "circuit-card__body";
  const badge = document.createElement("span");
  badge.className = "local-note-card__badge";
  setText(badge, INTERNAL_ASSET_BADGE);
  body.append(badge);

  const title = document.createElement("h3");
  title.className = "circuit-card__title";
  setText(title, galleryDisplayName(asset));
  body.append(title);

  const meta = document.createElement("p");
  meta.className = "circuit-card__meta";
  setText(meta, `${asset.filename} · ${asset.mime_type}`);
  body.append(meta);

  const claims = document.createElement("ul");
  claims.className = "local-note-card__claims";
  for (const line of internalClaimLines(asset)) {
    const item = document.createElement("li");
    setText(item, line);
    claims.append(item);
  }
  body.append(claims);

  const actions = document.createElement("div");
  actions.className = "circuit-card__actions";
  if (asset.mime_type === "image/png") {
    const useButton = document.createElement("button");
    useButton.type = "button";
    useButton.className = "circuit-card__use";
    setText(useButton, "Use in question");
    useButton.addEventListener("click", () => {
      void useGalleryAsset(asset, LOCAL_NOTE_GALLERY_ENDPOINT).catch((error) => {
        setText(
          elements.localNoteGalleryHint,
          error instanceof Error ? error.message : "Could not load the internal circuit image.",
        );
        elements.localNoteGalleryHint.hidden = false;
      });
    });
    actions.append(useButton);
  } else {
    const note = document.createElement("span");
    note.className = "circuit-card__view-only";
    setText(note, CIRCUIT_GALLERY_PREVIEW_ONLY_NOTE);
    actions.append(note);
  }
  body.append(actions);
  card.append(body);
  return card;
}

function createObjectUrl(file) {
  if (!file) return null;
  try {
    const url = URL.createObjectURL(file);
    if (url) objectUrlRegistry.add(url);
    return url || null;
  } catch {
    return null;
  }
}

function revokeObjectUrl(url) {
  if (!url || !objectUrlRegistry.delete(url)) return;
  try {
    URL.revokeObjectURL(url);
  } catch {
    // The URL is already removed from the registry.
  }
}

function releaseAllObjectUrls() {
  let released = 0;
  for (const url of objectUrlRegistry) {
    try {
      URL.revokeObjectURL(url);
      released += 1;
    } catch {
      // Best-effort cleanup for temporary previews.
    }
  }
  objectUrlRegistry.clear();
  return released;
}

function updateFileName(file) {
  setText(
    elements.fileName,
    file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KiB` : "No image selected",
  );
}

function updateComposerPreview(file) {
  const preview = elements.composerPreview;
  const image = elements.composerPreviewImage;
  if (!preview || !image) return;
  if (preview.dataset.objectUrl) {
    revokeObjectUrl(preview.dataset.objectUrl);
    preview.dataset.objectUrl = "";
  }
  if (!file) {
    preview.hidden = true;
    image.removeAttribute("src");
    return;
  }
  const url = createObjectUrl(file);
  if (!url) {
    preview.hidden = true;
    return;
  }
  preview.dataset.objectUrl = url;
  image.src = url;
  image.alt = `Preview of ${file.name}`;
  preview.hidden = false;
}

function assignImageFile(file) {
  if (typeof DataTransfer !== "function") {
    setText(elements.fileName, "Gallery selection is not supported in this browser.");
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.image.files = transfer.files;
  updateComposerPreview(file);
  updateFileName(file);
}

async function useGalleryAsset(asset, endpoint = CIRCUIT_GALLERY_ENDPOINT) {
  if (state.resettingConversation || state.conversationResetAmbiguous) return;
  if (asset.mime_type !== "image/png") return;
  const response = await fetch(galleryThumbnailSource(asset, endpoint));
  if (!response.ok) {
    throw new Error(`Could not load ${asset.filename} for the question.`);
  }
  const bytes = await response.arrayBuffer();
  const file = new File([bytes], asset.filename, { type: asset.mime_type });
  assignImageFile(file);
}

function appendTurnImagePreview(bubble, turn) {
  if (!bubble || !turn.imageFile) return;
  if (turn.imageObjectUrl) {
    revokeObjectUrl(turn.imageObjectUrl);
    turn.imageObjectUrl = null;
  }
  const image = document.createElement("img");
  image.className = "user-image-preview";
  image.alt = `Attached image: ${turn.imageFile.name}`;
  const url = createObjectUrl(turn.imageFile);
  turn.imageObjectUrl = url;
  if (url) image.src = url;
  bubble.append(image);
}

async function loadCircuitGallery() {
  if (!elements.gallery || !elements.galleryGrid || !elements.galleryHint) return;
  if (authState.status === "gated" || authState.status === "config_error") return;
  elements.gallery.hidden = false;
  elements.galleryGrid.hidden = true;
  elements.galleryGrid.replaceChildren();
  setText(elements.galleryHint, "Loading licensed circuit preview…");
  elements.galleryHint.hidden = false;
  try {
    const response = await fetch(CIRCUIT_GALLERY_ENDPOINT, AUTH_SAME_ORIGIN);
    if (response.status === 401 && authState.status === "authed") {
      handleExpiredSession("Your session is no longer valid. Sign in again to continue.");
      return;
    }
    if (!response.ok) {
      throw new Error(`Gallery manifest returned ${response.status}.`);
    }
    const manifest = await response.json();
    const assets = Array.isArray(manifest?.assets) ? manifest.assets : [];
    if (assets.length === 0) {
      setText(elements.galleryHint, "No licensed circuit assets are available.");
      return;
    }
    elements.galleryHint.hidden = true;
    for (const asset of assets) {
      if (!asset || typeof asset !== "object") continue;
      elements.galleryGrid.append(createGalleryCard(asset));
    }
    elements.galleryGrid.hidden = false;
  } catch (error) {
    setText(
      elements.galleryHint,
      error instanceof Error ? error.message : "Could not load the circuit gallery.",
    );
    elements.galleryHint.hidden = false;
  }
}

async function loadLocalNoteGallery() {
  if (
    !elements.localNoteGallery ||
    !elements.localNoteGalleryGrid ||
    !elements.localNoteGalleryHint
  ) {
    return;
  }
  if (authState.status === "gated" || authState.status === "config_error") return;
  elements.localNoteGallery.hidden = false;
  elements.localNoteGalleryGrid.hidden = true;
  elements.localNoteGalleryGrid.replaceChildren();
  setText(elements.localNoteGalleryHint, "Loading internal engineering preview assets…");
  elements.localNoteGalleryHint.hidden = false;
  try {
    const response = await fetch(LOCAL_NOTE_GALLERY_ENDPOINT, AUTH_SAME_ORIGIN);
    if (response.status === 401 && authState.status === "authed") {
      handleExpiredSession("Your session is no longer valid. Sign in again to continue.");
      return;
    }
    if (!response.ok) {
      throw new Error(`Internal gallery manifest returned ${response.status}.`);
    }
    const manifest = await response.json();
    const assets = Array.isArray(manifest?.assets) ? manifest.assets : [];
    if (assets.length === 0) {
      setText(elements.localNoteGalleryHint, "No internal engineering preview assets are available.");
      return;
    }
    elements.localNoteGalleryHint.hidden = true;
    for (const asset of assets) {
      if (!asset || typeof asset !== "object") continue;
      elements.localNoteGalleryGrid.append(createLocalNoteCard(asset));
    }
    elements.localNoteGalleryGrid.hidden = false;
  } catch (error) {
    setText(
      elements.localNoteGalleryHint,
      error instanceof Error ? error.message : "Could not load the internal gallery.",
    );
    elements.localNoteGalleryHint.hidden = false;
  }
}
// CIRCUIT_GALLERY_END

function renderVision(visionCard, payload) {
  const vision = payload.vision;
  const visionStatus = visionCard.querySelector(".vision-status");
  const visionJson = visionCard.querySelector(".vision-json");
  if (!vision || vision.status === "not_requested") {
    visionCard.hidden = true;
    return;
  }
  let summaryElement = visionCard.querySelector(".vision-summary");
  if (!summaryElement) {
    summaryElement = document.createElement("dl");
    summaryElement.className = "vision-summary answer-meta";
    visionJson.insertAdjacentElement("beforebegin", summaryElement);
  }
  renderVisionSummary(summaryElement, vision);
  setText(visionStatus, String(vision.status ?? "unknown").replaceAll("_", " "));
  setText(visionJson, JSON.stringify(vision, null, 2));
  visionCard.hidden = false;
}

function renderTurnAssistant(turnElement, turn) {
  const content = turnElement.querySelector(".assistant-content");
  const messageEl = turnElement.querySelector(".assistant-message");
  const statusEl = content.querySelector(".response-status");
  const timeEl = content.querySelector(".assistant-turn-time");
  const titleEl = content.querySelector(".answer-title");
  const bodyEl = content.querySelector(".answer-body");
  const metaEl = content.querySelector(".answer-meta");
  const workedSolutionCard = content.querySelector(".worked-solution-card");
  const toolchainCard = content.querySelector(".toolchain-card");
  const visionCard = content.querySelector(".vision-card");
  const abstainNote = content.querySelector(".abstain-note");
  const vlmStrip = content.querySelector(".vlm-strip");
  const errorEl = content.querySelector(".turn-error");
  const errorMessage = content.querySelector(".error-message");

  setTurnTimestamp(timeEl, turn.completedAt || turn.createdAt);

  metaEl.hidden = true;
  if (workedSolutionCard) workedSolutionCard.hidden = true;
  if (toolchainCard) toolchainCard.hidden = true;
  visionCard.hidden = true;
  if (abstainNote) abstainNote.hidden = true;
  if (vlmStrip) vlmStrip.hidden = true;
  errorEl.hidden = true;
  for (const modifier of [
    "assistant-message--loading",
    "assistant-message--error",
    "assistant-message--abstain",
    "assistant-message--ok",
  ]) {
    messageEl.classList.remove(modifier);
  }
  for (const modifier of [
    "response-status--loading",
    "response-status--error",
    "response-status--abstain",
    "response-status--ok",
  ]) {
    statusEl.classList.remove(modifier);
  }

  if (turn.status === "loading") {
    messageEl.classList.add("assistant-message--loading");
    statusEl.classList.add("response-status--loading");
    setText(statusEl, "Thinking · 0s");
    setText(titleEl, "Retrieving subject-grounded context");
    bodyEl.replaceChildren();
    for (const line of buildSkeletonLines()) {
      bodyEl.append(line);
    }
    return;
  }

  if (turn.status === "error") {
    messageEl.classList.add("assistant-message--error");
    statusEl.classList.add("response-status--error");
    setText(statusEl, "Unable to answer");
    setText(titleEl, "The request did not complete");
    setText(bodyEl, "");
    setText(errorMessage, turn.error || "Unexpected request failure.");
    errorEl.hidden = false;
    return;
  }

  const payload = turn.payload;
  if (!payload) return;

  const badge = payload.review?.badge || payload.mode || "unified";
  const needsInput = payload.status === "needs_input";
  setText(statusEl, needsInput ? "More information needed" : `${payload.status} · ${badge}`);
  if (needsInput || payload.status === "abstain" || payload.status === "needs_subject") {
    messageEl.classList.add("assistant-message--abstain");
    statusEl.classList.add("response-status--abstain");
    if (abstainNote) renderAbstainNote(abstainNote, payload);
  } else {
    messageEl.classList.add("assistant-message--ok");
    statusEl.classList.add("response-status--ok");
  }
  setText(
    titleEl,
    needsInput ? "More information needed" : payload.subject ? `${payload.subject} answer` : "Subject needed",
  );
  renderAnswerBody(bodyEl, payload.answer || "No answer was returned.");
  if (vlmStrip) renderVlmStrip(vlmStrip, payload.vision);
  if (elements.reviewBadge) {
    setText(elements.reviewBadge, String(badge).replaceAll("_", " "));
  }
  renderCalculatorSteps(workedSolutionCard, payload);
  renderMeta(metaEl, payload);
  renderToolchain(toolchainCard, payload);
  renderVision(visionCard, payload);
}

function createTurnElement(turn) {
  const fragment = elements.turnTemplate.content.cloneNode(true);
  const turnElement = fragment.querySelector(".turn");
  turnElement.dataset.turnId = turn.turnId;
  const userBubble = turnElement.querySelector(".user-bubble");
  userBubble.replaceChildren();
  appendTurnImagePreview(userBubble, turn);
  const userQuestion = document.createElement("span");
  userQuestion.className = "user-question";
  renderAnswerBody(userQuestion, turn.question);
  userBubble.append(userQuestion);
  setTurnTimestamp(turnElement.querySelector(".user-turn-time"), turn.createdAt);

  const retryButton = turnElement.querySelector(".retry-button");
  retryButton.addEventListener("click", () => {
    // Archive the transcript again after a retried turn completes so the
    // saved Recent Chat reflects the final visible state.
    void retryTurn(turn.turnId).then(() => archiveCurrentChat());
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
  renderComposerMathPreview();
  updateComposerPreview(null);
  updateFileName(null);
}

function buildFormData(turn) {
  const formData = new FormData();
  formData.append("subject", turn.subject || elements.subject.value);
  formData.append("mode", turn.mode || elements.mode?.value || "preview");
  formData.append("model", turn.model || elements.model.value);
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

function ensureConversationId() {
  if (!state.conversationId) state.conversationId = makeOpaqueId();
  return state.conversationId;
}

function captureServerConversationId(payload) {
  const serverConversationId = payload.conversation_id || payload.conversation?.id;
  if (serverConversationId && !state.conversationId) {
    state.conversationId = serverConversationId;
  }
  return state.conversationId;
}

function isConversationMismatchResponse(response, payload) {
  return (
    response.status === 403 &&
    payload !== null &&
    typeof payload === "object" &&
    payload.detail !== null &&
    typeof payload.detail === "object" &&
    payload.detail.code === "conversation_mismatch"
  );
}

async function requestAnswer(turn, controller) {
  const response = await fetch(`${API}/answer`, {
    method: "POST",
    body: buildFormData(turn),
    signal: controller.signal,
    credentials: "same-origin",
  });
  if (response.status === 401 && authState.status === "authed") {
    throw new SessionExpiredError("Your session is no longer valid. Sign in again to continue.");
  }
  const responseText = await response.text();
  let payload;
  try {
    payload = JSON.parse(responseText);
  } catch {
    throw new Error(`The service returned ${response.status}: ${responseText || "empty response"}`);
  }
  if (!response.ok) {
    const detail = payload.detail;
    if (isConversationMismatchResponse(response, payload)) {
      throw new ConversationMismatchError(detail?.message || "Request failed.");
    }
    throw new Error(typeof detail === "string" ? detail : detail?.message || "Request failed.");
  }
  return payload;
}

async function recoverConversationMismatch() {
  // Clear only the server-side binding (preserving the bounded teaching
  // context), then drop the client conversation id and mint a fresh opaque
  // id for the single retry. The visible thread, turns, current selection,
  // and image are all preserved. New chat remains the only path
  // that clears server-side history via resetConversationOnServer().
  await recoverConversationBindingOnServer();
  state.conversationId = null;
  ensureConversationId();
}

async function submitTurn(turn, { appendUserBubble }) {
  if (state.isBusy || state.resettingConversation || state.conversationResetAmbiguous) return;
  if (authState.status === "gated" || authState.status === "config_error") return;

  // The form path must prepare a restored transcript first. This defensive
  // guard prevents direct retry/caller paths from minting a binding early.
  if (state.restoredArchiveId) return;

  // One opaque conversation id per in-thread turn sequence. It stays stable
  // across turns and is regenerated by "New chat"; the backend binds each
  // sequence to the auth session and rejects mismatched ids (403).
  ensureConversationId();

  turn.status = "loading";
  turn.thinkingStartedAt = Date.now();
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
  startThinkingClock(turn, turnElement);

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), ANSWER_REQUEST_TIMEOUT_MS);
  setBusy(true);
  try {
    let payload;
    try {
      payload = await requestAnswer(turn, controller);
    } catch (error) {
      if (!(error instanceof ConversationMismatchError)) throw error;
      // The server-side binding was reset underneath this thread (page reload
      // or a concurrent New chat). Recover once with a fresh opaque id and
      // retry the exact same turn; a second mismatch surfaces the normal
      // error instead of looping.
      await recoverConversationMismatch();
      payload = await requestAnswer(turn, controller);
    }

    captureServerConversationId(payload);

    turn.status = "success";
    turn.completedAt = Date.now();
    turn.payload = payload;
    turn.error = undefined;
    if (turnElement) renderTurnAssistant(turnElement, turn);
    scrollToBottom();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    turn.status = "error";
    turn.completedAt = Date.now();
    turn.error =
      error?.name === "AbortError"
        ? `The request timed out after ${ANSWER_REQUEST_TIMEOUT_SECONDS} seconds. Please try again.`
        : error instanceof Error
          ? error.message
          : "Unexpected request failure.";
    if (turnElement) renderTurnAssistant(turnElement, turn);
    scrollToBottom();
  } finally {
    stopThinkingClock();
    window.clearTimeout(timeoutId);
    setBusy(false);
  }
}

async function retryTurn(turnId) {
  const turn = findTurn(turnId);
  if (
    !turn ||
    state.isBusy ||
    state.resettingConversation ||
    state.conversationResetAmbiguous
  ) {
    return;
  }
  await submitTurn(turn, { appendUserBubble: false });
}

async function prepareArchiveContinuation() {
  if (!state.restoredArchiveId) return true;
  if (state.isBusy || state.resettingConversation || state.conversationResetAmbiguous) {
    return false;
  }
  state.conversationResetAmbiguous = false;
  state.resettingConversation = true;
  setBusy(true);
  showConversationResetStatus("Starting new chat…", "pending");
  syncConversationResetUi();
  try {
    if (authState.status === "authed") {
      await resetConversationOnServer();
    }
    state.conversationId = null;
    state.restoredArchiveId = null;
    state.turns = [];
    const banner = elements.messages.querySelector(".archived-banner");
    setText(
      banner,
      "Continued from a local transcript. This question starts a fresh server conversation.",
    );
    renderRecentChats();
    showConversationResetStatus("New chat ready.", "ready");
    return true;
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      state.conversationResetAmbiguous = false;
      handleExpiredSession(error.message);
    } else {
      state.conversationResetAmbiguous = true;
      showConversationResetStatus(
        error instanceof Error
          ? error.message
          : "New chat was not confirmed. Your saved chat is preserved. Retry New Chat.",
        "error",
      );
    }
    return false;
  } finally {
    state.resettingConversation = false;
    setBusy(false);
    syncConversationResetUi();
  }
}

async function resetConversationOnServer() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    CONVERSATION_RESET_TIMEOUT_MS,
  );
  let response;
  try {
    response = await fetch("/auth/conversation/reset", {
      method: "POST",
      credentials: "same-origin",
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(
        `New chat was not confirmed within ${CONVERSATION_RESET_TIMEOUT_SECONDS} seconds. Your current chat is preserved. Retry New Chat.`,
      );
    }
    throw new Error(
      "The new-chat request could not be confirmed. Your current chat is preserved. Retry New Chat.",
    );
  } finally {
    window.clearTimeout(timeoutId);
  }
  if (response.status === 401 && authState.status === "authed") {
    throw new SessionExpiredError("Your session is no longer valid. Sign in again to continue.");
  }
  if (!response.ok) {
    throw new Error(
      "The server did not confirm a new chat. Your current chat is preserved. Retry New Chat.",
    );
  }
}

async function recoverConversationBindingOnServer() {
  const response = await fetch(AUTH_CONVERSATION_RECOVER_ENDPOINT, {
    method: "POST",
    credentials: "same-origin",
  });
  if (response.status === 401 && authState.status === "authed") {
    throw new SessionExpiredError("Your session is no longer valid. Sign in again to continue.");
  }
  if (!response.ok) {
    throw new Error("Could not recover the conversation binding on the server.");
  }
}

function lockConversationContext() {
  setConversationLock(true);
}

function setQuizSelectPlaceholder(select, text) {
  const option = document.createElement("option");
  option.value = "";
  option.textContent = text;
  option.selected = true;
  select.replaceChildren(option);
  select.disabled = true;
}

function clearQuizScopeControls() {
  setQuizSelectPlaceholder(elements.quizTopic, "Choose a subject first");
  setQuizSelectPlaceholder(elements.quizKnowledgePoint, "Choose a Topic first");
}

function resetQuizLocalUi() {
  state.quizInitialized = false;
  elements.quizExit.hidden = true;
  elements.quizSummary.replaceChildren();
  elements.quizSummary.hidden = true;
  elements.quizCard.hidden = true;
  elements.quizSetup.hidden = false;
  elements.quizFeedback.replaceChildren();
  elements.quizFeedback.hidden = true;
  elements.quizOptions.replaceChildren();
  elements.quizPrompt.textContent = "";
  elements.quizProgress.textContent = "";
  elements.quizError.textContent = "";
  elements.quizError.hidden = true;
  elements.quizSubject.value = "";
  elements.quizQuestionCount.value = "3";
  if (typeof setQuizGenerationState === "function") setQuizGenerationState(null);
  if (typeof clearQuizScopeControls === "function") clearQuizScopeControls();
}

function resetChatLocal() {
  // Persist the finished local transcript before clearing it so New chat,
  // logout, and expiry never lose a completed chat.
  archiveCurrentChat();
  resetConversationState();
  if (typeof resetQuizLocalUi === "function") resetQuizLocalUi();
  if (typeof setActivity === "function") setActivity("ask");
  elements.messages.replaceChildren();
  elements.form.reset();
  if (elements.subject) elements.subject.value = "";
  if (typeof closeSubjectMentionMenu === "function") closeSubjectMentionMenu();
  if (typeof closeMathDialog === "function") closeMathDialog();
  if (typeof renderComposerMathPreview === "function") renderComposerMathPreview();
  releaseAllObjectUrls();
  updateComposerPreview(null);
  elements.empty.hidden = false;
  elements.messages.hidden = true;
  updateFileName(null);
  setConversationLock(false);
  setArchivePreviewMode(false);
  setBusy(false);
  renderRecentChats();
  elements.question.focus();
  if (typeof syncConversationResetUi === "function") syncConversationResetUi();
}

// QUIZ_ACTIVITY_START
function quizErrorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail.message === "string") return detail.message;
  return fallback;
}

function showQuizError(message) {
  if (!elements.quizError) return;
  setText(elements.quizError, message || "");
  elements.quizError.hidden = !message;
}

function syncQuizStartEnabled() {
  const topic = state.quizTopics.find(
    (candidate) => candidate.id === elements.quizTopic.value,
  );
  const point = topic?.knowledge_points?.find(
    (candidate) => candidate.id === elements.quizKnowledgePoint.value,
  );
  const scope = point || topic;
  const count = Number(elements.quizQuestionCount.value);
  elements.quizStart.disabled =
    state.quizBusy ||
    state.catalogLoading ||
    !elements.quizSubject.value ||
    !elements.quizTopic.value ||
    !scope?.question_counts?.includes(count);
}

function quizSupportedCounts(scope) {
  if (scope) return scope.question_counts || [];
  return Array.from(
    new Set(
      state.quizTopics.flatMap((topic) => topic.question_counts || []),
    ),
  );
}

function syncQuizQuestionCountOptions(scope) {
  const supported = new Set(quizSupportedCounts(scope));
  for (const option of Array.from(elements.quizQuestionCount.options || [])) {
    option.disabled = !supported.has(Number(option.value));
  }
  if (!supported.size) return;
  const current = Number(elements.quizQuestionCount.value);
  if (supported.has(current)) return;
  const next = supported.has(3) ? 3 : Math.min(...supported);
  elements.quizQuestionCount.value = String(next);
}

function populateKnowledgePoints() {
  const topic = state.quizTopics.find(
    (candidate) => candidate.id === elements.quizTopic.value,
  );
  const previousPointId = elements.quizKnowledgePoint.value;
  setQuizSelectPlaceholder(elements.quizKnowledgePoint, "Whole topic");
  elements.quizKnowledgePoint.disabled = !topic;
  if (!topic) {
    syncQuizStartEnabled();
    return;
  }
  elements.quizKnowledgePoint.options[0].disabled =
    !(topic.question_counts || []).length;
  for (const point of topic.knowledge_points || []) {
    const option = document.createElement("option");
    option.value = point.id;
    option.textContent = point.level === 3 ? point.breadcrumb : point.label;
    option.disabled = !(point.question_counts || []).length;
    elements.quizKnowledgePoint.append(option);
  }
  const retainedPoint = (topic.knowledge_points || []).find(
    (point) => point.id === previousPointId && (point.question_counts || []).length,
  );
  elements.quizKnowledgePoint.value = retainedPoint?.id || "";
  syncQuizStartEnabled();
}

function reconcileQuizCapacitySelection() {
  let topic = state.quizTopics.find(
    (candidate) => candidate.id === elements.quizTopic.value,
  );
  if (topic && !(topic.question_counts || []).length) {
    topic = state.quizTopics.find((candidate) => (candidate.question_counts || []).length);
    elements.quizTopic.value = topic?.id || "";
  }
  const point = topic?.knowledge_points?.find(
    (candidate) => candidate.id === elements.quizKnowledgePoint.value,
  );
  syncQuizQuestionCountOptions(point || topic);
  populateKnowledgePoints();
  const selectedTopic = state.quizTopics.find(
    (candidate) => candidate.id === elements.quizTopic.value,
  );
  const selectedPoint = selectedTopic?.knowledge_points?.find(
    (candidate) => candidate.id === elements.quizKnowledgePoint.value,
  );
  syncQuizQuestionCountOptions(selectedPoint || selectedTopic);
}

async function loadQuizScopes() {
  const requestId = ++state.quizScopesRequestId;
  state.quizTopics = [];
  clearQuizScopeControls();
  syncQuizStartEnabled();
  if (!elements.quizSubject.value) return;
  showQuizError("");
  try {
    const path =
      "/api/v1/quiz/scopes?subject=" +
      encodeURIComponent(elements.quizSubject.value) +
      "&mode=" +
      encodeURIComponent(elements.mode.value);
    const response = await fetch(path, AUTH_SAME_ORIGIN);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (requestId !== state.quizScopesRequestId) return;
    if (response.status === 401 && authState.status === "authed") {
      throw new SessionExpiredError(
        "Your session is no longer valid. Sign in again to continue.",
      );
    }
    if (!response.ok) {
      throw new Error(quizErrorMessage(payload, "Quiz Topics are unavailable."));
    }
    state.quizTopics = Array.isArray(payload?.topics) ? payload.topics : [];
    setQuizSelectPlaceholder(elements.quizTopic, "Choose a Topic");
    for (const topic of state.quizTopics) {
      const option = document.createElement("option");
      option.value = topic.id;
      option.textContent = topic.label;
      elements.quizTopic.append(option);
    }
    elements.quizTopic.disabled =
      state.quizTopics.length === 0 || state.quizPayload?.status === "in_progress";
    if (state.quizTopics.length > 0 && state.quizPayload?.status !== "in_progress") {
      reconcileQuizCapacitySelection();
    } else {
      syncQuizStartEnabled();
    }
  } catch (error) {
    if (requestId !== state.quizScopesRequestId) return;
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(
      error instanceof Error ? error.message : "Quiz Topics are unavailable.",
    );
  }
}

function setQuizBusy(busy) {
  state.quizBusy = busy;
  setConversationLock(isConversationActive());
  for (const button of [
    elements.quizStart,
    elements.quizSubmit,
    elements.quizNext,
    elements.quizFinish,
    elements.quizExit,
  ]) {
    if (button) button.disabled = busy;
  }
  if (!busy) {
    const selected = elements.quizOptions.querySelector?.(
      'input[name="quiz-option"]:checked',
    );
    elements.quizSubmit.disabled = !selected || Boolean(state.quizPayload?.question?.submitted);
  }
  syncQuizStartEnabled();
  if (typeof syncConversationResetUi === "function") syncConversationResetUi();
}

function setQuizGenerationState(kind) {
  const active = kind === "start" || kind === "next";
  elements.quizLoading.hidden = !active;
  elements.quizActivity.setAttribute("aria-busy", String(active));
  setText(elements.quizStart, kind === "start" ? "Generating…" : "Start Quiz");
  setText(elements.quizNext, kind === "next" ? "Generating…" : "Next Question");
  if (!active) return;
  const total =
    kind === "start"
      ? Number(elements.quizQuestionCount.value)
      : Number(state.quizPayload?.requested_total || 1);
  const number =
    kind === "start" ? 1 : Number(state.quizPayload?.current_number || 0) + 1;
  setText(elements.quizLoadingLabel, `Generating question ${number} of ${total}…`);
  setText(
    elements.quizLoadingDetail,
    "The local model is selecting a source-bound question from the selected course notes.",
  );
}

function renderQuizSummary(payload) {
  elements.quizCard.hidden = true;
  elements.quizSetup.hidden = false;
  elements.quizSummary.replaceChildren();
  const title = document.createElement("h2");
  setText(title, "Quiz finished");
  const score = document.createElement("p");
  setText(
    score,
    "Score " +
      payload.score +
      " / " +
      payload.maximum_score +
      " · answered " +
      payload.answered_count +
      " of " +
      payload.requested_total,
  );
  const note = document.createElement("p");
  setText(
    note,
    payload.subject +
      (payload.topic ? " · " + payload.topic : "") +
      (payload.knowledge_point ? " · " + payload.knowledge_point : ""),
  );
  elements.quizSummary.append(title, score, note);
  if (payload.review_count > 0) {
    const review = document.createElement("button");
    review.type = "button";
    review.className = "quiz-primary";
    setText(review, "Review Answers");
    review.addEventListener("click", () => void loadQuizReview(1));
    elements.quizSummary.append(review);
  }
  elements.quizSummary.hidden = false;
  elements.quizTopic.disabled = state.quizTopics.length === 0;
  populateKnowledgePoints();
}

function appendQuizFeedbackCard(container, titleText, body, className = "") {
  const card = document.createElement("section");
  card.className = "quiz-feedback-card" + (className ? " " + className : "");
  const title = document.createElement("h3");
  setText(title, titleText);
  const content = document.createElement("div");
  renderAnswerBody(content, body);
  card.append(title, content);
  container.append(card);
  return card;
}

function formatQuizReference(reference) {
  const title = String(reference?.title || "").trim();
  const heading = String(reference?.heading || "").trim();
  if (!title) return heading;
  if (!heading) return title;
  const prefix = `${title} — `;
  if (heading.startsWith(prefix)) {
    const breadcrumb = heading.slice(prefix.length).trim();
    return breadcrumb ? `${title} · ${breadcrumb}` : title;
  }
  return `${title} · ${heading}`;
}

function renderStructuredQuizFeedback(container, payload, question) {
  appendQuizFeedbackCard(
    container,
    "Why this is correct",
    payload.why_correct || payload.explanation || "",
    "quiz-feedback-card--wide",
  );
  const wrongCard = document.createElement("section");
  wrongCard.className = "quiz-feedback-card quiz-feedback-card--wide";
  const wrongTitle = document.createElement("h3");
  setText(wrongTitle, "Why the other choices are wrong");
  const wrongList = document.createElement("ul");
  wrongList.className = "quiz-wrong-reasons";
  for (const item of payload.wrong_options || []) {
    const row = document.createElement("li");
    const label = document.createElement("strong");
    renderAnswerBody(label, `${item.option_id}. ${item.text}`);
    const reason = document.createElement("p");
    renderAnswerBody(reason, item.reason);
    row.append(label, reason);
    wrongList.append(row);
  }
  wrongCard.append(wrongTitle, wrongList);
  container.append(wrongCard);
  const reference = payload.reference || {};
  appendQuizFeedbackCard(
    container,
    "Reference note",
    formatQuizReference(reference),
    "quiz-feedback-card--wide",
  );
}

function renderQuizFeedback(feedback) {
  elements.quizFeedback.replaceChildren();
  if (!feedback) {
    elements.quizFeedback.hidden = true;
    return;
  }
  renderStructuredQuizFeedback(elements.quizFeedback, feedback, state.quizPayload?.question);
  elements.quizFeedback.hidden = false;
}

function renderQuiz() {
  const payload = state.quizPayload;
  elements.quizExit.hidden = !payload;
  elements.quizSummary.hidden = true;
  if (!payload) {
    elements.quizSetup.hidden = false;
    elements.quizCard.hidden = true;
    elements.quizTopic.disabled = state.quizTopics.length === 0;
    populateKnowledgePoints();
    setConversationLock(isConversationActive());
    return;
  }
  if (payload.status === "finished") {
    renderQuizSummary(payload);
    setConversationLock(isConversationActive());
    return;
  }
  const question = payload.question;
  elements.quizSetup.hidden = true;
  elements.quizCard.hidden = false;
  elements.quizTopic.disabled = true;
  elements.quizKnowledgePoint.disabled = true;
  elements.quizQuestionCount.disabled = true;
  setText(
    elements.quizProgress,
    "Question " +
      payload.current_number +
      " of " +
      payload.requested_total +
      " · Score " +
      payload.correct_count,
  );
  elements.quizProgressTrack.setAttribute("aria-valuemax", String(payload.requested_total));
  elements.quizProgressTrack.setAttribute("aria-valuenow", String(payload.answered_count));
  elements.quizProgressFill.style.width =
    (100 * payload.answered_count) / payload.requested_total + "%";
  renderAnswerBody(elements.quizPrompt, question.prompt);
  elements.quizOptions.replaceChildren();
  for (const option of question.options) {
    const label = document.createElement("label");
    label.className = "quiz-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "quiz-option";
    input.value = option.option_id;
    input.disabled = Boolean(question.submitted);
    input.checked = payload.feedback?.submitted_option_id === option.option_id;
    input.addEventListener("change", () => {
      elements.quizSubmit.disabled = false;
    });
    const text = document.createElement("span");
    text.className = "quiz-option-text";
    renderAnswerBody(text, option.option_id + ". " + option.text);
    if (question.submitted && option.option_id === payload.feedback?.correct_option_id) {
      label.classList.add("quiz-option--correct");
    } else if (
      question.submitted &&
      option.option_id === payload.feedback?.submitted_option_id
    ) {
      label.classList.add("quiz-option--incorrect");
    }
    label.append(input, text);
    if (question.submitted && option.option_id === payload.feedback?.correct_option_id) {
      const status = document.createElement("span");
      status.className = "quiz-option-status quiz-option-status--correct";
      setText(status, "Correct");
      label.append(status);
    } else if (
      question.submitted &&
      option.option_id === payload.feedback?.submitted_option_id
    ) {
      const status = document.createElement("span");
      status.className = "quiz-option-status quiz-option-status--incorrect";
      setText(status, "Your answer");
      label.append(status);
    }
    elements.quizOptions.append(label);
  }
  elements.quizSubmit.hidden = Boolean(question.submitted);
  elements.quizSubmit.disabled = true;
  elements.quizNext.hidden = !question.submitted || !payload.feedback?.has_next;
  elements.quizFinish.hidden = !question.submitted || Boolean(payload.feedback?.has_next);
  elements.quizExit.hidden = false;
  renderQuizFeedback(payload.feedback || null);
  setConversationLock(isConversationActive());
}

function renderQuizReview(payload) {
  elements.quizSummary.replaceChildren();
  const title = document.createElement("h2");
  setText(title, `Review ${payload.number} of ${payload.review_count}`);
  const prompt = document.createElement("h3");
  renderAnswerBody(prompt, payload.question.prompt);
  const options = document.createElement("ul");
  options.className = "quiz-review-options";
  for (const option of payload.question.options) {
    const item = document.createElement("li");
    renderAnswerBody(item, option.option_id + ". " + option.text);
    if (option.option_id === payload.correct_option_id) {
      item.classList.add("quiz-option--correct");
    } else if (option.option_id === payload.submitted_option_id) {
      item.classList.add("quiz-option--incorrect");
    }
    options.append(item);
  }
  const feedback = document.createElement("div");
  feedback.className = "quiz-feedback quiz-review-feedback";
  renderStructuredQuizFeedback(feedback, payload, payload.question);
  const navigation = document.createElement("div");
  navigation.className = "quiz-actions quiz-review-nav";
  const previous = document.createElement("button");
  previous.type = "button";
  previous.disabled = !payload.has_previous;
  setText(previous, "Previous");
  previous.addEventListener("click", () => void loadQuizReview(payload.number - 1));
  const next = document.createElement("button");
  next.type = "button";
  next.disabled = !payload.has_next;
  setText(next, "Next");
  next.addEventListener("click", () => void loadQuizReview(payload.number + 1));
  navigation.append(previous, next);
  elements.quizSummary.append(title, prompt, options, feedback);
  elements.quizSummary.append(navigation);
  elements.quizSummary.hidden = false;
}

async function loadQuizReview(number) {
  if (state.quizBusy) return;
  setQuizBusy(true);
  showQuizError("");
  try {
    const payload = await quizRequest(`/api/v1/quiz/review/${number}`);
    renderQuizReview(payload);
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(error instanceof Error ? error.message : "Quiz review is unavailable.");
  } finally {
    setQuizBusy(false);
  }
}

async function quizRequest(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (response.status === 401 && authState.status === "authed") {
    throw new SessionExpiredError("Your session is no longer valid. Sign in again to continue.");
  }
  if (!response.ok) {
    throw new Error(quizErrorMessage(payload, "The Quiz request failed."));
  }
  return payload;
}

async function loadCurrentQuiz() {
  setQuizBusy(true);
  showQuizError("");
  try {
    const response = await fetch("/api/v1/quiz/current", AUTH_SAME_ORIGIN);
    if (response.status === 404) {
      state.quizPayload = null;
      renderQuiz();
      return;
    }
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (response.status === 401 && authState.status === "authed") {
      throw new SessionExpiredError(
        "Your session is no longer valid. Sign in again to continue.",
      );
    }
    if (!response.ok) {
      throw new Error(quizErrorMessage(payload, "Sign in to start a Quiz."));
    }
    state.quizPayload = payload;
    renderQuiz();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(error instanceof Error ? error.message : "The Quiz could not be recovered.");
  } finally {
    setQuizBusy(false);
  }
}

async function initializeQuizSurface() {
  await loadCurrentQuiz();
  const recoveredSubject = state.quizPayload?.subject;
  const recoveredOption = Array.from(elements.quizSubject.options).find(
    (option) => option.value === recoveredSubject && !option.disabled,
  );
  if (recoveredOption) elements.quizSubject.value = recoveredOption.value;
  if (elements.quizSubject.value) await loadQuizScopes();
  renderQuiz();
}

async function ensureQuizInitialized() {
  if (
    state.activity !== "quiz" ||
    authState.status !== "authed" ||
    state.catalogLoading ||
    state.quizInitialized
  ) {
    return;
  }
  state.quizInitialized = true;
  await initializeQuizSurface();
}

function setActivity(activity) {
  const quizSelected = activity === "quiz";
  state.activity = quizSelected ? "quiz" : "ask";
  elements.askActivity.hidden = quizSelected;
  elements.quizActivity.hidden = !quizSelected;
  elements.composerShell.hidden = quizSelected;
  elements.askTab.setAttribute("aria-selected", String(!quizSelected));
  elements.quizTab.setAttribute("aria-selected", String(quizSelected));
  if (quizSelected) void ensureQuizInitialized();
}

async function startQuiz() {
  if (state.quizBusy) return;
  if (!elements.quizSubject.value) {
    showQuizError("Choose one available subject.");
    return;
  }
  if (!elements.quizTopic.value) {
    showQuizError("Choose a Topic before starting.");
    return;
  }
  setQuizBusy(true);
  setQuizGenerationState("start");
  showQuizError("");
  try {
    state.quizPayload = await quizRequest("/api/v1/quiz/start", {
      method: "POST",
      body: JSON.stringify({
        subject: elements.quizSubject.value,
        mode: elements.mode.value,
        topic_id: elements.quizTopic.value,
        knowledge_point_id: elements.quizKnowledgePoint.value || null,
        question_count: Number(elements.quizQuestionCount.value),
      }),
    });
    renderQuiz();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(error instanceof Error ? error.message : "The Quiz could not be started.");
  } finally {
    setQuizGenerationState(null);
    setQuizBusy(false);
  }
}

async function submitQuizAnswer() {
  if (state.quizBusy || !state.quizPayload?.question) return;
  const selection = elements.quizOptions.querySelector(
    'input[name="quiz-option"]:checked',
  );
  if (!selection) {
    showQuizError("Choose one option before submitting.");
    return;
  }
  setQuizBusy(true);
  showQuizError("");
  try {
    const feedback = await quizRequest("/api/v1/quiz/current/submit", {
      method: "POST",
      body: JSON.stringify({
        question_id: state.quizPayload.question.question_id,
        option_id: selection.value,
      }),
    });
    state.quizPayload = {
      ...state.quizPayload,
      answered_count: feedback.answered_count,
      correct_count: feedback.correct_count,
      question: { ...state.quizPayload.question, submitted: true },
      feedback,
    };
    renderQuiz();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(
      error instanceof Error ? error.message : "The answer could not be submitted.",
    );
  } finally {
    setQuizBusy(false);
  }
}

async function nextQuizQuestion() {
  if (state.quizBusy) return;
  setQuizBusy(true);
  setQuizGenerationState("next");
  showQuizError("");
  try {
    state.quizPayload = await quizRequest("/api/v1/quiz/current/next", {
      method: "POST",
    });
    renderQuiz();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(
      error instanceof Error ? error.message : "The next question could not be generated.",
    );
  } finally {
    setQuizGenerationState(null);
    setQuizBusy(false);
  }
}

async function finishQuiz() {
  if (state.quizBusy) return;
  setQuizBusy(true);
  showQuizError("");
  try {
    state.quizPayload = await quizRequest("/api/v1/quiz/current/finish", {
      method: "POST",
    });
    renderQuiz();
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(error instanceof Error ? error.message : "The Quiz could not be finished.");
  } finally {
    setQuizBusy(false);
  }
}

async function exitQuiz() {
  if (state.quizBusy || !state.quizPayload) return;
  if (!window.confirm("Exit this Quiz and discard its current progress?")) return;
  setQuizBusy(true);
  showQuizError("");
  try {
    await quizRequest("/api/v1/quiz/current/exit", { method: "POST" });
    state.quizPayload = null;
    resetQuizLocalUi();
    setActivity("ask");
  } catch (error) {
    if (error instanceof SessionExpiredError) {
      handleExpiredSession(error.message);
      return;
    }
    showQuizError(error instanceof Error ? error.message : "The Quiz could not be exited.");
  } finally {
    setQuizBusy(false);
  }
}
// QUIZ_ACTIVITY_END

function syncConversationResetUi() {
  const pending = state.resettingConversation;
  const ambiguous = state.conversationResetAmbiguous;
  const quizPending = Boolean(state.quizBusy);
  const label = elements.newChat?.querySelector?.(".new-chat-label");
  const buttonLabel = pending
    ? "Starting new chat…"
    : ambiguous
      ? "Retry new chat"
      : "New chat";
  setText(label, buttonLabel);
  if (elements.newChat) {
    elements.newChat.disabled = pending || quizPending;
    elements.newChat.setAttribute("aria-label", buttonLabel);
    elements.newChat.setAttribute("aria-busy", String(pending || quizPending));
  }
  if (elements.logoutButton) {
    elements.logoutButton.disabled = pending || quizPending || authState.submitting;
  }
  if (elements.composerPreviewRemove) {
    elements.composerPreviewRemove.disabled = pending || ambiguous;
  }
  syncComposerAuthState();
}

function showConversationResetStatus(message, status) {
  setText(elements.service, message);
  const serviceLine = document.querySelector(".service-line");
  serviceLine?.classList.toggle("service-line--error", status === "error");
  serviceLine?.classList.toggle("service-line--ready", status === "ready");
}

async function resetChat() {
  if (state.isBusy || state.resettingConversation || state.quizBusy) return;
  state.conversationResetAmbiguous = false;
  state.resettingConversation = true;
  showConversationResetStatus("Starting new chat…", "pending");
  syncConversationResetUi();
  // Keep the pending state observable for a click during boot. If /auth/me is
  // still unresolved on the next turn, preserve the current thread and expose
  // an explicit retry instead of silently dropping the click.
  if (authState.status === "unknown") {
    await Promise.resolve();
    if (authState.status === "unknown") {
      state.resettingConversation = false;
      state.conversationResetAmbiguous = true;
      showConversationResetStatus(
        "New chat is still waiting for auth status. Your current chat is preserved. Retry New Chat.",
        "error",
      );
      syncConversationResetUi();
      return;
    }
  }
  try {
    if (authState.status !== "authed") {
      resetChatLocal();
      showConversationResetStatus("New chat ready.", "ready");
      return;
    }
    try {
      await resetConversationOnServer();
    } catch (error) {
      if (error instanceof SessionExpiredError) {
        state.conversationResetAmbiguous = false;
        handleExpiredSession(error.message);
      } else {
        state.conversationResetAmbiguous = true;
        showConversationResetStatus(
          error instanceof Error
            ? error.message
            : "New chat was not confirmed. Your current chat is preserved. Retry New Chat.",
          "error",
        );
      }
      return;
    }
    resetChatLocal();
    showConversationResetStatus("New chat ready.", "ready");
  } finally {
    state.resettingConversation = false;
    syncConversationResetUi();
  }
}

function populateSubjectSelect(select, subjects, preferredSubject) {
  select.replaceChildren();
  const emptySubjectOption = document.createElement("option");
  emptySubjectOption.value = "";
  emptySubjectOption.textContent = "Choose a subject";
  emptySubjectOption.selected = true;
  select.append(emptySubjectOption);
  for (const subject of subjects) {
    const ready = Boolean(subject.preview_ready);
    const option = document.createElement("option");
    option.value = subject.subject;
    option.textContent = `${subject.subject} · ${subject.name}`;
    option.dataset.name = subject.name;
    option.disabled = !ready;
    select.append(option);
  }
  const preferredOption = Array.from(select.options).find(
    (option) => option.value === preferredSubject && !option.disabled,
  );
  select.value = preferredOption?.value || "";
}

async function loadCatalog() {
  const requestId = ++state.catalogRequestId;
  const selectedMode = "preview";
  const preferredSubject = elements.subject.value;
  const preferredQuizSubject = elements.quizSubject.value;
  state.catalogLoading = true;
  setConversationLock(isConversationActive());
  let health;
  let catalog;
  let modelCatalog;
  try {
    const [healthResponse, subjectResponse, modelResponse] = await Promise.all([
      fetch(`${API}/health`),
      fetch(`${API}/subjects`),
      fetch(`${API}/models`),
    ]);
    if (!healthResponse.ok || !subjectResponse.ok || !modelResponse.ok) {
      throw new Error("The preview service is unavailable.");
    }
    [health, catalog, modelCatalog] = await Promise.all([
      healthResponse.json(),
      subjectResponse.json(),
      modelResponse.json(),
    ]);
  } catch (error) {
    if (requestId !== state.catalogRequestId || selectedMode !== elements.mode.value) return;
    throw error;
  } finally {
    if (requestId === state.catalogRequestId && selectedMode === elements.mode.value) {
      state.catalogLoading = false;
      setConversationLock(isConversationActive());
    }
  }
  if (requestId !== state.catalogRequestId || selectedMode !== elements.mode.value) return;

  populateSubjectSelect(elements.subject, catalog.subjects, preferredSubject);
  populateSubjectSelect(elements.quizSubject, catalog.subjects, preferredQuizSubject);
  if (state.subjectMentionActive && document.activeElement === elements.question) {
    renderSubjectMentionMenu();
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
  if (!state.hasAvailableModel) {
    setUnavailableModelOption();
  }
  const previewReadyCount = catalog.subjects.filter((subject) => subject.preview_ready).length;
  const subjectCount = catalog.subjects.length;
  state.catalogLoading = false;
  setConversationLock(isConversationActive());
  syncComposerAuthState();
  updateSubjectState();
  updateModelNote();
  if (!state.hasAvailableModel) {
    setText(elements.modelNote, "The local model service is unavailable or could not be checked.");
  }

  setText(elements.modeBadge, `Engineering Preview · ${previewReadyCount}/${subjectCount}`);
  setText(
    elements.modeNotice,
    `Engineering Preview · ${previewReadyCount}/${subjectCount}: all subjects are available in preview.`,
  );
  setText(
    elements.service,
    `Engineering Preview · ${previewReadyCount}/${health.subject_count} indexes ready`,
  );
  document.querySelector(".service-line").classList.add("service-line--ready");
}

function setSidebarCollapsed(collapsed) {
  elements.sidebar.classList.toggle("sidebar--collapsed", collapsed);
  elements.sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  elements.sidebarToggle.setAttribute(
    "aria-label",
    collapsed ? "Expand sidebar" : "Collapse sidebar",
  );
  setText(elements.sidebarToggleLabel, collapsed ? "Expand sidebar" : "Collapse sidebar");
  setText(elements.sidebarToggleIcon, collapsed ? "»" : "«");
}

function setSidebarDrawerState(open) {
  elements.sidebar.classList.toggle("sidebar--open", open);
  elements.menu.setAttribute("aria-expanded", String(open));
  elements.menu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
}

if (elements.mode) {
  elements.mode.addEventListener("change", () => {
    state.catalogLoading = true;
    state.quizScopesRequestId += 1;
    state.quizTopics = [];
    clearQuizScopeControls();
    syncQuizStartEnabled();
    setConversationLock(isConversationActive());
    void loadCatalog()
      .then(() => {
        if (elements.quizSubject.value) void loadQuizScopes();
      })
      .catch((error) => {
        if (!state.hasAvailableModel) setUnavailableModelOption();
        setText(elements.service, error instanceof Error ? error.message : "Service unavailable");
      });
  });
}

elements.image.addEventListener("change", () => {
  const file = elements.image.files?.[0];
  updateFileName(file);
  updateComposerPreview(file);
});

elements.composerPreviewRemove?.addEventListener("click", () => {
  if (state.resettingConversation || state.conversationResetAmbiguous) return;
  elements.image.value = "";
  updateComposerPreview(null);
  updateFileName(null);
});

elements.subject.addEventListener("change", () => {
  updateSubjectState();
});
elements.quizSubject?.addEventListener("change", () => void loadQuizScopes());

elements.quizTopic?.addEventListener("change", reconcileQuizCapacitySelection);
elements.quizKnowledgePoint?.addEventListener("change", reconcileQuizCapacitySelection);
elements.quizQuestionCount?.addEventListener("change", reconcileQuizCapacitySelection);

elements.model.addEventListener("change", () => {
  updateModelNote();
});

elements.newChat.addEventListener("click", () => {
  void resetChat();
});
elements.menu.addEventListener("click", () =>
  setSidebarDrawerState(!elements.sidebar.classList.contains("sidebar--open")),
);
elements.sidebarClose?.addEventListener("click", () => {
  setSidebarDrawerState(false);
  elements.menu.focus();
});
elements.sidebarToggle?.addEventListener("click", () =>
  setSidebarCollapsed(!elements.sidebar.classList.contains("sidebar--collapsed")),
);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (elements.mathDialog?.open) return;
  if (elements.sidebar.classList.contains("sidebar--open")) {
    setSidebarDrawerState(false);
    elements.menu.focus();
    return;
  }
  if (elements.sidebar.classList.contains("sidebar--collapsed")) {
    setSidebarCollapsed(false);
  }
});

function applyExampleQuestion(example) {
  const subjectOption = Array.from(elements.subject.options).find(
    (option) => option.value === example.subject && !option.disabled,
  );
  if (!isConversationActive() && subjectOption) {
    elements.subject.value = subjectOption.value;
    updateSubjectState();
  }
  elements.question.value = example.question;
  renderComposerMathPreview();
  elements.question.focus();
}

function renderExampleQuestions() {
  const container = document.getElementById("example-questions");
  const previousOffset = Number.parseInt(sessionStorage.getItem(EXAMPLE_ROTATION_KEY), 10);
  const offset = Number.isInteger(previousOffset)
    ? (previousOffset + 3) % EXAMPLE_QUESTIONS.length
    : Math.floor(Math.random() * EXAMPLE_QUESTIONS.length);
  sessionStorage.setItem(EXAMPLE_ROTATION_KEY, String(offset));
  container.replaceChildren();
  const examples = EXAMPLE_QUESTIONS.map(
    (_, index) => EXAMPLE_QUESTIONS[(offset + index) % EXAMPLE_QUESTIONS.length],
  );
  for (let laneIndex = 0; laneIndex < 2; laneIndex += 1) {
    const lane = document.createElement("div");
    lane.className = `suggestion-lane suggestion-lane--${laneIndex + 1}`;
    const track = document.createElement("div");
    track.className = "suggestion-track";
    const laneExamples = examples.slice(laneIndex * 4, laneIndex * 4 + 4);
    for (let groupIndex = 0; groupIndex < 2; groupIndex += 1) {
      const group = document.createElement("div");
      group.className = "suggestion-group";
      const isClone = groupIndex === 1;
      if (isClone) group.setAttribute("aria-hidden", "true");
      for (const example of laneExamples) {
        const button = document.createElement("button");
        button.className = "suggestion";
        button.type = "button";
        button.dataset.subject = example.subject;
        button.dataset.question = example.question;
        if (isClone) {
          button.setAttribute("aria-hidden", "true");
          button.tabIndex = -1;
        }
        setText(button, example.label);
        button.addEventListener("click", () => applyExampleQuestion(example));
        group.append(button);
      }
      track.append(group);
    }
    lane.append(track);
    container.append(lane);
  }
}

renderExampleQuestions();

function setMathDialogError(message) {
  if (!elements.mathDialogError) return;
  setText(elements.mathDialogError, message);
  elements.mathDialogError.hidden = !message;
}

function openMathDialog() {
  if (!elements.mathDialog || !elements.mathField) return;
  const valueLength = elements.question.value.length;
  state.mathSelectionStart = Number.isFinite(elements.question.selectionStart)
    ? elements.question.selectionStart
    : valueLength;
  state.mathSelectionEnd = Number.isFinite(elements.question.selectionEnd)
    ? elements.question.selectionEnd
    : state.mathSelectionStart;
  closeSubjectMentionMenu();
  elements.mathField.value = "";
  elements.mathDisplayMode.value = "inline";
  setMathDialogError("");
  elements.mathDialog.showModal();
  elements.mathField.focus();
}

function closeMathDialog() {
  if (elements.mathDialog?.open) elements.mathDialog.close();
}

function insertMathFormula() {
  const latex = String(elements.mathField?.value || "").trim();
  if (!latex) {
    setMathDialogError("Enter a formula first.");
    return false;
  }
  const displayMode = elements.mathDisplayMode?.value === "display";
  const renderer = globalThis.katex;
  try {
    if (!renderer || typeof renderer.renderToString !== "function") throw new Error();
    renderer.renderToString(latex, mathRenderOptions(displayMode, true));
  } catch {
    setMathDialogError("This formula is not supported by the current renderer.");
    return false;
  }
  const delimiter = displayMode ? "$$" : "$";
  const insertion = `${delimiter}${latex}${delimiter}`;
  const value = elements.question.value;
  const nextValue = `${value.slice(0, state.mathSelectionStart)}${insertion}${value.slice(state.mathSelectionEnd)}`;
  if (elements.question.maxLength > 0 && nextValue.length > elements.question.maxLength) {
    setMathDialogError(`The question must stay within ${elements.question.maxLength} characters.`);
    return false;
  }
  elements.question.value = nextValue;
  const caret = state.mathSelectionStart + insertion.length;
  elements.question.setSelectionRange(caret, caret);
  closeMathDialog();
  elements.question.dispatchEvent(new Event("input", { bubbles: true }));
  return true;
}

function hasValidMathToken(value) {
  const renderer = globalThis.katex;
  if (!renderer || typeof renderer.renderToString !== "function") return false;
  return tokenizeAnswerMath(value).some((token) => {
    if (token.type !== "math") return false;
    try {
      renderer.renderToString(token.latex, mathRenderOptions(token.display, true));
      return true;
    } catch {
      return false;
    }
  });
}

function renderComposerMathPreview() {
  if (!elements.mathPreview || !elements.mathPreviewBody) return;
  const value = elements.question.value;
  if (!hasValidMathToken(value)) {
    elements.mathPreview.hidden = true;
    elements.mathPreviewBody.replaceChildren();
    return;
  }
  renderAnswerBody(elements.mathPreviewBody, value);
  elements.mathPreview.hidden = false;
}

function subjectMentionContext() {
  const cursor = Number.isFinite(elements.question.selectionStart)
    ? elements.question.selectionStart
    : elements.question.value.length;
  const before = elements.question.value.slice(0, cursor);
  const match = before.match(/(?:^|\s)@([A-Za-z0-9_-]*)$/);
  if (!match) return null;
  return {
    start: before.length - match[0].length + match[0].lastIndexOf("@"),
    end: cursor,
    query: match[1].toLowerCase(),
  };
}

function closeSubjectMentionMenu() {
  state.subjectMentionActive = false;
  state.subjectMentionIndex = -1;
  state.subjectMentionOptions = [];
  elements.subjectMentionMenu?.replaceChildren();
  if (elements.subjectMentionMenu) elements.subjectMentionMenu.hidden = true;
}

function renderSubjectMentionMenu() {
  const menu = elements.subjectMentionMenu;
  const context = subjectMentionContext();
  if (!menu || !context) {
    closeSubjectMentionMenu();
    return;
  }
  state.subjectMentionActive = true;
  const matchingOptions = Array.from(elements.subject.options || []).filter(
    (option) =>
      !option.disabled &&
      option.value &&
      `${option.value} ${option.textContent}`.toLowerCase().includes(context.query),
  );
  const exactCodeMatches = matchingOptions.filter(
    (option) => option.value.toLowerCase() === context.query,
  );
  const options = exactCodeMatches.length ? exactCodeMatches : matchingOptions;
  state.subjectMentionOptions = options.map((option) => option.value);
  menu.replaceChildren();
  if (options.length === 0) {
    state.subjectMentionIndex = -1;
    menu.hidden = true;
    return;
  }
  if (state.subjectMentionIndex < 0 || state.subjectMentionIndex >= options.length) {
    state.subjectMentionIndex = 0;
  }
  options.forEach((option, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "subject-mention-option";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(index === state.subjectMentionIndex));
    setText(button, `${option.value} · ${option.dataset.name || option.textContent}`);
    button.addEventListener("mousedown", (event) => event.preventDefault());
    button.addEventListener("click", () => selectSubjectMention(option.value));
    menu.append(button);
  });
  menu.hidden = false;
}

function selectSubjectMention(subject) {
  const context = subjectMentionContext();
  const option = Array.from(elements.subject.options || []).find(
    (candidate) => candidate.value === subject && !candidate.disabled,
  );
  if (!context || !option) return;
  const value = elements.question.value;
  const replacement = `@${option.value} `;
  elements.question.value = `${value.slice(0, context.start)}${replacement}${value.slice(context.end)}`;
  elements.subject.value = option.value;
  updateSubjectState();
  closeSubjectMentionMenu();
  const cursor = context.start + replacement.length;
  elements.question.setSelectionRange(cursor, cursor);
  elements.question.focus();
  renderComposerMathPreview();
}

function handleSubjectMentionKeydown(event) {
  if (event.key === "Escape" && state.subjectMentionActive) {
    event.preventDefault();
    closeSubjectMentionMenu();
    return true;
  }
  if (!state.subjectMentionOptions.length || elements.subjectMentionMenu?.hidden) return false;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const direction = event.key === "ArrowDown" ? 1 : -1;
    state.subjectMentionIndex =
      (state.subjectMentionIndex + direction + state.subjectMentionOptions.length) %
      state.subjectMentionOptions.length;
    renderSubjectMentionMenu();
    return true;
  }
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    selectSubjectMention(state.subjectMentionOptions[state.subjectMentionIndex]);
    return true;
  }
  return false;
}

elements.question.addEventListener("input", () => {
  if (subjectMentionContext()) {
    renderSubjectMentionMenu();
  } else {
    closeSubjectMentionMenu();
  }
  renderComposerMathPreview();
});

elements.question.addEventListener("blur", closeSubjectMentionMenu);

elements.question.addEventListener("keydown", (event) => {
  if (typeof handleSubjectMentionKeydown === "function" && handleSubjectMentionKeydown(event)) return;
  if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) {
    return;
  }
  event.preventDefault();
  if (typeof elements.form.requestSubmit === "function") {
    elements.form.requestSubmit();
  } else {
    elements.form.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

elements.formulaButton?.addEventListener("click", openMathDialog);
elements.mathDialogClose?.addEventListener("click", closeMathDialog);
elements.mathDialogCancel?.addEventListener("click", closeMathDialog);
elements.mathDialogInsert?.addEventListener("click", insertMathFormula);
elements.mathDialog?.addEventListener("close", () => {
  setMathDialogError("");
  elements.question.focus();
});
elements.mathDialog?.addEventListener("cancel", () => setMathDialogError(""));

if (elements.authForm) {
  elements.authForm.addEventListener("submit", (event) => {
    void submitAccessKey(event);
  });
}

if (elements.logoutButton) {
  elements.logoutButton.addEventListener("click", () => {
    void logoutSession();
  });
}

elements.quizStart?.addEventListener("click", () => void startQuiz());
elements.quizSubmit?.addEventListener("click", () => void submitQuizAnswer());
elements.quizNext?.addEventListener("click", () => void nextQuizQuestion());
elements.quizFinish?.addEventListener("click", () => void finishQuiz());
elements.quizExit?.addEventListener("click", () => void exitQuiz());
elements.askTab?.addEventListener("click", () => setActivity("ask"));
elements.quizTab?.addEventListener("click", () => setActivity("quiz"));

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isBusy || state.resettingConversation || state.conversationResetAmbiguous) return;
  if (authState.status === "gated" || authState.status === "config_error") return;

  if (state.restoredArchiveId && !(await prepareArchiveContinuation())) return;

  const question = elements.question.value.trim();
  if (!question) return;

  const imageFile = elements.image.files?.[0] || null;
  const subjectOption = elements.subject.selectedOptions[0];
  const modelOption = elements.model.selectedOptions[0];
  const turn = {
    turnId: makeOpaqueId(),
    question,
    imageFile,
    subject: elements.subject.value,
    subjectLabel: subjectOption?.textContent?.trim() || elements.subject.value,
    mode: elements.mode?.value || "preview",
    model: elements.model.value,
    modelLabel: modelOption?.textContent?.trim() || elements.model.value,
    createdAt: Date.now(),
    status: "loading",
  };

  state.turns.push(turn);
  resetComposerFields();
  await submitTurn(turn, { appendUserBubble: true });
  // Save the completed transcript after every submitted turn so Recent Chats
  // stays current even if the page is closed without using New chat.
  archiveCurrentChat();
});

initTheme();
configureMathField();
renderRecentChats();
setActivity("ask");
renderQuiz();
void bootApp();

async function bootApp() {
  const authStatus = await checkAuthStatus();
  if (authStatus === "config_error") return;
  if (authStatus !== "gated") {
    void loadCircuitGallery();
    void loadLocalNoteGallery();
  }
  try {
    await loadCatalog();
    await ensureQuizInitialized();
  } catch (error) {
    if (!state.hasAvailableModel) setUnavailableModelOption();
    setText(elements.service, error instanceof Error ? error.message : "Service unavailable");
    document.querySelector(".service-line").classList.add("service-line--error");
  }
}
