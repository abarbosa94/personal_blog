"use strict";

const editableFields = [
  "review_status",
  "reviewed_english",
  "reviewed_portuguese",
  "review_note",
];

const statusLabels = {
  needs_review: "Unreviewed",
  defer: "Deferred",
  accept: "Accepted",
  localized: "Localized",
  exclude: "Excluded",
};

const elements = {};
const state = {
  items: [],
  filtered: [],
  currentId: null,
  saveTimer: null,
  savePromise: null,
  undoStack: [],
};

function element(id) {
  return document.getElementById(id);
}

function remember(key, value) {
  try {
    window.localStorage.setItem(`translation-review:${key}`, value);
  } catch (_error) {
    // The app remains fully functional when storage is disabled.
  }
}

function recall(key, fallback = "") {
  try {
    return window.localStorage.getItem(`translation-review:${key}`) || fallback;
  } catch (_error) {
    return fallback;
  }
}

function currentItem() {
  return state.items.find((item) => item.alignment_id === state.currentId) || null;
}

function editableSnapshot(item) {
  return Object.fromEntries(editableFields.map((field) => [field, item[field] || ""]));
}

function formSnapshot(status = null) {
  return {
    review_status: status ?? currentItem()?.review_status ?? "needs_review",
    reviewed_english: elements.reviewedEnglish.value,
    reviewed_portuguese: elements.reviewedPortuguese.value,
    review_note: elements.reviewNote.value,
  };
}

function changesFrom(item, values) {
  return Object.fromEntries(
    editableFields
      .filter((field) => (item[field] || "") !== values[field])
      .map((field) => [field, values[field]]),
  );
}

function setSaveStatus(message, kind = "normal") {
  elements.saveStatus.textContent = message;
  elements.saveStatus.classList.toggle("error", kind === "error");
}

function showValidation(message = "") {
  elements.validationMessage.textContent = message;
  elements.validationMessage.hidden = !message;
  elements.reviewNote.classList.toggle("invalid", Boolean(message));
  if (message) elements.reviewNote.focus();
}

function validate(values) {
  const note = values.review_note.trim();
  if (values.review_status === "exclude" && !note) {
    return "Add a short rationale before excluding this alignment.";
  }
  if ((values.reviewed_english.trim() || values.reviewed_portuguese.trim()) && !note) {
    return "Explain the text override in the reviewer note before saving.";
  }
  return "";
}

function updateNoteHint(values) {
  const requiresNote = values.review_status === "exclude"
    || values.reviewed_english.trim()
    || values.reviewed_portuguese.trim();
  elements.noteHint.textContent = requiresNote
    ? "Required for exclusions and text overrides."
    : "Explain corrections, exclusions, or uncertainty.";
}

function updateDirtyState() {
  const item = currentItem();
  if (!item) return;
  const values = formSnapshot();
  updateNoteHint(values);
  const fieldMap = [
    [elements.reviewedEnglish, "reviewed_english"],
    [elements.reviewedPortuguese, "reviewed_portuguese"],
    [elements.reviewNote, "review_note"],
  ];
  for (const [control, field] of fieldMap) {
    control.classList.toggle("dirty", control.value !== (item[field] || ""));
  }
  if (!validate(values)) showValidation();
}

async function patchItem(id, changes) {
  const response = await fetch(`/api/items/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
  if (!response.ok) throw new Error(payload.error || `Save failed (${response.status})`);
  const index = state.items.findIndex((item) => item.alignment_id === id);
  if (index >= 0) state.items[index] = payload.item;
  return payload.item;
}

async function saveCurrentEdits({ announce = true } = {}) {
  window.clearTimeout(state.saveTimer);
  state.saveTimer = null;
  if (state.savePromise) {
    try {
      await state.savePromise;
    } catch (_error) {
      return false;
    }
    return saveCurrentEdits({ announce });
  }

  const item = currentItem();
  if (!item) return true;
  const id = item.alignment_id;
  const values = formSnapshot();
  const validation = validate(values);
  if (validation) {
    showValidation(validation);
    setSaveStatus("Not saved — a rationale is required", "error");
    return false;
  }
  const changes = changesFrom(item, values);
  if (!Object.keys(changes).length) {
    if (announce) setSaveStatus("All changes saved");
    updateDirtyState();
    return true;
  }

  setSaveStatus("Saving…");
  state.savePromise = patchItem(id, changes);
  try {
    await state.savePromise;
    showValidation();
    if (state.currentId === id) updateDirtyState();
    if (announce) setSaveStatus("Changes saved");
    updateProgress();
    return true;
  } catch (error) {
    showValidation(error.message);
    setSaveStatus(`Not saved — ${error.message}`, "error");
    return false;
  } finally {
    state.savePromise = null;
  }
}

function scheduleSave() {
  updateDirtyState();
  setSaveStatus("Unsaved edits…");
  window.clearTimeout(state.saveTimer);
  state.saveTimer = window.setTimeout(() => void saveCurrentEdits(), 650);
}

function updateProgress() {
  const total = state.items.length;
  const terminalStatuses = new Set(["accept", "localized", "exclude"]);
  const reviewed = state.items.filter((item) => terminalStatuses.has(item.review_status)).length;
  const remaining = total - reviewed;
  elements.progressLabel.textContent = `${reviewed} of ${total} reviewed`;
  elements.remainingLabel.textContent = `${remaining} remaining`;
  elements.progressBar.style.width = total ? `${(reviewed / total) * 100}%` : "0%";
  elements.progressBar.parentElement.setAttribute("aria-label", `${reviewed} of ${total} reviewed`);
}

function selectedFilters() {
  return {
    status: elements.statusFilter.value,
    priority: elements.priorityFilter.value,
  };
}

function itemMatchesFilters(item, filters) {
  const statusMatches = filters.status === "all"
    || item.review_status === filters.status
    || (filters.status === "pending" && ["needs_review", "defer"].includes(item.review_status));
  return statusMatches
    && (filters.priority === "all" || item.review_priority === filters.priority);
}

function rebuildFiltered(preferredId = null) {
  const filters = selectedFilters();
  remember("status-filter", filters.status);
  remember("priority-filter", filters.priority);
  state.filtered = state.items.filter((item) => itemMatchesFilters(item, filters));

  const candidate = preferredId || state.currentId;
  if (!state.filtered.some((item) => item.alignment_id === candidate)) {
    state.currentId = state.filtered[0]?.alignment_id || null;
  } else {
    state.currentId = candidate;
  }
  render();
}

function numberText(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function setBadge(target, text, variant = "") {
  target.textContent = text;
  target.className = `badge${variant ? ` badge-${variant}` : ""}`;
}

function render() {
  updateProgress();
  const item = currentItem();
  const visibleIndex = state.filtered.findIndex((row) => row.alignment_id === state.currentId);
  const hasItem = Boolean(item && visibleIndex >= 0);
  elements.emptyState.hidden = hasItem || state.filtered.length > 0;
  elements.reviewCard.hidden = !hasItem;
  if (!hasItem) {
    if (!state.filtered.length) setSaveStatus("No items match the current filters");
    return;
  }

  remember("current-id", item.alignment_id);
  setBadge(elements.alignmentBadge, item.alignment_id, "id");
  setBadge(
    elements.priorityBadge,
    item.review_priority === "high" ? "High priority" : "Normal priority",
    item.review_priority === "high" ? "high" : "",
  );
  setBadge(elements.typeBadge, `${item.alignment_type} alignment`);
  setBadge(elements.similarityBadge, `LaBSE ${numberText(item.labse_similarity)}`);
  setBadge(
    elements.statusBadge,
    statusLabels[item.review_status] || item.review_status,
    item.review_status === "needs_review" ? "" : item.review_status,
  );

  elements.warningText.textContent = item.automatic_warning || "";
  elements.warningText.hidden = !item.automatic_warning;
  elements.positionLabel.textContent = `${visibleIndex + 1} of ${state.filtered.length}`;
  elements.englishSentenceIds.textContent = item.english_sentence_ids;
  elements.portugueseSentenceIds.textContent = item.portuguese_sentence_ids;
  elements.englishText.textContent = item.english;
  elements.portugueseText.textContent = item.portuguese;
  elements.reviewedEnglish.value = item.reviewed_english || "";
  elements.reviewedPortuguese.value = item.reviewed_portuguese || "";
  elements.reviewNote.value = item.review_note || "";
  elements.cellPair.textContent = `EN ${item.english_cell} · PT ${item.portuguese_cell}`;
  elements.pairId.textContent = item.pair_id;
  elements.transitionScore.textContent = numberText(item.transition_score);
  elements.rawSimilarity.textContent = numberText(item.labse_similarity, 4);
  elements.previousButton.disabled = visibleIndex === 0;
  elements.nextButton.disabled = visibleIndex === state.filtered.length - 1;
  for (const button of document.querySelectorAll(".decision")) {
    button.classList.toggle("active", button.dataset.status === item.review_status);
  }
  showValidation();
  updateDirtyState();
  setSaveStatus("All changes saved");
}

async function navigate(offset) {
  const saved = await saveCurrentEdits({ announce: false });
  if (!saved) return;
  const index = state.filtered.findIndex((item) => item.alignment_id === state.currentId);
  const target = state.filtered[index + offset];
  if (!target) return;
  state.currentId = target.alignment_id;
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextCandidateId(ids, currentIndex) {
  return ids[currentIndex + 1] || ids[currentIndex - 1] || null;
}

async function decide(status) {
  window.clearTimeout(state.saveTimer);
  state.saveTimer = null;
  if (state.savePromise) {
    try {
      await state.savePromise;
    } catch (_error) {
      return;
    }
  }
  const item = currentItem();
  if (!item) return;

  const values = formSnapshot(status);
  updateNoteHint(values);
  const validation = validate(values);
  if (validation) {
    showValidation(validation);
    setSaveStatus("Decision not saved — add a rationale", "error");
    return;
  }

  const before = editableSnapshot(item);
  const changes = changesFrom(item, values);
  if (!Object.keys(changes).length) {
    await navigate(1);
    return;
  }
  const ids = state.filtered.map((row) => row.alignment_id);
  const oldIndex = ids.indexOf(item.alignment_id);
  const nextId = nextCandidateId(ids, oldIndex);
  setSaveStatus("Saving decision…");
  try {
    await patchItem(item.alignment_id, changes);
    state.undoStack.push({ id: item.alignment_id, values: before });
    if (state.undoStack.length > 50) state.undoStack.shift();
    showValidation();
    rebuildFiltered(nextId);
    setSaveStatus(`${statusLabels[status]} — saved`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    showValidation(error.message);
    setSaveStatus(`Decision not saved — ${error.message}`, "error");
  }
}

async function undoLastDecision() {
  const action = state.undoStack.pop();
  if (!action) {
    setSaveStatus("Nothing to undo");
    return;
  }
  setSaveStatus("Undoing…");
  try {
    await patchItem(action.id, action.values);
    const restored = state.items.find((item) => item.alignment_id === action.id);
    const filters = selectedFilters();
    const matchesFilters = restored && itemMatchesFilters(restored, filters);
    if (!matchesFilters) {
      elements.statusFilter.value = "all";
      elements.priorityFilter.value = "all";
    }
    rebuildFiltered(action.id);
    setSaveStatus("Last decision undone");
  } catch (error) {
    state.undoStack.push(action);
    setSaveStatus(`Undo failed — ${error.message}`, "error");
  }
}

async function jumpToId(rawId) {
  const id = rawId.trim().toLowerCase();
  const item = state.items.find((row) => row.alignment_id.toLowerCase() === id);
  if (!item) {
    setSaveStatus(`Unknown alignment ID: ${rawId.trim() || "(empty)"}`, "error");
    elements.jumpInput.focus();
    return;
  }
  const saved = await saveCurrentEdits({ announce: false });
  if (!saved) return;
  elements.statusFilter.value = "all";
  elements.priorityFilter.value = "all";
  state.currentId = item.alignment_id;
  rebuildFiltered(item.alignment_id);
  elements.jumpInput.value = "";
}

async function nextUnreviewed() {
  const saved = await saveCurrentEdits({ announce: false });
  if (!saved) return;
  const currentIndex = state.items.findIndex((item) => item.alignment_id === state.currentId);
  const ordered = [...state.items.slice(currentIndex + 1), ...state.items.slice(0, currentIndex + 1)];
  const target = ordered.find((item) => item.review_status === "needs_review");
  if (!target) {
    setSaveStatus("Every alignment has a decision");
    return;
  }
  elements.statusFilter.value = "needs_review";
  elements.priorityFilter.value = "all";
  rebuildFiltered(target.alignment_id);
}

function isEditingTarget(target) {
  return target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
    || target.isContentEditable;
}

function onKeydown(event) {
  const modifier = event.ctrlKey || event.metaKey;
  if (modifier && event.key.toLowerCase() === "s") {
    event.preventDefault();
    void saveCurrentEdits();
    return;
  }
  if (modifier && event.key === "Enter") {
    event.preventDefault();
    void saveCurrentEdits({ announce: false }).then((saved) => saved && navigate(1));
    return;
  }
  if (elements.shortcutsDialog.open || isEditingTarget(event.target)) return;

  const key = event.key.toLowerCase();
  const actions = {
    arrowleft: () => navigate(-1),
    p: () => navigate(-1),
    arrowright: () => navigate(1),
    n: () => navigate(1),
    "1": () => decide("accept"),
    "2": () => decide("localized"),
    "3": () => decide("exclude"),
    d: () => decide("defer"),
    u: () => undoLastDecision(),
    "?": () => elements.shortcutsDialog.showModal(),
  };
  if (actions[key]) {
    event.preventDefault();
    void actions[key]();
  }
}

function bindEvents() {
  elements.statusFilter.addEventListener("change", () => rebuildFiltered());
  elements.priorityFilter.addEventListener("change", () => rebuildFiltered());
  elements.clearFiltersButton.addEventListener("click", () => {
    elements.statusFilter.value = "all";
    elements.priorityFilter.value = "all";
    rebuildFiltered();
  });
  elements.previousButton.addEventListener("click", () => void navigate(-1));
  elements.nextButton.addEventListener("click", () => void navigate(1));
  elements.nextUnreviewedButton.addEventListener("click", () => void nextUnreviewed());
  elements.jumpButton.addEventListener("click", () => void jumpToId(elements.jumpInput.value));
  elements.jumpInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void jumpToId(elements.jumpInput.value);
    }
  });
  elements.shortcutsButton.addEventListener("click", () => elements.shortcutsDialog.showModal());
  for (const button of document.querySelectorAll(".decision")) {
    button.addEventListener("click", () => void decide(button.dataset.status));
  }
  for (const control of [elements.reviewedEnglish, elements.reviewedPortuguese, elements.reviewNote]) {
    control.addEventListener("input", scheduleSave);
    control.addEventListener("blur", () => void saveCurrentEdits());
  }
  document.addEventListener("keydown", onKeydown);
}

async function initialize() {
  for (const id of [
    "progressLabel", "remainingLabel", "progressBar", "statusFilter", "priorityFilter",
    "jumpInput", "jumpButton", "alignmentIds", "nextUnreviewedButton", "shortcutsButton",
    "emptyState", "clearFiltersButton", "reviewCard", "alignmentBadge", "priorityBadge",
    "typeBadge", "similarityBadge", "statusBadge", "warningText", "positionLabel",
    "englishSentenceIds", "portugueseSentenceIds", "englishText", "portugueseText",
    "reviewedEnglish", "reviewedPortuguese", "reviewNote", "noteHint", "validationMessage",
    "cellPair", "pairId", "transitionScore", "rawSimilarity", "previousButton", "nextButton",
    "saveStatus", "shortcutsDialog",
  ]) {
    elements[id] = element(id);
  }
  bindEvents();
  elements.statusFilter.value = recall("status-filter", "pending");
  elements.priorityFilter.value = recall("priority-filter", "all");

  try {
    const response = await fetch("/api/items", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Load failed (${response.status})`);
    state.items = payload.items;
    for (const item of state.items) {
      const option = document.createElement("option");
      option.value = item.alignment_id;
      elements.alignmentIds.append(option);
    }
    const rememberedId = recall("current-id");
    state.currentId = state.items.some((item) => item.alignment_id === rememberedId)
      ? rememberedId
      : state.items.find((item) => item.review_status === "needs_review")?.alignment_id
        || state.items[0]?.alignment_id
        || null;
    rebuildFiltered(state.currentId);
  } catch (error) {
    elements.reviewCard.hidden = true;
    elements.emptyState.hidden = false;
    elements.emptyState.querySelector("h2").textContent = "The review data could not be loaded.";
    setSaveStatus(error.message, "error");
  }
}

document.addEventListener("DOMContentLoaded", () => void initialize());
