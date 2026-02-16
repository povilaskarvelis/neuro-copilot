const state = {
  tasks: [],
  selectedTaskId: null,
  selectedDetail: null,
  activeRunId: null,
  pollTimer: null,
  isLoading: false,
  health: null,
  expanded: false,
  activityExpanded: false,
  latestRun: null,
  clarificationMessage: "",
  localEventsByTask: {},
  exportingPdf: false,
  reportStatusTaskId: null,
  reportStatusText: "",
  reportStatusError: false,
  pendingUserMessage: "",
};

const el = {
  workspace: document.getElementById("workspace"),
  tasksList: document.getElementById("tasksList"),
  taskTitle: document.getElementById("taskTitle"),
  messages: document.getElementById("messages"),
  promptInput: document.getElementById("promptInput"),
  sendBtn: document.getElementById("sendBtn"),
  composerForm: document.getElementById("composerForm"),
  newChatBtn: document.getElementById("newChatBtn"),
  notice: document.getElementById("notice"),
  loadingChip: document.getElementById("loadingChip"),
  activityCard: document.getElementById("activityCard"),
  activityWheel: document.getElementById("activityWheel"),
  activityTitle: document.getElementById("activityTitle"),
  activityStatus: document.getElementById("activityStatus"),
  activitySummary: document.getElementById("activitySummary"),
  activityPreview: document.getElementById("activityPreview"),
  activityDetails: document.getElementById("activityDetails"),
  reportPanel: document.getElementById("reportPanel"),
  reportTitle: document.getElementById("reportTitle"),
  reportStatus: document.getElementById("reportStatus"),
  reportContent: document.getElementById("reportContent"),
  exportPdfBtn: document.getElementById("exportPdfBtn"),
};

function formatDate(iso) {
  if (!iso) return "unknown";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function formatTime(iso) {
  if (!iso) return "unknown";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function toStatusClass(status) {
  return `status-${String(status || "").replace(/\s+/g, "_")}`;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return value;
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;
  let codeBuffer = [];

  const closeLists = () => {
    if (inUl) {
      html.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      html.push("</ol>");
      inOl = false;
    }
  };

  for (const rawLine of lines) {
    const line = String(rawLine || "");
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeLists();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
        codeBuffer = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      codeBuffer.push(line);
      continue;
    }

    if (!trimmed) {
      closeLists();
      continue;
    }

    const h3 = trimmed.match(/^###\s+(.+)$/);
    if (h3) {
      closeLists();
      html.push(`<h3>${inlineMarkdown(h3[1])}</h3>`);
      continue;
    }
    const h2 = trimmed.match(/^##\s+(.+)$/);
    if (h2) {
      closeLists();
      html.push(`<h2>${inlineMarkdown(h2[1])}</h2>`);
      continue;
    }
    const h1 = trimmed.match(/^#\s+(.+)$/);
    if (h1) {
      closeLists();
      html.push(`<h1>${inlineMarkdown(h1[1])}</h1>`);
      continue;
    }

    const quote = trimmed.match(/^>\s+(.+)$/);
    if (quote) {
      closeLists();
      html.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    const ul = trimmed.match(/^[-*]\s+(.+)$/);
    if (ul) {
      if (inOl) {
        html.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        html.push("<ul>");
        inUl = true;
      }
      html.push(`<li>${inlineMarkdown(ul[1])}</li>`);
      continue;
    }

    const ol = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (ol) {
      if (inUl) {
        html.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        const start = Number(ol[1] || "1");
        html.push(Number.isFinite(start) && start > 1 ? `<ol start="${start}">` : "<ol>");
        inOl = true;
      }
      html.push(`<li>${inlineMarkdown(ol[2])}</li>`);
      continue;
    }

    closeLists();
    html.push(`<p>${inlineMarkdown(trimmed)}</p>`);
  }

  closeLists();
  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
  }
  return html.join("");
}

function setExpanded(expanded) {
  state.expanded = !!expanded;
  document.body.classList.toggle("expanded", state.expanded);
  updatePromptPlaceholder();
}

function updatePromptPlaceholder() {
  el.promptInput.placeholder = state.expanded ? "Type a follow-up..." : "Ask anything.";
}

async function api(path, options = {}) {
  const config = {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  };
  const res = await fetch(path, config);
  const raw = await res.text();
  let parsed = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;
  } catch {
    parsed = raw;
  }
  if (!res.ok) {
    const detail =
      (parsed && parsed.detail) || (typeof parsed === "string" ? parsed : `HTTP ${res.status}`);
    throw new Error(detail);
  }
  return parsed;
}

function setNotice(message = "", isError = false) {
  if (!message) {
    el.notice.classList.add("hidden");
    el.notice.textContent = "";
    return;
  }
  el.notice.classList.remove("hidden");
  el.notice.textContent = message;
  el.notice.style.borderColor = isError ? "#4c232c" : "#303030";
  el.notice.style.background = isError ? "#1a0c10" : "#121212";
  el.notice.style.color = isError ? "#ffccd6" : "#f1f1f1";
}

function updateSendVisibility() {
  const hasText = el.promptInput.value.trim().length > 0;
  el.sendBtn.classList.toggle("hidden", !hasText);
  el.sendBtn.disabled = !hasText || state.isLoading || !state.health?.ok;
}

function setLoading(isLoading) {
  state.isLoading = isLoading;
  el.loadingChip.classList.toggle("hidden", !isLoading);
  updateSendVisibility();
}

function getScopeDecompositionText() {
  const stepOne = state.selectedDetail?.task?.steps?.[0];
  if (!stepOne || !stepOne.output) return "";
  return String(stepOne.output).trim();
}

function addLocalEvent(taskId, role, header, body) {
  if (!taskId || !body) return;
  if (!state.localEventsByTask[taskId]) state.localEventsByTask[taskId] = [];
  state.localEventsByTask[taskId].push({
    role,
    header,
    body,
    at: new Date().toISOString(),
  });
  state.localEventsByTask[taskId] = state.localEventsByTask[taskId].slice(-20);
}

function normalizePlanLine(text) {
  let value = String(text || "").trim();
  value = value.replace(/\*\*(.*?)\*\*/g, "$1");
  value = value.replace(/`([^`]+)`/g, "$1");
  value = value.replace(/^\[(.+?)\]\((.+?)\)$/g, "$1");
  value = value.replace(/\s+/g, " ").trim();
  return value;
}

function compactInstruction(text) {
  const cleaned = normalizePlanLine(String(text || ""));
  if (!cleaned) return "";
  const sentences = cleaned.split(/(?<=[.!?])\s+/).filter(Boolean);
  const firstSentence = sentences[0] || cleaned;
  const secondSentence = sentences[1] || "";
  const base = /^Prioritize user-requested tools:/i.test(firstSentence) && secondSentence
    ? `${firstSentence} ${secondSentence}`
    : firstSentence;
  const compact = base.replace(/^Use tools to\s+/i, "Gather ").replace(/^Produce /i, "Deliver ");
  return compact.length > 140 ? `${compact.slice(0, 137).trimEnd()}...` : compact;
}

function inlinePlanDetail(detail) {
  const value = normalizePlanLine(String(detail || ""));
  if (!value) return "";
  if (value.length < 2) return value.toLowerCase();
  const first = value[0];
  const second = value[1];
  if (/[A-Z]/.test(first) && /[a-z]/.test(second)) {
    return `${first.toLowerCase()}${value.slice(1)}`;
  }
  return value;
}

function buildPlanStepsFromVersion(task, version) {
  const steps = [];
  if (!version || !Array.isArray(version.steps)) return steps;

  const baseFrom = Number.isFinite(Number(version.base_from_step_index))
    ? Number(version.base_from_step_index)
    : 0;
  const frozenPrefix = Array.isArray(task?.steps) ? task.steps.slice(0, Math.max(0, baseFrom)) : [];

  for (const step of frozenPrefix) {
    steps.push({
      title: String(step?.title || "").trim(),
      instruction: String(step?.instruction || "").trim(),
      completed: true,
      recommendedTools: Array.isArray(step?.recommended_tools) ? step.recommended_tools : [],
      fallbackTools: Array.isArray(step?.fallback_tools) ? step.fallback_tools : [],
      allowedTools: Array.isArray(step?.allowed_tools) ? step.allowed_tools : [],
    });
  }
  for (const step of version.steps) {
    steps.push({
      title: String(step?.title || "").trim(),
      instruction: String(step?.instruction || "").trim(),
      completed: false,
      recommendedTools: Array.isArray(step?.recommended_tools) ? step.recommended_tools : [],
      fallbackTools: Array.isArray(step?.fallback_tools) ? step.fallback_tools : [],
      allowedTools: Array.isArray(step?.allowed_tools) ? step.allowed_tools : [],
    });
  }
  return steps;
}

function fallbackPlanSteps(task) {
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  return steps.map((step) => ({
    title: String(step?.title || "").trim(),
    instruction: String(step?.instruction || "").trim(),
    completed: String(step?.status || "") === "completed",
    recommendedTools: Array.isArray(step?.recommended_tools) ? step.recommended_tools : [],
    fallbackTools: Array.isArray(step?.fallback_tools) ? step.fallback_tools : [],
    allowedTools: Array.isArray(step?.allowed_tools) ? step.allowed_tools : [],
  }));
}

function isSynthesisPlanStep(step, idx, total) {
  const title = normalizePlanLine(String(step?.title || "").toLowerCase());
  const instruction = normalizePlanLine(String(step?.instruction || "").toLowerCase());
  if (/(synthes|report|final|recommend|summary)/.test(`${title} ${instruction}`)) return true;
  return idx === total - 1;
}

function selectPrimaryToolForPlanStep(step, idx, total) {
  if (!step) return "";
  if (isSynthesisPlanStep(step, idx, total)) return "";
  const candidates = uniqueNonEmpty(
    [
      ...(Array.isArray(step?.recommendedTools) ? step.recommendedTools : []),
      ...(Array.isArray(step?.allowedTools) ? step.allowedTools : []),
      ...(Array.isArray(step?.fallbackTools) ? step.fallbackTools : []),
    ].map((tool) => toolKeyName(tool) || normalizePlanLine(tool))
  );
  return candidates[0] || "";
}

function formatPlanText(task, version, { delta = null, reason = "" } = {}) {
  const steps = buildPlanStepsFromVersion(task, version);
  const planSteps = steps.length ? steps : fallbackPlanSteps(task);
  const lines = [];

  if (delta) {
    const summary = normalizePlanLine(String(delta?.summary || ""));
    if (summary) lines.push(`Updated from your feedback: ${summary}.`);
    const changed = [];
    if (Array.isArray(delta?.added_steps) && delta.added_steps.length) changed.push(`added: ${delta.added_steps.join(", ")}`);
    if (Array.isArray(delta?.modified_steps) && delta.modified_steps.length) changed.push(`modified: ${delta.modified_steps.join(", ")}`);
    if (Array.isArray(delta?.reordered_steps) && delta.reordered_steps.length) changed.push(`reordered: ${delta.reordered_steps.join(", ")}`);
    if (Array.isArray(delta?.removed_steps) && delta.removed_steps.length) changed.push(`removed: ${delta.removed_steps.join(", ")}`);
    if (changed.length) lines.push(`Changes: ${changed.join(" | ")}`);
    lines.push("");
  }

  planSteps.forEach((step, idx) => {
    const title = step.title || `Step ${idx + 1}`;
    const detail = compactInstruction(step.instruction) || "Execute this step with the defined constraints.";
    lines.push(`${idx + 1}. ${title}`);
    lines.push(`- ${detail}`);

    const primaryTool = selectPrimaryToolForPlanStep(step, idx, planSteps.length);
    if (primaryTool) {
      const source = displayToolSourceName(primaryTool);
      const action = toolUsageAction(primaryTool, {});
      lines.push(`- Primary tool/source: ${source} (used to ${action})`);
    } else {
      lines.push("- Primary method: structured synthesis to produce the final report-ready output");
    }
  });

  if (reason) {
    lines.push("");
    lines.push(`Gate reason: ${reason}`);
  }

  return lines.join("\n");
}

function extractSubtasks(scopeText) {
  const lines = String(scopeText || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const subtasks = [];
  for (const line of lines) {
    const cleaned = normalizePlanLine(line);
    const numbered = cleaned.match(/^\d+[).:-]?\s+(.+)$/);
    const bulleted = cleaned.match(/^[-*]\s+(.+)$/);
    let candidate = numbered?.[1] || bulleted?.[1] || "";
    candidate = normalizePlanLine(candidate);
    candidate = candidate.replace(/^[A-Za-z][A-Za-z\s/-]{1,24}:\s+/, "");
    if (!candidate || candidate.length < 4) continue;
    subtasks.push(candidate);
    if (subtasks.length >= 3) break;
  }
  return subtasks.length >= 2 ? subtasks : [];
}

function formatGateReason(reason) {
  const value = String(reason || "").trim();
  if (!value) return "";
  const map = {
    pre_evidence_execution: "Before bulk evidence collection",
    quality_gap_spike: "Quality/uncertainty spike detected",
    repeated_tool_failures: "Repeated tool failures detected",
    uncertainty_spike: "Uncertainty spike detected",
    pre_final_after_intent_change: "Intent changed before final synthesis",
    feedback_replan: "Plan updated from your feedback",
    queued_feedback_pending: "Queued feedback is ready to apply",
  };
  return map[value] || value.replaceAll("_", " ");
}

function inspectLatestPauseContext(task) {
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  if (!steps.length) {
    return {
      stepTitle: "",
      failure: null,
      criticalLine: "",
      highPriorityCriterion: false,
    };
  }

  let candidate = null;
  for (let idx = steps.length - 1; idx >= 0; idx -= 1) {
    const step = steps[idx];
    if (String(step?.status || "") !== "completed") continue;
    candidate = step;
    break;
  }
  if (!candidate) {
    return {
      stepTitle: "",
      failure: null,
      criticalLine: "",
      highPriorityCriterion: false,
    };
  }

  const title = String(candidate.title || "").trim() || "latest step";
  const titleLower = title.toLowerCase();
  const highPriorityCriterion = /(human genetics|genetic|safety liabilities|safety|risk signals|weighted criteria)/i.test(
    titleLower
  );
  const trace = Array.isArray(candidate.tool_trace) ? candidate.tool_trace : [];
  let bestFailure = null;
  let bestScore = -1;
  for (const entry of trace) {
    const outcome = String(entry?.outcome || "").trim().toLowerCase();
    if (!["error", "not_found_or_empty", "no_response", "degraded"].includes(outcome)) continue;
    const tool = String(entry?.tool_name || "a tool").trim();
    const detailRaw = normalizePlanLine(String(entry?.detail || ""));
    const detail = detailRaw ? detailRaw.slice(0, 180) : "";
    const detailLower = detail.toLowerCase();
    let score = 1;
    if (detailLower.includes("gwas")) score += 4;
    if (detailLower.includes("rate limit") || detailLower.includes("429")) score += 4;
    if (detailLower.includes("service unavailable") || detailLower.includes("unavailable")) score += 3;
    if (detailLower.includes("timed out") || detailLower.includes("timeout")) score += 3;
    if (outcome === "error") score += 2;
    if (outcome === "degraded") score += 1;
    if (score > bestScore) {
      bestScore = score;
      bestFailure = { tool, outcome, detail };
    }
  }
  if (bestFailure) {
    return {
      stepTitle: title,
      failure: bestFailure,
      criticalLine: "",
      highPriorityCriterion,
    };
  }

  const output = String(candidate.output || "");
  if (!output.trim()) {
    return {
      stepTitle: title,
      failure: null,
      criticalLine: "",
      highPriorityCriterion,
    };
  }
  const lines = output
    .split("\n")
    .map((line) => normalizePlanLine(line))
    .filter(Boolean);
  const marker =
    /(critical gap|service unavailable|failed|unable to retrieve|unresolved|limitation|contradict|error|aborted|timed out|timeout|rate limit|429)/i;
  const hit = lines.find((line) => marker.test(line));
  return {
    stepTitle: title,
    failure: null,
    criticalLine: hit ? hit.slice(0, 190) : "",
    highPriorityCriterion,
  };
}

function _failureReasonText(failure) {
  if (!failure) return "";
  const tool = String(failure.tool || "a required tool");
  const outcome = String(failure.outcome || "").toLowerCase();
  const detail = normalizePlanLine(String(failure.detail || ""));
  const lower = detail.toLowerCase();

  if (lower.includes("gwas") && (lower.includes("service unavailable") || lower.includes("unavailable"))) {
    return `${tool} could not reach the GWAS service`;
  }
  if (lower.includes("rate limit") || lower.includes("429")) {
    return `${tool} hit a rate limit`;
  }
  if (lower.includes("timed out") || lower.includes("timeout")) {
    return `${tool} timed out`;
  }
  if (lower.includes("service unavailable") || lower.includes("unavailable")) {
    return `${tool} could not reach an upstream service`;
  }
  if (lower.includes("aborted")) {
    return `${tool} request was aborted`;
  }
  if (outcome === "degraded") {
    return `${tool} returned degraded evidence`;
  }
  if (outcome === "no_response") {
    return `${tool} returned no response`;
  }
  if (outcome === "not_found_or_empty") {
    return `${tool} returned no usable data`;
  }
  if (outcome === "error") {
    return `${tool} returned an error`;
  }
  return `${tool} returned ${outcome || "an issue"}`;
}

function _buildEvidencePauseMessage(context, continueLabel) {
  const stepLabel = String(context?.stepTitle || "The latest evidence step");
  const failure = context?.failure || null;
  const criticalLine = String(context?.criticalLine || "").trim();
  const highPriority = Boolean(context?.highPriorityCriterion);

  let lead = "";
  if (failure) {
    const reason = _failureReasonText(failure);
    lead = `${stepLabel} could not be fully completed because ${reason}.`;
  } else if (criticalLine) {
    lead = `${stepLabel} surfaced a critical evidence gap: ${criticalLine}.`;
  } else {
    lead = `${stepLabel} has an unresolved evidence gap.`;
  }

  const importance = highPriority
    ? "This affects a high-priority criterion in the current decision."
    : "This affects one of the planned evidence paths."
  const continueImplication = `${continueLabel} will finish the report using available fallback evidence and clearly mark this part as provisional.`;
  const feedbackOption = "If you want a stricter evidence requirement or a different evidence path, send feedback before continuing.";
  return `${lead} ${importance} ${continueImplication} ${feedbackOption}`;
}

function checkpointPauseMessage(reason, task, continueLabel = "Continue") {
  const value = String(reason || "").trim().toLowerCase();
  const context = inspectLatestPauseContext(task);
  if (!value || value === "pre_evidence_execution" || value === "feedback_replan") return "";
  if (value === "quality_gap_spike" || value === "uncertainty_spike") {
    return _buildEvidencePauseMessage(context, continueLabel);
  }
  if (value === "repeated_tool_failures") {
    return _buildEvidencePauseMessage(context, continueLabel);
  }
  if (value === "pre_final_after_intent_change") {
    return `Your feedback changed the plan right before final synthesis. ${continueLabel} will finalize using the updated plan, or you can send one more refinement now.`;
  }
  if (value === "queued_feedback_pending") {
    return `Queued feedback is ready to apply at this checkpoint. You can send more feedback now, or use ${continueLabel} to proceed.`;
  }
  return `Checkpoint reason: ${formatGateReason(value)}. ${continueLabel} will proceed with the current evidence and keep any remaining gaps explicit, or you can send feedback to revise the plan.`;
}

function shouldRenderCheckpointAction(reason) {
  const value = String(reason || "").trim().toLowerCase();
  if (!value) return false;
  if (value === "pre_evidence_execution" || value === "feedback_replan") return false;
  return true;
}

function checkpointActionHtml(buttonLabel = "Continue") {
  return `
    <article class="message assistant checkpoint-action-message">
      <button class="primary-btn checkpoint-start-btn" data-action="checkpoint-start">${escapeHtml(buttonLabel)}</button>
    </article>
  `;
}

function checkpointHtml(planText, awaitingHitl, buttonLabel = "Start research") {
  const body = markdownToHtml(planText || "");
  const startButton = awaitingHitl
    ? `<button class="primary-btn checkpoint-start-btn" data-action="checkpoint-start">${escapeHtml(buttonLabel)}</button>`
    : "";
  return `
    <article class="message assistant checkpoint-message">
      <div class="message-body markdown-body">${body}</div>
      ${startButton}
    </article>
  `;
}

function primaryObjectiveText(text) {
  let value = String(text || "").trim();
  const markers = [
    "\nUser revision to scope/decomposition:",
    "\nRevision directives to apply:",
    "\nRequired revision constraints:",
    "\nUser clarification:",
    "\nUse this clarification as the intended meaning for ambiguous abbreviations.",
  ];
  for (const marker of markers) {
    if (value.includes(marker)) {
      value = value.split(marker, 1)[0].trim();
    }
  }
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines[0] || value;
}

function taskDisplayTitle(task) {
  const title = normalizePlanLine(String(task?.title || "").trim());
  if (title) return title;
  const objective = primaryObjectiveText(String(task?.objective || "").trim());
  return normalizePlanLine(objective) || "Untitled";
}

function summarizeRevisionTitle(text, maxWords = 12) {
  let value = normalizePlanLine(String(text || ""));
  if (!value) return "";
  value = value.replace(/^(please|also)\s+/i, "");
  value = value.replace(/^(make sure|ensure|can you)\s+/i, "");
  value = value.replace(/\s+/g, " ").trim();
  value = value.split(/[.!?]/, 1)[0].trim();
  if (!value) return "";
  const words = value.split(" ").filter(Boolean);
  if (words.length <= maxWords) return value;
  return `${words.slice(0, maxWords).join(" ")}...`;
}

function latestRevisionSummary(task) {
  const events = Array.isArray(task?.hitl_history) ? task.hitl_history : [];
  for (let idx = events.length - 1; idx >= 0; idx -= 1) {
    const raw = String(events[idx] || "").trim();
    if (!raw.toLowerCase().startsWith("revise:")) continue;
    const summary = summarizeRevisionTitle(raw.slice(7).trim());
    if (summary) return summary;
  }
  return "";
}

function compactReportTitle(text, maxChars = 110) {
  const value = normalizePlanLine(String(text || ""));
  if (!value) return "Research Report";
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars - 3).trim()}...`;
}

function reportProjectTitle(task) {
  const base = taskDisplayTitle(task) || "Research Report";
  const revision = latestRevisionSummary(task);
  if (!revision) return compactReportTitle(base);
  if (base.toLowerCase().includes(revision.toLowerCase())) return compactReportTitle(base);
  return compactReportTitle(`${base} — ${revision}`);
}

function inferProgress(run, task = null) {
  if (!run) return 0;
  const status = String(run.status || "");
  if (["awaiting_hitl", "completed", "failed", "needs_clarification"].includes(status)) return 100;
  if (status === "queued") return 6;
  if (status !== "running") return 12;

  const events = Array.isArray(run?.progress_events)
    ? run.progress_events
    : Array.isArray(task?.progress_events)
      ? task.progress_events
      : [];
  if (!events.length) {
    const logs = Array.isArray(run.logs) ? run.logs : [];
    let progress = 12;
    for (const entry of logs) {
      const msg = String(entry?.message || "").toLowerCase();
      if (msg.includes("checking for ambiguity")) progress = Math.max(progress, 18);
      if (msg.includes("routing intent")) progress = Math.max(progress, 32);
      if (msg.includes("executing workflow step 1")) progress = Math.max(progress, 52);
      if (msg.includes("running remaining workflow steps")) progress = Math.max(progress, 68);
      if (msg.includes("quality gate")) progress = Math.max(progress, 88);
    }
    return Math.min(97, progress);
  }

  const latest = events[events.length - 1] || {};
  const phase = String(latest?.phase || "").toLowerCase();
  const phaseBase = {
    intake: 16,
    plan: 30,
    execute: 48,
    search: 58,
    analyze: 68,
    synthesize: 82,
    checkpoint: 92,
    finalize: 95,
  };
  let progress = phaseBase[phase] || 20;

  const stepEvents = events.filter((event) => String(event?.type || "") === "step.completed");
  const lastStepEvent = stepEvents.length ? stepEvents[stepEvents.length - 1] : null;
  const metrics = lastStepEvent?.metrics || {};
  const completed = Number(metrics?.steps_completed || 0);
  const total = Number(metrics?.steps_total || 0);
  if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
    progress = Math.max(progress, Math.floor((Math.min(completed, total) / total) * 88));
  }

  return Math.min(97, Math.max(progress, 14));
}

function getRunStatusLabel(run) {
  const status = String(run?.status || "");
  if (status === "running") return "Running";
  if (status === "completed") return "Complete";
  if (status === "awaiting_hitl") return "Checkpoint";
  if (status === "needs_clarification") return "Needs Input";
  if (status === "failed") return "Failed";
  if (status === "queued") return "Queued";
  return "Idle";
}

function toolSourceName(toolName) {
  const raw = String(toolName || "").trim().toLowerCase();
  if (!raw) return "";

  const segments = raw.split(/[./:]/).filter(Boolean);
  const key = segments.length ? segments[segments.length - 1] : raw;

  const explicitSources = {
    search_diseases: "Open Targets",
    search_disease_targets: "Open Targets",
    get_target_info: "Open Targets",
    search_targets: "Open Targets",
    check_druggability: "Open Targets",
    get_target_drugs: "Open Targets",
    search_clinical_trials: "ClinicalTrials.gov",
    get_clinical_trial: "ClinicalTrials.gov",
    summarize_clinical_trials_landscape: "ClinicalTrials.gov",
    search_chembl_compounds_for_target: "ChEMBL",
    search_pubmed: "PubMed",
    get_pubmed_abstract: "PubMed",
    search_pubmed_advanced: "PubMed",
    get_pubmed_paper_details: "PubMed",
    get_pubmed_author_profile: "PubMed",
    search_openalex_works: "OpenAlex",
    search_openalex_authors: "OpenAlex",
    rank_researchers_by_activity: "OpenAlex",
    get_researcher_contact_candidates: "OpenAlex",
    get_gene_info: "NCBI Gene",
    search_clinvar_variants: "ClinVar",
    get_clinvar_variant_details: "ClinVar",
    search_gwas_associations: "GWAS Catalog",
    search_reactome_pathways: "Reactome",
    get_string_interactions: "STRING",
    expand_disease_context: "EBI OLS (EFO/MONDO)",
    summarize_target_expression_context: "Open Targets",
    infer_genetic_effect_direction: "GWAS Catalog",
    summarize_target_competitive_landscape: "Open Targets",
    summarize_target_safety_liabilities: "Open Targets",
    compare_targets_multi_axis: "Open Targets",
    list_local_datasets: "Local datasets",
    read_local_dataset: "Local datasets",
  };
  if (explicitSources[key]) return explicitSources[key];

  if (key.includes("pubmed")) return "PubMed";
  if (key.includes("openalex")) return "OpenAlex";
  if (key.includes("clinical") || key.includes("nct")) return "ClinicalTrials.gov";
  if (key.includes("chembl")) return "ChEMBL";
  if (key.includes("gwas")) return "GWAS Catalog";
  if (key.includes("clinvar")) return "ClinVar";
  if (key.includes("reactome")) return "Reactome";
  if (key.includes("string")) return "STRING";
  if (key.includes("target") || key.includes("druggability") || key.includes("expression") || key.includes("safety")) {
    return "Open Targets";
  }
  if (key.includes("gene")) return "NCBI Gene";
  if (key.includes("local")) return "Local datasets";
  return "";
}

function toolKeyName(toolName) {
  const raw = String(toolName || "").trim().toLowerCase();
  if (!raw) return "";
  const segments = raw.split(/[./:]/).filter(Boolean);
  return segments.length ? segments[segments.length - 1] : raw;
}

function compactArgValue(value, maxLen = 76) {
  if (value === null || value === undefined) return "";
  let text = "";
  if (Array.isArray(value)) {
    text = value
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .slice(0, 4)
      .join(", ");
  } else if (typeof value === "object") {
    try {
      text = JSON.stringify(value);
    } catch {
      text = String(value);
    }
  } else {
    text = String(value);
  }
  const normalized = normalizePlanLine(text);
  if (!normalized) return "";
  return normalized.length > maxLen ? `${normalized.slice(0, maxLen - 3).trimEnd()}...` : normalized;
}

function pickArg(args, keys) {
  const payload = args && typeof args === "object" ? args : {};
  for (const key of keys) {
    const value = compactArgValue(payload[key]);
    if (value) return value;
  }
  return "";
}

function toolUsageAction(toolName, args) {
  const key = toolKeyName(toolName);
  const query = pickArg(args, ["query", "search", "topic"]);
  const target = pickArg(args, ["targetId", "geneSymbol", "target", "query"]);
  const disease = pickArg(args, ["diseaseQuery", "diseaseId", "diseaseFilter", "query"]);
  const fromYear = pickArg(args, ["fromYear"]);
  const pmid = pickArg(args, ["pmid"]);
  const nctId = pickArg(args, ["nctId"]);
  const clinvarId = pickArg(args, ["clinvarId"]);
  const identifier = pickArg(args, ["identifier", "authorId", "authorName"]);

  switch (key) {
    case "search_diseases":
      return query ? `resolve disease entities/IDs for "${query}"` : "resolve disease entities/IDs";
    case "search_disease_targets":
      return disease ? `retrieve disease-associated targets for "${disease}"` : "retrieve disease-associated targets";
    case "get_target_info":
      return target ? `fetch target profile for "${target}"` : "fetch target profile";
    case "search_targets":
      return query ? `find target entities matching "${query}"` : "find target entities";
    case "check_druggability":
      return target ? `assess tractability for "${target}"` : "assess tractability";
    case "get_target_drugs":
      return target ? `retrieve known-drug evidence for "${target}"` : "retrieve known-drug evidence";
    case "search_clinical_trials":
      return query ? `search trials for "${query}"` : "search clinical trials";
    case "get_clinical_trial":
      return nctId ? `retrieve trial details for ${nctId}` : "retrieve trial details";
    case "summarize_clinical_trials_landscape":
      return query ? `summarize trial landscape for "${query}"` : "summarize trial landscape";
    case "search_chembl_compounds_for_target":
      return query ? `retrieve compound potency evidence for "${query}"` : "retrieve compound potency evidence";
    case "search_pubmed":
    case "search_pubmed_advanced":
      return query ? `search literature for "${query}"` : "search literature";
    case "get_pubmed_abstract":
      return pmid ? `retrieve abstract for PMID ${pmid}` : "retrieve paper abstract";
    case "get_pubmed_paper_details":
      return pmid ? `retrieve paper metadata/authors for PMID ${pmid}` : "retrieve paper metadata/authors";
    case "get_pubmed_author_profile":
      return identifier ? `summarize publication profile for "${identifier}"` : "summarize publication profile";
    case "search_openalex_works":
      return query ? `retrieve topic-matched works for "${query}"` : "retrieve topic-matched works";
    case "search_openalex_authors":
      return query ? `find author entities for "${query}"` : "find author entities";
    case "rank_researchers_by_activity":
      return query
        ? `rank researcher activity from topic-matched works for "${query}"${fromYear ? ` (from ${fromYear})` : ""}`
        : `rank researcher activity from topic-matched works${fromYear ? ` (from ${fromYear})` : ""}`;
    case "get_researcher_contact_candidates":
      return identifier ? `retrieve researcher profile/contact signals for "${identifier}"` : "retrieve researcher profile/contact signals";
    case "get_gene_info":
      return target ? `retrieve gene profile for "${target}"` : "retrieve gene profile";
    case "search_clinvar_variants":
      return query ? `retrieve variant records for "${query}"` : "retrieve variant records";
    case "get_clinvar_variant_details":
      return clinvarId ? `retrieve ClinVar record ${clinvarId}` : "retrieve ClinVar record details";
    case "search_gwas_associations":
      return query ? `retrieve GWAS associations for "${query}"` : "retrieve GWAS associations";
    case "search_reactome_pathways":
      return query ? `retrieve pathway matches for "${query}"` : "retrieve pathway matches";
    case "get_string_interactions":
      return identifier ? `retrieve interaction network for "${identifier}"` : "retrieve interaction network";
    case "expand_disease_context":
      return query ? `expand ontology terms/synonyms for "${query}"` : "expand ontology terms/synonyms";
    case "summarize_target_expression_context":
      return target ? `summarize tissue/cell expression context for "${target}"` : "summarize tissue/cell expression context";
    case "infer_genetic_effect_direction":
      return target || disease
        ? `estimate risk/protective genetic direction for ${target || "target"} in "${disease || "disease context"}"`
        : "estimate risk/protective genetic direction";
    case "summarize_target_competitive_landscape":
      return target || disease
        ? `summarize competitive landscape for ${target || "target"}${disease ? ` in "${disease}"` : ""}`
        : "summarize competitive landscape";
    case "summarize_target_safety_liabilities":
      return target ? `summarize safety liabilities for "${target}"` : "summarize safety liabilities";
    case "compare_targets_multi_axis":
      return disease ? `compare/rank targets in "${disease}"` : "compare/rank targets across multiple axes";
    case "list_local_datasets":
      return "list available local datasets";
    case "read_local_dataset":
      return identifier ? `inspect local dataset "${identifier}"` : "inspect local dataset";
    default: {
      const fallback = normalizePlanLine(key.replace(/[_-]+/g, " "));
      if (query) return `${fallback} for "${query}"`;
      if (target) return `${fallback} for "${target}"`;
      if (identifier) return `${fallback} for "${identifier}"`;
      return fallback || "run analysis operation";
    }
  }
}

function buildUsageLinesFromTrace(traceEntries, maxLines = 5) {
  const safeTrace = Array.isArray(traceEntries) ? traceEntries : [];
  const out = [];
  const seen = new Set();
  for (const entry of safeTrace) {
    const source = displayToolSourceName(entry?.tool_name || "");
    const action = toolUsageAction(entry?.tool_name || "", entry?.args || {});
    const outcome = toolOutcomeText(entry?.outcome);
    const line = `${source}: used to ${action} (${outcome}).`;
    const key = line.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(line);
    if (out.length >= maxLines) break;
  }
  return out;
}

function displayToolSourceName(toolName) {
  const raw = String(toolName || "").trim();
  if (!raw) return "Unknown source";
  const mapped = toolSourceName(raw);
  if (mapped) return mapped;
  const normalized = raw.toLowerCase();
  if (normalized.includes("compare") || normalized.includes("summarize") || normalized.includes("infer")) {
    return "Internal analysis engine";
  }
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\b[a-z]/g, (char) => char.toUpperCase());
}

function toolOutcomeText(outcomeRaw) {
  const outcome = String(outcomeRaw || "").trim().toLowerCase();
  if (outcome === "ok") return "success";
  if (outcome === "error") return "failed";
  if (outcome === "degraded") return "partial result";
  if (outcome === "not_found_or_empty") return "no usable data";
  if (outcome === "no_response") return "no response";
  if (outcome === "pending") return "pending";
  return outcome || "unknown";
}

function orderTraceEntries(traceEntries) {
  const safeTrace = Array.isArray(traceEntries) ? traceEntries : [];
  return safeTrace
    .map((entry, idx) => ({
      ...entry,
      _seq: Number.isFinite(Number(entry?.sequence)) ? Number(entry.sequence) : idx + 1,
      _idx: idx,
    }))
    .sort((a, b) => (a._seq === b._seq ? a._idx - b._idx : a._seq - b._seq));
}

function traceIssueCount(traceEntries) {
  let issues = 0;
  for (const entry of traceEntries || []) {
    const outcome = String(entry?.outcome || "").toLowerCase();
    if (["error", "not_found_or_empty", "no_response", "degraded"].includes(outcome)) issues += 1;
  }
  return issues;
}

function uniqueNonEmpty(items) {
  const out = [];
  const seen = new Set();
  for (const item of items) {
    const value = String(item || "").trim();
    if (!value) continue;
    const key = value.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(value);
  }
  return out;
}

function normalizeStepCompletionLabel(rawText, fallbackTitle = "") {
  const raw = normalizePlanLine(rawText || "");
  if (!raw) {
    return fallbackTitle ? `${normalizePlanLine(fallbackTitle)} complete` : "";
  }
  const stepMatch = raw.match(/^Step\s+\d+\s+completed:\s*(.+?)\.?$/i);
  if (stepMatch) {
    const title = normalizePlanLine(stepMatch[1]);
    return title ? `${title} complete` : "";
  }
  const alreadyComplete = raw.match(/^(.+?)\s+complete(?:d)?\.?$/i);
  if (alreadyComplete) {
    const title = normalizePlanLine(alreadyComplete[1]);
    return title ? `${title} complete` : "";
  }
  if (fallbackTitle) {
    return `${normalizePlanLine(fallbackTitle)} complete`;
  }
  return raw;
}

function collectCompletedStepLabels(task, events) {
  const fromTask = [];
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  for (const step of steps) {
    if (String(step?.status || "") !== "completed") continue;
    const title = normalizePlanLine(step?.title || "");
    if (title) fromTask.push(`${title} complete`);
  }
  if (fromTask.length) return uniqueNonEmpty(fromTask);

  const fromEvents = [];
  const safeEvents = Array.isArray(events) ? events : [];
  for (const event of safeEvents) {
    if (String(event?.type || "") !== "step.completed") continue;
    const label = normalizeStepCompletionLabel(
      String(event?.human_line || ""),
      String(event?.step_title || "")
    );
    if (label) fromEvents.push(label);
  }
  return uniqueNonEmpty(fromEvents);
}

function collectNextStepLabels(task) {
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  const labels = [];
  for (const step of steps) {
    const status = String(step?.status || "");
    if (status === "completed") continue;
    const title = normalizePlanLine(step?.title || "");
    if (!title) continue;
    labels.push(`${title} pending`);
    if (labels.length >= 2) break;
  }
  return labels;
}

function resolveStepProgress(task, events, completedLabels) {
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  const taskTotal = steps.length;
  const taskCompleted = steps.filter((step) => String(step?.status || "") === "completed").length;
  if (taskTotal > 0) {
    return {
      completed: taskCompleted,
      total: taskTotal,
    };
  }

  let eventTotal = 0;
  let eventCompleted = 0;
  const safeEvents = Array.isArray(events) ? events : [];
  for (const event of safeEvents) {
    if (String(event?.type || "") !== "step.completed") continue;
    const metrics = event?.metrics || {};
    const total = Number(metrics?.steps_total || 0);
    const completed = Number(metrics?.steps_completed || 0);
    if (Number.isFinite(total) && total > eventTotal) eventTotal = total;
    if (Number.isFinite(completed) && completed > eventCompleted) eventCompleted = completed;
  }
  if (eventTotal > 0) {
    return {
      completed: Math.min(eventCompleted, eventTotal),
      total: eventTotal,
    };
  }

  const fallbackCompleted = Array.isArray(completedLabels) ? completedLabels.length : 0;
  return {
    completed: fallbackCompleted,
    total: fallbackCompleted,
  };
}

function buildWorkflowMarkdown(task) {
  const steps = Array.isArray(task?.steps) ? task.steps : [];
  if (!steps.length) return "";

  const lines = ["### Workflow"];
  steps.forEach((step, idx) => {
    const title = normalizePlanLine(step?.title || `Step ${idx + 1}`);
    const statusRaw = String(step?.status || "pending").toLowerCase();
    const status = statusRaw === "completed"
      ? "complete"
      : statusRaw === "in_progress"
        ? "in progress"
        : statusRaw === "blocked"
          ? "blocked"
          : "pending";

    const orderedTrace = orderTraceEntries(step?.tool_trace);
    const usedSources = uniqueNonEmpty(
      orderedTrace.map((entry) => displayToolSourceName(entry?.tool_name || ""))
    );
    const usageLines = buildUsageLinesFromTrace(orderedTrace, idx === 1 ? 7 : 4);
    const totalCalls = orderedTrace.length;
    const successCalls = orderedTrace.filter((entry) => String(entry?.outcome || "").toLowerCase() === "ok").length;
    const issueCalls = traceIssueCount(orderedTrace);
    const evidenceRefs = uniqueNonEmpty(
      (Array.isArray(step?.evidence_refs) ? step.evidence_refs : []).map((ref) => String(ref || "").trim())
    );

    lines.push(`${idx + 1}. **${title}** (${status})`);
    if (totalCalls > 0) {
      lines.push(`- Tools used: ${usedSources.join(", ")}.`);
      for (const usageLine of usageLines) {
        lines.push(`- ${usageLine}`);
      }
      lines.push(
        `- Tool activity: ${totalCalls} call${totalCalls === 1 ? "" : "s"} (${successCalls} success${issueCalls ? `, ${issueCalls} with issues` : ""}).`
      );
      const firstIssue = orderedTrace.find((entry) =>
        ["error", "not_found_or_empty", "no_response", "degraded"].includes(String(entry?.outcome || "").toLowerCase())
      );
      if (firstIssue) {
        const source = displayToolSourceName(firstIssue?.tool_name || "");
        const action = toolUsageAction(firstIssue?.tool_name || "", firstIssue?.args || {});
        const issue = toolOutcomeText(firstIssue?.outcome);
        const detail = normalizePlanLine(String(firstIssue?.detail || ""));
        const detailSuffix = detail ? `: ${detail.length > 120 ? `${detail.slice(0, 117).trimEnd()}...` : detail}` : "";
        lines.push(`- First issue encountered: ${source} while trying to ${action} (${issue})${detailSuffix}`);
      }
    } else {
      lines.push("- Tools used: None recorded.");
      lines.push("- Tool activity: No external tool calls recorded.");
    }
    if (evidenceRefs.length) {
      const shownRefs = evidenceRefs.slice(0, 4);
      const remainder = evidenceRefs.length - shownRefs.length;
      lines.push(`- Evidence refs: ${shownRefs.join(", ")}${remainder > 0 ? ` (+${remainder} more)` : ""}.`);
    }

    if (idx === 1) {
      lines.push("- Step 2 tool sequence:");
      if (!orderedTrace.length) {
        lines.push("- No tool calls were captured for this step.");
      } else {
        orderedTrace.forEach((entry, traceIdx) => {
          const source = displayToolSourceName(entry?.tool_name || "");
          const action = toolUsageAction(entry?.tool_name || "", entry?.args || {});
          const outcome = toolOutcomeText(entry?.outcome);
          const phaseRaw = String(entry?.phase || "").trim().toLowerCase();
          const phase = phaseRaw && phaseRaw !== "main" ? ` (${phaseRaw.replace(/_/g, " ")})` : "";
          const detailRaw = normalizePlanLine(String(entry?.detail || ""));
          const detail = detailRaw && outcome !== "success"
            ? ` - ${detailRaw.length > 120 ? `${detailRaw.slice(0, 117).trimEnd()}...` : detailRaw}`
            : "";
          lines.push(`- Attempt ${traceIdx + 1}: **${source}** used to ${action} - ${outcome}${phase}${detail}`);
        });

        const pivotNotes = [];
        for (let pivotIdx = 0; pivotIdx < orderedTrace.length - 1; pivotIdx += 1) {
          const current = orderedTrace[pivotIdx];
          const next = orderedTrace[pivotIdx + 1];
          const outcome = String(current?.outcome || "").toLowerCase();
          if (!["error", "not_found_or_empty", "no_response", "degraded"].includes(outcome)) continue;
          const fromSource = displayToolSourceName(current?.tool_name || "");
          const toSource = displayToolSourceName(next?.tool_name || "");
          if (fromSource === toSource) continue;
          pivotNotes.push(
            `Attempt ${pivotIdx + 1}: pivoted from ${fromSource} to ${toSource} after a blocked/low-quality result.`
          );
        }
        const dedupedStepPivots = uniqueNonEmpty(pivotNotes);
        if (dedupedStepPivots.length) {
          lines.push("- Pivots during Step 2:");
          for (const note of dedupedStepPivots.slice(0, 8)) {
            lines.push(`- ${note}`);
          }
        }
      }
    }
  });

  const pivotNotes = [];
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx += 1) {
    const step = steps[stepIdx];
    const trace = Array.isArray(step?.tool_trace) ? step.tool_trace : [];
    for (let idx = 0; idx < trace.length - 1; idx += 1) {
      const current = trace[idx];
      const next = trace[idx + 1];
      const outcome = String(current?.outcome || "").toLowerCase();
      if (!["error", "not_found_or_empty", "no_response", "degraded"].includes(outcome)) continue;
      const fromSource = displayToolSourceName(current?.tool_name) || "one source";
      const toSource = displayToolSourceName(next?.tool_name) || "another source";
      if (fromSource === toSource) continue;
      const stepTitle = normalizePlanLine(step?.title || "step");
      pivotNotes.push(
        `Step ${stepIdx + 1} (${stepTitle}): pivoted from ${fromSource} to ${toSource} after a blocked/low-quality result.`
      );
    }
  }
  const dedupedPivots = uniqueNonEmpty(pivotNotes);
  if (dedupedPivots.length) {
    lines.push("");
    lines.push("### Pivots Across Workflow");
    for (const note of dedupedPivots.slice(0, 6)) {
      lines.push(`- ${note}`);
    }
  }

  return lines.join("\n");
}

function mountActivityCardInMessages() {
  if (!state.expanded) return;
  if (!el.messages || !el.activityCard) return;
  if (el.activityCard.parentElement !== el.messages) {
    el.messages.appendChild(el.activityCard);
    return;
  }
  el.messages.appendChild(el.activityCard);
}

function renderActivity() {
  const run = state.latestRun;
  const selectedTask = state.selectedDetail?.task || null;
  const runEvents = Array.isArray(run?.progress_events) ? run.progress_events : [];
  const runSummaries = Array.isArray(run?.progress_summaries) ? run.progress_summaries : [];
  const taskEvents = Array.isArray(selectedTask?.progress_events) ? selectedTask.progress_events : [];
  const taskSummaries = Array.isArray(selectedTask?.progress_summaries) ? selectedTask.progress_summaries : [];
  const hasRunLogs = !!(run && Array.isArray(run.logs) && run.logs.length > 0);
  const runStatus = String(run?.status || "");
  const taskStatus = String(selectedTask?.status || "");
  const hasAnyRunState = Boolean(run && (runStatus || runEvents.length || runSummaries.length || hasRunLogs));
  const shouldShow = Boolean(state.expanded && (state.isLoading || selectedTask || hasAnyRunState));

  if (!state.expanded || !shouldShow) {
    el.activityCard.classList.add("hidden");
    return;
  }

  mountActivityCardInMessages();
  el.activityCard.classList.remove("hidden");
  const activeEvents = runEvents.length ? runEvents : taskEvents;
  const activeSummaries = runSummaries.length ? runSummaries : taskSummaries;
  const latestSummary = activeSummaries.length ? activeSummaries[activeSummaries.length - 1] : null;
  const lastEvent = activeEvents.length ? activeEvents[activeEvents.length - 1] : null;
  const lastEventLine = String(lastEvent?.human_line || "").trim();
  const completedLabels = collectCompletedStepLabels(selectedTask, activeEvents);
  const nextLabels = collectNextStepLabels(selectedTask);
  const stepProgress = resolveStepProgress(selectedTask, activeEvents, completedLabels);
  const progressDerivedComplete =
    stepProgress.total > 0 && stepProgress.completed >= stepProgress.total;

  const effectiveStatus = runStatus || taskStatus;
  const isComplete = effectiveStatus === "completed" || progressDerivedComplete;
  const isError = effectiveStatus === "failed";
  const isPausedForCheckpoint =
    !isComplete && (runStatus === "awaiting_hitl" || Boolean(selectedTask?.awaiting_hitl));
  const isRunning = !isComplete && !isError && !isPausedForCheckpoint;

  el.activityCard.classList.toggle("is-running", isRunning);
  el.activityCard.classList.toggle("is-complete", isComplete);
  el.activityCard.classList.toggle("is-error", isError);
  el.activityCard.classList.toggle("is-paused", isPausedForCheckpoint);

  const titleText = isComplete
    ? "Research log"
    : (String(latestSummary?.headline || "").trim() || lastEventLine || "Preparing research steps");
  el.activityTitle.textContent = titleText;
  let statusLabel = run ? getRunStatusLabel(run || {}) : "History";
  if (!run && taskStatus) {
    statusLabel = getRunStatusLabel({ status: taskStatus });
  }
  if (isComplete) statusLabel = "Complete";
  if (isPausedForCheckpoint) statusLabel = "Checkpoint";
  if (isError) statusLabel = "Failed";
  el.activityStatus.textContent = statusLabel;

  let summary = "Tracking workflow progress.";
  if (runStatus === "running") summary = "Working through workflow steps";
  if (isComplete) {
    if (completedLabels.length) {
      const displayed = completedLabels.slice(0, 3);
      const remaining = completedLabels.length - displayed.length;
      summary = `${displayed.join(" • ")}${remaining > 0 ? ` • +${remaining} more` : ""}`;
    } else if (stepProgress.total > 0) {
      summary = `Workflow complete: ${stepProgress.completed}/${stepProgress.total} steps complete.`;
    } else {
      summary = "Workflow complete.";
    }
  }
  if (isPausedForCheckpoint) summary = "Waiting for checkpoint input";
  if (runStatus === "needs_clarification") summary = "Clarification required to continue";
  if (runStatus === "failed") summary = "Execution failed";
  if (!runStatus && taskStatus === "completed" && !isComplete) summary = "Completed run history is available.";
  if (!isComplete && latestSummary?.summary) summary = String(latestSummary.summary);
  el.activitySummary.textContent = summary;

  if (completedLabels.length) {
    const displayed = completedLabels.slice(0, 4);
    const remaining = completedLabels.length - displayed.length;
    el.activityPreview.textContent = `Completed: ${displayed.join(" • ")}${remaining > 0 ? ` • +${remaining} more` : ""}`;
  } else if (nextLabels.length) {
    el.activityPreview.textContent = `Next: ${nextLabels.join(" • ")}`;
  } else {
    const preview = activeEvents
      .slice(-2)
      .map((event) => String(event?.human_line || "").trim())
      .filter(Boolean);
    el.activityPreview.textContent = preview.length ? preview.join(" • ") : "Click for activity details";
  }

  const details = [];
  const workflowMarkdown = buildWorkflowMarkdown(selectedTask);
  if (workflowMarkdown) {
    details.push(workflowMarkdown);
  } else {
    details.push("### Workflow");
    details.push("- Workflow details are not available yet.");
  }
  el.activityDetails.innerHTML = markdownToHtml(details.join("\n"));
  el.activityCard.classList.toggle("expanded", state.activityExpanded);
  el.activityCard.setAttribute("aria-expanded", state.activityExpanded ? "true" : "false");
  el.activityDetails.classList.toggle("hidden", !state.activityExpanded);
}

function renderTasks() {
  if (!state.tasks.length) {
    el.tasksList.innerHTML = '<p class="muted">No chats yet.</p>';
    return;
  }
  el.tasksList.innerHTML = state.tasks
    .map((task) => {
      const active = task.task_id === state.selectedTaskId ? "active" : "";
      const shortObjective = escapeHtml(taskDisplayTitle(task));
      const status = escapeHtml(task.status || "unknown");
      return `
        <article class="task-item ${active}" data-task-id="${escapeHtml(task.task_id)}">
          <div class="task-line">
            <span><span class="status-dot ${toStatusClass(status)}"></span>${status}</span>
            <span>${formatDate(task.updated_at)}</span>
          </div>
          <div class="task-objective">${shortObjective}</div>
        </article>
      `;
    })
    .join("");
}

function messageHtml(role, header, body) {
  const head = String(header || "").trim();
  const renderedBody =
    role === "assistant" || role === "system"
      ? `<div class="message-body markdown-body">${markdownToHtml(body)}</div>`
      : `<pre class="message-body">${escapeHtml(body)}</pre>`;
  return `
    <article class="message ${role}">
      ${head ? `<div class="message-head">${escapeHtml(head)}</div>` : ""}
      ${renderedBody}
    </article>
  `;
}

function setReportStatus(taskId, message = "", isError = false) {
  state.reportStatusTaskId = taskId || null;
  state.reportStatusText = String(message || "");
  state.reportStatusError = !!isError;
}

function renderReportPanel() {
  const detail = state.selectedDetail;
  const taskId = detail?.task?.task_id || null;
  el.reportTitle.textContent = reportProjectTitle(detail?.task || null);
  const taskStatus = String(detail?.task?.status || "");
  const persistedReportMarkdown = String(detail?.report_markdown || "").trim();
  const latestRunReport =
    state.latestRun?.task_id === taskId ? String(state.latestRun?.final_report || "").trim() : "";
  const reportMarkdown = persistedReportMarkdown || latestRunReport;
  const hasReport = Boolean(taskId && reportMarkdown);
  const showPanel = state.expanded && Boolean(taskId) && (taskStatus === "completed" || hasReport);

  el.workspace.classList.toggle("report-open", showPanel);
  el.reportPanel.classList.toggle("hidden", !showPanel);
  if (!showPanel) {
    el.reportTitle.textContent = "Research Report";
    el.reportContent.innerHTML = "";
    el.exportPdfBtn.classList.add("hidden");
    el.reportStatus.classList.add("hidden");
    el.reportStatus.classList.remove("error");
    return;
  }

  el.reportContent.innerHTML = hasReport
    ? markdownToHtml(reportMarkdown)
    : '<p>Final report is not available yet. Reopen this task in a moment.</p>';
  el.exportPdfBtn.classList.toggle("hidden", !hasReport);
  el.exportPdfBtn.disabled = state.exportingPdf;

  const shouldShowStatus =
    state.reportStatusTaskId === taskId && String(state.reportStatusText || "").trim().length > 0;
  el.reportStatus.classList.toggle("hidden", !shouldShowStatus);
  el.reportStatus.classList.toggle("error", shouldShowStatus && state.reportStatusError);
  el.reportStatus.textContent = shouldShowStatus ? state.reportStatusText : "";
}

async function exportFinalReportPdf(taskId) {
  if (!taskId || state.exportingPdf) return;
  state.exportingPdf = true;
  setReportStatus(taskId, "Preparing PDF export...");
  renderReportPanel();

  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/report.pdf`, {
      method: "GET",
    });
    if (!response.ok) {
      const raw = await response.text();
      let detail = `HTTP ${response.status}`;
      try {
        const parsed = raw ? JSON.parse(raw) : null;
        detail =
          (parsed && parsed.detail) || (typeof parsed === "string" && parsed) || detail;
      } catch {
        if (raw) detail = raw;
      }
      throw new Error(detail);
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const contentDisposition = response.headers.get("Content-Disposition") || "";
    const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    const filename = filenameMatch?.[1] || `report-${taskId}.pdf`;

    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    setReportStatus(taskId, "PDF export complete.");
  } catch (err) {
    setReportStatus(taskId, `PDF export failed: ${err.message}`, true);
  } finally {
    state.exportingPdf = false;
    renderReportPanel();
  }
}

function renderMessages() {
  const detail = state.selectedDetail;
  const pendingQuery = String(state.pendingUserMessage || state.latestRun?.query || "").trim();

  if (!detail?.task && state.clarificationMessage) {
    const parts = [];
    const runQuery = pendingQuery;
    if (runQuery) parts.push(messageHtml("user", "", runQuery));
    parts.push(messageHtml("assistant", "", state.clarificationMessage));
    el.messages.innerHTML = parts.join("");
    el.messages.scrollTop = el.messages.scrollHeight;
    return;
  }

  if (!detail || !detail.task) {
    if (pendingQuery) {
      el.messages.innerHTML = messageHtml("user", "", pendingQuery);
      el.messages.scrollTop = el.messages.scrollHeight;
      return;
    }
    el.messages.innerHTML = "";
    return;
  }

  const task = detail.task;
  const parts = [messageHtml("user", "", primaryObjectiveText(task.objective) || "(empty objective)")];
  const persistedReportMarkdown = String(detail?.report_markdown || "").trim();
  const latestRunReport =
    state.latestRun?.task_id === task.task_id ? String(state.latestRun?.final_report || "").trim() : "";
  const reportMarkdown = persistedReportMarkdown || latestRunReport;
  const planVersions = Array.isArray(task?.plan_versions) ? task.plan_versions : [];
  const activePlan = detail?.active_plan_version || null;
  const hitlEvents = Array.isArray(task.hitl_history) ? task.hitl_history : [];
  const currentCheckpointReason = String(detail?.checkpoint_reason || task?.checkpoint_reason || "").trim();
  const hasContinueHistory = hitlEvents.some((event) => String(event || "").trim().toLowerCase() === "continue");
  const activeButtonLabel = hasContinueHistory ? "Continue" : "Start research";
  const activeCheckpointNeedsAction =
    Boolean(task.awaiting_hitl) && shouldRenderCheckpointAction(currentCheckpointReason);

  let planCursor = 0;
  if (planVersions.length) {
    const firstPlan = planVersions[0];
    const initialIsActive =
      Boolean(task.awaiting_hitl) &&
      Boolean(activePlan?.version_id) &&
      String(activePlan.version_id) === String(firstPlan.version_id);
    const initialPlanText = formatPlanText(task, planVersions[0], {
      delta: null,
      reason: "",
    });
    parts.push(
      checkpointHtml(
        initialPlanText,
        initialIsActive && !activeCheckpointNeedsAction,
        initialIsActive ? activeButtonLabel : "Start research"
      )
    );
    planCursor = 1;
  } else {
    const subtasks = extractSubtasks(getScopeDecompositionText());
    if (subtasks.length) {
      parts.push(
        checkpointHtml(
          subtasks.map((x, i) => `${i + 1}. ${x}`).join("\n"),
          Boolean(task.awaiting_hitl),
          activeButtonLabel
        )
      );
    }
  }

  let renderedCurrentPauseReason = false;
  let renderedCurrentPauseAction = false;
  for (const event of hitlEvents) {
    const normalized = String(event || "").trim();
    if (!normalized) continue;
    const lowered = normalized.toLowerCase();
    if (lowered.startsWith("gate_ack:")) continue;

    if (lowered.startsWith("checkpoint:")) {
      const reasonCode = normalized.slice("checkpoint:".length).trim();
      const isCurrentActiveReason =
        Boolean(task.awaiting_hitl) &&
        reasonCode &&
        reasonCode.toLowerCase() === currentCheckpointReason.toLowerCase();
      if (
        isCurrentActiveReason
      ) {
        renderedCurrentPauseReason = true;
      }
      const prose = checkpointPauseMessage(reasonCode, task, activeButtonLabel);
      if (prose) {
        parts.push(messageHtml("assistant", "", prose));
      }
      if (isCurrentActiveReason && shouldRenderCheckpointAction(reasonCode)) {
        parts.push(checkpointActionHtml(activeButtonLabel));
        renderedCurrentPauseAction = true;
      }
      continue;
    }

    if (lowered.startsWith("revise:")) {
      const feedback = normalized.slice(7).trim();
      if (feedback) {
        parts.push(messageHtml("user", "", feedback));
      }
      const nextVersion = planVersions[planCursor] || null;
      if (nextVersion) {
        const isActive =
          Boolean(task.awaiting_hitl) &&
          Boolean(activePlan?.version_id) &&
          String(activePlan.version_id) === String(nextVersion.version_id);
        const planText = formatPlanText(task, nextVersion, {
          delta: isActive ? (detail?.latest_plan_delta || null) : null,
          reason: "",
        });
        parts.push(checkpointHtml(planText, isActive && !activeCheckpointNeedsAction, "Continue"));
        planCursor += 1;
      }
      continue;
    }

    if (lowered === "continue") {
      parts.push(messageHtml("user", "", "Continue"));
      continue;
    }

    if (lowered === "stop") {
      parts.push(messageHtml("user", "", "Stop"));
    }
  }

  if (task.awaiting_hitl && currentCheckpointReason && !renderedCurrentPauseReason) {
    const prose = checkpointPauseMessage(currentCheckpointReason, task, activeButtonLabel);
    if (prose) {
      parts.push(messageHtml("assistant", "", prose));
    }
  }
  if (task.awaiting_hitl && shouldRenderCheckpointAction(currentCheckpointReason) && !renderedCurrentPauseAction) {
    parts.push(checkpointActionHtml(activeButtonLabel));
  }

  const localEvents = state.localEventsByTask[task.task_id] || [];
  for (const event of localEvents) {
    parts.push(messageHtml(event.role || "assistant", "", event.body || ""));
  }

  const hasFinalReport = task.status === "completed" && reportMarkdown.length > 0;
  if (hasFinalReport) {
    el.messages.innerHTML = parts.join("");
    el.messages.scrollTop = el.messages.scrollHeight;
    return;
  }

  el.messages.innerHTML = parts.join("");
  el.messages.scrollTop = el.messages.scrollHeight;
}

function renderTaskHeader() {
  const detail = state.selectedDetail;
  if (!detail?.task) {
    el.taskTitle.textContent = "New chat";
    return;
  }
  const task = detail.task;
  const checkpointOpen = task.awaiting_hitl || String(task.checkpoint_state || "") === "open";
  const checkpointText = checkpointOpen ? "checkpoint open" : "checkpoint closed";
  el.taskTitle.textContent = `${taskDisplayTitle(task)} · ${task.status} · ${checkpointText}`;
}

function renderAll() {
  setExpanded(state.expanded);
  renderTasks();
  renderTaskHeader();
  renderMessages();
  renderReportPanel();
  renderActivity();
  setLoading(state.isLoading);
}

async function refreshHealth() {
  try {
    state.health = await api("/api/health");
    if (!state.health.ok) setNotice(state.health.error || "Backend is not ready.", true);
    else setNotice("");
    updateSendVisibility();
  } catch (err) {
    setNotice(`Health check failed: ${err.message}`, true);
    state.health = { ok: false };
    updateSendVisibility();
  }
}

async function refreshTasks({ keepSelection = true } = {}) {
  const payload = await api("/api/tasks");
  state.tasks = payload.tasks || [];

  if (!keepSelection) {
    state.selectedTaskId = null;
    state.selectedDetail = null;
    renderAll();
    return;
  }

  const selectedExists = state.selectedTaskId
    ? state.tasks.some((task) => task.task_id === state.selectedTaskId)
    : false;

  if (!selectedExists) {
    state.selectedTaskId = null;
    state.selectedDetail = null;
  }

  if (state.selectedTaskId) await selectTask(state.selectedTaskId, { silent: true });
  else renderAll();
}

async function selectTask(taskId, { silent = false } = {}) {
  state.selectedTaskId = taskId;
  if (!taskId) {
    state.selectedDetail = null;
    renderAll();
    return;
  }
  try {
    const detail = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
    state.selectedDetail = detail;
    state.pendingUserMessage = "";
    setExpanded(true);
    renderAll();
  } catch (err) {
    if (!silent) setNotice(`Failed to load task: ${err.message}`, true);
  }
}

function stopRunPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.activeRunId = null;
  setLoading(false);
}

function startNewChatDraft() {
  if (state.isLoading) {
    setNotice("A workflow is currently running.", true);
    return;
  }
  state.selectedTaskId = null;
  state.selectedDetail = null;
  state.latestRun = null;
  state.clarificationMessage = "";
  state.activityExpanded = false;
  state.exportingPdf = false;
  state.pendingUserMessage = "";
  setReportStatus(null, "");
  setExpanded(false);
  el.promptInput.value = "";
  el.promptInput.focus();
  setNotice("");
  updateSendVisibility();
  renderAll();
}

function handleTerminalRunState(run) {
  if (!run) return;
  state.latestRun = run;
  renderActivity();

  const terminalStates = new Set(["completed", "failed", "awaiting_hitl", "needs_clarification", "queued"]);
  if (!terminalStates.has(run.status)) return;

  stopRunPolling();

  if (run.task_id) {
    state.selectedTaskId = run.task_id;
    setExpanded(true);
  }

  refreshTasks({ keepSelection: true })
    .then(() => {
      if (run.task_id) return selectTask(run.task_id, { silent: true });
      return null;
    })
    .catch((err) => setNotice(`Could not refresh chats: ${err.message}`, true));

  if (run.status === "failed") {
    setNotice(run.error || "Run failed.", true);
    return;
  }

  if (run.status === "needs_clarification") {
    state.clarificationMessage = run.clarification || "Clarification needed before execution.";
    setExpanded(true);
    renderMessages();
    setNotice("");
    return;
  }

  if (run.status === "queued") {
    const taskId = run.task_id || state.selectedTaskId;
    if (taskId) {
      addLocalEvent(
        taskId,
        "assistant",
        "Update",
        "Feedback queued. I will apply it at the next checkpoint."
      );
    }
    renderMessages();
    setNotice("");
    return;
  }

  if (run.status === "awaiting_hitl") {
    setNotice("");
    return;
  }

  state.clarificationMessage = "";
  setNotice("");
}

function startRunPolling(runId) {
  stopRunPolling();
  state.activeRunId = runId;
  setLoading(true);

  const poll = async () => {
    try {
      const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
      handleTerminalRunState(run);
    } catch (err) {
      stopRunPolling();
      setNotice(`Run polling failed: ${err.message}`, true);
    }
  };

  poll();
  state.pollTimer = setInterval(poll, 1000);
}

async function submitNewQuery(query) {
  state.pendingUserMessage = String(query || "").trim();
  state.clarificationMessage = "";
  setExpanded(true);
  renderAll();

  const payload = await api("/api/query", {
    method: "POST",
    body: JSON.stringify({ query }),
  });

  state.latestRun = payload;
  state.activityExpanded = false;
  renderActivity();
  startRunPolling(payload.run_id);
}

async function submitContinue(taskId) {
  state.clarificationMessage = "";
  const planVersionId = state.selectedDetail?.active_plan_version?.version_id || null;
  let payload;
  try {
    payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/start`, {
      method: "POST",
      body: JSON.stringify({ plan_version_id: planVersionId }),
    });
  } catch (err) {
    if (String(err?.message || "").trim() !== "Not Found") throw err;
    payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/continue`, {
      method: "POST",
      body: JSON.stringify({}),
    });
  }
  state.latestRun = payload;
  state.activityExpanded = false;
  renderActivity();
  startRunPolling(payload.run_id);
}

async function submitFeedback(taskId, message) {
  state.clarificationMessage = "";
  let payload;
  try {
    payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/feedback`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
  } catch (err) {
    if (String(err?.message || "").trim() !== "Not Found") throw err;
    payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/revise`, {
      method: "POST",
      body: JSON.stringify({ scope: message }),
    });
  }
  state.latestRun = payload;
  state.activityExpanded = false;
  renderActivity();
  startRunPolling(payload.run_id);
}

function bindEvents() {
  el.tasksList.addEventListener("click", (event) => {
    const item = event.target.closest("[data-task-id]");
    if (!item) return;
    selectTask(item.dataset.taskId).catch((err) => setNotice(err.message, true));
  });

  el.messages.addEventListener("click", (event) => {
    const startBtn = event.target.closest('[data-action="checkpoint-start"]');
    if (!startBtn) return;
    const taskId = state.selectedTaskId;
    if (!taskId) return;
    setNotice("");
    submitContinue(taskId).catch((err) => {
      setNotice(`Start failed: ${err.message}`, true);
      setLoading(false);
    });
  });

  el.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = el.promptInput.value.trim();
    if (!query) return;

    const activeTask = state.selectedDetail?.task;
    el.promptInput.value = "";
    updateSendVisibility();
    setNotice("");

    if (activeTask?.awaiting_hitl && activeTask?.task_id) {
      submitFeedback(activeTask.task_id, query).catch((err) => {
        setNotice(`Feedback failed: ${err.message}`, true);
        setLoading(false);
      });
      return;
    }

    if (activeTask?.task_id && state.isLoading && state.selectedTaskId === activeTask.task_id) {
      submitFeedback(activeTask.task_id, query).catch((err) => {
        setNotice(`Feedback queue failed: ${err.message}`, true);
        setLoading(false);
      });
      return;
    }

    submitNewQuery(query).catch((err) => {
      setNotice(`Failed to start query: ${err.message}`, true);
      setLoading(false);
    });
  });

  el.promptInput.addEventListener("input", () => {
    updateSendVisibility();
  });

  el.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.composerForm.requestSubmit();
    }
  });

  el.newChatBtn.addEventListener("click", () => {
    startNewChatDraft();
  });

  el.exportPdfBtn.addEventListener("click", () => {
    const taskId = state.selectedDetail?.task?.task_id || null;
    if (!taskId) return;
    exportFinalReportPdf(taskId);
  });

  const toggleActivity = () => {
    if (el.activityCard.classList.contains("hidden")) return;
    state.activityExpanded = !state.activityExpanded;
    renderActivity();
  };

  el.activityCard.addEventListener("click", () => toggleActivity());
  el.activityCard.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleActivity();
    }
  });
}

async function bootstrap() {
  bindEvents();
  setExpanded(false);
  updatePromptPlaceholder();
  updateSendVisibility();
  renderActivity();
  await refreshHealth();
  await refreshTasks({ keepSelection: true });
  renderAll();
}

bootstrap().catch((err) => setNotice(`UI initialization failed: ${err.message}`, true));
