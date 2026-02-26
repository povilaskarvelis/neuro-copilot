const state = {
  conversations: [],
  selectedConversationId: null,
  selectedConversationDetail: null,
  selectedReportTaskId: null,
  activeRunId: null,
  pollTimer: null,
  isLoading: false,
  health: null,
  latestRun: null,
  clarificationMessage: "",
  pendingUserMessage: "",
  exportingPdf: false,
  reportStatusTaskId: null,
  reportStatusText: "",
  reportStatusError: false,
  activityExpandedByTask: {},
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

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  value = value.replace(/(^|[\s(])((?:https?:\/\/)[^\s<)]+)(?=$|[\s).,;:!?])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
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
  if (inCode) html.push(`<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
  return html.join("");
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
    const detail = (parsed && parsed.detail) || (typeof parsed === "string" ? parsed : `HTTP ${res.status}`);
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

function setExpanded(expanded) {
  document.body.classList.toggle("expanded", !!expanded);
  updatePromptPlaceholder();
}

function updatePromptPlaceholder() {
  el.promptInput.placeholder = state.selectedConversationId ? "Type a follow-up..." : "Ask any biomedical question....";
}

function updateSendVisibility() {
  const hasText = el.promptInput.value.trim().length > 0;
  el.sendBtn.classList.toggle("hidden", !hasText);
  el.sendBtn.disabled = !hasText || state.isLoading || !state.health?.ok;
}

function setLoading(isLoading) {
  state.isLoading = !!isLoading;
  updateSendVisibility();
}

function conversationTitle(conversation) {
  const text = String(conversation?.title || "").trim();
  return text || "Untitled Research";
}

function latestIteration(detail) {
  const iterations = Array.isArray(detail?.iterations) ? detail.iterations : [];
  if (!iterations.length) return null;
  return iterations[iterations.length - 1] || null;
}

function findIteration(detail, taskId) {
  if (!taskId) return null;
  const iterations = Array.isArray(detail?.iterations) ? detail.iterations : [];
  for (const iteration of iterations) {
    if (String(iteration?.task?.task_id || "") === String(taskId)) return iteration;
  }
  return null;
}

function latestCompletedTaskId(detail) {
  const iterations = Array.isArray(detail?.iterations) ? detail.iterations : [];
  for (let idx = iterations.length - 1; idx >= 0; idx -= 1) {
    const iteration = iterations[idx];
    if (String(iteration?.task?.status || "") === "completed") {
      return String(iteration?.task?.task_id || "");
    }
  }
  return "";
}

function planTextForIteration(iteration) {
  const activePlan = iteration?.active_plan_version;
  const task = iteration?.task || {};
  const steps = Array.isArray(activePlan?.steps)
    ? activePlan.steps
    : Array.isArray(task?.steps)
      ? task.steps
      : [];
  if (!steps.length) return "Plan is not available yet.";
  const lines = ["Here is the proposed plan:", ""];
  steps.forEach((step, idx) => {
    const title = String(step?.title || `Step ${idx + 1}`).trim();
    const instruction = String(step?.instruction || "").trim();
    lines.push(`${idx + 1}. ${title}`);
    if (instruction) lines.push(`- ${instruction}`);
  });
  return lines.join("\n");
}

function checkpointHtml(taskId, planText, showAction, buttonLabel) {
  const body = markdownToHtml(planText || "");
  const action = showAction
    ? `<button class="primary-btn checkpoint-start-btn" data-action="checkpoint-start" data-task-id="${escapeHtml(taskId)}">${escapeHtml(buttonLabel || "Start research")}</button>`
    : "";
  return `
    <article class="message assistant checkpoint-message">
      <div class="message-body markdown-body">${body}</div>
      ${action}
    </article>
  `;
}

function reportCardHtml(iteration) {
  const task = iteration?.task || {};
  const report = iteration?.report || {};
  const taskId = String(task.task_id || "");
  if (!taskId || !report.has_report) return "";
  const isActive = String(state.selectedReportTaskId || "") === taskId;
  const branchLabel = String(iteration?.branch_label || "").trim();
  return `
    <article class="message assistant report-card ${isActive ? "active" : ""}" data-action="open-report" data-task-id="${escapeHtml(taskId)}" role="button" tabindex="0">
      <div class="report-card-head">
        <strong>Report ${Number(iteration?.iteration_index || 0) || 1}</strong>
        <span>${escapeHtml(formatDate(task.updated_at))}</span>
      </div>
      <div class="report-card-title">${escapeHtml(String(task.title || task.user_query || "Research report"))}</div>
      ${branchLabel ? `<div class="branch-badge">${escapeHtml(branchLabel)}</div>` : ""}
    </article>
  `;
}

function followUpSuggestionsHtml(iteration) {
  const task = iteration?.task || {};
  if (String(task.status || "") !== "completed") return "";
  const suggestions = Array.isArray(iteration?.follow_up_suggestions) ? iteration.follow_up_suggestions : [];
  const lines = ["What do you want to do next?"];
  if (suggestions.length) {
    for (const item of suggestions.slice(0, 5)) {
      lines.push(`- ${String(item || "").trim()}`);
    }
  } else {
    lines.push("- Ask a focused follow-up to deepen or stress-test the recommendation.");
  }
  return `
    <article class="message assistant">
      <div class="message-body markdown-body">${markdownToHtml(lines.join("\n"))}</div>
    </article>
  `;
}

function activityExpansionKey(taskId) {
  const value = String(taskId || "").trim();
  return value || "__pending__";
}

function isActivityExpanded(taskId) {
  return !!state.activityExpandedByTask[activityExpansionKey(taskId)];
}

function setActivityExpanded(taskId, expanded) {
  state.activityExpandedByTask[activityExpansionKey(taskId)] = !!expanded;
}

function getStatusLabel(status) {
  const normalized = String(status || "").trim();
  if (normalized === "running" || normalized === "in_progress") return "Running";
  if (normalized === "completed") return "Complete";
  if (normalized === "awaiting_hitl") return "Checkpoint";
  if (normalized === "needs_clarification") return "Needs Input";
  if (normalized === "failed") return "Failed";
  if (normalized === "queued") return "Queued";
  return "Idle";
}

function reactTraceLines({ trace = "", phases = null } = {}) {
  const lines = [];
  const normalizedTrace = String(trace || "").trim();
  const phaseMap = phases && typeof phases === "object" ? phases : null;
  const order = ["reason", "act", "observe", "conclude"];
  const hasPhases = phaseMap && order.some((key) => String(phaseMap[key] || "").trim());
  if (!normalizedTrace && !hasPhases) return lines;

  lines.push("**ReAct Trace**");
  lines.push("");

  if (hasPhases) {
    for (const key of order) {
      const value = String(phaseMap[key] || "").trim();
      if (!value) continue;
      const label = key.charAt(0).toUpperCase() + key.slice(1);
      lines.push(`- **${label}:** ${value}`);
    }
  } else {
    for (const rawLine of normalizedTrace.split("\n")) {
      const line = String(rawLine || "").trim();
      if (!line) continue;
      const m = line.match(/^(REASON|ACT|OBSERVE|CONCLUDE)\s*:\s*(.+)$/i);
      if (m) {
        const label = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
        lines.push(`- **${label}:** ${m[2].trim()}`);
      } else {
        lines.push(`- ${line}`);
      }
    }
  }
  lines.push("");
  return lines;
}

function buildActivitySnapshot({ taskId = "", status = "", events = [], summaries = [] } = {}) {
  const normalizedStatus = String(status || "").trim();
  const safeEvents = Array.isArray(events) ? events : [];
  const safeSummaries = Array.isArray(summaries) ? summaries : [];
  const visibleStatuses = new Set(["running", "queued", "awaiting_hitl", "needs_clarification", "failed", "completed", "in_progress"]);
  const shouldShow = visibleStatuses.has(normalizedStatus) || safeEvents.length > 0 || safeSummaries.length > 0;
  if (!shouldShow) return null;

  const latestSummary = safeSummaries.length ? safeSummaries[safeSummaries.length - 1] : null;
  const latestEvent = safeEvents.length ? safeEvents[safeEvents.length - 1] : null;
  const latestLine = String(latestEvent?.human_line || latestEvent?.type || "").trim();

  const stepEvents = safeEvents.filter((e) => e?.type === "step.completed" && e?.metrics?.step_id);
  const stepsCompleted = Number(latestSummary?.steps_completed || stepEvents.length || 0);
  const stepsTotal = Number(latestSummary?.steps_total || 0);
  const stepDetails = Array.isArray(latestSummary?.step_details) ? latestSummary.step_details : [];

  const summaryByStatus = {
    queued: "Preparing research workflow.",
    running: "Executing the approved plan.",
    in_progress: "Executing the approved plan.",
    awaiting_hitl: "Waiting at a checkpoint for your input.",
    completed: "Research run completed.",
    failed: "Run ended with an error.",
    needs_clarification: "Clarification required to proceed.",
  };

  let title = "";
  if (normalizedStatus === "completed") {
    title = "Completed";
  } else if (stepsTotal > 0) {
    title = `Step ${stepsCompleted}/${stepsTotal}`;
  } else if (normalizedStatus === "queued") {
    title = "Starting";
  } else {
    title = "Research log";
  }

  let summary = "";
  if (stepEvents.length > 0) {
    const lastStep = stepEvents[stepEvents.length - 1];
    const sid = String(lastStep?.metrics?.step_id || "").trim();
    const goal = String(lastStep?.metrics?.goal || "").trim();
    summary = sid && goal ? `${sid}: ${goal}` : latestLine;
  } else {
    summary = String(latestSummary?.summary || "").trim() || latestLine || summaryByStatus[normalizedStatus] || "Tracking workflow progress.";
  }

  let preview = "";
  const stepPips = stepEvents.map((e) => {
    const sid = String(e?.metrics?.step_id || "").trim();
    const st = String(e?.metrics?.step_status || "completed").trim();
    const icon = st === "completed" ? "✓" : st === "blocked" ? "✗" : "…";
    return `${sid} ${icon}`;
  });
  if (stepPips.length) {
    preview = stepPips.join("  ·  ");
    if (stepsTotal > stepPips.length) {
      const remaining = stepsTotal - stepPips.length;
      preview += `  ·  ${remaining} remaining`;
    }
  } else if (latestLine) {
    preview = latestLine;
  } else if (normalizedStatus === "completed") {
    preview = "Execution finished. Expand to inspect details.";
  } else {
    preview = "Click for activity details";
  }

  const details = [];

  if (stepDetails.length > 0) {
    for (const step of stepDetails) {
      const sid = String(step.id || "").trim();
      const st = String(step.status || "pending").trim();
      const goal = String(step.goal || "").trim();
      const icon = st === "completed" ? "✓" : st === "blocked" ? "✗" : st === "pending" ? "○" : "…";
      details.push(`### ${icon} ${sid} — ${goal}`);
      if (st === "completed" || st === "blocked") {
        const progressNote = String(step.step_progress_note || "").trim();
        if (progressNote) details.push(progressNote);
        const reasoningTrace = String(step.reasoning_trace || "").trim();
        if (reasoningTrace) {
          details.push(...reactTraceLines({ trace: reasoningTrace }));
        }
        const resultSummary = String(step.result_summary || "").trim();
        if (resultSummary) {
          details.push(resultSummary);
        }
        const tools = Array.isArray(step.tools_called) ? step.tools_called : [];
        if (tools.length) {
          details.push(`**Sources:** ${tools.map((t) => `\`${t}\``).join(", ")}`);
        }
        const evidence = Array.isArray(step.evidence_ids) ? step.evidence_ids : [];
        if (evidence.length) {
          details.push(`**Evidence:** ${evidence.slice(0, 6).map((e) => `\`${e}\``).join(", ")}`);
        }
        const gaps = Array.isArray(step.open_gaps) ? step.open_gaps : [];
        if (gaps.length) {
          details.push(`**Gaps:** ${gaps.slice(0, 3).join("; ")}`);
        }
      }
      details.push("");
    }
  } else if (stepEvents.length > 0) {
    for (const event of stepEvents) {
      const m = event?.metrics || {};
      const sid = String(m.step_id || "").trim();
      const st = String(m.step_status || "completed").trim();
      const goal = String(m.goal || "").trim();
      const icon = st === "completed" ? "✓" : st === "blocked" ? "✗" : "…";
      details.push(`### ${icon} ${sid} — ${goal}`);
      const findings = String(m.findings || "").trim();
      if (findings) details.push(findings);
      const reactTrace = String(m.react_trace || "").trim();
      const reactPhases = m.react_phases && typeof m.react_phases === "object" ? m.react_phases : null;
      if (reactTrace || reactPhases) {
        details.push(...reactTraceLines({ trace: reactTrace, phases: reactPhases }));
      }
      const tools = Array.isArray(m.tools) ? m.tools : [];
      if (tools.length) details.push(`**Sources:** ${tools.map((t) => `\`${t}\``).join(", ")}`);
      const evidence = Array.isArray(m.evidence) ? m.evidence : [];
      if (evidence.length) details.push(`**Evidence:** ${evidence.slice(0, 6).map((e) => `\`${e}\``).join(", ")}`);
      const progress = String(m.progress || "").trim();
      if (progress) details.push(`_${progress}_`);
      details.push("");
    }
  } else if (safeEvents.length > 0) {
    details.push("### Activity");
    for (const event of safeEvents.slice(-12)) {
      const line = String(event?.human_line || event?.type || "").trim();
      if (!line) continue;
      details.push(`- ${line}`);
    }
  } else {
    details.push("- Waiting for workflow events.");
  }

  return {
    taskId: String(taskId || "").trim() || "pending",
    status: normalizedStatus,
    title,
    summary,
    preview,
    detailsHtml: markdownToHtml(details.join("\n")),
  };
}

function activityCardClassNames(status, expanded) {
  const normalized = String(status || "").trim();
  return [
    "activity-card",
    expanded ? "expanded" : "",
    (normalized === "running" || normalized === "in_progress" || normalized === "queued") ? "is-running" : "",
    normalized === "completed" ? "is-complete" : "",
    normalized === "failed" ? "is-error" : "",
    normalized === "awaiting_hitl" ? "is-paused" : "",
  ].filter(Boolean).join(" ");
}

function activityCardHtml(snapshot) {
  if (!snapshot) return "";
  const expanded = isActivityExpanded(snapshot.taskId);
  const status = String(snapshot.status || "").trim();
  const classNames = activityCardClassNames(status, expanded);

  return `
    <section
      class="${classNames}"
      data-role="activity-card"
      data-action="toggle-activity"
      data-task-id="${escapeHtml(snapshot.taskId)}"
      role="button"
      tabindex="0"
      aria-expanded="${expanded ? "true" : "false"}"
    >
      <div class="activity-head">
        <div class="activity-main">
          <span class="activity-wheel" aria-hidden="true"></span>
          <div class="activity-title-stack">
            <span class="activity-title">${escapeHtml(snapshot.title)}</span>
            <div class="activity-summary">${escapeHtml(snapshot.summary)}</div>
          </div>
        </div>
      </div>
      <div class="activity-preview">${escapeHtml(snapshot.preview)}</div>
      <div class="activity-details markdown-body ${expanded ? "" : "hidden"}">${snapshot.detailsHtml}</div>
    </section>
  `;
}

function patchActivityCardElement(card, snapshot) {
  if (!card || !snapshot) return;
  const expanded = isActivityExpanded(snapshot.taskId);
  card.className = activityCardClassNames(snapshot.status, expanded);
  card.dataset.taskId = snapshot.taskId;
  card.setAttribute("aria-expanded", expanded ? "true" : "false");

  const titleEl = card.querySelector(".activity-title");
  if (titleEl) titleEl.textContent = snapshot.title;
  const summaryEl = card.querySelector(".activity-summary");
  if (summaryEl) summaryEl.textContent = snapshot.summary;
  const previewEl = card.querySelector(".activity-preview");
  if (previewEl) previewEl.textContent = snapshot.preview;
  const detailsEl = card.querySelector(".activity-details");
  if (detailsEl) {
    detailsEl.innerHTML = snapshot.detailsHtml;
    detailsEl.classList.toggle("hidden", !expanded);
  }
}

function iterationActivitySnapshot(iteration) {
  const task = iteration?.task || {};
  const taskId = String(task.task_id || "").trim();
  if (!taskId) return null;
  const run = String(state.latestRun?.task_id || "").trim() === taskId ? state.latestRun : null;
  const researchLog = iteration?.research_log || {};
  const events = run
    ? (Array.isArray(run.progress_events) ? run.progress_events : [])
    : (Array.isArray(researchLog.events) ? researchLog.events : []);
  const summaries = run
    ? (Array.isArray(run.progress_summaries) ? run.progress_summaries : [])
    : (Array.isArray(researchLog.summaries) ? researchLog.summaries : []);
  const status = String(run?.status || task.status || researchLog.status || "").trim();
  return buildActivitySnapshot({ taskId, status, events, summaries });
}

function pendingActivitySnapshot() {
  const run = state.latestRun || {};
  const taskId = String(run.task_id || "").trim() || "pending";
  const status = String(run.status || "queued").trim();
  const events = Array.isArray(run.progress_events) ? run.progress_events : [];
  const summaries = Array.isArray(run.progress_summaries) ? run.progress_summaries : [];
  return buildActivitySnapshot({ taskId, status, events, summaries });
}

function taskHasStarted(task) {
  const history = Array.isArray(task?.hitl_history) ? task.hitl_history : [];
  return history.includes("continue");
}

function placeActivityAfterPlan(task, activeRun) {
  const status = String(activeRun?.status || task?.status || "").trim();
  if (["running", "queued", "in_progress", "completed", "failed", "needs_clarification"].includes(status)) return true;
  if (status === "awaiting_hitl" && taskHasStarted(task)) return true;
  return false;
}

function updateInlineActivityCard(run) {
  if (!run || !el.messages) return;
  const runTaskId = String(run.task_id || "").trim();
  const snapshot = buildActivitySnapshot({
    taskId: runTaskId || "pending",
    status: String(run.status || "").trim(),
    events: Array.isArray(run.progress_events) ? run.progress_events : [],
    summaries: Array.isArray(run.progress_summaries) ? run.progress_summaries : [],
  });
  if (!snapshot) return;

  const cards = Array.from(el.messages.querySelectorAll('[data-role="activity-card"]'));
  if (!cards.length) return;
  const targetCards = cards.filter((card) => {
    const cardTaskId = String(card.dataset.taskId || "").trim();
    if (runTaskId) return cardTaskId === runTaskId || cardTaskId === "pending";
    return cardTaskId === "pending";
  });
  if (!targetCards.length) return;

  if (runTaskId && isActivityExpanded("pending") && !isActivityExpanded(runTaskId)) {
    setActivityExpanded(runTaskId, true);
  }

  for (const card of targetCards) {
    patchActivityCardElement(card, snapshot);
  }
}

function renderSidebar() {
  if (!state.conversations.length) {
    el.tasksList.innerHTML = '<p class="muted">No chats yet.</p>';
    return;
  }
  el.tasksList.innerHTML = state.conversations
    .map((conversation) => {
      const conversationId = String(conversation.conversation_id || "");
      const active = conversationId === state.selectedConversationId ? "active" : "";
      const title = escapeHtml(conversationTitle(conversation));
      const status = escapeHtml(String(conversation.latest_status || "unknown"));
      const count = Number(conversation.iteration_count || 0);
      return `
        <article class="task-item ${active}" data-conversation-id="${escapeHtml(conversationId)}">
          <div class="task-line">
            <span><span class="status-dot status-${status.replace(/\s+/g, "_")}"></span>${status}</span>
            <span>${formatDate(conversation.updated_at)}</span>
          </div>
          <div class="task-objective">${title}</div>
          <div class="task-line"><span>${count} iteration${count === 1 ? "" : "s"}</span></div>
        </article>
      `;
    })
    .join("");
}

function renderTaskHeader() {
  const conversation = state.selectedConversationDetail?.conversation;
  if (!conversation) {
    el.taskTitle.textContent = "New chat";
    return;
  }
  const title = conversationTitle(conversation);
  const status = String(conversation.latest_status || "");
  const count = Number(conversation.iteration_count || 0);
  el.taskTitle.textContent = `${title} · ${status} · ${count} iteration${count === 1 ? "" : "s"}`;
}

function renderMessages() {
  const detail = state.selectedConversationDetail;

  if (!detail && state.clarificationMessage) {
    const parts = [];
    if (state.pendingUserMessage) {
      parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`);
      const pendingCard = activityCardHtml(pendingActivitySnapshot());
      if (pendingCard) parts.push(pendingCard);
    }
    parts.push(`<article class="message assistant"><div class="message-body markdown-body">${markdownToHtml(state.clarificationMessage)}</div></article>`);
    el.messages.innerHTML = parts.join("");
    el.messages.scrollTop = el.messages.scrollHeight;
    return;
  }

  if (!detail) {
    if (state.pendingUserMessage) {
      const parts = [
        `<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`,
      ];
      const pendingCard = activityCardHtml(pendingActivitySnapshot());
      if (pendingCard) parts.push(pendingCard);
      el.messages.innerHTML = parts.join("");
      el.messages.scrollTop = el.messages.scrollHeight;
      return;
    }
    el.messages.innerHTML = "";
    return;
  }

  const iterations = Array.isArray(detail.iterations) ? detail.iterations : [];
  const parts = [];

  for (const iteration of iterations) {
    const task = iteration?.task || {};
    const userText = String(task.user_query || task.objective || "").trim() || "(empty query)";
    parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(userText)}</pre></article>`);
    const runForTask = String(state.latestRun?.task_id || "").trim() === String(task.task_id || "").trim() ? state.latestRun : null;
    const activityCard = activityCardHtml(iterationActivitySnapshot(iteration));
    const shouldPlaceAfterPlan = placeActivityAfterPlan(task, runForTask);
    if (activityCard && !shouldPlaceAfterPlan) parts.push(activityCard);

    const branchLabel = String(iteration?.branch_label || "").trim();
    if (branchLabel) {
      parts.push(
        `<article class="message assistant"><div class="message-body"><span class="branch-badge">${escapeHtml(branchLabel)}</span></div></article>`
      );
    }

    const planText = planTextForIteration(iteration);
    const awaiting = Boolean(task.awaiting_hitl);
    const buttonLabel = task.hitl_history && task.hitl_history.includes("continue") ? "Continue" : "Start research";
    parts.push(checkpointHtml(task.task_id, planText, awaiting, buttonLabel));
    if (activityCard && shouldPlaceAfterPlan) parts.push(activityCard);

    const reportCard = reportCardHtml(iteration);
    if (reportCard) parts.push(reportCard);

    const suggestions = followUpSuggestionsHtml(iteration);
    if (suggestions) parts.push(suggestions);
  }

  if (state.pendingUserMessage) {
    parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`);
    const pendingCard = activityCardHtml(pendingActivitySnapshot());
    if (pendingCard) parts.push(pendingCard);
  }

  el.messages.innerHTML = parts.join("");
  el.messages.scrollTop = el.messages.scrollHeight;
}

function setReportStatus(taskId, message = "", isError = false) {
  state.reportStatusTaskId = taskId || null;
  state.reportStatusText = String(message || "");
  state.reportStatusError = !!isError;
}

function currentReportIteration() {
  const detail = state.selectedConversationDetail;
  if (!detail) return null;
  const explicit = findIteration(detail, state.selectedReportTaskId);
  if (explicit) return explicit;
  const selected = String(detail?.conversation?.selected_report_task_id || "").trim();
  if (selected) return findIteration(detail, selected);
  const latestDone = latestCompletedTaskId(detail);
  if (latestDone) return findIteration(detail, latestDone);
  return latestIteration(detail);
}

function renderReportPanel() {
  const iteration = currentReportIteration();
  const showPanel = Boolean(state.selectedConversationId && iteration && iteration?.report?.has_report);

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

  const task = iteration.task || {};
  const taskId = String(task.task_id || "");
  const reportMarkdown = String(iteration?.report?.report_markdown || "").trim();
  el.reportTitle.textContent = String(task.title || task.user_query || "Research Report");
  el.reportContent.innerHTML = markdownToHtml(reportMarkdown);
  el.exportPdfBtn.classList.toggle("hidden", !taskId);
  el.exportPdfBtn.disabled = state.exportingPdf;

  const shouldShowStatus = state.reportStatusTaskId === taskId && String(state.reportStatusText || "").trim().length > 0;
  el.reportStatus.classList.toggle("hidden", !shouldShowStatus);
  el.reportStatus.classList.toggle("error", shouldShowStatus && state.reportStatusError);
  el.reportStatus.textContent = shouldShowStatus ? state.reportStatusText : "";
}

function renderAll() {
  const expanded = Boolean(state.selectedConversationId || state.pendingUserMessage || state.clarificationMessage);
  setExpanded(expanded);
  renderSidebar();
  renderTaskHeader();
  renderMessages();
  renderReportPanel();
  setLoading(state.isLoading);
}

async function refreshHealth() {
  try {
    state.health = await api("/api/health");
    if (!state.health.ok) setNotice(state.health.error || "Backend is not ready.", true);
    else setNotice("");
  } catch (err) {
    state.health = { ok: false };
    setNotice(`Health check failed: ${err.message}`, true);
  }
  updateSendVisibility();
}

async function refreshConversations({ keepSelection = true } = {}) {
  const payload = await api("/api/conversations");
  state.conversations = Array.isArray(payload?.conversations) ? payload.conversations : [];

  if (!keepSelection) {
    state.selectedConversationId = null;
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    renderAll();
    return;
  }

  const stillExists = state.selectedConversationId
    ? state.conversations.some((conversation) => String(conversation.conversation_id || "") === state.selectedConversationId)
    : false;

  if (!stillExists) {
    state.selectedConversationId = null;
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    renderAll();
    return;
  }

  if (state.selectedConversationId) {
    await selectConversation(state.selectedConversationId, { silent: true });
  } else {
    renderAll();
  }
}

async function selectConversation(conversationId, { silent = false } = {}) {
  state.selectedConversationId = conversationId;
  if (!conversationId) {
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    renderAll();
    return;
  }
  try {
    const detail = await api(`/api/conversations/${encodeURIComponent(conversationId)}`);
    state.selectedConversationDetail = detail;
    const defaultReportTaskId = String(detail?.conversation?.selected_report_task_id || "").trim();
    const latestDone = latestCompletedTaskId(detail);
    if (!state.selectedReportTaskId || !findIteration(detail, state.selectedReportTaskId)) {
      state.selectedReportTaskId = defaultReportTaskId || latestDone || "";
    }
    state.pendingUserMessage = "";
    state.clarificationMessage = "";
    renderAll();
  } catch (err) {
    if (!silent) setNotice(`Failed to load conversation: ${err.message}`, true);
  }
}

async function exportFinalReportPdf(taskId) {
  if (!taskId || state.exportingPdf) return;
  state.exportingPdf = true;
  setReportStatus(taskId, "Preparing PDF export...");
  renderReportPanel();

  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/report.pdf`, { method: "GET" });
    if (!response.ok) {
      const raw = await response.text();
      let detail = `HTTP ${response.status}`;
      try {
        const parsed = raw ? JSON.parse(raw) : null;
        detail = (parsed && parsed.detail) || (typeof parsed === "string" && parsed) || detail;
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

function stopRunPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.activeRunId = null;
  setLoading(false);
}

function startRunPolling(runId) {
  stopRunPolling();
  state.activeRunId = runId;
  setLoading(true);

  const poll = async () => {
    try {
      const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
      state.latestRun = run;
      updateInlineActivityCard(run);
      await handleTerminalRunState(run);
    } catch (err) {
      stopRunPolling();
      setNotice(`Run polling failed: ${err.message}`, true);
    }
  };

  poll();
  state.pollTimer = setInterval(poll, 1000);
}

async function handleTerminalRunState(run) {
  if (!run) return;
  const status = String(run.status || "");
  const kind = String(run.kind || "");
  const isQueuedFeedbackAck = status === "queued" && kind === "feedback_task";
  const terminalStates = new Set(["completed", "failed", "awaiting_hitl", "needs_clarification"]);
  if (!terminalStates.has(status) && !isQueuedFeedbackAck) return;

  stopRunPolling();

  if (status === "failed") {
    setNotice(run.error || "Run failed.", true);
  } else if (status === "needs_clarification") {
    state.clarificationMessage = run.clarification || "Clarification required before execution.";
    setNotice("");
  } else {
    setNotice("");
  }

  if (run.task_id) {
    try {
      const taskDetail = await api(`/api/tasks/${encodeURIComponent(run.task_id)}`);
      const conversationId = String(taskDetail?.task?.conversation_id || "").trim() || `conv_${run.task_id}`;
      await refreshConversations({ keepSelection: true });
      await selectConversation(conversationId, { silent: true });
      if (status === "completed") {
        state.selectedReportTaskId = run.task_id;
      }
    } catch (err) {
      setNotice(`Could not refresh conversations: ${err.message}`, true);
    }
  } else {
    try {
      await refreshConversations({ keepSelection: true });
    } catch (err) {
      setNotice(`Could not refresh conversations: ${err.message}`, true);
    }
  }

  renderAll();
}

async function submitNewQuery(query, { conversationId = null, parentTaskId = null } = {}) {
  state.pendingUserMessage = String(query || "").trim();
  state.clarificationMessage = "";
  setActivityExpanded("pending", false);
  state.latestRun = null;
  renderAll();

  const requestBody = { query };
  if (conversationId) requestBody.conversation_id = conversationId;
  if (parentTaskId) requestBody.parent_task_id = parentTaskId;

  const payload = await api("/api/query", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });

  state.latestRun = payload;
  renderMessages();
  startRunPolling(payload.run_id);
}

async function submitContinue(taskId) {
  const detail = state.selectedConversationDetail;
  const iteration = findIteration(detail, taskId);
  const planVersionId = iteration?.active_plan_version?.version_id || null;
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
  renderMessages();
  startRunPolling(payload.run_id);
}

async function submitFeedback(taskId, message) {
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
  renderMessages();
  startRunPolling(payload.run_id);
}

function clearDraft() {
  state.selectedConversationId = null;
  state.selectedConversationDetail = null;
  state.selectedReportTaskId = null;
  state.latestRun = null;
  state.clarificationMessage = "";
  state.pendingUserMessage = "";
  state.activityExpandedByTask = {};
  setReportStatus(null, "");
  el.promptInput.value = "";
  updateSendVisibility();
  setNotice("");
  renderAll();
}

function bindEvents() {
  el.tasksList.addEventListener("click", (event) => {
    const item = event.target.closest("[data-conversation-id]");
    if (!item) return;
    selectConversation(item.dataset.conversationId).catch((err) => setNotice(err.message, true));
  });

  el.messages.addEventListener("click", (event) => {
    const activityCard = event.target.closest('[data-action="toggle-activity"]');
    if (activityCard) {
      const taskId = String(activityCard.dataset.taskId || "").trim();
      const expanded = !isActivityExpanded(taskId);
      setActivityExpanded(taskId, expanded);
      activityCard.classList.toggle("expanded", expanded);
      activityCard.setAttribute("aria-expanded", expanded ? "true" : "false");
      const details = activityCard.querySelector(".activity-details");
      if (details) details.classList.toggle("hidden", !expanded);
      return;
    }

    const startBtn = event.target.closest('[data-action="checkpoint-start"]');
    if (startBtn) {
      if (startBtn.disabled) return;
      const taskId = String(startBtn.dataset.taskId || "").trim();
      if (!taskId) return;
      const originalLabel = startBtn.textContent;
      startBtn.disabled = true;
      startBtn.textContent = "Starting...";
      submitContinue(taskId).catch((err) => {
        startBtn.disabled = false;
        startBtn.textContent = originalLabel;
        setNotice(`Start failed: ${err.message}`, true);
      });
      return;
    }

    const reportCard = event.target.closest('[data-action="open-report"]');
    if (reportCard) {
      const taskId = String(reportCard.dataset.taskId || "").trim();
      if (!taskId) return;
      state.selectedReportTaskId = taskId;
      renderAll();
    }
  });

  el.messages.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const activityCard = event.target.closest('[data-action="toggle-activity"]');
    if (activityCard) {
      event.preventDefault();
      const taskId = String(activityCard.dataset.taskId || "").trim();
      const expanded = !isActivityExpanded(taskId);
      setActivityExpanded(taskId, expanded);
      activityCard.classList.toggle("expanded", expanded);
      activityCard.setAttribute("aria-expanded", expanded ? "true" : "false");
      const details = activityCard.querySelector(".activity-details");
      if (details) details.classList.toggle("hidden", !expanded);
      return;
    }

    const reportCard = event.target.closest('[data-action="open-report"]');
    if (!reportCard) return;
    event.preventDefault();
    const taskId = String(reportCard.dataset.taskId || "").trim();
    if (!taskId) return;
    state.selectedReportTaskId = taskId;
    renderAll();
  });

  el.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = el.promptInput.value.trim();
    if (!query) return;

    const detail = state.selectedConversationDetail;
    const active = latestIteration(detail);

    el.promptInput.value = "";
    updateSendVisibility();
    setNotice("");

    if (active?.task?.awaiting_hitl && active?.task?.task_id) {
      submitFeedback(active.task.task_id, query).catch((err) => setNotice(`Feedback failed: ${err.message}`, true));
      return;
    }

    if (state.selectedConversationId) {
      const anchorTaskId =
        state.selectedReportTaskId
        || String(detail?.conversation?.selected_report_task_id || "").trim()
        || latestCompletedTaskId(detail)
        || "";
      submitNewQuery(query, {
        conversationId: state.selectedConversationId,
        parentTaskId: anchorTaskId || null,
      }).catch((err) => setNotice(`Failed to start query: ${err.message}`, true));
      return;
    }

    submitNewQuery(query).catch((err) => setNotice(`Failed to start query: ${err.message}`, true));
  });

  el.promptInput.addEventListener("input", () => updateSendVisibility());
  el.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.composerForm.requestSubmit();
    }
  });

  el.newChatBtn.addEventListener("click", () => clearDraft());

  el.exportPdfBtn.addEventListener("click", () => {
    const iteration = currentReportIteration();
    const taskId = String(iteration?.task?.task_id || "").trim();
    if (!taskId) return;
    exportFinalReportPdf(taskId);
  });
}

async function bootstrap() {
  bindEvents();
  updatePromptPlaceholder();
  updateSendVisibility();
  renderAll();
  await refreshHealth();
  await refreshConversations({ keepSelection: true });
  renderAll();
}

bootstrap().catch((err) => setNotice(`UI initialization failed: ${err.message}`, true));
