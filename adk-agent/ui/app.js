const state = {
  conversations: [],
  selectedConversationId: null,
  selectedConversationDetail: null,
  selectedReportTaskId: null,
  activeRunIds: new Set(),
  runsByRunId: {},
  runsByTaskId: {},
  pendingRunId: null,
  pollTimer: null,
  pollInFlight: false,
  isLoading: false,
  health: null,
  clarificationMessage: "",
  pendingUserMessage: "",
  exportingPdf: false,
  reportStatusTaskId: null,
  reportStatusText: "",
  reportStatusError: false,
  renderedReportTaskId: "",
  renderedReportTitle: "",
  renderedReportMarkdown: "",
  debugOpen: false,
  debugByTaskId: {},
  activityExpandedByTask: {},
  handlingTerminalRunIds: new Set(),
  startingTaskIds: new Set(),
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
  debugToggleBtn: document.getElementById("debugToggleBtn"),
  debugPanel: document.getElementById("debugPanel"),
  debugSummary: document.getElementById("debugSummary"),
  debugJson: document.getElementById("debugJson"),
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
  let value = text.replace(/<a\s+id="[^"]*">\s*<\/a>/gi, "");
  value = escapeHtml(value);
  value = value.replace(/\[([^\]]+)\]\((#[^)\s]+)\)/g, '<a href="$2">$1</a>');
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
  let inCode = false;
  let codeBuffer = [];
  let inBlockquote = false;
  const listStack = [];

  const closeCurrentList = () => {
    const current = listStack.pop();
    if (!current) return;
    if (current.liOpen) html.push("</li>");
    html.push(`</${current.type}>`);
  };
  const closeListsToIndent = (targetIndent = -1) => {
    while (listStack.length && listStack[listStack.length - 1].indent > targetIndent) {
      closeCurrentList();
    }
  };
  const closeLists = () => {
    closeListsToIndent(-1);
  };
  const closeBlockquote = () => {
    if (inBlockquote) { html.push("</blockquote>"); inBlockquote = false; }
  };
  const openList = (type, indent, start = 1) => {
    if (type === "ol") {
      html.push(Number.isFinite(start) && start > 1 ? `<ol start="${start}">` : "<ol>");
    } else {
      html.push("<ul>");
    }
    listStack.push({ type, indent, liOpen: false });
  };
  const renderListItem = ({ type, indent, content, anchorHtml: itemAnchorHtml, start = 1 }) => {
    closeBlockquote();
    closeListsToIndent(indent);

    let current = listStack[listStack.length - 1];
    if (!current || indent > current.indent) {
      openList(type, indent, start);
      current = listStack[listStack.length - 1];
    } else if (current.type !== type || current.indent !== indent) {
      closeCurrentList();
      openList(type, indent, start);
      current = listStack[listStack.length - 1];
    }

    if (current.liOpen) html.push("</li>");
    html.push("<li>");
    current.liOpen = true;
    html.push(`${itemAnchorHtml}${inlineMarkdown(content)}`);
  };
  const looksLikeTableRow = (value) => {
    const text = String(value || "").trim();
    if (!text.includes("|")) return false;
    const pipeCount = (text.match(/\|/g) || []).length;
    return pipeCount >= 2 || (pipeCount === 1 && text.startsWith("|"));
  };
  const isTableSeparator = (value) => {
    const text = String(value || "").trim();
    if (!looksLikeTableRow(text)) return false;
    const normalized = text.replace(/^\|/, "").replace(/\|$/, "");
    const cells = normalized.split("|").map((cell) => cell.trim());
    if (!cells.length) return false;
    return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  };
  const splitTableCells = (value) => {
    const text = String(value || "").trim().replace(/^\|/, "").replace(/\|$/, "");
    const cells = [];
    let current = "";
    let escaped = false;
    for (const ch of text) {
      if (escaped) {
        current += ch;
        escaped = false;
        continue;
      }
      if (ch === "\\") {
        escaped = true;
        continue;
      }
      if (ch === "|") {
        cells.push(current.trim());
        current = "";
        continue;
      }
      current += ch;
    }
    cells.push(current.trim());
    return cells;
  };

  for (let idx = 0; idx < lines.length; idx += 1) {
    const rawLine = lines[idx];
    const line = String(rawLine || "");
    let trimmed = line.trim();

    // Extract <a id="..."></a> anchors — re-emit as real HTML anchor targets
    let anchorHtml = "";
    const am = trimmed.match(/^<a\s+id="([^"]+)">\s*<\/a>/);
    if (am) {
      anchorHtml = `<span id="${escapeHtml(am[1])}"></span>`;
      trimmed = trimmed.slice(am[0].length).trim();
    }

    if (trimmed.startsWith("```")) {
      closeBlockquote();
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
      closeBlockquote();
      closeLists();
      continue;
    }

    // Horizontal rule
    if (/^(?:---+|\*\*\*+|___+)$/.test(trimmed)) {
      closeBlockquote();
      closeLists();
      html.push("<hr>");
      continue;
    }

    const h3 = trimmed.match(/^###\s+(.+)$/);
    if (h3) {
      closeBlockquote();
      closeLists();
      html.push(`${anchorHtml}<h3>${inlineMarkdown(h3[1])}</h3>`);
      continue;
    }
    const h2 = trimmed.match(/^##\s+(.+)$/);
    if (h2) {
      closeBlockquote();
      closeLists();
      html.push(`${anchorHtml}<h2>${inlineMarkdown(h2[1])}</h2>`);
      continue;
    }
    const h1 = trimmed.match(/^#\s+(.+)$/);
    if (h1) {
      closeBlockquote();
      closeLists();
      html.push(`${anchorHtml}<h1>${inlineMarkdown(h1[1])}</h1>`);
      continue;
    }

    // Blockquotes — merge consecutive > lines into a single <blockquote>
    const quoteMatch = trimmed.match(/^>\s?(.*)$/);
    if (quoteMatch) {
      closeLists();
      if (!inBlockquote) {
        html.push("<blockquote>");
        inBlockquote = true;
      }
      const content = (quoteMatch[1] || "").trim();
      if (content) {
        html.push(`<p>${inlineMarkdown(content)}</p>`);
      }
      continue;
    }

    closeBlockquote();

    const nextLine = idx + 1 < lines.length ? String(lines[idx + 1] || "") : "";
    if (looksLikeTableRow(trimmed) && isTableSeparator(nextLine)) {
      closeLists();
      const headerCells = splitTableCells(trimmed);
      const rows = [];
      idx += 2;
      while (idx < lines.length) {
        const candidate = String(lines[idx] || "");
        const candidateTrimmed = candidate.trim();
        if (!candidateTrimmed || !looksLikeTableRow(candidateTrimmed) || isTableSeparator(candidateTrimmed)) {
          idx -= 1;
          break;
        }
        rows.push(splitTableCells(candidateTrimmed));
        idx += 1;
      }

      html.push("<table><thead><tr>");
      headerCells.forEach((cell) => {
        html.push(`<th>${inlineMarkdown(cell)}</th>`);
      });
      html.push("</tr></thead>");
      if (rows.length) {
        html.push("<tbody>");
        rows.forEach((row) => {
          html.push("<tr>");
          headerCells.forEach((_, cellIdx) => {
            html.push(`<td>${inlineMarkdown(row[cellIdx] || "")}</td>`);
          });
          html.push("</tr>");
        });
        html.push("</tbody>");
      }
      html.push("</table>");
      continue;
    }

    const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.+)$/);
    if (listMatch) {
      const indent = (listMatch[1] || "").replace(/\t/g, "  ").length;
      const marker = listMatch[2] || "";
      const content = listMatch[3] || "";
      const type = /^\d+\.$/.test(marker) ? "ol" : "ul";
      const start = type === "ol" ? Number.parseInt(marker, 10) || 1 : 1;
      renderListItem({ type, indent, content, anchorHtml, start });
      continue;
    }

    closeLists();
    html.push(`${anchorHtml}<p>${inlineMarkdown(trimmed)}</p>`);
  }

  closeBlockquote();
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

function setInnerHtmlIfChanged(element, html) {
  if (!element) return false;
  if (element.innerHTML === html) return false;
  element.innerHTML = html;
  return true;
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

function planHtmlForIteration(iteration) {
  const activePlan = iteration?.active_plan_version;
  const task = iteration?.task || {};
  const steps = Array.isArray(activePlan?.steps)
    ? activePlan.steps
    : Array.isArray(task?.steps)
      ? task.steps
      : [];
  if (!steps.length) return "";

  let html = "<p>To answer your query I will:</p><ol class=\"plan-steps\">";
  steps.forEach((step, idx) => {
    const title = escapeHtml(String(step?.title || `Step ${idx + 1}`).trim());
    let source = String(step?.source || "").trim();
    let completion = String(step?.completion_condition || "").trim();

    // Fall back to parsing the instruction string if structured fields are absent
    if (!source || !completion) {
      const instruction = String(step?.instruction || "").trim();
      if (!source) {
        const m = instruction.match(/(?:Potential source|Source):\s*([^.]+)/i);
        if (m) source = m[1].trim();
      }
      if (!completion) {
        const m = instruction.match(/Done when:\s*(.+)$/i);
        if (m) completion = m[1].trim();
      }
    }

    html += `<li><span class="plan-step-title">${title}</span>`;
    if (source || completion) {
      html += `<ul class="plan-step-details">`;
      if (source) {
        html += `<li><span class="plan-step-label">Potential source</span>${escapeHtml(source)}</li>`;
      }
      if (completion) {
        html += `<li><span class="plan-step-label">Done when</span>${escapeHtml(completion)}</li>`;
      }
      html += `</ul>`;
    }
    html += `</li>`;
  });
  html += `</ol><p class="plan-followup">You can revise the plan, share suggestions, or continue when you're ready.</p>`;
  return html;
}

function checkpointHtml(taskId, planHtml, showAction, buttonLabel) {
  if (!planHtml && !showAction) return "";
  const normalizedTaskId = String(taskId || "").trim();
  const isStarting = state.startingTaskIds.has(normalizedTaskId);
  const action = showAction
    ? `<button class="primary-btn checkpoint-start-btn" data-action="checkpoint-start" data-task-id="${escapeHtml(normalizedTaskId)}" ${isStarting ? 'disabled aria-disabled="true"' : ""}>${escapeHtml(isStarting ? "Starting..." : (buttonLabel || "Start research"))}</button>`
    : "";
  return `
    <article class="message assistant checkpoint-message">
      <div class="message-body markdown-body">${planHtml}</div>
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
  if (iteration?.is_direct_response) return "";
  const task = iteration?.task || {};
  if (String(task.status || "") !== "completed") return "";
  const report = iteration?.report || {};
  if (!report.has_report) return "";
  const suggestions = Array.isArray(iteration?.follow_up_suggestions) ? iteration.follow_up_suggestions : [];
  const lines = ["What would you like to do next?"];
  lines.push("");
  lines.push("For example, we could:");
  if (suggestions.length) {
    for (const item of suggestions.slice(0, 3)) {
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

function normalizeActivityText(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function activityStepStatusLabel(status) {
  const normalized = String(status || "").trim();
  if (normalized === "completed") return "Completed";
  if (normalized === "blocked") return "Blocked";
  if (normalized === "in_progress") return "In progress";
  return "Pending";
}

function activityStepStatusClass(status) {
  const normalized = String(status || "").trim();
  if (normalized === "completed") return "is-complete";
  if (normalized === "blocked") return "is-blocked";
  if (normalized === "in_progress") return "is-running";
  return "";
}

function summarizeActivityToolEntry(entry) {
  const summary = normalizeActivityText(entry?.summary || "");
  const result = normalizeActivityText(entry?.result || "");
  const status = String(entry?.status || "done").trim();
  const tool = normalizeActivityText(entry?.tool || entry?.raw_tool || "Tool");
  if (summary && result) return `${summary} -> ${result}`;
  if (summary) return status === "called" ? `${summary}...` : summary;
  return `${tool} querying...`;
}

function collectActivitySources(step) {
  const sources = [];
  const addSource = (value) => {
    const normalized = normalizeActivityText(value);
    if (!normalized || sources.includes(normalized)) return;
    sources.push(normalized);
  };

  const dataSources = Array.isArray(step?.data_sources) ? step.data_sources : [];
  for (const source of dataSources) {
    addSource(source);
  }

  if (sources.length) return sources;

  addSource(step?.source || step?.tool_hint || "");

  const toolLog = Array.isArray(step?.tool_log) ? step.tool_log : [];
  for (const entry of toolLog) {
    addSource(entry?.tool || "");
  }

  return sources;
}

function renderActivitySectionHtml(label, items = [], extraClass = "") {
  const lines = (Array.isArray(items) ? items : []).filter(Boolean);
  if (!lines.length) return "";
  const classNames = ["activity-log-step-section", extraClass].filter(Boolean).join(" ");
  return `
    <div class="${classNames}">
      <div class="activity-log-step-label">${escapeHtml(label)}</div>
      <div class="activity-log-step-list">
        ${lines.map((line) => `<div class="activity-log-step-item">${inlineMarkdown(line)}</div>`).join("")}
      </div>
    </div>
  `;
}

function buildActivityDetailsHtml(stepDetails = []) {
  const visibleSteps = (Array.isArray(stepDetails) ? stepDetails : [])
    .filter((step) => String(step?.status || "pending").trim() !== "pending");

  if (!visibleSteps.length) {
    return '<div class="activity-log-empty">Activity details will appear here as steps begin.</div>';
  }

  return `
    <div class="activity-log">
      ${visibleSteps.map((step) => {
        const sid = String(step?.id || "").trim() || "Step";
        const status = String(step?.status || "pending").trim();
        const statusLabel = activityStepStatusLabel(status);
        const statusClass = activityStepStatusClass(status);
        const goal = normalizeActivityText(step?.goal || "") || sid;
        const sources = collectActivitySources(step);
        const toolLog = Array.isArray(step?.tool_log) ? step.tool_log : [];
        const toolLines = toolLog.map((entry) => summarizeActivityToolEntry(entry)).filter(Boolean);
        const resultSummary = normalizeActivityText(step?.result_summary || "");
        const openGaps = (Array.isArray(step?.open_gaps) ? step.open_gaps : [])
          .map((item) => normalizeActivityText(item))
          .filter(Boolean);
        const evidenceIds = (Array.isArray(step?.evidence_ids) ? step.evidence_ids : [])
          .map((item) => normalizeActivityText(item))
          .filter(Boolean);
        const summaryLines = [];
        if (resultSummary) summaryLines.push(resultSummary);
        if (!summaryLines.length) {
          if (status === "in_progress") {
            summaryLines.push(sources.length ? `Working with ${sources.join(", ")}.` : "Working on this step.");
          } else if (status === "blocked") {
            summaryLines.push("This step needs follow-up before it can continue.");
          } else {
            summaryLines.push("Step completed.");
          }
        }

        const metaParts = [statusLabel];
        if (sources.length) metaParts.push(`Sources: ${sources.join(", ")}`);

        return `
          <div class="activity-log-step ${statusClass}">
            <div class="activity-log-step-head">
              <span class="activity-log-step-dot" aria-hidden="true"></span>
              <div class="activity-log-step-copy">
                <div class="activity-log-step-title">
                  <span class="activity-log-step-id">${escapeHtml(sid)}</span>
                  <span class="activity-log-step-goal">${escapeHtml(goal)}</span>
                </div>
                <div class="activity-log-step-meta">${escapeHtml(metaParts.join(" · "))}</div>
              </div>
            </div>
            <div class="activity-log-step-body">
              ${renderActivitySectionHtml("Summary", summaryLines, "is-summary")}
              ${renderActivitySectionHtml("Activity", toolLines)}
              ${renderActivitySectionHtml("Evidence", evidenceIds)}
              ${renderActivitySectionHtml("Open questions", openGaps)}
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function reactTraceLines({ trace = "", phases = null } = {}) {
  const lines = [];
  const normalizedTrace = String(trace || "").trim();
  const phaseMap = phases && typeof phases === "object" ? phases : null;
  const order = ["reason", "act", "observe", "conclude"];
  const hasPhases = phaseMap && order.some((key) => String(phaseMap[key] || "").trim());
  if (!normalizedTrace && !hasPhases) return lines;

  lines.push("**Tool Trace**");
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

function buildActivitySnapshot({ taskId = "", status = "", events = [], summaries = [], planApproved = true } = {}) {
  const normalizedStatus = String(status || "").trim();
  const safeEvents = (Array.isArray(events) ? events : []).filter((e) => String(e?.type || "") !== "plan.initializing");
  const safeSummaries = Array.isArray(summaries) ? summaries : [];
  const visibleStatuses = new Set(["running", "queued", "awaiting_hitl", "needs_clarification", "failed", "completed", "in_progress"]);
  const shouldShow = visibleStatuses.has(normalizedStatus) || safeEvents.length > 0 || safeSummaries.length > 0;
  if (!shouldShow) return null;

  const latestSummary = safeSummaries.length ? safeSummaries[safeSummaries.length - 1] : null;
  const latestEvent = safeEvents.length ? safeEvents[safeEvents.length - 1] : null;
  const latestLine = String(latestEvent?.human_line || latestEvent?.type || "").trim();

  const stepEvents = safeEvents.filter((e) => e?.type === "step.completed" && e?.metrics?.step_id);
  const toolCalledEvents = safeEvents.filter((e) => e?.type === "tool.called");
  const toolFailedEvents = safeEvents.filter((e) => e?.type === "tool.failed");
  const stepRetryEvents = safeEvents.filter((e) => e?.type === "step.retry");
  const stepStartedEvents = safeEvents.filter((e) => e?.type === "step.started" && e?.metrics?.step_id);
  const latestToolCalled = toolCalledEvents.length ? toolCalledEvents[toolCalledEvents.length - 1] : null;
  const latestToolFailed = toolFailedEvents.length ? toolFailedEvents[toolFailedEvents.length - 1] : null;
  const latestStepRetry = stepRetryEvents.length ? stepRetryEvents[stepRetryEvents.length - 1] : null;
  const latestStepStarted = stepStartedEvents.length ? stepStartedEvents[stepStartedEvents.length - 1] : null;
  const stepsCompleted = Number(latestSummary?.steps_completed || stepEvents.length || 0);
  const stepsTotal = Number(latestSummary?.steps_total || 0);
  const stepDetails = Array.isArray(latestSummary?.step_details) ? latestSummary.step_details : [];

  const currentStepFromDetails = stepDetails.find((s) => String(s?.status || "") === "in_progress")
    || stepDetails.find((s) => String(s?.status || "") === "pending");

  const summaryByStatus = {
    queued: "Preparing research workflow.",
    running: "Preparing to execute the plan\u2026",
    in_progress: "Preparing to execute the plan\u2026",
    awaiting_hitl: "Waiting at a checkpoint for your input.",
    completed: "Research run completed.",
    failed: "Run ended with an error.",
    needs_clarification: "Clarification required to proceed.",
  };

  const title = "Research log";

  let summary = "";
  if (normalizedStatus === "completed" && (stepsTotal > 0 || latestSummary?.summary)) {
    summary = String(latestSummary?.summary || "").trim() || `${stepsCompleted}/${stepsTotal} plan steps executed`;
  } else if (latestToolFailed) {
    summary = String(latestToolFailed?.human_line || "").trim() || "A tool call failed.";
  } else if (latestStepRetry) {
    summary = String(latestStepRetry?.human_line || "").trim() || "Retrying the current step.";
  } else if (latestToolCalled) {
    summary = String(latestToolCalled?.human_line || "").trim() || "Querying sources\u2026";
  } else if (latestStepStarted) {
    summary = String(latestStepStarted?.human_line || "").trim() || "Executing step\u2026";
  } else if (currentStepFromDetails) {
    const sid = String(currentStepFromDetails?.id || "").trim();
    const goal = String(currentStepFromDetails?.goal || "").trim();
    const source = String(currentStepFromDetails?.source || currentStepFromDetails?.tool_hint || "").trim();
    summary = source ? `Querying ${source}\u2026` : (sid && goal ? `${sid}: ${goal}` : "Executing step\u2026");
  } else if (stepEvents.length > 0) {
    summary = String(latestSummary?.summary || "").trim() || summaryByStatus[normalizedStatus] || "Tracking workflow progress.";
  } else {
    summary = String(latestSummary?.summary || "").trim() || latestLine || summaryByStatus[normalizedStatus] || "Tracking workflow progress.";
  }

  let preview = "";
  const completedStepIds = new Set(stepEvents.map((e) => String(e?.metrics?.step_id || "").trim()).filter(Boolean));
  const stepPips = stepEvents.map((e) => {
    const sid = String(e?.metrics?.step_id || "").trim();
    const st = String(e?.metrics?.step_status || "completed").trim();
    const icon = st === "completed" ? "✓" : st === "blocked" ? "✗" : "…";
    return `${sid} ${icon}`;
  });
  const inProgressStepId = latestStepStarted && !completedStepIds.has(String(latestStepStarted?.metrics?.step_id || "").trim())
    ? String(latestStepStarted.metrics?.step_id || "").trim()
    : "";
  if (inProgressStepId) {
    stepPips.push(`${inProgressStepId} …`);
  }
  if (stepPips.length) {
    preview = stepPips.join("  ·  ");
  } else if (latestLine) {
    preview = latestLine;
  } else if (normalizedStatus === "completed") {
    preview = "Execution finished. Expand to inspect details.";
  } else {
    preview = "Click for activity details";
  }

  return {
    taskId: String(taskId || "").trim() || "pending",
    status: normalizedStatus,
    title,
    summary,
    preview,
    detailsHtml: buildActivityDetailsHtml(stepDetails),
    planApproved: !!planApproved,
  };
}

function activityCardClassNames(status, expanded, planApproved = true) {
  const normalized = String(status || "").trim();
  const isActive = normalized === "running" || normalized === "in_progress" || normalized === "queued";
  const showSpinner = planApproved && isActive;
  return [
    "activity-card",
    expanded ? "expanded" : "",
    showSpinner ? "is-running" : "",
    normalized === "completed" ? "is-complete" : "",
    normalized === "failed" ? "is-error" : "",
    (normalized === "awaiting_hitl" || (isActive && !planApproved)) ? "is-paused" : "",
  ].filter(Boolean).join(" ");
}

function activityCardHtml(snapshot) {
  if (!snapshot) return "";
  const expanded = isActivityExpanded(snapshot.taskId);
  const status = String(snapshot.status || "").trim();
  const planApproved = snapshot.planApproved !== false;
  const classNames = activityCardClassNames(status, expanded, planApproved);

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
      <div class="activity-details ${expanded ? "" : "hidden"}">${snapshot.detailsHtml}</div>
    </section>
  `;
}

function patchActivityCardElement(card, snapshot) {
  if (!card || !snapshot) return;
  const expanded = isActivityExpanded(snapshot.taskId);
  const planApproved = snapshot.planApproved !== false;
  const classNames = activityCardClassNames(snapshot.status, expanded, planApproved);
  if (card.className !== classNames) card.className = classNames;
  if (card.dataset.taskId !== snapshot.taskId) card.dataset.taskId = snapshot.taskId;
  const ariaExpanded = expanded ? "true" : "false";
  if (card.getAttribute("aria-expanded") !== ariaExpanded) card.setAttribute("aria-expanded", ariaExpanded);

  const titleEl = card.querySelector(".activity-title");
  if (titleEl && titleEl.textContent !== snapshot.title) titleEl.textContent = snapshot.title;
  const summaryEl = card.querySelector(".activity-summary");
  if (summaryEl && summaryEl.textContent !== snapshot.summary) summaryEl.textContent = snapshot.summary;
  const previewEl = card.querySelector(".activity-preview");
  if (previewEl && previewEl.textContent !== snapshot.preview) previewEl.textContent = snapshot.preview;
  const detailsEl = card.querySelector(".activity-details");
  if (detailsEl) {
    if (detailsEl.innerHTML !== snapshot.detailsHtml) detailsEl.innerHTML = snapshot.detailsHtml;
    detailsEl.classList.toggle("hidden", !expanded);
  }
}

function getRunForTask(taskId) {
  const tid = String(taskId || "").trim();
  return tid ? (state.runsByTaskId[tid] || null) : null;
}

const RESEARCH_EVENT_TYPES = new Set([
  "plan.generated", "plan.retry", "plan.failed", "task.created", "step.completed", "step.started", "step.retry", "tool.called", "tool.failed",
  "synthesis.completed", "checkpoint.opened", "execution.paused", "execution.running", "run.completed",
]);

function hasResearchProgress(run) {
  const evs = Array.isArray(run?.progress_events) ? run.progress_events : [];
  return evs.some((e) => RESEARCH_EVENT_TYPES.has(String(e?.type || "")));
}

function minimalLoadingSpinnerHtml(label = "") {
  const labelHtml = label ? `<span class="loading-label">${escapeHtml(label)}</span>` : "";
  return `<article class="message assistant loading-spinner-only" data-role="loading-spinner"><span class="activity-wheel" aria-hidden="true"></span>${labelHtml}</article>`;
}

function pendingRunLabel() {
  const run = state.pendingRunId ? (state.runsByRunId[state.pendingRunId] || state.runsByTaskId["__pending__"]) : null;
  const evs = Array.isArray(run?.progress_events) ? run.progress_events : [];
  if (evs.some((e) => String(e?.type || "") === "plan.initializing")) return "Preparing a plan\u2026";
  return "";
}

function updateLoadingSpinnerLabel() {
  if (!el.messages) return;
  const spinner = el.messages.querySelector('[data-role="loading-spinner"]');
  if (!spinner) return;
  const label = pendingRunLabel();
  let labelEl = spinner.querySelector(".loading-label");
  if (label) {
    if (!labelEl) {
      labelEl = document.createElement("span");
      labelEl.className = "loading-label";
      spinner.appendChild(labelEl);
    }
    labelEl.textContent = label;
  } else if (labelEl) {
    labelEl.remove();
  }
}

function iterationActivitySnapshot(iteration) {
  if (iteration?.is_direct_response) return null;
  const task = iteration?.task || {};
  const taskId = String(task.task_id || "").trim();
  if (!taskId) return null;
  if (state.startingTaskIds.has(taskId)) {
    return buildActivitySnapshot({
      taskId,
      status: "running",
      events: [{ type: "execution.running", human_line: "Executing plan..." }],
      summaries: [],
      planApproved: true,
    });
  }
  const run = getRunForTask(taskId);
  const taskStatus = String(task.status || "").trim();
  const runStatus = String(run?.status || "").trim();
  const hasActiveRun = run && ["running", "queued", "in_progress"].includes(runStatus);
  if (!hasActiveRun && !taskHasStarted(task) && taskStatus !== "completed" && taskStatus !== "failed") return null;
  const researchLog = iteration?.research_log || {};
  const runEvents = Array.isArray(run?.progress_events) ? run.progress_events : [];
  const runSummaries = Array.isArray(run?.progress_summaries) ? run.progress_summaries : [];
  const logEvents = Array.isArray(researchLog.events) ? researchLog.events : [];
  const logSummaries = Array.isArray(researchLog.summaries) ? researchLog.summaries : [];
  const status = String(run?.status || task.status || researchLog.status || "").trim();
  // For completed runs: prefer whichever source has more step completion data (handles edge cases
  // where run may be stale or research_log from persisted task has fuller data after refresh)
  const stepCount = (arr) => arr.filter((e) => e?.type === "step.completed" && e?.metrics?.step_id).length;
  const useLogForCompleted =
    (status === "completed" || status === "failed") &&
    stepCount(logEvents) > stepCount(runEvents);
  const events = useLogForCompleted ? logEvents : runEvents.length ? runEvents : logEvents;
  const summaries = useLogForCompleted ? logSummaries : runSummaries.length ? runSummaries : logSummaries;
  // Plan is approved if we have an active run executing, or task's hitl_history shows continue
  const planApproved = (run && ["running", "queued", "in_progress"].includes(status)) || taskHasStarted(task);
  return buildActivitySnapshot({ taskId, status, events, summaries, planApproved });
}

function pendingActivitySnapshot() {
  const run = state.pendingRunId ? (state.runsByRunId[state.pendingRunId] || state.runsByTaskId["__pending__"]) : null;
  const taskId = String(run?.task_id || "").trim() || "pending";
  const status = String(run?.status || "queued").trim();
  const events = Array.isArray(run?.progress_events) ? run.progress_events : [];
  const summaries = Array.isArray(run?.progress_summaries) ? run.progress_summaries : [];
  return buildActivitySnapshot({ taskId, status, events, summaries });
}

function taskHasStarted(task) {
  const history = Array.isArray(task?.hitl_history) ? task.hitl_history : [];
  return history.includes("approve") || history.includes("continue");
}

function placeActivityAfterPlan(task, activeRun) {
  const taskId = String(task?.task_id || "").trim();
  if (taskId && state.startingTaskIds.has(taskId)) return true;
  const status = String(activeRun?.status || task?.status || "").trim();
  // When we have an active run that's executing, plan was approved - place below immediately
  // (avoids delay until conversation is re-fetched with updated hitl_history)
  if (activeRun && ["running", "queued", "in_progress"].includes(status)) return true;
  // When plan hasn't been approved/started, always show research log above the plan
  if (!taskHasStarted(task)) return false;
  if (["running", "queued", "in_progress", "completed", "failed", "needs_clarification"].includes(status)) return true;
  if (status === "awaiting_hitl") return true;
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
  let html = "";
  if (!state.conversations.length) {
    html = '<p class="muted">No chats yet.</p>';
    setInnerHtmlIfChanged(el.tasksList, html);
    return;
  }
  html = state.conversations
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
  setInnerHtmlIfChanged(el.tasksList, html);
}

function renderTaskHeader() {
  const conversation = state.selectedConversationDetail?.conversation;
  if (!conversation) {
    if (el.taskTitle.textContent !== "New chat") el.taskTitle.textContent = "New chat";
    return;
  }
  const title = conversationTitle(conversation);
  const status = String(conversation.latest_status || "");
  const count = Number(conversation.iteration_count || 0);
  const nextTitle = `${title} · ${status} · ${count} iteration${count === 1 ? "" : "s"}`;
  if (el.taskTitle.textContent !== nextTitle) el.taskTitle.textContent = nextTitle;
}

function renderMessages() {
  const detail = state.selectedConversationDetail;

  if (!detail && state.clarificationMessage) {
    const parts = [];
    if (state.pendingUserMessage) {
      parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`);
      const pendingRun = state.pendingRunId ? (state.runsByRunId[state.pendingRunId] || state.runsByTaskId["__pending__"]) : null;
      if (pendingRun && hasResearchProgress(pendingRun)) {
        const pendingCard = activityCardHtml(pendingActivitySnapshot());
        if (pendingCard) parts.push(pendingCard);
      } else {
        parts.push(minimalLoadingSpinnerHtml(pendingRunLabel()));
      }
    }
    parts.push(`<article class="message assistant"><div class="message-body markdown-body">${markdownToHtml(state.clarificationMessage)}</div></article>`);
    if (setInnerHtmlIfChanged(el.messages, parts.join(""))) {
      el.messages.scrollTop = el.messages.scrollHeight;
    }
    return;
  }

  if (!detail) {
    if (state.pendingUserMessage) {
      const parts = [
        `<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`,
      ];
      const pendingRun = state.pendingRunId ? (state.runsByRunId[state.pendingRunId] || state.runsByTaskId["__pending__"]) : null;
      if (pendingRun && hasResearchProgress(pendingRun)) {
        const pendingCard = activityCardHtml(pendingActivitySnapshot());
        if (pendingCard) parts.push(pendingCard);
      } else {
        parts.push(minimalLoadingSpinnerHtml(pendingRunLabel()));
      }
      if (setInnerHtmlIfChanged(el.messages, parts.join(""))) {
        el.messages.scrollTop = el.messages.scrollHeight;
      }
      return;
    }
    setInnerHtmlIfChanged(el.messages, "");
    return;
  }

  const iterations = Array.isArray(detail.iterations) ? detail.iterations : [];
  const parts = [];

  for (const iteration of iterations) {
    const task = iteration?.task || {};
    const userText = String(task.user_query || task.objective || "").trim() || "(empty query)";
    parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(userText)}</pre></article>`);

    if (iteration?.is_direct_response) {
      const directText = String(iteration?.direct_response_text || "").trim();
      if (directText) {
        parts.push(`<article class="message assistant"><div class="message-body markdown-body">${markdownToHtml(directText)}</div></article>`);
      }
      continue;
    }

    const runForTask = getRunForTask(task.task_id);
    const activityCard = activityCardHtml(iterationActivitySnapshot(iteration));
    const shouldPlaceAfterPlan = placeActivityAfterPlan(task, runForTask);
    if (activityCard && !shouldPlaceAfterPlan) parts.push(activityCard);

    const branchLabel = String(iteration?.branch_label || "").trim();
    if (branchLabel) {
      parts.push(
        `<article class="message assistant"><div class="message-body"><span class="branch-badge">${escapeHtml(branchLabel)}</span></div></article>`
      );
    }

    const planHtml = planHtmlForIteration(iteration);
    const awaiting = Boolean(task.awaiting_hitl);
    const buttonLabel = task.hitl_history && (task.hitl_history.includes("approve") || task.hitl_history.includes("continue")) ? "Approve plan" : "Start research";
    parts.push(checkpointHtml(task.task_id, planHtml, awaiting, buttonLabel));
    if (activityCard && shouldPlaceAfterPlan) parts.push(activityCard);

    const reportCard = reportCardHtml(iteration);
    if (reportCard) parts.push(reportCard);

    const suggestions = followUpSuggestionsHtml(iteration);
    if (suggestions) parts.push(suggestions);
  }

  if (state.pendingUserMessage) {
    parts.push(`<article class="message user"><pre class="message-body">${escapeHtml(state.pendingUserMessage)}</pre></article>`);
    const pendingRun = state.pendingRunId ? (state.runsByRunId[state.pendingRunId] || state.runsByTaskId["__pending__"]) : null;
    if (pendingRun && hasResearchProgress(pendingRun)) {
      const pendingCard = activityCardHtml(pendingActivitySnapshot());
      if (pendingCard) parts.push(pendingCard);
    } else {
      parts.push(minimalLoadingSpinnerHtml(pendingRunLabel()));
    }
  }

  if (setInnerHtmlIfChanged(el.messages, parts.join(""))) {
    el.messages.scrollTop = el.messages.scrollHeight;
  }
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

function currentDebugTaskId() {
  const iteration = currentReportIteration();
  return String(iteration?.task?.task_id || "").trim();
}

function clearRenderedReportPanel() {
  state.renderedReportTaskId = "";
  state.renderedReportTitle = "";
  state.renderedReportMarkdown = "";
  el.reportTitle.textContent = "Research Report";
  el.reportContent.innerHTML = "";
}

function setRenderedReportTitle(taskId, title) {
  const normalizedTaskId = String(taskId || "").trim();
  const normalizedTitle = String(title || "");
  if (
    state.renderedReportTaskId === normalizedTaskId
    && state.renderedReportTitle === normalizedTitle
  ) {
    return;
  }
  el.reportTitle.textContent = normalizedTitle || "Research Report";
  state.renderedReportTaskId = normalizedTaskId;
  state.renderedReportTitle = normalizedTitle;
}

function setRenderedReportMarkdown(taskId, markdown, { preserveScroll = true } = {}) {
  const normalizedTaskId = String(taskId || "").trim();
  const normalizedMarkdown = String(markdown || "");
  const sameTask = preserveScroll && state.renderedReportTaskId === normalizedTaskId;
  if (sameTask && state.renderedReportMarkdown === normalizedMarkdown) return;
  const scrollTop = sameTask ? el.reportContent.scrollTop : 0;
  const scrollLeft = sameTask ? el.reportContent.scrollLeft : 0;
  el.reportContent.innerHTML = markdownToHtml(normalizedMarkdown);
  state.renderedReportTaskId = normalizedTaskId;
  state.renderedReportMarkdown = normalizedMarkdown;
  if (sameTask) {
    el.reportContent.scrollTop = scrollTop;
    el.reportContent.scrollLeft = scrollLeft;
  }
}

function setDebugSummaryHtml(taskId, html) {
  const normalizedTaskId = String(taskId || "").trim();
  if (
    String(el.debugSummary.dataset.taskId || "") === normalizedTaskId
    && el.debugSummary.innerHTML === html
  ) {
    return;
  }
  el.debugSummary.innerHTML = html;
  el.debugSummary.dataset.taskId = normalizedTaskId;
}

function setDebugJsonText(taskId, text) {
  const normalizedTaskId = String(taskId || "").trim();
  const sameTask = String(el.debugJson.dataset.taskId || "") === normalizedTaskId;
  if (sameTask && el.debugJson.textContent === text) return;
  const scrollTop = sameTask ? el.debugJson.scrollTop : 0;
  const scrollLeft = sameTask ? el.debugJson.scrollLeft : 0;
  el.debugJson.textContent = text;
  el.debugJson.dataset.taskId = normalizedTaskId;
  if (sameTask) {
    el.debugJson.scrollTop = scrollTop;
    el.debugJson.scrollLeft = scrollLeft;
  }
}

function countCollectionItems(value) {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === "object") return Object.keys(value).length;
  return 0;
}

function escapePreformattedJson(value) {
  return escapeHtml(JSON.stringify(value || {}, null, 2));
}

function debugSummaryHtml(debugPayload) {
  const payload = debugPayload && typeof debugPayload === "object" ? debugPayload : {};
  const rawState = payload.state && typeof payload.state === "object" ? payload.state : {};
  const workflow = rawState.workflow_task_state && typeof rawState.workflow_task_state === "object"
    ? rawState.workflow_task_state
    : {};
  const evidenceStore = workflow.evidence_store && typeof workflow.evidence_store === "object"
    ? workflow.evidence_store
    : {};
  const executionMetrics = workflow.execution_metrics && typeof workflow.execution_metrics === "object"
    ? workflow.execution_metrics
    : {};
  const executionSummary = executionMetrics.summary && typeof executionMetrics.summary === "object"
    ? executionMetrics.summary
    : {};
  const steps = Array.isArray(workflow.steps) ? workflow.steps : [];
  const completedSteps = steps.filter((step) => String(step?.status || "").trim() === "completed").length;
  const priorResearch = Array.isArray(rawState.co_scientist_prior_research) ? rawState.co_scientist_prior_research : [];
  const watchlist = Array.isArray(executionSummary.specialization_watchlist)
    ? executionSummary.specialization_watchlist
    : [];
  const metrics = [
    { label: "Source", value: String(payload.source || "none") || "none" },
    { label: "Plan", value: String(workflow.plan_status || "unknown") || "unknown" },
    { label: "Steps", value: steps.length ? `${completedSteps}/${steps.length}` : "0" },
    {
      label: "Evidence",
      value: `${countCollectionItems(evidenceStore.entities)} entities / ${countCollectionItems(evidenceStore.claims)} claims / ${countCollectionItems(evidenceStore.evidence)} evidence`,
    },
    {
      label: "Exec metrics",
      value: String(executionSummary.total_steps || steps.length || 0),
    },
    { label: "Prior research", value: String(priorResearch.length) },
    { label: "Approval pending", value: rawState.co_scientist_plan_pending_approval ? "yes" : "no" },
    {
      label: "Persisted",
      value: payload.persisted_updated_at ? formatDate(payload.persisted_updated_at) : "n/a",
    },
  ];

  const metricHtml = metrics
    .map((item) => `
      <div class="debug-metric">
        <span class="debug-metric-label">${escapeHtml(item.label)}</span>
        <span class="debug-metric-value">${escapeHtml(item.value)}</span>
      </div>
    `)
    .join("");

  const watchlistHtml = watchlist.length
    ? `
      <div class="debug-watchlist">
        ${watchlist.slice(0, 6).map((item) => `<span class="debug-pill">${escapeHtml(String(item || "").trim())}</span>`).join("")}
      </div>
    `
    : "";

  return `
    <div class="debug-metrics-grid">${metricHtml}</div>
    ${watchlistHtml}
  `;
}

async function refreshCurrentDebugState({ force = false } = {}) {
  const taskId = currentDebugTaskId();
  if (!taskId || !state.debugOpen) return;
  const existing = state.debugByTaskId[taskId];
  if (existing?.loading) return;
  if (!force && existing && existing.data) return;

  const hasExistingData = Boolean(existing?.data);
  state.debugByTaskId[taskId] = hasExistingData
    ? { loading: true, error: "", data: existing.data }
    : { loading: true, error: "", data: null };
  if (!hasExistingData) renderReportPanel();

  try {
    const payload = await api(`/api/tasks/${encodeURIComponent(taskId)}/debug/workflow-state`);
    state.debugByTaskId[taskId] = { loading: false, error: "", data: payload };
  } catch (err) {
    state.debugByTaskId[taskId] = hasExistingData
      ? { loading: false, error: "", data: existing.data }
      : {
          loading: false,
          error: String(err?.message || "Failed to load debug state."),
          data: null,
        };
  }

  renderReportPanel();
}

function renderReportPanel() {
  const iteration = currentReportIteration();
  const showPanel = Boolean(state.selectedConversationId && iteration && iteration?.report?.has_report);
  const debugTaskId = currentDebugTaskId();
  const debugEntry = debugTaskId ? state.debugByTaskId[debugTaskId] : null;
  const showDebugToggle = Boolean(showPanel && debugTaskId);
  const showDebugPanel = Boolean(showPanel && state.debugOpen && debugTaskId);

  el.workspace.classList.toggle("report-open", showPanel);
  el.reportPanel.classList.toggle("hidden", !showPanel);

  if (!showPanel) {
    clearRenderedReportPanel();
    el.exportPdfBtn.classList.add("hidden");
    el.debugToggleBtn.classList.add("hidden");
    el.reportStatus.classList.add("hidden");
    el.reportStatus.classList.remove("error");
    el.debugPanel.classList.add("hidden");
    setDebugSummaryHtml("", "");
    setDebugJsonText("", "");
    return;
  }

  const task = iteration.task || {};
  const taskId = String(task.task_id || "");
  const reportMarkdown = String(iteration?.report?.report_markdown || "").trim();
  const reportTaskChanged = state.renderedReportTaskId !== taskId;
  setRenderedReportTitle(taskId, String(task.title || task.user_query || "Research Report"));
  setRenderedReportMarkdown(taskId, reportMarkdown, { preserveScroll: !reportTaskChanged });
  el.exportPdfBtn.classList.toggle("hidden", !taskId);
  el.exportPdfBtn.disabled = state.exportingPdf;
  el.debugToggleBtn.classList.toggle("hidden", !showDebugToggle);
  el.debugToggleBtn.textContent = state.debugOpen ? "Hide Debug" : "Show Debug";

  const shouldShowStatus = state.reportStatusTaskId === taskId && String(state.reportStatusText || "").trim().length > 0;
  el.reportStatus.classList.toggle("hidden", !shouldShowStatus);
  el.reportStatus.classList.toggle("error", shouldShowStatus && state.reportStatusError);
  el.reportStatus.textContent = shouldShowStatus ? state.reportStatusText : "";

  el.debugPanel.classList.toggle("hidden", !showDebugPanel);
  if (!showDebugPanel) {
    setDebugSummaryHtml("", "");
    setDebugJsonText("", "");
    return;
  }

  if (!debugEntry || (debugEntry.loading && !debugEntry.data)) {
    setDebugSummaryHtml(debugTaskId, '<div class="debug-empty">Loading workflow debug state...</div>');
    setDebugJsonText(debugTaskId, "");
    return;
  }

  if (debugEntry.error && !debugEntry.data) {
    setDebugSummaryHtml(debugTaskId, `<div class="debug-empty debug-error">${escapeHtml(debugEntry.error)}</div>`);
    setDebugJsonText(debugTaskId, "");
    return;
  }

  const debugPayload = debugEntry.data || {};
  setDebugSummaryHtml(debugTaskId, debugSummaryHtml(debugPayload));
  setDebugJsonText(debugTaskId, JSON.stringify(debugPayload.state || {}, null, 2));
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

async function refreshConversations({ keepSelection = true, skipRender = false } = {}) {
  const payload = await api("/api/conversations");
  state.conversations = Array.isArray(payload?.conversations) ? payload.conversations : [];

  if (!keepSelection) {
    state.selectedConversationId = null;
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    if (!skipRender) renderAll();
    return;
  }

  const stillExists = state.selectedConversationId
    ? state.conversations.some((conversation) => String(conversation.conversation_id || "") === state.selectedConversationId)
    : false;

  if (!stillExists) {
    state.selectedConversationId = null;
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    if (!skipRender) renderAll();
    return;
  }

  if (state.selectedConversationId) {
    await selectConversation(state.selectedConversationId, { silent: true, skipRender });
  } else {
    if (!skipRender) renderAll();
  }
}

async function selectConversation(conversationId, { silent = false, skipRender = false } = {}) {
  state.selectedConversationId = conversationId;
  if (!conversationId) {
    state.selectedConversationDetail = null;
    state.selectedReportTaskId = null;
    if (!skipRender) renderAll();
    return;
  }
  try {
    const detail = await api(`/api/conversations/${encodeURIComponent(conversationId)}`);
    state.selectedConversationDetail = detail;
    const iterations = Array.isArray(detail?.iterations) ? detail.iterations : [];
    for (const it of iterations) {
      const runId = it?.task?.active_run_id;
      if (runId && !state.activeRunIds.has(runId)) {
        try {
          const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
          storeRunData(run);
          const terminal = ["completed", "failed", "awaiting_hitl", "needs_clarification"].includes(String(run?.status || ""));
          if (!terminal) startRunPolling(runId);
        } catch {
          /* run may have finished, ignore */
        }
      }
    }
    const defaultReportTaskId = String(detail?.conversation?.selected_report_task_id || "").trim();
    const latestDone = latestCompletedTaskId(detail);
    if (!state.selectedReportTaskId || !findIteration(detail, state.selectedReportTaskId)) {
      state.selectedReportTaskId = defaultReportTaskId || latestDone || "";
    }
    state.pendingUserMessage = "";
    state.clarificationMessage = "";
    if (!skipRender) renderAll();
    refreshCurrentDebugState().catch((err) => setNotice(`Failed to load debug state: ${err.message}`, true));
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

function storeRunData(run) {
  if (!run?.run_id) return;
  state.runsByRunId[run.run_id] = run;
  if (run.task_id) {
    state.runsByTaskId[run.task_id] = run;
  } else if (run.run_id === state.pendingRunId) {
    state.runsByTaskId["__pending__"] = run;
  }
}

function stopRunPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.activeRunIds.clear();
  state.pollInFlight = false;
  setLoading(false);
}

function ensurePollTimerRunning() {
  if (state.pollTimer || state.activeRunIds.size === 0) return;
  setLoading(true);
  const poll = async () => {
    if (state.pollInFlight) return;
    const ids = Array.from(state.activeRunIds);
    if (ids.length === 0) {
      if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
      setLoading(false);
      return;
    }
    state.pollInFlight = true;
    try {
      const results = await Promise.allSettled(
        ids.map((runId) => api(`/api/runs/${encodeURIComponent(runId)}`))
      );
      for (let i = 0; i < ids.length; i++) {
        const runId = ids[i];
        const r = results[i];
        if (r.status === "rejected") {
          state.activeRunIds.delete(runId);
          setNotice(`Run polling failed: ${r.reason?.message || "Unknown error"}`, true);
          continue;
        }
        const run = r.value;
        storeRunData(run);
        updateInlineActivityCard(run);
        updateLoadingSpinnerLabel();
        await handleTerminalRunState(run);
      }
    } finally {
      state.pollInFlight = false;
      if (state.activeRunIds.size === 0 && state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        setLoading(false);
      }
    }
  };
  poll();
  state.pollTimer = setInterval(poll, 1000);
}

function startRunPolling(runId) {
  if (!runId) return;
  state.activeRunIds.add(runId);
  ensurePollTimerRunning();
}

async function handleTerminalRunState(run) {
  if (!run) return;
  const status = String(run.status || "");
  const kind = String(run.kind || "");
  const isQueuedFeedbackAck = status === "queued" && kind === "feedback_task";
  const terminalStates = new Set(["completed", "failed", "awaiting_hitl", "needs_clarification"]);
  if (!terminalStates.has(status) && !isQueuedFeedbackAck) return;
  if (state.handlingTerminalRunIds.has(run.run_id)) return;

  state.handlingTerminalRunIds.add(run.run_id);
  try {
    if (run.task_id) {
      state.startingTaskIds.delete(String(run.task_id || "").trim());
    }

    state.activeRunIds.delete(run.run_id);
    if (state.pendingRunId === run.run_id) state.pendingRunId = null;
    if (state.activeRunIds.size === 0 && state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      setLoading(false);
    }

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
        state.selectedConversationId = conversationId;
        await refreshConversations({ keepSelection: true, skipRender: true });
        if (status === "completed") {
          state.selectedReportTaskId = run.task_id;
        }
      } catch (err) {
        setNotice(`Could not refresh conversations: ${err.message}`, true);
      }
    } else {
      try {
        await refreshConversations({ keepSelection: true, skipRender: true });
      } catch (err) {
        setNotice(`Could not refresh conversations: ${err.message}`, true);
      }
    }

    renderAll();
  } finally {
    state.handlingTerminalRunIds.delete(run.run_id);
  }
}

async function submitNewQuery(query, { conversationId = null, parentTaskId = null } = {}) {
  state.pendingUserMessage = String(query || "").trim();
  state.clarificationMessage = "";
  setActivityExpanded("pending", false);
  renderAll();

  const requestBody = { query };
  if (conversationId) requestBody.conversation_id = conversationId;
  if (parentTaskId) requestBody.parent_task_id = parentTaskId;

  const payload = await api("/api/query", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });

  state.pendingRunId = payload.run_id;
  storeRunData(payload);
  renderMessages();
  startRunPolling(payload.run_id);
}

async function submitContinue(taskId) {
  const normalizedTaskId = String(taskId || "").trim();
  if (!normalizedTaskId || state.startingTaskIds.has(normalizedTaskId)) return;
  const detail = state.selectedConversationDetail;
  const iteration = findIteration(detail, normalizedTaskId);
  const task = iteration?.task || null;
  const planVersionId = iteration?.active_plan_version?.version_id || null;
  state.startingTaskIds.add(normalizedTaskId);
  if (task) task.awaiting_hitl = false;
  renderMessages();
  let payload;
  try {
    try {
      payload = await api(`/api/tasks/${encodeURIComponent(normalizedTaskId)}/start`, {
        method: "POST",
        body: JSON.stringify({ plan_version_id: planVersionId }),
      });
    } catch (err) {
      if (String(err?.message || "").trim() !== "Not Found") {
        throw err;
      }
      payload = await api(`/api/tasks/${encodeURIComponent(normalizedTaskId)}/continue`, {
        method: "POST",
        body: JSON.stringify({}),
      });
    }
  } catch (err) {
    state.startingTaskIds.delete(normalizedTaskId);
    renderMessages();
    throw err;
  }
  storeRunData(payload);
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
  storeRunData(payload);
  renderMessages();
  startRunPolling(payload.run_id);
}

function clearDraft() {
  state.selectedConversationId = null;
  state.selectedConversationDetail = null;
  state.selectedReportTaskId = null;
  state.debugOpen = false;
  state.debugByTaskId = {};
  stopRunPolling();
  state.runsByRunId = {};
  state.runsByTaskId = {};
  state.pendingRunId = null;
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
      if (activityCard.classList.contains("expanded") && event.target.closest(".activity-details")) return;
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
      startBtn.disabled = true;
      startBtn.textContent = "Starting...";
      submitContinue(taskId).catch((err) => {
        renderMessages();
        setNotice(`Start failed: ${err.message}`, true);
      });
      return;
    }

    const reportCard = event.target.closest('[data-action="open-report"]');
    if (reportCard) {
      const taskId = String(reportCard.dataset.taskId || "").trim();
      if (!taskId) return;
      const scrollTop = el.messages.scrollTop;
      state.selectedReportTaskId = taskId;
      renderAll();
      refreshCurrentDebugState().catch((err) => setNotice(`Failed to load debug state: ${err.message}`, true));
      el.messages.scrollTop = scrollTop;
    }
  });

  el.messages.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const activityCard = event.target.closest('[data-action="toggle-activity"]');
    if (activityCard) {
      if (activityCard.classList.contains("expanded") && event.target.closest(".activity-details")) return;
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
    const scrollTop = el.messages.scrollTop;
    state.selectedReportTaskId = taskId;
    renderAll();
    refreshCurrentDebugState().catch((err) => setNotice(`Failed to load debug state: ${err.message}`, true));
    el.messages.scrollTop = scrollTop;
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

  el.debugToggleBtn.addEventListener("click", () => {
    state.debugOpen = !state.debugOpen;
    renderReportPanel();
    refreshCurrentDebugState().catch((err) => setNotice(`Failed to load debug state: ${err.message}`, true));
  });

  const exampleContainer = document.getElementById("exampleQueries");
  if (exampleContainer) {
    exampleContainer.addEventListener("click", (event) => {
      const chip = event.target.closest(".example-query");
      if (!chip) return;
      const query = chip.dataset.query || chip.textContent.trim();
      el.promptInput.value = query;
      updateSendVisibility();
      el.promptInput.focus();
    });
  }
}

async function bootstrap() {
  bindEvents();
  updatePromptPlaceholder();
  updateSendVisibility();
  renderAll();
  await refreshHealth();
  await refreshConversations({ keepSelection: true, skipRender: true });
  renderAll();
}

bootstrap().catch((err) => setNotice(`UI initialization failed: ${err.message}`, true));
