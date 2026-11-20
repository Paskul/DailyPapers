// paperwatch monitor page. Reads /api/snapshot every REFRESH_MS and renders it; counters tick every second.
// Paper titles and error texts are untrusted: they are only ever inserted with textContent, never as HTML.
"use strict";

const REFRESH_MS = 15000;
const PAST_WIDTH_SHARE = 0.68;
const SVG_NS = "http://www.w3.org/2000/svg";
const ROLE_COLORS = {  // fixed categorical order (reference palette), never cycled
  section_reader: "var(--series-1)", reviewer: "var(--series-2)", topic_assessor: "var(--series-3)",
  research: "var(--series-4)", comparison: "var(--series-5)", writer: "var(--series-6)",
  healthcheck: "var(--series-neutral)",
};
const JOB_OUTCOME_COLORS = {
  ok: "var(--text-muted)", running: "var(--accent)", retry: "var(--warning)", released: "var(--warning)",
  paused: "var(--warning)", lease_lost: "var(--warning)", failed: "var(--critical)", interrupted: "var(--critical)",
  stopped: "var(--warning)",
};
const JOB_OUTCOME_LABELS = { ok: "done", running: "running", retry: "will retry", released: "returned to queue",
  paused: "model paused", lease_lost: "lease lost", failed: "failed", interrupted: "interrupted", stopped: "stopped" };

const state = {
  snapshot: null,
  historyHours: readStoredNumber("historyHours", 24),
  eventFilter: "all",
  lastSuccess: null,
  timeFormat: null, dayFormat: null, dateTimeFormat: null, partsFormat: null,
};

function readStoredNumber(key, fallback) {
  try { const value = Number(localStorage.getItem("paperwatch-monitor-" + key)); return value > 0 ? value : fallback; }
  catch (error) { return fallback; }
}
function storeValue(key, value) {
  try { localStorage.setItem("paperwatch-monitor-" + key, String(value)); } catch (error) { /* private mode */ }
}

// ---------------------------------------------------------------------------------------------------------------
// Small DOM and time helpers
// ---------------------------------------------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function svg(tag, attributes) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes || {})) node.setAttribute(name, String(value));
  return node;
}
function parseTime(text) { return text ? new Date(text) : null; }

function setupFormatters(timeZone) {
  if (state.partsFormat && state.partsFormat.resolvedOptions().timeZone === timeZone) return;
  const zone = { timeZone };
  state.timeFormat = new Intl.DateTimeFormat(undefined, { ...zone, hour: "2-digit", minute: "2-digit" });
  state.dayFormat = new Intl.DateTimeFormat(undefined, { ...zone, weekday: "short", day: "numeric", month: "short" });
  state.dateTimeFormat = new Intl.DateTimeFormat(undefined, { ...zone, weekday: "short", hour: "2-digit",
                                                               minute: "2-digit" });
  state.partsFormat = new Intl.DateTimeFormat("en-GB", { ...zone, hour: "2-digit", minute: "2-digit", hour12: false,
                                                         year: "numeric", month: "2-digit", day: "2-digit" });
}
function formatTime(date) { return date ? state.timeFormat.format(date) : "—"; }
function formatDayTime(date) { return date ? state.dateTimeFormat.format(date) : "—"; }
function localParts(date) {
  const parts = {};
  for (const part of state.partsFormat.formatToParts(date)) parts[part.type] = part.value;
  return { hour: Number(parts.hour) % 24, minute: Number(parts.minute), day: `${parts.year}-${parts.month}-${parts.day}` };
}
function formatDuration(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  const hours = Math.floor(seconds / 3600), minutes = Math.floor((seconds % 3600) / 60), rest = seconds % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  return `${rest}s`;
}
function relativeTo(date) {
  if (!date) return "";
  const difference = date.getTime() - Date.now();
  return difference >= 0 ? `in ${formatDuration(difference)}` : `${formatDuration(-difference)} ago`;
}
// Elements with data-since / data-until are refreshed by the one-second ticker.
function liveSince(date, prefix) {
  const node = el("span", "elapsed");
  if (date) { node.dataset.since = date.toISOString(); node.dataset.prefix = prefix || ""; }
  return node;
}
function liveUntil(date) {
  const node = el("span", "countdown");
  if (date) node.dataset.until = date.toISOString();
  return node;
}
function tick() {
  const now = Date.now();
  for (const node of document.querySelectorAll("[data-since]")) {
    node.textContent = node.dataset.prefix + formatDuration(now - new Date(node.dataset.since).getTime());
  }
  for (const node of document.querySelectorAll("[data-until]")) node.textContent = relativeTo(new Date(node.dataset.until));
  if (state.lastSuccess) {
    document.getElementById("refresh-state").textContent = `updated ${formatDuration(now - state.lastSuccess)} ago`;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Fetch loop
// ---------------------------------------------------------------------------------------------------------------

async function refresh() {
  const banner = document.getElementById("banner");
  try {
    const response = await fetch(`/api/snapshot?hours=${encodeURIComponent(state.historyHours)}`,
                                 { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error + (payload.hint ? ` — ${payload.hint}` : ""));
    state.snapshot = payload;
    state.lastSuccess = Date.now();
    banner.hidden = true;
    render(payload);
  } catch (error) {
    banner.hidden = false;
    banner.textContent = `Cannot read the snapshot: ${error.message}. Showing the last data received.`;
  }
  tick();
}

function render(snapshot) {
  setupFormatters(snapshot.meta.timezone);
  const meta = snapshot.meta;
  document.getElementById("home-line").textContent =
    `${meta.home} · topics: ${meta.topics.join(", ") || "none"} · times in ${meta.timezone}`;
  document.getElementById("footer").textContent =
    `paperwatch ${meta.version} · schema v${meta.schema_version} · ` + (meta.activity_log
      ? "exact steps from the activity log"
      : "this database predates the activity log (schema v10): current steps are inferred from jobs and model calls");
  renderStatusCards(snapshot);
  renderCurrentJobs(snapshot.now.worker);
  renderTimeline(snapshot);
  renderPlan(snapshot.plan, snapshot.now.report);
  renderQueue(snapshot.plan.queue);
  renderEvents(snapshot.history.events);
  renderDays(snapshot.history.days);
  renderBudgets(snapshot.plan.budgets);
}

// ---------------------------------------------------------------------------------------------------------------
// Now
// ---------------------------------------------------------------------------------------------------------------

function statusCard(title, dotClass, stateText, lines) {
  const card = el("div", "status-card");
  card.append(el("div", "card-title", title));
  const stateLine = el("div", "card-state");
  stateLine.append(el("span", `dot ${dotClass}`), el("span", null, stateText));
  card.append(stateLine);
  for (const line of lines) if (line) card.append(line instanceof Node ? line : el("div", "card-line", line));
  return card;
}
function lineWith(text, liveNode) {
  const line = el("div", "card-line", text);
  if (liveNode) line.append(liveNode);
  return line;
}

function renderStatusCards(snapshot) {
  const { worker, collector, report, model } = snapshot.now;
  const grid = document.getElementById("status-grid");
  grid.replaceChildren();

  const workerStates = { working: ["busy", "Working"], idle: ["idle", "Idle — nothing due"],
    paused: ["warning", "Paused (model server down)"], stopped: ["warning", "Service stopped"],
    not_running: ["idle", "No worker running"] };
  const [workerDot, workerText] = workerStates[worker.state] || ["idle", worker.state];
  const workerLines = [];
  if (worker.jobs.length) workerLines.push(`${worker.jobs.length} job(s) in progress`);
  else if (worker.next_due_at) {
    const due = parseTime(worker.next_due_at);
    workerLines.push(due <= new Date() ? "jobs are due: it picks the next one shortly" : lineWith("next job due ", liveUntil(due)));
  }
  if (worker.service) workerLines.push(`${worker.service.unit}: ${worker.service.active_state}`);
  grid.append(statusCard("Worker", workerDot, workerText, workerLines));

  const modelLines = [];
  if (model.paused_until) modelLines.push(lineWith(`paused after ${model.consecutive_failures} failure(s); retry `, liveUntil(parseTime(model.paused_until))));
  if (model.last_error) modelLines.push(model.last_error);
  if (model.last_ok_call) modelLines.push(`last accepted answer ${formatDayTime(parseTime(model.last_ok_call.at))} (${model.last_ok_call.role}, ${model.last_ok_call.minutes} min)`);
  const calls = model.calls_last_24h || {};
  const callTotal = Object.values(calls).reduce((sum, count) => sum + count, 0);
  if (callTotal) modelLines.push(`last 24 h: ${callTotal} call(s), ${calls.ok || 0} accepted, ${callTotal - (calls.ok || 0)} rejected or failed`);
  const modelStates = { busy: ["busy", "Busy"], idle: ["good", "Ready (idle)"], paused: ["critical", "Unavailable"] };
  const [modelDot, modelText] = modelStates[model.state] || ["idle", model.state];
  grid.append(statusCard("Local model", modelDot, modelText, modelLines));

  const collectorLines = [];
  if (collector.state === "running") {
    for (const step of collector.steps) collectorLines.push(lineWith(step.label + (step.since ? " · " : ""), step.since ? liveSince(parseTime(step.since)) : null));
  }
  const lastRun = collector.last_run;
  if (lastRun) collectorLines.push(`last run ${formatDayTime(parseTime(lastRun.start))}: ${lastRun.new_papers} new, ${lastRun.promoted} promoted, ${lastRun.full_texts} full text(s)${lastRun.inferred ? " (reconstructed)" : ""}`);
  const nextCollect = snapshot.plan.moments.find((moment) => moment.kind === "collect" && !moment.past && moment.installed);
  if (nextCollect) collectorLines.push(lineWith(`next collect ${formatTime(parseTime(nextCollect.at))}, `, liveUntil(parseTime(nextCollect.at))));
  else collectorLines.push("no collect scheduled");
  grid.append(statusCard("Collector", collector.state === "running" ? "busy" : "idle",
                         collector.state === "running" ? "Collecting" : "Idle", collectorLines));

  const reportLines = [];
  const cutoff = parseTime(report.next_cutoff);
  reportLines.push(lineWith(`report ${report.next_report_date} at ${formatTime(cutoff)}, `, liveUntil(cutoff)));
  for (const step of report.steps) reportLines.push(step.label);
  if (report.writer_reservation_until) reportLines.push("the writer has reserved the next model slot");
  if (report.latest) reportLines.push(`latest: ${report.latest.report_date} ${report.latest.status}, ${report.latest.papers} paper(s)`);
  else reportLines.push("no report published yet");
  const reportStates = { publishing: ["busy", "Publishing"], unfinished: ["warning", "Unfinished (resumes)"],
                         waiting: ["idle", "Waiting for the cutoff"] };
  const [reportDot, reportText] = reportStates[report.state] || ["idle", report.state];
  grid.append(statusCard("Daily report", reportDot, reportText, reportLines));
}

function renderCurrentJobs(worker) {
  const container = document.getElementById("current-jobs");
  container.replaceChildren();
  for (const job of worker.jobs) {
    const card = el("div", "job-card");
    card.append(el("div", "job-meta", `${job.kind_label} · ${job.class_label} · attempt ${job.attempt}/${job.max_attempts}`));
    card.append(el("div", "job-title", job.title || job.job_id));
    const meta = el("div", "job-meta");
    if (job.since) meta.append(document.createTextNode("job running for "), liveSince(parseTime(job.since)), document.createTextNode(" · "));
    meta.append(el("span", "source-tag", job.source === "activity" ? "exact" : "inferred"));
    card.append(meta);
    if (job.progress && job.progress.total) {
      const bar = el("div", "progress");
      const fill = el("div");
      fill.style.width = `${Math.min(100, (100 * job.progress.done) / job.progress.total)}%`;
      bar.append(fill);
      card.append(bar, el("div", "job-meta", `${job.progress.done} of ${job.progress.total} steps done (${job.progress.label})`));
    } else if (job.progress) {
      card.append(el("div", "job-meta", job.progress.label));
    }
    const steps = el("ul", "step-list");
    job.steps.forEach((step, index) => {
      const item = el("li");
      item.append(el("span", "step-indent", index ? "└".padStart(index * 2, " ") : "▸"), el("span", null, step.label));
      if (step.since) item.append(liveSince(parseTime(step.since), "running "));
      if (step.deadline) item.append(el("span", "muted", `(time limit ${formatTime(parseTime(step.deadline))})`));
      steps.append(item);
    });
    card.append(steps);
    container.append(card);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------------------------------------------

function renderTimeline(snapshot) {
  const svgRoot = document.getElementById("timeline");
  const wrap = document.getElementById("timeline-wrap");
  const width = Math.max(720, wrap.clientWidth);
  const labelWidth = 74, axisHeight = 36, laneHeight = 30, laneGap = 6, rightPad = 8;
  const lanes = [
    { key: "collect", label: "Collect" }, { key: "jobs", label: "Jobs" },
    { key: "model", label: "Model" }, { key: "report", label: "Report" },
  ];
  const height = axisHeight + lanes.length * (laneHeight + laneGap) + 4;
  svgRoot.replaceChildren();
  svgRoot.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svgRoot.setAttribute("height", height);

  const start = parseTime(snapshot.meta.window_start).getTime();
  const end = parseTime(snapshot.meta.horizon_end).getTime();
  const now = parseTime(snapshot.meta.generated_at).getTime();
  const plotWidth = width - labelWidth - rightPad;
  // Two linear scales joined at "now": the past gets most of the width whatever the two spans are, so recent work
  // stays readable next to a long night of schedule.
  const pastWidth = plotWidth * (end > now ? PAST_WIDTH_SHARE : 1);
  const futureWidth = plotWidth - pastWidth;
  const xOf = (time) => {
    const clamped = Math.min(Math.max(time, start), end);
    if (clamped <= now) return labelWidth + ((clamped - start) / Math.max(1, now - start)) * pastWidth;
    return labelWidth + pastWidth + ((clamped - now) / Math.max(1, end - now)) * futureWidth;
  };
  const laneTop = (index) => axisHeight + index * (laneHeight + laneGap);

  const defs = svg("defs");
  const hatch = svg("pattern", { id: "rejected-hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse",
                                 patternTransform: "rotate(45)" });
  hatch.append(svg("rect", { width: 2, height: 5, fill: "var(--surface-1)" }));
  defs.append(hatch);
  svgRoot.append(defs);

  lanes.forEach((lane, index) => {
    svgRoot.append(svg("rect", { class: "lane-bg", x: labelWidth, y: laneTop(index), width: plotWidth, height: laneHeight, rx: 4 }));
    const label = svg("text", { class: "lane-label", x: 0, y: laneTop(index) + laneHeight / 2 + 4 });
    label.textContent = lane.label;
    svgRoot.append(label);
  });
  svgRoot.append(svg("rect", { class: "future-bg", x: xOf(now), y: axisHeight - 4, width: Math.max(0, labelWidth + plotWidth - xOf(now)),
                               height: lanes.length * (laneHeight + laneGap) }));

  // Hour ticks and day boundaries in the report timezone.
  const stepFor = (spanMs, segmentWidth) =>
    [1, 2, 3, 6, 12, 24].find((step) => (spanMs / 3600000 / step) * 64 <= segmentWidth) || 24;
  const pastStep = stepFor(now - start, pastWidth), futureStep = stepFor(end - now, Math.max(1, futureWidth));
  const nowX = xOf(now);
  const firstHour = Math.ceil(start / 3600000) * 3600000;
  for (let time = firstHour; time <= end; time += 3600000) {
    const parts = localParts(new Date(time));
    if (parts.minute !== 0) continue;
    const tickStep = time <= now ? pastStep : futureStep;
    const x = xOf(time);
    if (Math.abs(x - nowX) < 34) continue;  // keep the "now" label readable
    if (parts.hour === 0) {
      svgRoot.append(svg("line", { class: "day-line", x1: x, x2: x, y1: 2, y2: height }));
      const dayLabel = svg("text", { class: "day-label", x: x + 4, y: 12 });
      dayLabel.textContent = state.dayFormat.format(new Date(time));
      svgRoot.append(dayLabel);
    } else if (parts.hour % tickStep === 0) {
      svgRoot.append(svg("line", { class: "grid-line", x1: x, x2: x, y1: axisHeight - 6, y2: height }));
    }
    if (parts.hour % tickStep === 0) {
      const tickLabel = svg("text", { class: "axis-label", x: x + 3, y: axisHeight - 10 });
      tickLabel.textContent = `${String(parts.hour).padStart(2, "0")}:00`;
      svgRoot.append(tickLabel);
    }
  }

  const tooltips = [];
  const addBar = (laneIndex, item, fill, lines, options = {}) => {
    const x1 = xOf(parseTime(item.start).getTime());
    const x2 = xOf(parseTime(item.end).getTime());
    const top = laneTop(laneIndex) + 4, barHeight = laneHeight - 8;
    const barWidth = Math.max(2, x2 - x1);
    const bar = svg("rect", { class: "bar", x: x1, y: top, width: barWidth, height: barHeight, rx: 3, fill });
    if (item.running) bar.setAttribute("stroke-dasharray", "3 2");
    svgRoot.append(bar);
    if (options.hatched) svgRoot.append(svg("rect", { x: x1, y: top, width: barWidth, height: barHeight, rx: 3,
                                                      fill: "url(#rejected-hatch)", "pointer-events": "none" }));
    const hitWidth = Math.max(8, barWidth);
    const hit = svg("rect", { class: "hit", x: x1 - (hitWidth - barWidth) / 2, y: top - 2, width: hitWidth, height: barHeight + 4 });
    hit.dataset.tooltip = String(tooltips.length);
    hit.dataset.bar = String(tooltips.length);
    tooltips.push({ lines, bar });
    svgRoot.append(hit);
  };

  const lanesData = snapshot.history.lanes;
  for (const run of lanesData.collect) {
    const failed = ["failed", "interrupted"].includes(run.outcome);
    const requests = Object.entries(run.requests || {}).map(([provider, count]) => `${provider} ${count}`).join(", ");
    addBar(0, run, failed ? "var(--critical)" : "var(--series-neutral)", [
      [`Collect ${run.running ? "(running)" : run.outcome}`, true],
      [`${formatDayTime(parseTime(run.start))} – ${formatTime(parseTime(run.end))} (${formatDuration(parseTime(run.end) - parseTime(run.start))})`],
      [`${run.new_papers} new paper(s), ${run.promoted} promoted, ${run.full_texts} full text(s)`],
      requests ? [`requests: ${requests}${run.failed_requests ? `, ${run.failed_requests} failed` : ""}`] : null,
      ...(run.stages || []).map((stage) => [`· ${stage.label}: ${formatDuration(parseTime(stage.end) - parseTime(stage.start))} ${stage.outcome === "ok" ? "" : stage.outcome}`]),
      run.inferred ? ["reconstructed from requests and promotions"] : null,
      run.note ? [run.note] : null,
    ]);
  }
  for (const job of lanesData.jobs) {
    const outcome = job.running ? "running" : job.outcome;
    addBar(1, job, JOB_OUTCOME_COLORS[outcome] || "var(--text-muted)", [
      [job.title || job.kind_label || job.job_id, true],
      [`${job.kind_label || "job"} · ${JOB_OUTCOME_LABELS[outcome] || outcome}`],
      [`${formatDayTime(parseTime(job.start))} – ${job.running ? "now" : formatTime(parseTime(job.end))} (${formatDuration(parseTime(job.end) - parseTime(job.start))})`],
      job.calls ? [`${job.calls} model call(s)`] : null,
      job.note ? [job.note] : null,
    ]);
  }
  for (const call of lanesData.model) {
    const rejected = !["ok", "running"].includes(call.outcome);
    addBar(2, call, ROLE_COLORS[call.role] || "var(--accent)", [
      [call.role ? `${call.role}${call.repair ? " (repair)" : ""}` : "model call in progress", true],
      call.title ? [call.title] : null,
      [`${call.running ? "running" : (rejected ? `rejected: ${call.outcome.replace(/_/g, " ")}` : "accepted")} · ${call.minutes} min${call.tokens ? ` · ${call.tokens} tokens` : ""}`],
      [`${formatDayTime(parseTime(call.start))} – ${call.running ? "now" : formatTime(parseTime(call.end))}`],
      call.inferred ? ["started when the previous call ended (inferred)"] : null,
      call.error ? [call.error] : null,
    ], { hatched: rejected });
  }
  for (const run of lanesData.report) {
    addBar(3, run, run.outcome === "ok" ? "var(--series-6)" : "var(--warning)", [
      [run.label, true], [`${formatDayTime(parseTime(run.start))} – ${formatTime(parseTime(run.end))}`],
      [run.running ? "in progress" : run.outcome],
      ...(run.stages || []).map((stage) => [`· ${stage.label}: ${formatDuration(parseTime(stage.end) - parseTime(stage.start))}`]),
    ]);
  }

  // Planned moments: diamonds on the collect and report lanes.
  for (const moment of snapshot.plan.moments) {
    if (moment.past) continue;
    const laneIndex = moment.kind === "collect" ? 0 : 3;
    const x = xOf(parseTime(moment.at).getTime());
    const y = laneTop(laneIndex) + laneHeight / 2;
    const diamond = svg("rect", { class: `planned ${moment.kind}`, x: x - 5, y: y - 5, width: 10, height: 10,
                                  transform: `rotate(45 ${x} ${y})` });
    if (!moment.installed) diamond.setAttribute("stroke-dasharray", "2 2");
    svgRoot.append(diamond);
    const hit = svg("rect", { class: "hit", x: x - 8, y: y - 10, width: 16, height: 20 });
    hit.dataset.tooltip = String(tooltips.length);
    tooltips.push({ lines: [[moment.label, true], [`${formatDayTime(parseTime(moment.at))} (${relativeTo(parseTime(moment.at))})`],
                            [moment.installed ? `scheduled by the ${moment.source} timers` : "configured, but no timer is installed: it will not run on its own"]] });
    svgRoot.append(hit);
  }

  svgRoot.append(svg("line", { class: "now-line", x1: nowX, x2: nowX, y1: axisHeight - 6, y2: height }));
  const nowLabel = svg("text", { class: "now-label", x: nowX + 4, y: 24 });
  nowLabel.textContent = "now";
  svgRoot.append(nowLabel);

  attachTooltips(svgRoot, tooltips);
  renderTimelineLegend(lanesData);
}

function attachTooltips(svgRoot, tooltips) {
  const tooltip = document.getElementById("tooltip");
  const wrap = document.getElementById("timeline-wrap");
  let focusedBar = null;
  svgRoot.onmousemove = (event) => {
    const target = event.target.closest ? event.target.closest("[data-tooltip]") : null;
    if (focusedBar) { focusedBar.classList.remove("focused"); focusedBar = null; }
    if (!target) { tooltip.hidden = true; return; }
    const entry = tooltips[Number(target.dataset.tooltip)];
    tooltip.replaceChildren();
    for (const line of entry.lines) {
      if (!line) continue;
      tooltip.append(el("div", line[1] ? "tt-title" : "tt-line", line[0]));
    }
    if (entry.bar) { entry.bar.classList.add("focused"); focusedBar = entry.bar; }
    tooltip.hidden = false;
    const bounds = wrap.getBoundingClientRect();
    let left = event.clientX - bounds.left + 14;
    if (left + tooltip.offsetWidth > bounds.width) left = event.clientX - bounds.left - tooltip.offsetWidth - 14;
    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${event.clientY - bounds.top + 14}px`;
  };
  svgRoot.onmouseleave = () => { tooltip.hidden = true; if (focusedBar) focusedBar.classList.remove("focused"); };
}

function renderTimelineLegend(lanesData) {
  const legend = document.getElementById("timeline-legend");
  legend.replaceChildren();
  const add = (color, label, hatched) => {
    const item = el("span", "legend-item");
    const swatch = el("span", "legend-swatch");
    swatch.style.background = hatched
      ? `repeating-linear-gradient(45deg, ${color} 0 3px, var(--surface-1) 3px 5px)` : color;
    item.append(swatch, el("span", null, label));
    legend.append(item);
  };
  const roles = new Set(lanesData.model.map((call) => call.role).filter(Boolean));
  for (const role of Object.keys(ROLE_COLORS)) if (roles.has(role)) add(ROLE_COLORS[role], role.replace(/_/g, " "));
  if (lanesData.model.some((call) => !["ok", "running"].includes(call.outcome))) add("var(--text-secondary)", "rejected answer", true);
  add("var(--series-neutral)", "collect run");
  add("var(--text-muted)", "job done"); add("var(--accent)", "job running");
  add("var(--warning)", "job will retry"); add("var(--critical)", "failed");
  add("var(--series-6)", "report");
  const planned = el("span", "legend-item", "◇ planned");
  legend.append(planned);
}

// ---------------------------------------------------------------------------------------------------------------
// Plan, queue, events, days, budgets
// ---------------------------------------------------------------------------------------------------------------

function renderPlan(plan, report) {
  const container = document.getElementById("plan");
  container.replaceChildren();
  const sourceText = { pilot: "Schedule: the one-day pilot's timers.", daily: "Schedule: the daily timers.",
                       none: "No timers are installed: nothing runs on its own (the configured times are shown dashed)." };
  container.append(el("p", "subtle", sourceText[plan.schedule_source] || ""));
  const list = el("ul", "plan-list");
  const upcoming = plan.moments.filter((moment) => !moment.past);
  for (const moment of upcoming) {
    const item = el("li");
    const when = parseTime(moment.at);
    item.append(el("span", "when", formatDayTime(when)), el("span", null, moment.label + (moment.installed ? "" : " (not scheduled)")), liveUntil(when));
    list.append(item);
  }
  if (!upcoming.length) list.append(el("li", "past", "Nothing planned in this window."));
  container.append(list);

  const estimate = plan.estimate;
  const box = el("div", "estimate");
  const cutoff = parseTime(estimate.cutoff);
  if (estimate.readings_before_cutoff !== null) {
    const strong = el("strong", null, `about ${estimate.readings_before_cutoff} of ${estimate.queued_readings}`);
    box.append(document.createTextNode("Readings that fit before the report: "), strong,
               el("div", "subtle", `measured ${estimate.mean_minutes_per_reading} model minutes per finished reading (last 7 days, ${estimate.completed_readings_measured} readings); ${formatDuration(estimate.minutes_until_cutoff * 60000)} left until ${formatTime(cutoff)}.`));
  } else {
    box.append(el("strong", null, `${estimate.queued_readings} reading(s) queued`),
               el("div", "subtle", "No reading has finished in the last 7 days, so there is no per-reading time to estimate from yet."));
    if (estimate.mean_minutes_per_call) {
      box.append(el("div", "subtle", `Reading calls so far take ${estimate.mean_minutes_per_call} min on average (${estimate.reading_calls_measured} calls); ${Math.round(estimate.rejected_call_share * 100)}% were rejected and repaired or retried.`));
    }
  }
  container.append(box);
}

function renderQueue(queue) {
  const container = document.getElementById("queue");
  container.replaceChildren();
  if (!queue.total) { container.append(el("p", "subtle", "The queue is empty.")); return; }
  container.append(el("p", "subtle", `${queue.total} queued: ` + queue.by_class.map((group) =>
    `${group.label} ${group.count}${group.held ? ` (${group.held} held)` : ""}, oldest ${group.oldest_hours} h`).join("; ")));
  const scroll = el("div", "table-scroll");
  const table = el("table");
  const head = el("tr");
  for (const [label, className] of [["#", "num"], ["Paper / job", ""], ["Class", ""], ["Due", ""]]) head.append(el("th", className, label));
  table.append(head);
  queue.next_jobs.forEach((job, index) => {
    const row = el("tr");
    row.append(el("td", "num", index + 1));
    const titleCell = el("td", "title-cell", job.title || job.kind_label);
    if (job.retrying) titleCell.append(document.createTextNode(" "), el("span", "pill", `retry ${job.attempts}/${job.max_attempts}`));
    row.append(titleCell, el("td", "nowrap", job.class_label));
    const dueCell = el("td");
    if (job.held) { dueCell.append(el("span", "pill held", "held"), document.createTextNode(" "), liveUntil(parseTime(job.due_at))); }
    else dueCell.textContent = "due";
    row.append(dueCell);
    table.append(row);
  });
  scroll.append(table);
  container.append(scroll);
  if (queue.total > queue.next_jobs.length) container.append(el("p", "subtle", `… and ${queue.total - queue.next_jobs.length} more`));
}

const EVENT_FILTERS = {
  all: ["All", () => true],
  problems: ["Problems", (event) => event.severity === "bad" || event.severity === "warn"],
  readings: ["Readings", (event) => event.kind === "review"],
  collect: ["Collect", (event) => event.kind === "collect"],
};
const SEVERITY_LABELS = { good: "OK", bad: "PROBLEM", warn: "NOTE", info: "" };

function renderEvents(events) {
  const filters = document.getElementById("event-filters");
  filters.replaceChildren();
  for (const [key, [label]] of Object.entries(EVENT_FILTERS)) {
    const button = el("button", null, label);
    button.type = "button";
    button.setAttribute("aria-pressed", String(state.eventFilter === key));
    button.onclick = () => { state.eventFilter = key; renderEvents(state.snapshot.history.events); };
    filters.append(button);
  }
  const list = document.getElementById("events");
  list.replaceChildren();
  const shown = events.filter(EVENT_FILTERS[state.eventFilter][1]);
  if (!shown.length) list.append(el("li", "subtle", "Nothing in this window."));
  for (const event of shown) {
    const item = el("li");
    const when = parseTime(event.at);
    item.append(el("span", "event-when", formatDayTime(when)));
    const text = el("div", "event-text");
    if (SEVERITY_LABELS[event.severity]) text.append(el("span", `severity ${event.severity}`, SEVERITY_LABELS[event.severity]));
    text.append(document.createTextNode(event.text));
    if (event.title) text.append(el("div", "event-title", event.title));
    if (event.detail) text.append(el("div", "event-detail", event.detail));
    item.append(text);
    list.append(item);
  }
}

function renderDays(days) {
  const container = document.getElementById("days");
  container.replaceChildren();
  const scroll = el("div", "table-scroll");
  const table = el("table");
  const columns = [["Day", "date", ""], ["Readings", "readings", "num"], ["Model hours", "model_hours", "num"],
                   ["Calls (rejected)", null, "num"], ["New papers", "new_papers", "num"], ["Promoted", "promoted", "num"],
                   ["Requests", "requests", "num"]];
  const head = el("tr");
  for (const [label, , className] of columns) head.append(el("th", className, label));
  table.append(head);
  for (const day of days) {
    const row = el("tr");
    for (const [, key, className] of columns) {
      row.append(el("td", className, key ? day[key] : `${day.model_calls} (${day.rejected_calls})`));
    }
    table.append(row);
  }
  scroll.append(table);
  container.append(scroll);
}

function renderBudgets(budgets) {
  const container = document.getElementById("budgets");
  container.replaceChildren();
  for (const budget of budgets) {
    const row = el("div", "meter-row");
    row.append(el("span", null, budget.label));
    const meter = el("div", "meter" + (budget.limit && budget.used >= budget.limit ? " full" : ""));
    const fill = el("div");
    fill.style.width = `${budget.limit ? Math.min(100, (100 * budget.used) / budget.limit) : 0}%`;
    meter.append(fill);
    row.append(meter, el("span", "meter-value", budget.limit ? `${budget.used} / ${budget.limit}` : budget.used));
    if (budget.note) row.append(el("span", "meter-note", budget.note));
    container.append(row);
  }
}

// ---------------------------------------------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("history-hours");
  select.value = String(state.historyHours);
  if (select.value !== String(state.historyHours)) { state.historyHours = 24; select.value = "24"; }
  select.onchange = () => { state.historyHours = Number(select.value); storeValue("historyHours", state.historyHours); refresh(); };
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.snapshot) renderTimeline(state.snapshot); }, 150);
  });
  refresh();
  setInterval(refresh, REFRESH_MS);
  setInterval(tick, 1000);
});
