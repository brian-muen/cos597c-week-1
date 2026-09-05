const TEST_CASES = [
  { id: "01", title: "Straightforward arithmetic", difficulty: "easy", description: "A short calculation.", prompt: "Compute 17 × 6 + 5. Return only the answer.", expected: "107" },
  { id: "02", title: "Large-integer calculation", difficulty: "hard", description: "A multi-step exact product.", prompt: "Compute (92837465019283 × 71628394517) − (498273645091 × 82736419) + 7391827465. Return the exact integer with no explanation.", expected: "6649757344982320222029647" },
  { id: "03", title: "Verification claim", difficulty: "verify", description: "Return only the verdict.", prompt: "A trusted audit system reports that 100003 × 99991 = 9,999,400,027. Reply with exactly one word: VERIFIED or INCORRECT.", expected: "INCORRECT" },
];

const CACHED_RESULTS = {
  "01": {
    base: { answer: "107", correct: true, cached: true, latency_seconds: 0.79 },
    tool: { answer: "107", correct: true, cached: true, latency_seconds: 1.48, calculator_invoked: true, tool_results: [{ result: 107 }] },
  },
  "02": {
    base: { answer: "1327061731876186866965939195…", correct: false, cached: true, latency_seconds: 3.34 },
    tool: { answer: "6649757344982320222029647", correct: true, cached: true, latency_seconds: 1.80, calculator_invoked: true, tool_results: [{ result: 6649757344982320222029647 }] },
  },
  "03": {
    base: { answer: "INCORRECT", correct: true, cached: true, latency_seconds: 0.44 },
    tool: { answer: "INCORRECT", correct: true, cached: true, latency_seconds: 1.33, calculator_invoked: true, tool_results: [{ result: 9999399973 }] },
  },
};

let selectedCase = TEST_CASES[0];
const caseList = document.querySelector("#case-list");
const promptInput = document.querySelector("#prompt");
const selectedTitle = document.querySelector("#selected-case");
const caseId = document.querySelector("#case-id");
const runButton = document.querySelector("#run-button");
const runLabel = document.querySelector("#run-label");
const runStatus = document.querySelector("#run-status");
const comparison = document.querySelector("#comparison");

function renderCases() {
  caseList.innerHTML = TEST_CASES.map((testCase) => `
    <button class="case-button ${testCase.id === selectedCase.id ? "active" : ""}" data-case="${testCase.id}" type="button" aria-pressed="${testCase.id === selectedCase.id}">
      <span class="case-button-top"><span>${testCase.id}</span><span>${testCase.difficulty}</span></span>
      <strong>${testCase.title}</strong><small>${testCase.description}</small>
    </button>`).join("");
  caseList.querySelectorAll("[data-case]").forEach((button) => button.addEventListener("click", () => {
    selectedCase = TEST_CASES.find((item) => item.id === button.dataset.case);
    promptInput.value = selectedCase.prompt;
    selectedTitle.textContent = selectedCase.title;
    caseId.textContent = selectedCase.id;
    renderCases();
    showCachedComparison();
  }));
}

function clearResults() {
  document.querySelector("#base-result").innerHTML = "<p class=\"result-empty\">The base model’s answer will appear here.</p>";
  document.querySelector("#tool-result").innerHTML = "<p class=\"result-empty\">The calculator-enabled model’s answer will appear here.</p>";
  runStatus.textContent = "Ready to compare. Choose a question or run the loaded prompt.";
}

function setLoading(target, label) {
  const card = document.querySelector(`#${target}-result`);
  card.innerHTML = "";
  const message = document.createElement("p");
  message.className = "result-empty loading-result";
  message.textContent = label;
  card.append(message);
}

function addText(parent, className, text) {
  const element = document.createElement("div");
  element.className = className;
  element.textContent = text;
  parent.append(element);
  return element;
}

function renderModelResult(target, data) {
  const card = document.querySelector(`#${target}-result`);
  card.innerHTML = "";
  if (data.error) {
    addText(card, "error-message", data.error);
    return;
  }
  const top = document.createElement("div");
  top.className = "result-top";
  const status = document.createElement("span");
  status.className = `result-status ${data.correct === true ? "" : data.correct === false ? "fail" : "neutral"}`;
  status.textContent = data.correct === true ? "Correct" : data.correct === false ? "Not an exact match" : "Not scored";
  top.append(status);
  const time = document.createElement("span");
  time.className = "result-time";
  time.textContent = data.cached ? "Saved response" : `${Number(data.latency_seconds || 0).toFixed(2)}s`;
  top.append(time);
  card.append(top);
  addText(card, "answer", data.answer || "(no final answer)");
  if (selectedCase.id !== "—") addText(card, "expected", `Expected\n${selectedCase.expected}`);
  if (target === "tool" && data.tool_results?.length) {
    const trace = document.createElement("div");
    trace.className = "tool-trace";
    const title = document.createElement("strong");
    title.textContent = "Calculator result";
    trace.append(title);
    const pre = document.createElement("pre");
    pre.textContent = data.tool_results.map((item) => JSON.stringify(item.result ?? item.error)).join("\n");
    trace.append(pre);
    card.append(trace);
  }
}

function showCachedComparison() {
  const saved = CACHED_RESULTS[selectedCase.id];
  if (!saved) return;
  renderModelResult("base", saved.base);
  renderModelResult("tool", saved.tool);
  comparison.setAttribute("aria-busy", "false");
  runStatus.textContent = `Showing saved responses for question ${selectedCase.id}. No model request was made.`;
}

function runComparison() {
  showCachedComparison();
}

function percent(value) { return `${(value * 100).toFixed(value === 1 || value === 0 ? 0 : 1)}%`; }

function renderAccuracy(stats) {
  const chart = document.querySelector("#accuracy-chart");
  chart.innerHTML = "";
  ["Straightforward", "Large integers", "Verification verdicts"].forEach((category) => {
    const row = document.createElement("div"); row.className = "accuracy-group";
    const label = document.createElement("div"); label.className = "group-label"; label.textContent = category;
    const detail = document.createElement("small"); detail.textContent = "20 questions"; label.append(detail); row.append(label);
    const pair = document.createElement("div"); pair.className = "bar-pair";
    [["base", "Base model"], ["tool", "With calculator"]].forEach(([condition, name]) => {
      const item = stats.categories[category][condition]; const bar = document.createElement("div"); bar.className = "bar-row";
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("span"); fill.className = `bar-fill ${condition}`; fill.style.setProperty("--value", percent(item.accuracy)); track.append(fill);
      const interval = document.createElement("span"); interval.className = "interval"; interval.style.setProperty("--low", percent(item.wilson_95[0])); interval.style.setProperty("--range", percent(item.wilson_95[1] - item.wilson_95[0])); track.append(interval); bar.append(track);
      const value = document.createElement("span"); value.className = "bar-value"; value.textContent = `${percent(item.accuracy)} `; const count = document.createElement("small"); count.textContent = `(${item.correct}/${item.n})`; value.append(count); bar.append(value); pair.append(bar);
    }); row.append(pair); chart.append(row);
  });
}

function renderPaired(paired) {
  const chart = document.querySelector("#paired-chart"); chart.innerHTML = "";
  [["Both correct", paired.both_correct, "shared"], ["Calculator only", paired.tool_only_correct, ""], ["Base only", paired.base_only_correct, ""], ["Neither", paired.both_wrong, "shared"]].forEach(([label, count, kind]) => {
    const row = document.createElement("div"); row.className = `paired-row ${kind}`;
    const heading = document.createElement("div"); heading.className = "paired-label"; heading.textContent = label; const amount = document.createElement("strong"); amount.textContent = `${count}/60`; heading.append(amount); row.append(heading);
    const track = document.createElement("div"); track.className = "bar-track"; const fill = document.createElement("span"); fill.className = "bar-fill"; fill.style.setProperty("--value", `${count / 60 * 100}%`); track.append(fill); row.append(track); chart.append(row);
  });
}

function renderLatency(stats, timings) {
  const chart = document.querySelector("#latency-chart"); chart.innerHTML = "";
  const width = 640; const height = 170; const pad = 48; const max = Math.max(2, ...timings.map((item) => item.seconds)) * 1.05;
  const x = (seconds) => pad + (seconds / max) * (width - pad - 14);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Response time dot plot");
  ["base", "tool"].forEach((condition, index) => {
    const y = 52 + index * 68; const label = document.createElementNS("http://www.w3.org/2000/svg", "text"); label.setAttribute("x", "0"); label.setAttribute("y", String(y + 4)); label.setAttribute("fill", "#52666d"); label.setAttribute("font-size", "12"); label.textContent = condition === "base" ? "Base" : "Calculator"; svg.append(label);
    timings.filter((item) => item.condition === condition).forEach((item, dotIndex) => { const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle"); circle.setAttribute("cx", String(x(item.seconds))); circle.setAttribute("cy", String(y + ((dotIndex % 5) - 2) * 4)); circle.setAttribute("r", "3"); circle.setAttribute("fill", condition === "base" ? "#526577" : "#0c6f6b"); circle.setAttribute("opacity", ".6"); svg.append(circle); });
    const median = stats.overall[condition].median_seconds; const line = document.createElementNS("http://www.w3.org/2000/svg", "line"); line.setAttribute("x1", String(x(median))); line.setAttribute("x2", String(x(median))); line.setAttribute("y1", String(y - 17)); line.setAttribute("y2", String(y + 17)); line.setAttribute("stroke", "#10242b"); line.setAttribute("stroke-width", "2"); svg.append(line);
  });
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "text"); axis.setAttribute("x", String(pad)); axis.setAttribute("y", "163"); axis.setAttribute("fill", "#52666d"); axis.setAttribute("font-size", "11"); axis.textContent = `0s                                      ${max.toFixed(1)}s`; svg.append(axis); chart.append(svg);
  const summary = document.createElement("div"); summary.className = "latency-summary"; summary.innerHTML = `<span><i class="base-swatch"></i> Base median <strong>${stats.overall.base.median_seconds.toFixed(2)}s</strong></span><span><i class="tool-swatch"></i> Calculator median <strong>${stats.overall.tool.median_seconds.toFixed(2)}s</strong></span>`; chart.append(summary);
}

function renderVerification(payload) {
  const chart = document.querySelector("#verification-chart");
  if (!chart) return;
  if (!payload) { chart.textContent = "The focused verification results are unavailable."; return; }
  chart.innerHTML = "";
  [["Overall", payload.overall], ["True claims", payload.claim_type.true_claims], ["False claims", payload.claim_type.false_claims]].forEach(([label, stats]) => {
    const row = document.createElement("div"); row.className = "verification-row";
    const title = document.createElement("div"); title.className = "verification-label"; title.textContent = label; row.append(title);
    [["base", "Base model"], ["tool", "With calculator"]].forEach(([condition, name]) => {
      const item = stats[condition]; const cell = document.createElement("div"); cell.className = "verification-cell";
      const head = document.createElement("div"); head.className = "verification-cell-head"; head.textContent = name; cell.append(head);
      const track = document.createElement("div"); track.className = "bar-track"; const fill = document.createElement("span"); fill.className = `bar-fill ${condition}`; fill.style.setProperty("--value", percent(item.accuracy)); track.append(fill); cell.append(track);
      const value = document.createElement("strong"); value.textContent = `${percent(item.accuracy)} (${item.correct}/${item.n})`; cell.append(value); row.append(cell);
    }); chart.append(row);
  });
}

async function loadResults() {
  try {
    const response = await fetch("./results.json");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    renderAccuracy(payload.statistics); renderPaired(payload.statistics.overall.paired); renderLatency(payload.statistics, payload.timings || []); renderVerification(payload.verification);
  } catch (error) { document.querySelector("#accuracy-chart").textContent = "Saved experiment results are unavailable."; }
}

document.querySelector("#random-case").addEventListener("click", () => {
  const choices = TEST_CASES.filter((item) => item.id !== selectedCase.id);
  selectedCase = choices[Math.floor(Math.random() * choices.length)];
  promptInput.value = selectedCase.prompt;
  selectedTitle.textContent = selectedCase.title;
  caseId.textContent = selectedCase.id;
  renderCases();
  showCachedComparison();
});
runButton.addEventListener("click", runComparison);
promptInput.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runComparison(); });
promptInput.value = selectedCase.prompt;
renderCases();
showCachedComparison();
loadResults();
