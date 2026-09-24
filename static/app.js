"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const colors = ["#528573", "#8a79a9", "#789953", "#c39263", "#6b93a5", "#b47685", "#948e57", "#588e94"];
const state = {recordings: [], current: null, analysis: null, segment: null, channels: [], position: 0, window: 0, playing: false, timer: null, request: 0, selection: 0, busy: false};
const time = (s) => `${Math.floor(s / 60).toString().padStart(2, "0")}:${Math.floor(s % 60).toString().padStart(2, "0")}`;
const text = (selector, value) => { $(selector).textContent = value; };
const node = (tag, cls, value) => { const n = document.createElement(tag); if (cls) n.className = cls; if (value !== undefined) n.textContent = value; return n; };
function notice(message, error = false) { const n = $("#notice"); n.hidden = !message; n.textContent = message; n.classList.toggle("error", error); }
async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {...options, headers: {"X-EEG-Local": "1", ...(options.body && !(options.body instanceof FormData) ? {"Content-Type": "application/json"} : {}), ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Please check the values you entered.");
  return data;
}
async function busy(work, message) {
  if (state.busy) return;
  state.busy = true; stop();
  const buttons = ["#demo-button", "#sample-button", "#welcome-demo", "#analyze-button", "#open-import"];
  buttons.forEach(s => $(s).disabled = true); notice(message);
  try { await work(); notice(""); } catch (err) { notice(err.message, true); }
  finally { state.busy = false; buttons.forEach(s => $(s).disabled = false); }
}
async function refreshLibrary() {
  state.recordings = await api("/recordings");
  const lib = $("#library"); lib.replaceChildren();
  text("#library-count", state.recordings.length);
  state.recordings.forEach(rec => {
    const button = node("button", rec.id === state.current?.id ? "active" : "", rec.name);
    button.append(node("small", "", `${rec.source === "synthetic" ? "SYNTHETIC" : rec.source === "public" ? "PUBLIC DATA" : "LOCAL FILE"} · ${time(rec.duration)}`));
    button.onclick = () => { if (!state.busy) busy(() => selectRecording(rec), "Opening recording…"); };
    lib.append(button);
  });
}
async function selectRecording(rec) {
  rec = await api(`/recordings/${rec.id}`);
  stop(); state.selection++; state.request++; state.current = rec; state.analysis = null; state.segment = null;
  state.channels = Array.from({length: Math.min(rec.channels.length, 4)}, (_, i) => i);
  state.position = 0; state.window = 0;
  $("#welcome").hidden = true; $("#workspace").hidden = false; $("#export-button").disabled = true;
  text("#recording-title", rec.name);
  text("#recording-subtitle", rec.source === "synthetic" ? "A generated signal with known changes. A place to learn the tools." : "Inspect the recording, compare patterns, and add your observations.");
  text("#source-badge", rec.source === "synthetic" ? "SYNTHETIC DEMO" : rec.source === "public" ? "PUBLIC RESEARCH DATA" : "LOCAL RECORDING");
  $("#source-badge").className = `badge ${rec.source}`;
  text("#recording-spec", `${rec.channels.length} channels · ${rec.sfreq} Hz · µV`);
  text("#metric-duration", time(rec.duration));
  text("#end-time", time(rec.duration)); text("#timeline-end", time(rec.duration));
  ["#metric-windows", "#metric-deviations", "#metric-quality"].forEach(s => text(s, "—"));
  $("#timeline").replaceChildren();
  ["#insight-status", "#insight-reason", "#quality-reasons", "#band-bars"].forEach(s => $(s).replaceChildren());
  text("#score", "—"); text("#baseline-warning", "Analysis has not been completed for this recording.");
  $("#baseline-start").value = 0; $("#baseline-end").value = Math.min(20, Math.floor(rec.duration / 2));
  if (Number($("#baseline-end").value) < 12) $("#baseline-end").value = 12;
  renderChannels(); renderMetadata(); renderNotes(); setTab("signal");
  await refreshLibrary(); await loadSegment(); await analyze();
}
function renderChannels() {
  $("#channel-picker").replaceChildren();
  state.current.channels.forEach((name, i) => {
    const button = node("button", state.channels.includes(i) ? "active" : "", name);
    button.setAttribute("aria-pressed", state.channels.includes(i));
    button.title = "Select up to 8 channels for display. Analysis uses all channels.";
    button.onclick = async () => {
      if (state.channels.includes(i)) { if (state.channels.length === 1) return; state.channels = state.channels.filter(x => x !== i); }
      else { if (state.channels.length === 8) return notice("Select up to 8 display channels. Deselect one first."); state.channels.push(i); }
      renderChannels(); renderInsight(); await safeSegment();
    };
    $("#channel-picker").append(button);
  });
}
async function analyze() {
  if (!state.current) return;
  const selection = state.selection;
  const options = {baseline_start: Number($("#baseline-start").value), baseline_end: Number($("#baseline-end").value), threshold: Number($("#threshold").value), mains: Number($("#mains").value)};
  let result;
  try {
    result = await api(`/recordings/${state.current.id}/analysis`, {method: "POST", body: JSON.stringify(options)});
  } catch (error) {
    if (state.analysis) {
      $("#baseline-start").value = state.analysis.baseline[0];
      $("#baseline-end").value = state.analysis.baseline[1];
      $("#threshold").value = state.analysis.threshold;
      $("#mains").value = state.analysis.mains_hz;
      throw new Error(`${error.message} Previous analysis settings have been restored.`);
    }
    throw error;
  }
  if (selection !== state.selection) return;
  state.analysis = result;
  text("#metric-windows", result.summary.windows);
  text("#metric-deviations", result.summary.deviations);
  text("#metric-quality", result.summary.quality_flagged);
  const unavailable = Object.entries(result.baseline_windows_per_channel).filter(([, n]) => n < 5).map(([name]) => name);
  text("#baseline-warning", unavailable.length ? `No usable reference for ${unavailable.join(", ")}; excluded from scoring.` : "Quality-flagged channels are excluded from each window’s score. Scores use all eligible channels.");
  $("#export-button").disabled = false;
  renderTimeline(); selectWindow(Math.min(state.window, result.windows.length - 1), false);
  await loadSegment();
}
function renderTimeline() {
  const root = $("#timeline"); root.replaceChildren();
  state.analysis.windows.forEach((window, i) => {
    const b = node("button", `${window.status} ${window.quality.length ? "quality-flag" : ""} ${i === state.window ? "selected" : ""}`);
    b.style.height = `${Math.max(12, Math.min(100, (window.score || 0) / 15 * 100))}%`;
    b.title = `${time(window.start)}–${time(window.end)} · ${window.status} · score ${window.score ?? "unavailable"}${window.quality.length ? " · quality flags" : ""}`;
    b.setAttribute("aria-label", b.title); b.onclick = () => selectWindow(i);
    root.append(b);
  });
}
function selectWindow(i, seek = true) {
  state.window = i; if (!state.analysis) return;
  [...$("#timeline").children].forEach((el, j) => el.classList.toggle("selected", i === j));
  renderInsight();
  if (seek) { stop(); state.position = Math.min(state.analysis.windows[i].start, maxPosition()); safeSegment(); }
}
function renderInsight() {
  if (!state.analysis) return;
  const window = state.analysis.windows[state.window];
  text("#insight-time", `${time(window.start)} – ${time(window.end)}`);
  const names = {baseline: "Reference interval", deviation: "Pattern deviation", typical: "Within selected threshold", quality: "Not scored"};
  text("#insight-status", names[window.status]); $("#insight-status").className = `insight-status ${window.status}`;
  text("#score", window.score === null ? "—" : window.score.toFixed(1));
  text("#insight-reason", window.reason);
  $("#quality-reasons").replaceChildren(...window.quality.map(q => node("div", "quality-note", `${q.channel}: ${q.reasons.join("; ")}`)));
  text("#band-channel", state.current.channels[state.channels[0]]);
  const powers = window.relative_power[state.channels[0]];
  $("#band-bars").replaceChildren(...["Delta", "Theta", "Alpha", "Beta"].map((band, i) => {
    const row = node("div", "band-row"); const track = node("div", "band-track"); const fill = node("div", "band-fill");
    fill.style.width = `${powers[i] * 100}%`; fill.style.background = colors[i]; track.append(fill);
    row.append(node("span", "", band), track, node("span", "", `${Math.round(powers[i] * 100)}%`)); return row;
  }));
}
function maxPosition() { return Math.max(0, state.current.duration - Number($("#window-size").value)); }
async function loadSegment() {
  if (!state.current) return;
  const request = ++state.request;
  state.position = Math.max(0, Math.min(state.position, maxPosition()));
  $("#seek").max = maxPosition(); $("#seek").value = state.position; text("#play-time", time(state.position));
  const params = new URLSearchParams({start: state.position, seconds: $("#window-size").value, channels: state.channels.join(","), filtered: $("#filtered").checked, mains: state.analysis?.mains_hz ?? $("#mains").value});
  const segment = await api(`/recordings/${state.current.id}/segment?${params}`);
  if (request !== state.request) return;
  state.segment = segment; drawWave(); drawSpectrum();
  text("#wave-caption", $("#filtered").checked ? "Filtered · 1–30 Hz · microvolts" : "Raw signal · microvolts · original reference");
}
async function safeSegment() { try { await loadSegment(); } catch (err) { stop(); notice(err.message, true); } }
function canvasContext(selector, height) {
  const canvas = $(selector); const width = canvas.getBoundingClientRect().width;
  if (!width) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.font = "9px -apple-system, sans-serif";
  return {ctx, width, height};
}
function drawWave() {
  if (!state.segment) return;
  const seg = state.segment, h = 60 * seg.channels.length + 32;
  const dimensions = canvasContext("#waveform", h); if (!dimensions) return;
  const {ctx, width} = dimensions; const left = 42, right = width - 8, plot = right - left;
  const duration = seg.end - seg.start;
  ctx.strokeStyle = "#eff2ec"; ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) { const x = left + plot * i / 10; ctx.beginPath(); ctx.moveTo(x, 5); ctx.lineTo(x, h - 24); ctx.stroke(); if (i % 2 === 0) { ctx.fillStyle = "#9aa494"; ctx.fillText(time(seg.start + duration * i / 10), Math.min(x - 10, width - 30), h - 6); } }
  seg.channels.forEach((name, c) => {
    const y = 31 + c * 60;
    const magnitude = Math.max(40, ...seg.low[c].map(Math.abs), ...seg.high[c].map(Math.abs));
    const gain = 22 / magnitude;
    ctx.strokeStyle = "#e6ece3"; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = colors[c]; ctx.fillText(name.slice(0, 7), 0, y - 5);
    ctx.fillStyle = "#97a18e"; ctx.font = "7px -apple-system, sans-serif"; ctx.fillText(`±${Math.ceil(magnitude)} µV`, 0, y + 8); ctx.font = "9px -apple-system, sans-serif";
    ctx.strokeStyle = colors[c]; ctx.lineWidth = .85; ctx.beginPath();
    seg.times.forEach((t, i) => { const x = left + (t - seg.start) / duration * plot; const lo = y - seg.low[c][i] * gain, hi = y - seg.high[c][i] * gain; if (!i) ctx.moveTo(x, lo); else ctx.lineTo(x, lo); ctx.lineTo(x, hi); }); ctx.stroke();
  });
}
function drawSpectrum() {
  if (!state.segment) return;
  const dimensions = canvasContext("#spectrum", 190); if (!dimensions) return;
  const {ctx, width, height} = dimensions; const left = 40, right = width - 12, top = 20, bottom = height - 28;
  const psd = state.segment.psd.map(row => row.map(v => 10 * Math.log10(Math.max(v, 1e-10))));
  const max = Math.ceil(Math.max(...psd.flat()) / 10) * 10;
  const min = Math.max(-100, max - 70);
  const x = f => left + (f - 1) / 39 * (right - left), y = v => bottom - Math.max(0, (v - min) / (max - min)) * (bottom - top);
  [[1,4,"δ"],[4,8,"θ"],[8,13,"α"],[13,30,"β"]].forEach(([lo,hi,label],i) => { ctx.fillStyle = `${colors[i]}09`; ctx.fillRect(x(lo), top, x(hi)-x(lo), bottom-top); ctx.fillStyle = "#9ba58f"; ctx.fillText(label, x((lo+hi)/2), 12); });
  ctx.strokeStyle = "#edf0e9";
  for (let n = 0; n <= 4; n++) { const value = min + (max-min)*n/4; const py = y(value); ctx.beginPath(); ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();ctx.fillStyle="#97a18e";ctx.fillText(value.toFixed(0), 6,py+3); }
  [1,10,20,30,40].forEach(f => { ctx.fillStyle="#97a18e";ctx.fillText(`${f}`,x(f)-4,height-10); });
  ctx.fillText("Hz",right-5,height-10);ctx.fillText("dB µV²/Hz",2,10);
  psd.forEach((row,c) => {ctx.strokeStyle=colors[c];ctx.lineWidth=1.5;ctx.beginPath();row.forEach((v,i)=>{const px=x(state.segment.frequencies[i]),py=y(v);if(!i)ctx.moveTo(px,py);else ctx.lineTo(px,py);});ctx.stroke();});
}
function stop() { state.playing = false; clearTimeout(state.timer); text("#play-button", "▶"); $("#play-button").setAttribute("aria-label", "Play recording"); }
async function tick() {
  if (!state.playing) return;
  state.position = Math.min(state.position + 1, maxPosition()); await safeSegment();
  if (state.analysis) selectWindow(Math.min(Math.floor(state.position / state.analysis.hop_seconds), state.analysis.windows.length - 1), false);
  if (state.position >= maxPosition()) return stop();
  if (state.playing) state.timer = setTimeout(tick, 1000);
}
function setTab(tab) {
  $$("[data-tab]").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab));
  $$(".tab-content").forEach(el => el.hidden = el.id !== `tab-${tab}`);
  if (tab === "signal") { drawWave(); drawSpectrum(); } else stop();
}
function renderMetadata() {
  const rec = state.current; const root = $("#details"); root.replaceChildren();
  const fields = {"Source": rec.source, "Sampling rate": `${rec.sfreq} Hz`, "Duration": `${rec.duration.toFixed(2)} seconds`, "Channel names": rec.channels.join(", "), "Signal units": "Microvolts (µV)", "Reference": "As recorded; no rereferencing applied", "Imported at": rec.created_at, "Signal SHA-256": rec.sha256, ...rec.provenance};
  Object.entries(fields).forEach(([key, value]) => { if (value !== null) root.append(node("dt", "", key), node("dd", "", String(value))); });
  const events = $("#source-events"); events.replaceChildren();
  if (!rec.events.length) events.append(node("p", "muted", "No source events in this recording."));
  rec.events.forEach(e => { const b = node("button", "event-button", e.label); b.append(node("small", "", `${time(e.start)} · ${e.duration.toFixed(1)} seconds`)); b.onclick = () => {stop();state.position = Math.min(e.start,maxPosition());safeSegment();if(state.analysis)selectWindow(Math.min(Math.floor(e.start/2),state.analysis.windows.length-1),false);}; events.append(b); });
  Object.entries(rec.context).forEach(([k,v]) => { const field = $("#context-form").elements.namedItem(k); if (field) field.value = v; });
}
function renderNotes() {
  const root = $("#annotations"); root.replaceChildren(); text("#note-count", state.current.annotations.length);
  if (!state.current.annotations.length) root.append(node("p", "muted", "No observations yet. Add one from the signal explorer or start here."));
  state.current.annotations.forEach(note => {
    const el = node("div", "annotation"), head = node("div", "annotation-head"), jump = node("button", "", `${time(note.start)} – ${time(note.end)} ↗`), remove = node("button", "", "Remove");
    jump.onclick = () => {setTab("signal");stop();state.position=Math.min(note.start,maxPosition());safeSegment();};
    remove.onclick = async () => {try {state.current=await api(`/recordings/${state.current.id}/annotations/${note.id}`,{method:"DELETE"});renderNotes();}catch(e){notice(e.message,true);}};
    head.append(jump,remove);el.append(head,node("p","",note.label));root.append(el);
  });
}
function openNote() {
  if (!state.current) return;
  const window = state.analysis?.windows[state.window]; const form = $("#note-form");
  form.elements.start.value = window?.start ?? state.position; form.elements.end.value = window?.end ?? Math.min(state.position+4,state.current.duration);
  form.elements.label.value = ""; form.querySelector(".form-error").textContent = ""; $("#note-dialog").showModal();
}
function exportSession() {
  if (!state.analysis) return;
  const output = {schema_version:"1.0", exported_at: new Date().toISOString(), recording:state.current, analysis:state.analysis, interpretation:"Exploratory signal descriptors; not disease or symptom predictions. No raw signal samples included."};
  const url = URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:"application/json"})); const a=node("a");a.href=url;a.download=`eeg-session-${state.current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("#demo-button").onclick = $("#welcome-demo").onclick = () => busy(async () => selectRecording(await api("/demo", {method:"POST"})), "Preparing the synthetic walkthrough…");
$("#sample-button").onclick = () => busy(async () => selectRecording(await api("/sample", {method:"POST"})), "Downloading the public PhysioNet sample…");
$("#open-import").onclick = () => {$("#import-form .form-error").textContent="";$("#import-dialog").showModal();};
$("#method-button").onclick = () => $("#method-dialog").showModal();
$$(".close-dialog").forEach(b => b.onclick = () => b.closest("dialog").close());
$$("[data-tab]").forEach(b => b.onclick = () => setTab(b.dataset.tab));
$("#analyze-button").onclick = () => busy(analyze, "Comparing windows with the selected reference…");
$("#filtered").onchange = safeSegment;
$("#window-size").onchange = safeSegment;
$("#seek").oninput = () => {stop();state.position=Number($("#seek").value);safeSegment();if(state.analysis)selectWindow(Math.min(Math.floor(state.position/2),state.analysis.windows.length-1),false);};
$("#play-button").onclick = () => {if(state.playing)return stop();if(state.position>=maxPosition())state.position=0;state.playing=true;text("#play-button","Ⅱ");$("#play-button").setAttribute("aria-label","Pause recording");tick();};
$("#annotate-window").onclick = $("#add-note").onclick = openNote;
$("#export-button").onclick = exportSession;
$("#import-form").onsubmit = async e => {e.preventDefault();const form=e.currentTarget;const button=form.querySelector("button.primary");button.disabled=true;try {const rec=await api("/import",{method:"POST",body:new FormData(form)});$("#import-dialog").close();await busy(()=>selectRecording(rec),"Analyzing imported recording…");}catch(err){form.querySelector(".form-error").textContent=err.message;}finally{button.disabled=false;}};
$("#note-form").onsubmit = async e => {e.preventDefault();const form=e.currentTarget;const button=form.querySelector("button.primary");button.disabled=true;try {state.current=await api(`/recordings/${state.current.id}/annotations`,{method:"POST",body:JSON.stringify({start:Number(form.elements.start.value),end:Number(form.elements.end.value),label:form.elements.label.value})});renderNotes();$("#note-dialog").close();}catch(err){form.querySelector(".form-error").textContent=err.message;}finally{button.disabled=false;}};
$("#context-form").onsubmit = async e => {e.preventDefault();try {state.current=await api(`/recordings/${state.current.id}/context`,{method:"PUT",body:JSON.stringify(Object.fromEntries(new FormData(e.currentTarget)))});notice("Session context saved on this computer.");}catch(err){notice(err.message,true);}};
window.addEventListener("resize", () => {drawWave();drawSpectrum();});
(async () => {try {await refreshLibrary();if(state.recordings.length)await busy(()=>selectRecording(state.recordings[0]),"Opening your latest recording…");}catch(err){notice(`Could not connect to the local application. ${err.message}`,true);}})();

async function refreshArchive() {
  try {
    const result = await api("/archive/status");
    text("#archive-state", result.available ? `Download state: ${result.status.state} · updated ${new Date(result.status.updated_at).toLocaleString()}` : result.message);
    if (!result.available) {for (const id of ["location","progress","cohorts","plan"]) text(`#archive-${id}`, ""); return;}
    const {status:s, summary:p} = result;
    const gb = n => ((n || 0) / 1e9).toFixed(2);
    text("#archive-location", result.path);
    text("#archive-progress", `${s.verified_files.toLocaleString()} / ${s.total_files.toLocaleString()} files verified · ${gb(s.verified_bytes)} GB · ${s.failures.length} failed files`);
    text("#archive-cohorts", `Downloaded EEG: healthy/control ${gb(s.cohort_bytes.healthy_control)} GB · neurodegenerative ${gb(s.cohort_bytes.neurodegenerative)} GB`);
    text("#archive-plan", `Queued across ${p.datasets} datasets: ${gb(p.known_bytes_by_cohort.healthy_control)} GB controls and ${gb(p.known_bytes_by_cohort.neurodegenerative)} GB neurodegenerative EEG, plus up to 44.6 GB of ALS/control recordings whose split is measured as files arrive.`);
  } catch (e) {text("#archive-state", e.message);}
}
$("#archive-button").onclick = () => {$("#archive-dialog").showModal();refreshArchive();};
setInterval(() => {if ($("#archive-dialog").open) refreshArchive();}, 5000);
