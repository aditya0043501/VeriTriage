/*
 * Auto-playing demo driver for the Wells' DVT pathway.
 *
 * INERT BY DEFAULT: this file loads with the page but does absolutely
 * nothing — no button, no timers — unless the URL has ?demo=1 or the page
 * sets window.ENABLE_DEMO = true before this script runs.
 *
 * The demo drives the real UI (sendMessage, chip clicks) against the real
 * backend. No mock data, no bypasses.
 *
 * Note on beats: the script taps the "Not sure" chip on the paralysis
 * question (the second unclear answer) so the Live Reasoning panel shows
 * the amber "Not established" state. Tapping "No, I can move normally"
 * would resolve the field as a genuine No and that state could never
 * appear — the two outcomes are mutually exclusive by design.
 */
(function () {
  var params = new URLSearchParams(window.location.search);
  var ENABLED = params.get("demo") === "1" || window.ENABLE_DEMO === true;
  if (!ENABLED) return;
  var AUTOSTART = params.get("autostart") === "1";

  var STEP_DELAY_MS = 1900;      // natural pause between messages
  var AFTER_NOT_SURE_MS = 2600;  // longer beat so the escalation chips are visible
  var BEFORE_REVEAL_MS = 2200;   // pause before auto-opening "Show me why"
  var BEFORE_REPORT_MS = 2200;   // pause before scrolling to the doctor report

  // Mapped onto the real Wells' question order (age -> cancer -> paralysis ->
  // bedridden -> tenderness -> whole-leg -> calf>3cm -> pitting -> collateral
  // -> context). Every message is a real API call.
  var SCRIPT = [
    { text: "I have pain in my leg" },
    { text: "My calf is swollen and sore — started two days ago" },
    { text: "I'm 45" },
    { text: "No, I don't have cancer" },
    { text: "not sure", pauseAfter: AFTER_NOT_SURE_MS },
    { tapChip: "Not sure" },   // second unclear answer -> "Not established"
    { text: "no" },
    { text: "it's really sore when I press on my calf" },
    { text: "No, just the calf" },
    { text: "yes it's swollen, clearly bigger than the other side" },
    { text: "no" },
    { text: "No new veins, but I also have some numbness in my foot" },
    { text: "no" }
  ];

  var timers = [];
  var playing = false;
  var stopRequested = false;

  function later(fn, ms) {
    var id = setTimeout(function () {
      timers = timers.filter(function (t) { return t !== id; });
      fn();
    }, ms);
    timers.push(id);
    return id;
  }

  function cancelAll() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  // ---- Demo control bar ----
  var bar = document.createElement("div");
  bar.id = "demoBar";
  bar.style.cssText = "position:fixed;top:14px;right:24px;z-index:200;display:none;gap:8px;align-items:center;";

  function pill(label, bg) {
    var b = document.createElement("button");
    b.textContent = label;
    b.style.cssText =
      "padding:8px 18px;border:none;border-radius:999px;background:" + bg + ";" +
      "color:#fff;font-family:'IBM Plex Mono',monospace;font-size:11px;" +
      "letter-spacing:0.08em;text-transform:uppercase;cursor:pointer;" +
      "box-shadow:0 2px 10px rgba(0,0,0,0.35);";
    return b;
  }

  var startBtn = pill("\u25B6 Start Demo", "#2563eb");
  var stopBtn = pill("\u23F9 Stop Demo", "#b91c1c");
  var resetBtn = pill("\u21BA Reset Demo", "#475569");

  startBtn.onclick = startDemo;
  stopBtn.onclick = stopDemo;
  resetBtn.onclick = resetDemo;

  bar.appendChild(startBtn);
  document.body.appendChild(bar);

  function showBar(mode) {
    bar.innerHTML = "";
    bar.style.display = "flex";
    if (mode === "landing") bar.appendChild(startBtn);
    if (mode === "playing") { bar.appendChild(stopBtn); bar.appendChild(resetBtn); }
    if (mode === "done") { bar.appendChild(resetBtn); bar.appendChild(startBtn); }
  }

  function setComposerLocked(locked) {
    var input = document.getElementById("userInput");
    var send = document.getElementById("sendButton");
    if (input) {
      input.disabled = locked;
      input.placeholder = locked ? "Demo playing — sit back…" : "Describe your symptoms in your own words...";
    }
    if (send) send.style.visibility = locked ? "hidden" : "visible";
  }

  function waitIdle(cb) {
    var send = document.getElementById("sendButton");
    if (send && !send.disabled) return later(cb, 150);
    later(function () { waitIdle(cb); }, 120);
  }

  function runStep(i) {
    if (stopRequested) return;
    if (i >= SCRIPT.length) return finishDemo();
    var step = SCRIPT[i];
    var input = document.getElementById("userInput");
    if (step.tapChip) {
      var chips = document.querySelectorAll("#chipContainer .chip");
      var hit = null;
      chips.forEach(function (c) { if (c.textContent.trim() === step.tapChip) hit = c; });
      if (hit) { hit.click(); } else if (input) { input.value = step.tapChip; sendMessage(); }
    } else {
      if (!input) return;
      input.value = step.text;
      sendMessage();
    }
    setComposerLocked(true);
    waitIdle(function () {
      if (stopRequested) return;
      // sendMessage re-enables the input on each response; re-lock between steps
      setComposerLocked(true);
      later(function () { runStep(i + 1); }, step.pauseAfter || STEP_DELAY_MS);
    });
  }

  function startDemo() {
    if (playing) return;
    // conversationId is a top-level `let` in index.html (global lexical
    // scope, not a window property) — reference it directly.
    if (typeof conversationId !== "undefined" && conversationId) return;
    playing = true;
    stopRequested = false;
    showBar("playing");
    setComposerLocked(true);
    later(function () { runStep(0); }, 800);
  }

  function finishDemo() {
    if (stopRequested) return;
    // Beat (l): open the explanation layer
    later(function () {
      if (stopRequested) return;
      var why = document.getElementById("explainToggleBtn");
      if (why) why.click();
      // Beat (m): there is no separate doctor-report button — the report
      // renders automatically, so scroll it into view instead.
      later(function () {
        if (stopRequested) return;
        var report = document.getElementById("doctorReport");
        if (report) report.scrollIntoView({ behavior: "smooth", block: "start" });
        playing = false;
        setComposerLocked(false);
        showBar("done");
      }, BEFORE_REPORT_MS);
    }, BEFORE_REVEAL_MS);
  }

  function stopDemo() {
    stopRequested = true;
    playing = false;
    cancelAll();
    setComposerLocked(false);
    showBar("done");
  }

  function resetDemo() {
    stopDemo();
    if (typeof resetConversation === "function") resetConversation();
    stopRequested = false;
    setComposerLocked(false);
    showBar("landing");
  }

  // Only visible on the landing page, before any conversation exists.
  var hasConversation = (typeof conversationId !== "undefined" && conversationId);
  if (!hasConversation) showBar("landing");
  if (AUTOSTART && !hasConversation) later(startDemo, 600);
})();
