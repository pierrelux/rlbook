// Replay only: all fields and diagnostics come from the offline numerical PDE.
export function createPlumeView({ $, heatmap, setup, stat }) {
  $("plume").innerHTML = `
    <div class="toolbar">
      <label>Heating schedule<select id="plume-schedule" aria-label="Plume heating schedule"><option value="constant">Constant heating</option><option value="boost">Scheduled upstream heat pulse</option></select></label>
      <label>Display<select id="plume-display" aria-label="Plume field display"><option value="absolute">Physical fields</option><option value="difference">Change caused by the cold pulse</option></select></label>
      <label>Comparison<select id="plume-comparison" aria-label="Plume comparison"><option value="single">One schedule</option><option value="both">Both heating schedules</option></select></label>
    </div>
    <div class="timeline">
      <button id="plume-play" aria-label="Play PDE dynamics">Play</button>
      <button id="plume-reset">Reset</button>
      <input id="plume-time" aria-label="PDE time" type="range" min="0" max="320" value="0" step="1">
      <output id="plume-clock" class="time">0.00 s</output>
      <label>Speed<select id="plume-speed" aria-label="PDE playback speed"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label>
    </div>
    <p id="plume-event" class="caption"></p>
    <p id="plume-note" class="caption"></p>
    <div id="plume-panels" class="plume-panels"></div>
    <div class="plume-flow"><canvas id="plume-velocity" aria-label="Prescribed downstream speed is faster in the middle"></canvas><p>Material near the middle moves faster than material near the walls. The velocity is prescribed; only temperature and reactant fields evolve.</p></div>
    <details><summary>What moves, and what heats it?</summary>
      <p>A cold pulse enters through the lower part of the inlet from 6 to 14 seconds. Flow is faster near the middle of the reactor, so different parts of the pulse travel different distances. Heat transport, reaction, and heating change its shape and temperature. Colder material reacts more slowly, leaving a moving region with less conversion.</p>
      <p>The second schedule raises heater H1 from 48% to 60% during 6–18 seconds. It is a prescribed input, not an MPPI decision. Both schedules begin from the same warm state. The difference view subtracts a separate numerical run with the same heating and no cold pulse.</p>
      <p id="plume-refinement"></p>
      <p>The marked 1070 K temperature limit and 95% outlet-conversion target are comparison guides. These heating schedules can violate them; they are not validated control policies. There is no random forcing or learned prediction in this view.</p>
      <p><a href="../_static/thermal_reactor/plume-storyboard.pdf">Four-time storyboard (PDF)</a> · <a href="../artifacts/thermal_plume/README.md">Scenario and regeneration notes</a></p>
    </details>`;
  let data,
    shown = [],
    raf = null,
    lastStamp = 0,
    elapsed = 0,
    generation = 0,
    initialized = false;
  const base = "./thermal-plume-data/",
    cache = new Map();
  const names = {
    constant: "Constant heating",
    boost: "Scheduled upstream heat pulse",
  };
  function pause() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
    $("plume-play").textContent =
      +$("plume-time").value === 320 ? "Replay" : "Play";
    $("plume-play").setAttribute("aria-label", "Play PDE dynamics");
  }
  function fail(e) {
    pause();
    $("plume-note").textContent = e.message;
    $("plume-note").classList.add("error");
  }
  async function read(desc) {
    if (!cache.has(desc.file))
      cache.set(
        desc.file,
        (async () => {
          const r = await fetch(base + desc.file, { cache: "no-store" });
          if (!r.ok) throw Error(`Cannot load plume recording: ${desc.file}`);
          const b = await r.arrayBuffer();
          if (b.byteLength !== desc.shape.reduce((a, v) => a * v, 1) * 4)
            throw Error("Incorrect plume recording size");
          return new Float32Array(b);
        })().catch((e) => {
          cache.delete(desc.file);
          throw e;
        }),
      );
    return cache.get(desc.file);
  }
  function panel(id) {
    return `<article class="panel"><div class="labelrow"><h2>${names[id]}</h2><span class="badge">Numerical PDE response</span></div><h3 id="plume-T-label-${id}">Temperature</h3><canvas id="plume-T-${id}" class="map" aria-label="${names[id]} temperature field"></canvas><h3 id="plume-c-label-${id}">Conversion</h3><canvas id="plume-c-${id}" class="map" aria-label="${names[id]} conversion field"></canvas><div id="plume-stats-${id}" class="stats"></div><div id="plume-heaters-${id}" class="heaterbars"></div><p id="plume-summary-${id}" class="caption"></p></article>`;
  }
  async function load() {
    pause();
    const token = ++generation;
    const both = $("plume-comparison").value === "both";
    const ids = both ? ["constant", "boost"] : [$("plume-schedule").value];
    const difference = $("plume-display").value === "difference";
    $("plume-panels").classList.add("loading");
    $("plume-play").disabled = true;
    const loaded = [];
    for (const id of ids) {
      const run = data.runs.find((r) => r.id === id);
      const array = await read(run.fields);
      const reference = difference
        ? await read(data.runs.find((r) => r.id === run.reference_id).fields)
        : null;
      loaded.push({ run, array, reference });
    }
    if (token !== generation) return;
    shown = loaded;
    $("plume-panels").classList.toggle("compare", both);
    $("plume-panels").innerHTML = ids.map(panel).join("");
    $("plume-panels").classList.remove("loading");
    $("plume-play").disabled = false;
    $("plume-note").classList.remove("error");
    draw();
  }
  function velocity() {
    const { ctx, width } = setup($("plume-velocity"), 82);
    const left = 40,
      span = Math.max(80, width - 150);
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#607079";
    ctx.fillText("Flow →", 0, 13);
    for (const [y, v, label] of [
      [24, 0.16, "near wall"],
      [45, 0.22, "middle"],
      [66, 0.16, "near wall"],
    ]) {
      const end = left + (span * v) / 0.22;
      ctx.strokeStyle = "#176787";
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(end, y);
      ctx.lineTo(end - 5, y - 3);
      ctx.moveTo(end, y);
      ctx.lineTo(end - 5, y + 3);
      ctx.stroke();
      ctx.fillText(`${v.toFixed(2)} m/s · ${label}`, end + 7, y + 3);
    }
  }
  function draw() {
    if (!data || !shown.length || $("plume").hidden) return;
    const k = +$("plume-time").value,
      t = data.times_s[k];
    const difference = $("plume-display").value === "difference";
    $("plume-clock").textContent = t.toFixed(2) + " s";
    $("plume-event").textContent =
      t < 6
        ? "The inlet pulse begins at 6 s. Material is already flowing through the warm reactor."
        : t <= 14
          ? "A cold pulse enters through the lower part of the inlet. Its leading edge is already traveling downstream."
          : t < 45
            ? "The inlet is back to normal. Follow the colder material as it travels, heats up, and reacts."
            : "The reactor returns toward its warm operating state as the pulse moves out.";
    $("plume-note").textContent = difference
      ? "Pulse run minus a no-pulse run with identical heating. Blue means colder material or less conversion. These are physical changes, not surrogate prediction errors."
      : "Temperature and conversion are numerical PDE fields. Fresh feed enters on the left; conversion increases as the material reacts. The four outlined regions are the heaters.";
    for (const { run, array, reference } of shown) {
      const id = run.id;
      for (const [ch, key] of [
        [0, "T"],
        [1, "c"],
      ]) {
        $("plume-" + key + "-label-" + id).textContent =
          (difference ? "Change in " : "") +
          (ch === 0
            ? difference
              ? "temperature"
              : "Temperature"
            : difference
              ? "conversion"
              : "Conversion");
        heatmap($("plume-" + key + "-" + id), array, k, ch, {
          dataset: data,
          errorAgainst: reference,
          differenceLabel: true,
        });
      }
      const peak = run.peak_temperature_K[k],
        conversion = run.outlet_conversion[k];
      $("plume-stats-" + id).innerHTML =
        stat(
          peak.toFixed(1) + " K",
          "Peak temperature · limit 1070 K",
          peak > 1070,
        ) +
        stat(
          (100 * conversion).toFixed(2) + "%",
          "Outlet conversion · flux weighted",
          conversion < 0.95,
        ) +
        stat(
          Math.max(0, peak - 1070).toFixed(1) + " K",
          "Temperature above limit",
          peak > 1070,
        ) +
        stat(run.energy_s[k].toFixed(2) + " s", "Normalized heating energy");
      $("plume-heaters-" + id).innerHTML = run.controls[k]
        .map(
          (v, j) =>
            `<span>H${j + 1} <i><em style="width:${v * 100}%"></em></i> ${(v * 100).toFixed(0)}%</span>`,
        )
        .join("");
      const s = run.summary;
      $("plume-summary-" + id).textContent =
        `Over 80 s: peak ${s.maximum_temperature_K.toFixed(1)} K; temperature above the limit for ${s.temperature_violation_duration_s.toFixed(2)} s; outlet conversion below 95% for ${s.conversion_violation_duration_s.toFixed(2)} s (checks every 0.25 s).`;
    }
    velocity();
  }
  function play() {
    if (!shown.length || $("plume").hidden) return;
    if (raf !== null) {
      pause();
      return;
    }
    if (+$("plume-time").value >= 320) $("plume-time").value = 0;
    elapsed = data.times_s[+$("plume-time").value];
    lastStamp = performance.now();
    $("plume-play").textContent = "Pause";
    $("plume-play").setAttribute("aria-label", "Pause PDE dynamics");
    function tick(stamp) {
      elapsed = Math.min(
        data.duration_s,
        elapsed + ((stamp - lastStamp) / 1000) * +$("plume-speed").value,
      );
      lastStamp = stamp;
      const frame = Math.min(
        320,
        Math.floor((elapsed + 1e-6) / data.frame_dt_s),
      );
      if (frame !== +$("plume-time").value) {
        $("plume-time").value = frame;
        draw();
      }
      if (elapsed >= data.duration_s) {
        pause();
        return;
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
  }
  $("plume-play").onclick = play;
  $("plume-reset").onclick = () => {
    $("plume-time").value = 0;
    pause();
    draw();
  };
  $("plume-time").oninput = () => {
    pause();
    draw();
  };
  for (const id of ["plume-schedule", "plume-display", "plume-comparison"])
    $(id).onchange = () => load().catch(fail);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  return {
    pause,
    draw,
    async init() {
      if (initialized) {
        draw();
        return;
      }
      try {
        const r = await fetch(base + "manifest.json", { cache: "no-store" });
        if (!r.ok)
          throw Error(
            "Plume artifacts missing. Run: python scripts/build_thermal_reactor.py plume",
          );
        data = await r.json();
        const v = data.numerical_validation;
        $("plume-refinement").textContent =
          `The mesh has 64 × 32 cells, each storing temperature and reactant fraction. Saved fields are 0.25 s apart; integration substeps are at most 0.1 s. Halving the timestep changes temperature by ${v.time_refinement.temperature_rmse_K.toFixed(3)} K RMSE; doubling the mesh changes it by ${v.space_refinement.temperature_rmse_K.toFixed(2)} K RMSE. Spreading includes numerical diffusion from first-order upwind transport, which exceeds physical diffusion at this resolution.`;
        await load();
        initialized = true;
        if (
          !matchMedia("(prefers-reduced-motion: reduce)").matches &&
          !document.hidden
        )
          play();
      } catch (e) {
        fail(e);
      }
    },
  };
}
