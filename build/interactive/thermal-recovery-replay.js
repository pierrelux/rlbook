// Saved direct-PDE field restoration. No browser-side optimization or simulation.
export function createRecoveryView({ $, heatmap, setup, stat }) {
  $("recovery").innerHTML = `
    <p class="recovery-intro">The cold pulse is inside the reactor. Restore the desired temperature and conversion fields while keeping the temperature below 1070 K and outlet conversion above 95%. Control starts at 14 s; the inlet stays normal afterwards.</p>
    <div class="recovery-target-row"><div><h3>Desired temperature</h3><canvas id="recovery-target-T" aria-label="Desired temperature field"></canvas></div><div><h3>Desired conversion</h3><canvas id="recovery-target-c" aria-label="Desired conversion field"></canvas></div><p>The target is the warm operating field at <b>46% nominal heating</b>. MPPI can increase or decrease any heater. These runs share the same disturbed initial state.</p></div>
    <div class="tabs" role="tablist" aria-label="Recovery views"><button role="tab" aria-selected="true" data-recovery-panel="runs">Field restoration</button><button role="tab" aria-selected="false" data-recovery-panel="samples">Sampled control update</button></div>
    <p id="recovery-status" role="status">Loading saved recovery…</p>
    <section id="recovery-runs">
      <div class="toolbar"><label>Compare MPPI with<select id="recovery-comparison" aria-label="Recovery comparison"><option value="unchanged">Unchanged heating</option><option value="seed">Supplied feasible schedule</option></select></label><label>Display<select id="recovery-display" aria-label="Recovery field display"><option value="absolute">Physical fields</option><option value="error">Error relative to the target</option></select></label></div>
      <div class="timeline"><button id="recovery-play" aria-label="Play field recovery">Play</button><button id="recovery-reset">Reset</button><input id="recovery-time" aria-label="Recovery time" type="range" min="0" max="264" step="1" value="0"><output id="recovery-clock" class="time">14.00 s</output><label>Speed<select id="recovery-speed" aria-label="Recovery playback speed"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label></div>
      <p id="recovery-display-note" class="caption"></p><div id="recovery-panels" class="plume-panels compare"></div>
      <h2 class="recovery-heading">When and where the heaters act</h2><p class="caption">Signed deviations from 46% heating. Positive values add heat; negative values reduce it. Each control is held for 2 s.</p><canvas id="recovery-controls" aria-label="MPPI heater deviations from nominal over time"></canvas>
      <h2 class="recovery-heading">Recovery over the whole recording</h2><div class="table-wrap"><table><thead><tr><th>Schedule</th><th>Integrated field error</th><th>Heating energy</th><th>Peak temperature</th><th>Lowest outlet conversion</th></tr></thead><tbody id="recovery-results"></tbody></table></div><p id="recovery-results-note" class="caption"></p>
    </section>
    <section id="recovery-samples" hidden>
      <div class="toolbar"><label>Decision<select id="recovery-decision" aria-label="Recovery planning decision"></select></label><label>Field<select id="recovery-candidate-field" aria-label="Recovery candidate field"><option value="0">Temperature error</option><option value="1">Conversion error</option></select></label></div>
      <p id="recovery-sampling-note"></p><div class="timeline"><button id="recovery-future-play">Play futures</button><button id="recovery-future-reset">Reset</button><input id="recovery-future" aria-label="Recovery prediction time ahead" type="range" min="0" max="160" step="1" value="0"><output id="recovery-future-clock" class="time">+0.00 s</output></div>
      <div id="recovery-candidates" class="candidates"></div><h2 class="recovery-heading">The weighted update and the plan kept</h2><p id="recovery-update-note"></p><div class="plume-panels compare"><article class="panel"><h3 id="recovery-weighted-title">Weighted proposal</h3><canvas id="recovery-weighted" class="map" aria-label="Weighted proposal PDE field"></canvas><canvas id="recovery-weighted-controls" aria-label="Weighted proposal heater sequence"></canvas></article><article class="panel"><h3>Plan kept after validation</h3><canvas id="recovery-used" class="map" aria-label="Validated recovery plan PDE field"></canvas><canvas id="recovery-used-controls" aria-label="Validated plan heater sequence"></canvas></article></div>
      <p class="caption">The Gaussian weights average 24 latent parameters. A bounded transform and interpolation turn those parameters into a heating sequence. Each movie is a new PDE rollout under its displayed controls; candidate field images are never averaged. The main replay executes only the first action before replanning.</p>
    </section>
    <details><summary>Objective, initialization, and numerical checks</summary><p>Field error is the cell average of (temperature error / 20 K)² + (conversion error / 0.1)². The 40 s objective averages this error with control-deviation and slew penalties, then adds terminal field error. Both warmer and colder departures from the desired field are penalized.</p><p>The supplied feasible schedule starts H1 at 50%, then returns to 46%; the other heaters remain at 46%. The optimizer begins from this known schedule. Its fixed Gaussian reference centers sampling around the initial schedule of each decision. It makes four updates using 256 candidates, with a Gaussian proposal correction as the sampling mean moves. A damped update is used when the full update fails the improvement or feasibility checks.</p><p>Rejected updates retain the exact feasible control tape. Each accepted proposal is checked every 0.25 s over its 40 s horizon and another 40 s at nominal heating. Neither these grid checks nor the finite-sample optimizer establish a continuous-time guarantee or a globally optimal solution.</p><p id="recovery-validation-note"></p><p id="recovery-runtime-note"></p><h3>Opening-decision sampling sensitivity</h3><div class="table-wrap"><table><thead><tr><th>Candidates</th><th>Seed</th><th>Initial → final objective</th><th>Mean ESS</th><th>Runtime</th></tr></thead><tbody id="recovery-sensitivity"></tbody></table></div><p class="caption">These seeds vary the optimizer's samples at the same physical state. The inlet and the PDE are deterministic. Sampling variability is not physical uncertainty.</p><p><a href="../artifacts/thermal_recovery/README.md">Model and regeneration notes</a> · <a href="../_static/thermal_reactor/recovery-storyboard.pdf">Recovery storyboard (PDF)</a> · <a href="../_static/thermal_reactor/recovery-control.pdf">Control and tracking figure (PDF)</a></p></details>`;
  const names = {
    unchanged: "Unchanged heating",
    seed: "Supplied feasible schedule",
    mppi: "MPPI recovery",
  };
  const colors = ["#176787", "#b95525", "#298066", "#92704f"];
  let data,
    target,
    runs = [],
    snapshot,
    initPromise,
    runGeneration = 0,
    sampleGeneration = 0,
    raf = null;
  const cache = new Map(),
    base = "./thermal-recovery-data/";
  function pause() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
    $("recovery-play").textContent = "Play";
    $("recovery-future-play").textContent = "Play futures";
  }
  async function read(desc) {
    if (!cache.has(desc.file))
      cache.set(
        desc.file,
        (async () => {
          const r = await fetch(base + desc.file, { cache: "no-store" });
          if (!r.ok) throw Error(`Cannot load ${desc.file}`);
          const b = await r.arrayBuffer();
          if (b.byteLength !== desc.shape.reduce((a, v) => a * v, 1) * 4)
            throw Error("Unexpected recovery field size");
          return new Float32Array(b);
        })().catch((e) => {
          cache.delete(desc.file);
          throw e;
        }),
      );
    return cache.get(desc.file);
  }
  function fail(e) {
    pause();
    $("recovery-status").textContent = e.message;
    $("recovery-status").className = "error";
  }
  function panel(id) {
    return `<article class="panel"><div class="labelrow"><h2>${names[id]}</h2><span class="badge">PDE execution</span></div><h3 id="recovery-T-label-${id}">Temperature</h3><canvas id="recovery-T-${id}" class="map" aria-label="${names[id]} temperature field"></canvas><h3 id="recovery-c-label-${id}">Conversion</h3><canvas id="recovery-c-${id}" class="map" aria-label="${names[id]} conversion field"></canvas><div id="recovery-stats-${id}" class="stats"></div><div id="recovery-heaters-${id}" class="heaterbars"></div><p id="recovery-plan-${id}" class="caption"></p></article>`;
  }
  async function loadRuns() {
    if (!data || !target) return;
    pause();
    const token = ++runGeneration;
    const ids = [$("recovery-comparison").value, "mppi"];
    const loaded = [];
    $("recovery-panels").classList.add("loading");
    for (const id of ids) {
      const run = data.runs.find((r) => r.id === id);
      loaded.push({ run, array: await read(run.fields) });
    }
    if (token !== runGeneration) return;
    runs = loaded;
    $("recovery-panels").innerHTML = ids.map(panel).join("");
    $("recovery-panels").classList.remove("loading");
    draw();
  }
  function controls(canvas, tape, cursor, mini = false) {
    const { ctx, width, height } = setup(canvas, mini ? 90 : 200),
      left = mini ? 28 : 44,
      right = mini ? 8 : 18,
      top = mini ? 14 : width < 550 ? 48 : 28,
      bottom = mini ? 20 : 32,
      w = width - left - right,
      h = height - top - bottom;
    const bound = 0.12,
      y = (v) => top + h * (0.5 - v / (2 * bound));
    ctx.font = "11px system-ui";
    ctx.strokeStyle = "#cbd5d1";
    ctx.fillStyle = "#607079";
    for (const v of mini ? [0] : [-0.05, 0, 0.05]) {
      ctx.beginPath();
      ctx.moveTo(left, y(v));
      ctx.lineTo(left + w, y(v));
      ctx.stroke();
      ctx.fillText(
        (v > 0 ? "+" : "") + (v * 100).toFixed(0),
        mini ? 4 : 10,
        y(v) + 3,
      );
    }
    const H = tape.length;
    tape[0].forEach((_, j) => {
      ctx.strokeStyle = colors[j];
      ctx.lineWidth = mini ? 1.2 : 1.8;
      ctx.beginPath();
      for (let k = 0; k < H; k++) {
        const value = tape[k][j] - 0.46;
        ctx.lineTo(left + (w * k) / H, y(value));
        ctx.lineTo(left + (w * (k + 1)) / H, y(value));
      }
      ctx.stroke();
    });
    ctx.strokeStyle = "#24313a";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(left + (w * cursor) / H, top);
    ctx.lineTo(left + (w * cursor) / H, top + h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#607079";
    ctx.fillText(mini ? "0 s" : "14 s", left, height - 5);
    ctx.fillText((mini ? H * 2 : 14 + H * 2) + " s", left + w - 28, height - 5);
    if (!mini) {
      ctx.fillText("Percentage points above 46%", left, 14);
      colors.forEach((color, j) => {
        ctx.fillStyle = color;
        ctx.fillText(
          "H" + (j + 1),
          (width < 550 ? left : width - 220) + j * 48,
          width < 550 ? 30 : 14,
        );
      });
    }
  }
  function drawRuns() {
    if (!data || !runs.length) return;
    const requested = +$("recovery-time").value;
    const t = data.start_s + requested * 0.25;
    const error = $("recovery-display").value === "error";
    $("recovery-clock").textContent = t.toFixed(2) + " s";
    $("recovery-display-note").textContent = error
      ? "Each field minus the same desired operating field. Blue means colder material or less conversion; red means warmer material or more conversion. Scalar diagnostics describe the actual run."
      : "Numerical PDE fields under the displayed controls. The desired fields above are fixed throughout recovery.";
    for (const { run, array } of runs) {
      const k = Math.min(requested, run.fields.shape[0] - 1),
        id = run.id,
        u = run.controls[Math.min(Math.floor(k / 8), run.controls.length - 1)];
      for (const [ch, key] of [
        [0, "T"],
        [1, "c"],
      ]) {
        $("recovery-" + key + "-label-" + id).textContent =
          (ch ? "Conversion" : "Temperature") + (error ? " error" : "");
        heatmap($("recovery-" + key + "-" + id), array, k, ch, {
          dataset: data,
          errorAgainst: error ? target : null,
          referenceFrame: 0,
        });
      }
      const e =
          run.temperature_tracking_error[k] + run.conversion_tracking_error[k],
        peak = run.peak_temperature_K[k],
        conv = run.outlet_conversion[k];
      $("recovery-stats-" + id).innerHTML =
        stat(e.toFixed(4), "Normalized field error") +
        stat(peak.toFixed(1) + " K", "Peak · limit 1070 K", peak > 1070) +
        stat(
          (conv * 100).toFixed(2) + "%",
          "Outlet · target ≥95%",
          conv < 0.95,
        ) +
        stat(
          (
            run.energy_s?.[k] ??
            run.controls
              .slice(0, Math.floor(k / 8))
              .reduce((s, u) => s + (u.reduce((a, b) => a + b, 0) / 4) * 2, 0) +
              ((k % 8) * 0.25 * u.reduce((a, b) => a + b, 0)) / 4
          ).toFixed(2) + " s",
          "Heating energy since 14 s",
        );
      $("recovery-heaters-" + id).innerHTML = u
        .map(
          (v, j) =>
            `<span>H${j + 1} <i><em style="width:${v * 100}%"></em></i> ${(v * 100).toFixed(1)}%</span>`,
        )
        .join("");
      const rec =
        run.decisions[Math.min(Math.floor(k / 8), run.decisions.length - 1)];
      $("recovery-plan-" + id).textContent = rec
        ? `Decision at ${rec.time_s} s: ${rec.accepted} accepted, ${rec.rejected} rejected updates; ${rec.runtime_s.toFixed(2)} s offline planning. ${k === run.fields.shape[0] - 1 ? "Recording complete; the last applied control is shown." : "The control shown applies to the current 2 s interval."}`
        : id === "seed"
          ? "A supplied feasible recovery schedule, not an optimized result."
          : "All four heaters remain at 46%.";
    }
    const r = data.runs.find((r) => r.id === "mppi");
    controls($("recovery-controls"), r.controls, requested / 8);
  }
  async function loadSnapshot() {
    if (!data) return;
    pause();
    const token = ++sampleGeneration;
    const s = data.snapshots[+$("recovery-decision").value];
    const arrays = [];
    for (const c of s.candidates) arrays.push(await read(c.fields));
    arrays.push(await read(s.weighted.fields), await read(s.used.fields));
    if (token !== sampleGeneration) return;
    snapshot = { s, arrays };
    $("recovery-candidates").innerHTML = s.candidates
      .map(
        (c, j) =>
          `<article class="panel candidate"><h3>Candidate ${c.index + 1}</h3><canvas id="recovery-mini-${j}" class="mini" aria-label="Recovery candidate ${c.index + 1} field"></canvas><div class="caption">Cost ${c.objective?.toFixed(4) ?? "invalid"} · weight ${(100 * c.weight).toFixed(2)}%${c.coarse_feasible ? "" : " · violates a constraint"}</div><canvas id="recovery-spark-${j}" aria-label="Recovery candidate ${c.index + 1} heating sequence"></canvas></article>`,
      )
      .join("");
    drawSnapshot();
  }
  function drawSnapshot() {
    if (!snapshot) return;
    const { s, arrays } = snapshot,
      k = +$("recovery-future").value,
      ch = +$("recovery-candidate-field").value,
      rec = s.iteration;
    $("recovery-future-clock").textContent = "+" + (k * 0.25).toFixed(2) + " s";
    $("recovery-sampling-note").textContent =
      `Decision at ${s.time_s} s, update ${rec.iteration} of ${data.planner.iterations}. Six of ${data.planner.candidates} candidates; ${rec.feasible_candidates} passed the coarse constraint checks. ESS ${rec.ess.toFixed(1)}. Displayed weights come from the full batch and include the Gaussian proposal correction. Every movie uses the same known normal inlet.`;
    s.candidates.forEach((c, j) => {
      heatmap($("recovery-mini-" + j), arrays[j], k, ch, {
        dataset: data,
        errorAgainst: target,
        referenceFrame: 0,
        mini: true,
      });
      controls($("recovery-spark-" + j), c.controls, k / 8, true);
    });
    heatmap($("recovery-weighted"), arrays[6], k, ch, {
      dataset: data,
      errorAgainst: target,
      referenceFrame: 0,
    });
    heatmap($("recovery-used"), arrays[7], k, ch, {
      dataset: data,
      errorAgainst: target,
      referenceFrame: 0,
    });
    controls($("recovery-weighted-controls"), s.weighted.controls, k / 8, true);
    controls($("recovery-used-controls"), s.used.controls, k / 8, true);
    $("recovery-weighted-title").textContent = s.weighted_proposal_exists
      ? "Weighted proposal"
      : "No valid weighted proposal";
    $("recovery-update-note").textContent =
      (rec.accepted
        ? `Accepted at update fraction ${rec.alpha}. `
        : "Rejected; the exact feasible incumbent was retained. ") +
      `Incumbent objective ${rec.incumbent_before_objective.toFixed(5)} → ${rec.incumbent_after_objective.toFixed(5)}. Raw weighted-proposal objective ${rec.weighted_proposal_objective?.toFixed(5) ?? "unavailable"}. Maps show error relative to the target.`;
  }
  function draw() {
    if (!data || !target || $("recovery").hidden) return;
    heatmap($("recovery-target-T"), target, 0, 0, {
      dataset: data,
      mini: true,
    });
    heatmap($("recovery-target-c"), target, 0, 1, {
      dataset: data,
      mini: true,
    });
    if (!$("recovery-runs").hidden) drawRuns();
    else drawSnapshot();
  }
  function play(future = false) {
    if (!data || !runs.length || (future && !snapshot)) return;
    if (raf !== null) {
      pause();
      return;
    }
    const id = future ? "recovery-future" : "recovery-time",
      max = future ? 160 : 264,
      button = future ? "recovery-future-play" : "recovery-play";
    if (+$(id).value === max) $(id).value = 0;
    let t = +$(id).value * 0.25,
      stamp = performance.now();
    $(button).textContent = "Pause";
    function tick(now) {
      t = Math.min(
        max * 0.25,
        t + ((now - stamp) / 1000) * (future ? 4 : +$("recovery-speed").value),
      );
      stamp = now;
      const k = Math.min(max, Math.floor(t / 0.25));
      if (k !== +$(id).value) {
        $(id).value = k;
        future ? drawSnapshot() : drawRuns();
      }
      if (k === max) {
        pause();
        return;
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
  }
  $("recovery-play").onclick = () => play();
  $("recovery-future-play").onclick = () => play(true);
  for (const [time, reset] of [
    ["recovery-time", "recovery-reset"],
    ["recovery-future", "recovery-future-reset"],
  ]) {
    $(time).oninput = () => {
      pause();
      draw();
    };
    $(reset).onclick = () => {
      pause();
      $(time).value = 0;
      draw();
    };
  }
  $("recovery-comparison").onchange = () => loadRuns().catch(fail);
  $("recovery-display").onchange = draw;
  $("recovery-candidate-field").onchange = draw;
  $("recovery-decision").onchange = () => loadSnapshot().catch(fail);
  document.querySelectorAll("[data-recovery-panel]").forEach(
    (b) =>
      (b.onclick = async () => {
        pause();
        document
          .querySelectorAll("[data-recovery-panel]")
          .forEach((x) => x.setAttribute("aria-selected", String(x === b)));
        for (const id of ["runs", "samples"])
          $("recovery-" + id).hidden = id !== b.dataset.recoveryPanel;
        if (b.dataset.recoveryPanel === "samples" && !snapshot)
          await loadSnapshot().catch(fail);
        draw();
      }),
  );
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  return {
    pause,
    draw,
    async init() {
      if (initPromise) return initPromise;
      initPromise = (async () => {
        try {
          const r = await fetch(base + "manifest.json", { cache: "no-store" });
          if (!r.ok)
            throw Error(
              "Recovery artifacts are not ready. Run: python scripts/build_thermal_reactor.py recovery",
            );
          data = await r.json();
          target = await read(data.target);
          $("recovery-decision").innerHTML = data.snapshots
            .map(
              (s, i) =>
                `<option value="${i}">${s.time_s} s · final update</option>`,
            )
            .join("");
          $("recovery-results").innerHTML = data.runs
            .map((r) => {
              const s = r.summary;
              return `<tr><td>${names[r.id]}</td><td>${s.integrated_field_error_s.toFixed(3)} s</td><td>${s.energy_s.toFixed(2)} s</td><td>${s.maximum_temperature_K.toFixed(1)} K</td><td>${(s.minimum_outlet_conversion * 100).toFixed(2)}%</td></tr>`;
            })
            .join("");
          const mp = data.runs.find((r) => r.id === "mppi"),
            un = data.runs.find((r) => r.id === "unchanged"),
            seed = data.runs.find((r) => r.id === "seed");
          $("recovery-results-note").textContent =
            `MPPI integrated field error changes by ${(100 * (mp.summary.integrated_field_error_s / un.summary.integrated_field_error_s - 1)).toFixed(1)}% relative to unchanged heating and ${(100 * (mp.summary.integrated_field_error_s / seed.summary.integrated_field_error_s - 1)).toFixed(1)}% relative to the supplied schedule. Unchanged heating falls below 95% conversion for ${un.summary.quality_violation_duration_s.toFixed(2)} s. MPPI temperature/quality violations: ${mp.summary.temperature_violation_duration_s.toFixed(2)} / ${mp.summary.quality_violation_duration_s.toFixed(2)} s. All durations use the 0.25 s output grid.`;
          $("recovery-status").className = "note";
          $("recovery-status").textContent =
            "Recorded direct-PDE MPPI · simulated control interval 2 s · all optimization ran offline";
          const v = data.numerical_validation;
          $("recovery-validation-note").textContent =
            `Halving the integration timestep changes temperature by ${v.time_refinement.temperature_rmse_K.toFixed(3)} K RMSE. Doubling the mesh changes temperature by ${v.space_refinement.temperature_rmse_K.toFixed(2)} K RMSE and outlet conversion by ${v.space_refinement.outlet_conversion_mae.toFixed(5)} on average. On the finer mesh, peak temperature is ${v.space_refinement.maximum_temperature_K.toFixed(2)} K and minimum outlet conversion is ${(100 * v.space_refinement.minimum_outlet_conversion).toFixed(2)}%. Numerical spreading is significant with this first-order transport scheme.`;
          $("recovery-runtime-note").textContent = data.runtime_note;
          $("recovery-sensitivity").innerHTML = data.sensitivity
            .map(
              (r) =>
                `<tr><td>${r.candidates}</td><td>${r.seed}</td><td>${r.initial_objective.toFixed(5)} → ${r.final_objective.toFixed(5)}</td><td>${(r.ess.reduce((s, v) => s + v, 0) / r.ess.length).toFixed(1)}</td><td>${r.runtime_s.toFixed(2)} s</td></tr>`,
            )
            .join("");
          await loadRuns();
        } catch (e) {
          initPromise = null;
          fail(e);
        }
      })();
      return initPromise;
    },
  };
}
