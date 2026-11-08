// Replay saved PDE fields. This module performs no optimization or field simulation.
export function createStirringView({ $, setup, stat }) {
  $("stirring").innerHTML = `
    <p>Warm and cold material keeps entering the channel. Three reversible stirring regions change where it travels. The objective is uniform temperature and conversion in the final 2 m, using limited stirring effort.</p>
    <p class="caption">Jacket temperature: 1050 K · known inlet cycle: 32 s · no direct heaters · instantaneous, prescribed incompressible flow modes</p>
    <div class="tabs" role="tablist" aria-label="Stirring views"><button role="tab" aria-selected="true" data-stirring-panel="runs">Mixing the flow</button><button role="tab" aria-selected="false" data-stirring-panel="samples">Sampled stirring update</button></div>
    <p id="stirring-status" role="status">Loading stirring recordings…</p>
    <div class="toolbar"><label>Display<select id="stirring-display" aria-label="Stirring field display"><option value="absolute">Physical fields</option><option value="variation">Departure from cross-section mean</option></select></label><label>Flow arrows<select id="stirring-arrows" aria-label="Stirring flow arrows"><option value="on">Show</option><option value="off">Hide</option></select></label></div>
    <p id="stirring-display-note" class="caption"></p>
    <section id="stirring-runs"><div class="toolbar"><label>Compare MPPI with<select id="stirring-comparison" aria-label="Stirring comparison"><option value="unstirred">No stirring</option><option value="constant">Best grid constant</option></select></label><label>Optimizer seed<select id="stirring-seed" aria-label="Stirring optimizer seed"><option value="19744">19744</option><option value="19745">19745</option><option value="19746">19746</option></select></label></div>
    <div class="timeline"><button id="stirring-play" aria-label="Play stirring replay">Play</button><button id="stirring-reset">Reset</button><input id="stirring-time" aria-label="Stirring time" type="range" min="0" max="640" step="1" value="0"><output id="stirring-clock" class="time">0.00 s</output><label>Speed<select id="stirring-speed" aria-label="Stirring playback speed"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label></div>
    <div id="stirring-panels" class="plume-panels compare"></div>
    <h2>Where the stirring effort goes</h2><p class="caption">S1, S2, and S3 act at 2, 4, and 6 m. Positive controls add counterclockwise circulation; negative controls reverse it. The displayed arrows include the downstream background flow.</p><canvas id="stirring-controls" aria-label="Three signed MPPI stirring controls over time"></canvas>
    <h2>Mixing over the whole recording</h2><div class="table-wrap"><table><thead><tr><th>Control</th><th>Mean variation</th><th>Downstream T RMS</th><th>Outlet T std.</th><th>Stirring effort</th><th>Lowest conversion</th></tr></thead><tbody id="stirring-results"></tbody></table></div><p id="stirring-results-note" class="caption"></p></section>
    <section id="stirring-samples" hidden><div class="toolbar"><label>Decision<select id="stirring-decision" aria-label="Stirring planning decision"></select></label><label>Field<select id="stirring-candidate-field" aria-label="Stirring candidate field"><option value="0">Temperature</option><option value="1">Conversion</option></select></label></div><p id="stirring-sampling-note"></p>
    <div class="timeline"><button id="stirring-future-play">Play futures</button><button id="stirring-future-reset">Reset</button><input id="stirring-future" aria-label="Stirring prediction time ahead" type="range" min="0" max="160" step="1" value="0"><output id="stirring-future-clock" class="time">+0.00 s</output></div>
    <p class="caption">Signed control traces share a −1 to +1 scale: <span style="color:#0072B2">S1 —</span> · <span style="color:#D55E00">S2 – –</span> · <span style="color:#009E73">S3 · —</span>. The vertical cursor marks the movie time.</p><div id="stirring-candidates" class="candidates"></div>
    <h2>The weighted proposal and the plan kept</h2><p id="stirring-update-note"></p><div class="plume-panels compare"><article class="panel"><h3>Weighted proposal</h3><canvas id="stirring-weighted" class="map" aria-label="Weighted stirring proposal PDE field"></canvas><canvas id="stirring-weighted-controls" aria-label="Weighted stirring proposal controls"></canvas></article><article class="panel"><h3>Plan kept after validation</h3><canvas id="stirring-used" class="map" aria-label="Validated stirring plan PDE field"></canvas><canvas id="stirring-used-controls" aria-label="Validated stirring controls"></canvas></article></div>
    <p class="caption">Weights average 60 independent latent coordinates. A temporal filter and capacity projection produce the new controls. Every movie is a separate PDE rollout under those controls; field images are never averaged. The executed replay applies one action before replanning.</p></section>
    <details><summary>Objective, flow model, and numerical evidence</summary><p>The jacket exchanges heat toward 1050 K. The three control inputs multiply prescribed velocity fields; velocity and pressure are not additional simulated states. The 1070 K check is primarily a physical and numerical consistency check here. The outlet conversion requirement remains 95%.</p><p>Mixing error averages (temperature departure / 20 K)² + (conversion departure / 0.1)² over the final 2 m. Each departure is relative to that cross-section's mean. The objective adds normalized effort and control-change penalties, plus terminal mixing error. There is no reward for reversing or alternating the controls.</p><p id="stirring-method-note"></p><p id="stirring-refinement-note"></p><p id="stirring-transport-note"></p><h3>Candidate-count sensitivity at the opening state</h3><div class="table-wrap"><table><thead><tr><th>Candidates</th><th>Seed</th><th>Initial → final objective</th><th>Mean ESS</th><th>Planning time</th></tr></thead><tbody id="stirring-sensitivity"></tbody></table></div><p class="caption">Seeds change optimizer samples, not the physical inlet. All planning is offline; the 2 s decision interval is simulated time.</p><p><a href="../artifacts/thermal_stirring/README.md">Model and regeneration notes</a> · <a href="../_static/thermal_reactor/stirring-storyboard.pdf">Mixing storyboard (PDF)</a> · <a href="../_static/thermal_reactor/stirring-control.pdf">Control and comparison figure (PDF)</a></p></details>`;
  const colors = ["#0072B2", "#D55E00", "#009E73"];
  const base = "./thermal-stirring-data/",
    cache = new Map();
  let data,
    initialization,
    runs = [],
    snapshot,
    runToken = 0,
    sampleToken = 0,
    raf = null;
  const name = (id) =>
    id === "unstirred"
      ? "No stirring"
      : id === "constant"
        ? "Best grid constant"
        : "MPPI · seed " + id.split("-")[1];
  function pause() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
    $("stirring-play").textContent = "Play";
    $("stirring-future-play").textContent = "Play futures";
  }
  function fail(e) {
    pause();
    $("stirring-status").textContent = e.message;
    $("stirring-status").className = "error";
  }
  async function read(desc) {
    if (!cache.has(desc.file))
      cache.set(
        desc.file,
        (async () => {
          const response = await fetch(base + desc.file, { cache: "no-store" });
          if (!response.ok) throw Error("Cannot load " + desc.file);
          const bytes = await response.arrayBuffer();
          if (bytes.byteLength !== desc.shape.reduce((a, b) => a * b, 1) * 4)
            throw Error("Unexpected stirring field size");
          return new Float32Array(bytes);
        })().catch((e) => {
          cache.delete(desc.file);
          throw e;
        }),
      );
    return cache.get(desc.file);
  }
  function action(tape, frame) {
    return tape[Math.min(tape.length - 1, Math.floor(frame / 8))];
  }
  function map(canvas, array, frame, channel, controls, mini = false) {
    if (!data) return;
    const variation = $("stirring-display").value === "variation",
      cfg = data.config;
    const available = canvas.clientWidth || 480;
    const { ctx, width, height } = setup(
      canvas,
      mini
        ? Math.max(50, available / 4 + 10)
        : Math.max(145, (available - 56) / 4 + 96),
    );
    const left = mini ? 5 : 42,
      top = mini ? 4 : 30,
      w = width - left - (mini ? 5 : 14),
      h = w / 4;
    const limits =
      data.limits[
        variation
          ? channel
            ? "conversion_error"
            : "temperature_error_K"
          : channel
            ? "conversion"
            : "temperature_K"
      ];
    const lut =
      data.colormaps[variation ? "RdBu_r" : channel ? "viridis" : "magma"];
    const count = cfg.nx * cfg.ny,
      offset = (frame * 2 + channel) * count;
    const means = new Float64Array(cfg.nx);
    if (variation)
      for (let y = 0; y < cfg.ny; y++)
        for (let x = 0; x < cfg.nx; x++)
          means[x] += array[offset + y * cfg.nx + x] / cfg.ny;
    const pixel = document.createElement("canvas");
    pixel.width = cfg.nx;
    pixel.height = cfg.ny;
    const pc = pixel.getContext("2d"),
      im = pc.createImageData(cfg.nx, cfg.ny);
    for (let y = 0; y < cfg.ny; y++)
      for (let x = 0; x < cfg.nx; x++) {
        let value = array[offset + y * cfg.nx + x];
        value = variation
          ? (value - means[x]) * (channel ? -1 : 1)
          : channel
            ? 1 - value
            : value;
        const index = Math.round(
          255 *
            Math.max(
              0,
              Math.min(1, (value - limits[0]) / (limits[1] - limits[0])),
            ),
        );
        const dest = ((cfg.ny - 1 - y) * cfg.nx + x) * 4;
        im.data.set([...lut[index], 255], dest);
      }
    pc.putImageData(im, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(pixel, left, top, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, w, h);
    ctx.clip();
    // The shaded boundary identifies the objective region, not a solid wall.
    ctx.strokeStyle = variation ? "#727b80" : "#ffffffbb";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(left + w * 0.75, top);
    ctx.lineTo(left + w * 0.75, top + h);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!mini && $("stirring-arrows").value === "on") {
      const ys = data.arrow_y_m,
        xs = data.arrow_x_m,
        b = data.arrow_basis;
      for (let iy = 0; iy < ys.length; iy++)
        for (let ix = 0; ix < xs.length; ix++) {
          let u = b[0][0][iy][ix],
            v = b[0][1][iy][ix];
          for (let j = 0; j < 3; j++) {
            u += controls[j] * b[j + 1][0][iy][ix];
            v += controls[j] * b[j + 1][1][iy][ix];
          }
          const x = left + (xs[ix] / 8) * w,
            y = top + (1 - ys[iy] / 2) * h,
            dx = (u * w) / 8,
            dy = (-v * h) / 2;
          const angle = Math.atan2(dy, dx),
            tipx = x + dx,
            tipy = y + dy;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(tipx, tipy);
          ctx.moveTo(
            tipx - 3 * Math.cos(angle - 0.5),
            tipy - 3 * Math.sin(angle - 0.5),
          );
          ctx.lineTo(tipx, tipy);
          ctx.lineTo(
            tipx - 3 * Math.cos(angle + 0.5),
            tipy - 3 * Math.sin(angle + 0.5),
          );
          ctx.strokeStyle = variation ? "#26383d99" : "#00000080";
          ctx.lineWidth = variation ? 1 : 2.6;
          ctx.stroke();
          if (!variation) {
            ctx.strokeStyle = "#ffffffcc";
            ctx.lineWidth = 0.9;
            ctx.stroke();
          }
        }
    }
    ctx.restore();
    if (mini) return;
    ctx.fillStyle = "#607079";
    ctx.font = "11px system-ui";
    ctx.fillText("Inlet →", left, 14);
    ctx.fillText("Outlet", left + w - 33, 14);
    [2, 4, 6].forEach((x, j) => {
      ctx.fillStyle = colors[j];
      ctx.fillText("S" + (j + 1), left + (w * x) / 8 - 7, 26);
    });
    ctx.fillStyle = "#607079";
    ctx.fillText("2 m", 5, top + 5);
    ctx.fillText("0 m", 5, top + h);
    [0, 4, 8].forEach((x) =>
      ctx.fillText(x + " m", left + (x / 8) * w - (x ? 12 : 0), top + h + 17),
    );
    const barW = Math.min(220, w * 0.6),
      barY = top + h + 30;
    for (let i = 0; i < 256; i++) {
      ctx.fillStyle = `rgb(${lut[i].join(",")})`;
      ctx.fillRect(left + (i / 256) * barW, barY, barW / 256 + 1, 7);
    }
    ctx.fillStyle = "#607079";
    ctx.fillText(limits[0], left, barY + 20);
    ctx.fillText(limits[1], left + barW - 18, barY + 20);
    ctx.fillText(
      variation
        ? channel
          ? "Δ conversion"
          : "Δ T (K)"
        : channel
          ? "Conversion"
          : "T (K)",
      left + barW + 10,
      barY + 7,
    );
  }
  function controlPlot(canvas, tape, frame, mini = false, start = 0) {
    const { ctx, width, height } = setup(canvas, mini ? 90 : 200);
    const left = mini ? 22 : 40,
      right = 12,
      top = mini ? 10 : 34,
      bottom = 24,
      w = width - left - right,
      h = height - top - bottom;
    const Y = (v) => top + ((1 - v) * h) / 2;
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#607079";
    for (const v of [-1, 0, 1]) {
      ctx.strokeStyle = "#d9e1dd";
      ctx.beginPath();
      ctx.moveTo(left, Y(v));
      ctx.lineTo(left + w, Y(v));
      ctx.stroke();
      ctx.fillText(v > 0 ? "+1" : String(v), 0, Y(v) + 3);
    }
    for (let j = 0; j < 3; j++) {
      ctx.strokeStyle = colors[j];
      ctx.lineWidth = mini ? 1.3 : 1.8;
      ctx.setLineDash(j === 1 ? [5, 3] : j === 2 ? [6, 2, 1, 2] : []);
      ctx.beginPath();
      tape.forEach((a, k) => {
        ctx.lineTo(left + (k / tape.length) * w, Y(a[j]));
        ctx.lineTo(left + ((k + 1) / tape.length) * w, Y(a[j]));
      });
      ctx.stroke();
      if (!mini) {
        ctx.fillStyle = colors[j];
        ctx.fillText(
          `S${j + 1} · ${2 + 2 * j} m`,
          left + j * Math.min(110, w / 3),
          16,
        );
      }
    }
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "#24313a";
    ctx.beginPath();
    const cursor = left + (frame / (8 * tape.length)) * w;
    ctx.moveTo(cursor, top);
    ctx.lineTo(cursor, top + h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#607079";
    ctx.fillText(start + " s", left, height - 5);
    ctx.fillText(start + 2 * tape.length + " s", left + w - 34, height - 5);
  }
  function panel(id) {
    return `<article class="panel"><div class="labelrow"><h2>${name(id)}</h2><span class="badge">PDE execution</span></div><h3>Temperature</h3><canvas id="stirring-T-${id}" class="map" aria-label="${name(id)} temperature field"></canvas><h3>Conversion</h3><canvas id="stirring-c-${id}" class="map" aria-label="${name(id)} conversion field"></canvas><div id="stirring-stats-${id}" class="stats"></div><p id="stirring-actions-${id}" class="caption"></p><p id="stirring-planning-${id}" class="caption"></p></article>`;
  }
  async function loadRuns() {
    if (!data) return;
    pause();
    const token = ++runToken;
    const ids = [
      $("stirring-comparison").value,
      "mppi-" + $("stirring-seed").value,
    ];
    const loaded = await Promise.all(
      ids.map(async (id) => {
        const run = data.runs.find((r) => r.id === id);
        return { run, array: await read(run.fields) };
      }),
    );
    if (token !== runToken) return;
    runs = loaded;
    $("stirring-panels").innerHTML = ids.map(panel).join("");
    draw();
  }
  function drawRuns() {
    if (!runs.length) return;
    const frame = +$("stirring-time").value;
    $("stirring-clock").textContent = (frame * 0.25).toFixed(2) + " s";
    for (const { run, array } of runs) {
      const id = run.id,
        a = action(run.controls, frame);
      map($("stirring-T-" + id), array, frame, 0, a);
      map($("stirring-c-" + id), array, frame, 1, a);
      const error =
        run.temperature_mixing_error[frame] +
        run.conversion_mixing_error[frame];
      $("stirring-stats-" + id).innerHTML =
        stat(error.toFixed(4), "Downstream variation") +
        stat(
          run.outlet_temperature_std_K[frame].toFixed(2) + " K",
          "Outlet temperature std.",
        ) +
        stat(
          (100 * run.outlet_conversion[frame]).toFixed(2) + "%",
          "Outlet conversion · ≥95%",
          run.outlet_conversion[frame] < 0.95,
        ) +
        stat(
          run.peak_temperature_K[frame].toFixed(1) + " K",
          "Peak · limit 1070 K",
          run.peak_temperature_K[frame] > 1070,
        );
      $("stirring-actions-" + id).innerHTML =
        a
          .map(
            (v, j) =>
              `<span style="color:${colors[j]}">S${j + 1}: ${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`,
          )
          .join(" · ") +
        ` · capacity used ${(100 * a.reduce((s, v) => s + v * v, 0)).toFixed(0)}%`;
      const rec =
        run.decisions[
          Math.min(run.decisions.length - 1, Math.floor(frame / 8))
        ];
      $("stirring-planning-" + id).textContent = rec
        ? `Decision at ${rec.time_s} s: ${rec.accepted} accepted, ${rec.rejected} rejected updates; ${rec.runtime_s.toFixed(1)} s offline planning. ${frame === 640 ? "The final applied controls are shown." : "Controls are held for 2 s."}`
        : id === "unstirred"
          ? "Background downstream flow; all three stirring inputs are zero."
          : "A fixed vector selected from the declared grid, with no subsequent replanning.";
    }
    controlPlot($("stirring-controls"), runs[1].run.controls, frame);
  }
  async function loadSnapshot() {
    if (!data) return;
    pause();
    const token = ++sampleToken,
      s = data.snapshots[+$("stirring-decision").value];
    const arrays = await Promise.all(
      [...s.candidates, s.weighted, s.used].map((c) => read(c.fields)),
    );
    if (token !== sampleToken) return;
    snapshot = { s, arrays };
    $("stirring-candidates").innerHTML = s.candidates
      .map(
        (c, j) =>
          `<article class="candidate"><h3>Candidate ${c.index}</h3><canvas id="stirring-mini-${j}" aria-label="Stirring candidate ${c.index} PDE future"></canvas><p class="caption">Cost ${c.objective.toFixed(4)} · weight ${(100 * c.weight).toFixed(2)}%${c.coarse_feasible ? "" : " · violates a constraint"}</p><canvas id="stirring-mini-controls-${j}" aria-label="Stirring candidate ${c.index} controls"></canvas></article>`,
      )
      .join("");
    draw();
  }
  function drawSnapshot() {
    if (!snapshot) return;
    const { s, arrays } = snapshot,
      k = +$("stirring-future").value,
      ch = +$("stirring-candidate-field").value,
      it = s.iteration;
    $("stirring-future-clock").textContent = "+" + (k * 0.25).toFixed(2) + " s";
    $("stirring-sampling-note").textContent =
      `Seed 19744, decision at ${s.time_s} s, update ${it.iteration} of 4. Showing weight ranks 1, 2, 3, 65, 129, and 256; ${it.feasible_candidates} candidates passed coarse checks. ESS ${it.ess.toFixed(1)}. Weights are from the full batch and include the Gaussian proposal correction. The inlet future is identical in every rollout.`;
    s.candidates.forEach((c, j) => {
      map(
        $("stirring-mini-" + j),
        arrays[j],
        k,
        ch,
        action(c.controls, k),
        true,
      );
      controlPlot($("stirring-mini-controls-" + j), c.controls, k, true);
    });
    for (const [name, index] of [
      ["weighted", 6],
      ["used", 7],
    ]) {
      map(
        $("stirring-" + name),
        arrays[index],
        k,
        ch,
        action(s[name].controls, k),
      );
      controlPlot(
        $("stirring-" + name + "-controls"),
        s[name].controls,
        k,
        true,
      );
    }
    $("stirring-update-note").textContent =
      (it.accepted
        ? `Accepted at update fraction ${it.alpha}.`
        : "The weighted update was rejected; the feasible incumbent was retained.") +
      ` Incumbent objective ${it.incumbent_before_objective.toFixed(5)} → ${it.incumbent_after_objective.toFixed(5)}. Raw weighted objective ${it.weighted_proposal_objective === null ? "unavailable" : it.weighted_proposal_objective.toFixed(5)}.`;
  }
  function draw() {
    if (!data || $("stirring").hidden) return;
    $("stirring-display-note").textContent =
      $("stirring-display").value === "variation"
        ? "Each field minus its mean across the width at the same downstream position. Blue and red mark opposite departures within that run, not target errors or surrogate errors. Fixed scales: ±60 K and ±0.2 conversion. Dashed line: start of the final 2 m used in the cost."
        : "Actual temperature and conversion from the numerical PDE. Fixed scales: 800–1070 K and 0–1 conversion. Arrow displacement corresponds to 1 s at the instantaneous flow velocity. Dashed line: start of the final 2 m used in the cost.";
    if (!$("stirring-runs").hidden) drawRuns();
    else drawSnapshot();
  }
  function play(future = false) {
    if (!data || (future ? !snapshot : !runs.length)) return;
    if (raf !== null) {
      pause();
      return;
    }
    const slider = $(future ? "stirring-future" : "stirring-time"),
      button = $(future ? "stirring-future-play" : "stirring-play"),
      max = +slider.max;
    if (+slider.value >= max) slider.value = 0;
    button.textContent = "Pause";
    let start = null,
      initial = +slider.value;
    const advance = (stamp) => {
      if (start === null) start = stamp;
      const speed = future ? 4 : +$("stirring-speed").value;
      slider.value = Math.min(
        max,
        initial + Math.floor(((stamp - start) * speed) / 250),
      );
      draw();
      if (+slider.value >= max) pause();
      else raf = requestAnimationFrame(advance);
    };
    raf = requestAnimationFrame(advance);
  }
  async function init() {
    if (initialization) return initialization;
    initialization = (async () => {
      const response = await fetch(base + "manifest.json", {
        cache: "no-store",
      });
      if (!response.ok) throw Error("Stirring recordings are unavailable");
      data = await response.json();
      $("stirring-decision").innerHTML = data.snapshots
        .map(
          (s, i) =>
            `<option value="${i}">${s.time_s} s · final update</option>`,
        )
        .join("");
      $("stirring-results").innerHTML = data.runs
        .map((r) => {
          const s = r.summary;
          return `<tr><td>${name(r.id)}</td><td>${s.mean_mixing_error.toFixed(4)}</td><td>${s.downstream_temperature_rms_K.toFixed(2)} K</td><td>${s.mean_outlet_temperature_std_K.toFixed(2)} K</td><td>${s.effort_s.toFixed(2)} s</td><td>${(100 * s.minimum_outlet_conversion).toFixed(3)}%</td></tr>`;
        })
        .join("");
      const varied = data.optimizer_variability,
        primary = data.runs.find((r) => r.id === "mppi-19744"),
        baseline = data.runs.find((r) => r.id === "unstirred"),
        constant = data.runs.find((r) => r.id === "constant");
      $("stirring-results-note").textContent =
        `Across three optimizer seeds, mean normalized variation is ${varied.mean_mixing_error_mean.toFixed(4)} ± ${varied.mean_mixing_error_sample_sd.toFixed(4)} (sample standard deviation). Seed 19744 changes integrated variation by ${(100 * (primary.summary.mean_mixing_error / baseline.summary.mean_mixing_error - 1)).toFixed(1)}% versus no stirring and ${(100 * (primary.summary.mean_mixing_error / constant.summary.mean_mixing_error - 1)).toFixed(1)}% versus the best grid constant. ${constant.controls.every((a) => a.every((v) => v === 0)) ? "The best constant on this grid is zero stirring, so the two baseline recordings coincide. " : ""}Effort is a normalized actuator-load integral, not motor energy in joules.`;
      const decisions = data.runs
          .filter((r) => r.decisions.length)
          .flatMap((r) => r.decisions),
        runtime =
          decisions.reduce((s, d) => s + d.runtime_s, 0) / decisions.length;
      const updates = decisions.flatMap((d) => d.iterations),
        rejected = updates.filter((u) => !u.accepted).length,
        meanEss = updates.reduce((s, u) => s + u.ess, 0) / updates.length,
        maxPeak = Math.max(
          ...data.runs.map((r) => r.summary.maximum_temperature_K),
        ),
        temperatureViolations = data.runs.reduce(
          (s, r) => s + r.summary.temperature_violation_s,
          0,
        ),
        conversionViolations = data.runs.reduce(
          (s, r) => s + r.summary.conversion_violation_s,
          0,
        );
      $("stirring-method-note").textContent =
        `256 candidates, four updates, and 60 independent latent parameters per decision. Across all three seeds: ${rejected} of ${updates.length} updates rejected; mean ESS ${meanEss.toFixed(1)}. Candidate evaluation backend: ${data.planner.backend}. Selected controls are validated and executed in the JAX PDE. Mean recorded planning time: ${runtime.toFixed(1)} s per decision. Fine validation covers 40 s of proposed controls plus 40 s at zero stirring. The reference stays fixed through the updates of a decision. Across all recordings, peak temperature is ${maxPeak.toFixed(2)} K; total recorded temperature and conversion violation durations are ${temperatureViolations.toFixed(2)} s and ${conversionViolations.toFixed(2)} s.`;
      const refin = data.numerical_validation["mppi-19744"];
      const fineReduction =
        100 *
        (1 -
          refin.space_refinement.mean_mixing_error /
            data.numerical_validation.unstirred.space_refinement
              .mean_mixing_error);
      $("stirring-refinement-note").textContent =
        `For the same MPPI controls, halving the timestep changes temperature by ${refin.time_refinement.temperature_rmse_K.toFixed(3)} K RMSE. Doubling the mesh changes temperature by ${refin.space_refinement.temperature_rmse_K.toFixed(2)} K RMSE and normalized variation by ${(100 * refin.space_refinement.mixing_error_relative_change).toFixed(1)}%. The saved controls reduce variation by ${fineReduction.toFixed(1)}% relative to no stirring on the finer mesh. These comparisons preserve the controller ordering while exposing numerical smoothing; they do not establish fully resolved physical mixing.`;
      const bench = data.advection_benchmark.results.filter(
        (r) => r.mesh === 64,
      );
      $("stirring-transport-note").textContent =
        `In a diffusion-free periodic transport test at 64 × 64, the new scheme retains ${(100 * bench.find((r) => r.order === 2).variance_retained).toFixed(1)}% of scalar variance after one circuit; first-order upwind retains ${(100 * bench.find((r) => r.order === 1).variance_retained).toFixed(1)}%. Lost variance in that test is numerical smoothing, not physical mixing.`;
      $("stirring-sensitivity").innerHTML = data.sensitivity
        .map(
          (r) =>
            `<tr><td>${r.candidates}</td><td>${r.seed}</td><td>${r.initial_objective.toFixed(4)} → ${r.final_objective.toFixed(4)}</td><td>${(r.iterations.reduce((s, i) => s + i.ess, 0) / r.iterations.length).toFixed(1)}</td><td>${r.runtime_s.toFixed(1)} s</td></tr>`,
        )
        .join("");
      await loadRuns();
      if (!$("stirring-samples").hidden && !snapshot) await loadSnapshot();
      $("stirring-status").textContent =
        "Saved numerical PDE responses · offline MPPI · identical deterministic feed in every comparison";
    })().catch((e) => {
      initialization = null;
      fail(e);
      throw e;
    });
    return initialization;
  }
  $("stirring-comparison").onchange = () => loadRuns().catch(fail);
  $("stirring-seed").onchange = () => loadRuns().catch(fail);
  $("stirring-display").onchange = draw;
  $("stirring-arrows").onchange = draw;
  $("stirring-candidate-field").onchange = draw;
  $("stirring-decision").onchange = () => loadSnapshot().catch(fail);
  $("stirring-time").oninput = () => {
    pause();
    draw();
  };
  $("stirring-future").oninput = () => {
    pause();
    draw();
  };
  $("stirring-speed").onchange = pause;
  $("stirring-play").onclick = () => play();
  $("stirring-future-play").onclick = () => play(true);
  $("stirring-reset").onclick = () => {
    pause();
    $("stirring-time").value = 0;
    draw();
  };
  $("stirring-future-reset").onclick = () => {
    pause();
    $("stirring-future").value = 0;
    draw();
  };
  document.querySelectorAll("[data-stirring-panel]").forEach((button) => {
    button.onclick = async () => {
      pause();
      document
        .querySelectorAll("[data-stirring-panel]")
        .forEach((b) => b.setAttribute("aria-selected", String(b === button)));
      const samples = button.dataset.stirringPanel === "samples";
      $("stirring-runs").hidden = samples;
      $("stirring-samples").hidden = !samples;
      try {
        if (samples && !snapshot) await loadSnapshot();
        draw();
      } catch (e) {
        fail(e);
      }
    };
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  return { init, draw, pause };
}
