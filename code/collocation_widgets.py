"""Browser-native teaching widgets for the collocation chapter."""

from __future__ import annotations

import json
import re
import uuid


LINEAR_CONTROL_FALLBACK_ID = "fig-linear-control-trapezoid-fallback"


def _dom_id(prefix: str | None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "linear-control-area").strip("-")
    if not stem:
        stem = "linear-control-area"
    return f"{stem}-{uuid.uuid4().hex[:10]}"


def _safe_json(value: str) -> str:
    return (
        json.dumps(value, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_linear_control_area(
    *,
    widget_id: str | None = None,
    fallback_id: str = LINEAR_CONTROL_FALLBACK_ID,
) -> str:
    """Return an offline SVG widget linking integrated control to state change."""

    root = _dom_id(widget_id)
    title_id = f"{root}-title"
    description_id = f"{root}-description"
    clip_id = f"{root}-clip"

    template = r'''
<section id="__ROOT__" class="linear-control-area" tabindex="0" aria-labelledby="__TITLE_ID__" aria-describedby="__DESCRIPTION_ID__">
  <style>
    #__ROOT__ {
      --paper:#F6F7F4; --raised:#FFFFFF; --ink:#1B2430; --muted:#66727D; --rule:#CDD5D5;
      --blue:#0072B2; --sky:#56B4E9; --orange:#E69F00; --orange-ink:#815800; --accent:#2F6F8F;
      background:var(--paper); border:1px solid var(--rule); border-radius:9px; color:var(--ink);
      color-scheme:light; box-sizing:border-box; font:14px/1.4 "IBM Plex Sans",system-ui,sans-serif;
      margin-inline:auto; max-width:64rem; padding:clamp(.75rem,2vw,1.1rem); width:100%;
    }
    #__ROOT__[data-theme="dark"] {
      --paper:#121920; --raised:#1B2430; --ink:#EDF1F1; --muted:#A8B2B9; --rule:#34434D;
      --blue:#56B4E9; --sky:#56B4E9; --orange:#F0B94A; --orange-ink:#F0C76D; --accent:#72A8C2;
      color-scheme:dark;
    }
    #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { box-sizing:border-box; }
    #__ROOT__ [hidden] { display:none !important; }
    #__ROOT__:focus-visible { outline:3px solid color-mix(in srgb,var(--accent) 35%,transparent); outline-offset:3px; }
    #__ROOT__ h3 { font:500 clamp(1.2rem,3vw,1.5rem)/1.12 Newsreader,Georgia,serif; margin:0; }
    #__ROOT__ .lede { color:var(--muted); font-size:.88rem; margin:.28rem 0 .75rem; max-width:78ch; }
    #__ROOT__ .toolbar { align-items:center; border-block:1px solid var(--rule); display:flex; flex-wrap:wrap; gap:.55rem 1rem; padding:.65rem 0; }
    #__ROOT__ button { appearance:none; background:var(--raised); border:1px solid #98A5A9; border-radius:5px; color:var(--ink); font:inherit; font-size:.8rem; min-height:2rem; padding:.3rem .62rem; }
    #__ROOT__ button:hover, #__ROOT__ button:focus-visible { border-color:var(--accent); outline:2px solid color-mix(in srgb,var(--accent) 23%,transparent); outline-offset:1px; }
    #__ROOT__ .slider-group { align-items:center; display:grid; gap:.35rem; grid-template-columns:auto minmax(6rem,9rem) 3.2rem; }
    #__ROOT__ .slider-group label { font-size:.76rem; font-weight:600; white-space:nowrap; }
    #__ROOT__ input[type="range"] { accent-color:var(--accent); margin:0; width:100%; }
    #__ROOT__ output { font:600 .75rem "IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; text-align:right; }
    #__ROOT__ .stage { display:grid; gap:.8rem; grid-template-columns:minmax(0,1.45fr) minmax(15rem,.85fr); margin-top:.8rem; }
    #__ROOT__ .plot-card, #__ROOT__ .equation-card { background:color-mix(in srgb,var(--raised) 67%,var(--paper)); border:1px solid var(--rule); border-radius:7px; min-width:0; padding:.65rem; }
    #__ROOT__ .plot { display:block; height:auto; overflow:visible; width:100%; }
    #__ROOT__ .plot .axis, #__ROOT__ .plot .grid, #__ROOT__ .plot .reference { vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .axis { stroke:var(--ink); stroke-width:1.15; }
    #__ROOT__ .plot .grid { stroke:var(--rule); stroke-dasharray:2 5; stroke-width:1; }
    #__ROOT__ .plot .reference { stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1; }
    #__ROOT__ .plot .area-bg { fill:var(--sky); opacity:.10; }
    #__ROOT__ .plot .rectangle { fill:var(--sky); opacity:.42; stroke:var(--blue); stroke-width:1; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .triangle { fill:var(--orange); opacity:.48; stroke:var(--orange-ink); stroke-width:1; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .control-line { fill:none; stroke:var(--ink); stroke-linecap:round; stroke-width:2.4; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .endpoint { fill:var(--ink); stroke:var(--paper); stroke-width:2; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .playhead { stroke:var(--accent); stroke-width:1.5; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot .current { fill:var(--accent); stroke:var(--paper); stroke-width:2; vector-effect:non-scaling-stroke; }
    #__ROOT__ .plot text { fill:var(--muted); font:12px "IBM Plex Mono",ui-monospace,monospace; }
    #__ROOT__ .plot .endpoint-label { fill:var(--ink); font-weight:600; paint-order:stroke; stroke:var(--paper); stroke-width:3px; }
    #__ROOT__ .plot .area-label { font:600 12px "IBM Plex Sans",system-ui,sans-serif; paint-order:stroke; stroke:var(--paper); stroke-width:3px; }
    #__ROOT__ .plot .rectangle-label { fill:var(--blue); }
    #__ROOT__ .plot .triangle-label { fill:var(--orange-ink); }
    #__ROOT__ .equation-card { display:flex; flex-direction:column; justify-content:center; }
    #__ROOT__ .eyebrow { color:var(--muted); font-size:.66rem; font-weight:700; letter-spacing:.08em; margin-bottom:.25rem; text-transform:uppercase; }
    #__ROOT__ .equation { font:500 clamp(1rem,2.4vw,1.24rem)/1.35 Newsreader,Georgia,serif; margin:.15rem 0; }
    #__ROOT__ .equation sub { font-size:.68em; }
    #__ROOT__ .equation .value { color:var(--blue); font-weight:650; }
    #__ROOT__ .breakdown { border-block:1px solid var(--rule); display:grid; gap:.3rem; margin:.7rem 0; padding:.55rem 0; }
    #__ROOT__ .breakdown div { align-items:center; display:grid; font-size:.74rem; gap:.5rem; grid-template-columns:.7rem 1fr auto; }
    #__ROOT__ .swatch { border:1px solid currentColor; display:inline-block; height:.65rem; width:.65rem; }
    #__ROOT__ .swatch.rectangle-swatch { background:color-mix(in srgb,var(--sky) 42%,transparent); color:var(--blue); }
    #__ROOT__ .swatch.triangle-swatch { background:color-mix(in srgb,var(--orange) 48%,transparent); color:var(--orange-ink); }
    #__ROOT__ .state-card { margin-top:.55rem; }
    #__ROOT__ .state-head { align-items:center; color:var(--muted); display:flex; font-size:.7rem; justify-content:space-between; }
    #__ROOT__ .state-track { background:var(--rule); border-radius:999px; height:.35rem; margin:.7rem .3rem .3rem; position:relative; }
    #__ROOT__ .state-progress { background:var(--blue); border-radius:inherit; height:100%; left:0; position:absolute; top:0; width:100%; }
    #__ROOT__ .state-dot { background:var(--blue); border:2px solid var(--paper); border-radius:50%; box-shadow:0 0 0 1px var(--blue); height:.8rem; left:100%; position:absolute; top:50%; transform:translate(-50%,-50%); width:.8rem; }
    #__ROOT__ .status { color:var(--muted); font-size:.68rem; margin:.45rem 0 0; }
    #__ROOT__ .sr-only { clip:rect(0 0 0 0); clip-path:inset(50%); height:1px; overflow:hidden; position:absolute; white-space:nowrap; width:1px; }
    @media (max-width:760px) {
      #__ROOT__ .stage { grid-template-columns:1fr; }
      #__ROOT__ .slider-group { flex:1 1 14rem; }
    }
    @media (max-width:520px) {
      #__ROOT__ .toolbar { align-items:stretch; flex-direction:column; }
      #__ROOT__ .slider-group { width:100%; }
    }
    @media (prefers-reduced-motion:reduce) {
      #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { animation-duration:.001ms !important; scroll-behavior:auto !important; transition-duration:.001ms !important; }
    }
    @media print { #__ROOT__ .toolbar { display:none; } }
  </style>
  <header>
    <h3 id="__TITLE_ID__">Area becomes state change</h3>
    <p id="__DESCRIPTION_ID__" class="lede">Drag either endpoint, or play the accumulation. Because ẋ = u, every strip of shaded area adds directly to the state.</p>
  </header>
  <div class="toolbar" aria-label="Linear-control illustration controls">
    <button type="button" data-action="play" aria-keyshortcuts="Space">Play accumulation</button>
    <button type="button" data-action="reset" aria-keyshortcuts="Home">Reset</button>
    <div class="slider-group">
      <label for="__ROOT__-u0">U<sub>0</sub></label>
      <input id="__ROOT__-u0" data-input="u0" type="range" min="0.10" max="2.40" step="0.05" value="0.75">
      <output data-output="u0" for="__ROOT__-u0">0.75</output>
    </div>
    <div class="slider-group">
      <label for="__ROOT__-u1">U<sub>1</sub></label>
      <input id="__ROOT__-u1" data-input="u1" type="range" min="0.10" max="2.40" step="0.05" value="1.65">
      <output data-output="u1" for="__ROOT__-u1">1.65</output>
    </div>
    <div class="slider-group">
      <label for="__ROOT__-h">width h</label>
      <input id="__ROOT__-h" data-input="h" type="range" min="0.25" max="2.00" step="0.05" value="1.00">
      <output data-output="h" for="__ROOT__-h">1.00</output>
    </div>
  </div>
  <div class="stage">
    <div class="plot-card">
      <svg class="plot" viewBox="0 0 700 350" role="img" aria-label="Linear control with rectangle and triangle areas accumulated to the current playhead">
        <defs><clipPath id="__CLIP_ID__"><rect data-clip x="70" y="20" width="550" height="270"></rect></clipPath></defs>
        <line class="grid" x1="70" y1="80" x2="620" y2="80"></line>
        <line class="grid" x1="70" y1="185" x2="620" y2="185"></line>
        <line class="axis" x1="70" y1="290" x2="640" y2="290"></line>
        <line class="axis" x1="70" y1="305" x2="70" y2="20"></line>
        <path class="area-bg" data-area-bg></path>
        <g clip-path="url(#__CLIP_ID__)">
          <rect class="rectangle" data-rectangle></rect>
          <path class="triangle" data-triangle></path>
        </g>
        <line class="reference" data-reference></line>
        <path class="control-line" data-control-line></path>
        <circle class="endpoint" data-endpoint="u0" r="6"></circle>
        <circle class="endpoint" data-endpoint="u1" r="6"></circle>
        <text class="endpoint-label" data-label="u0"></text>
        <text class="endpoint-label" data-label="u1" text-anchor="end"></text>
        <text class="area-label rectangle-label" data-area-label="rectangle" text-anchor="middle">rectangle</text>
        <text class="area-label triangle-label" data-area-label="triangle" text-anchor="middle">triangle</text>
        <line class="playhead" data-playhead y1="24" y2="290"></line>
        <circle class="current" data-current r="5"></circle>
        <text data-time text-anchor="middle"></text>
        <text x="347" y="338" text-anchor="middle">time t</text>
        <text x="18" y="155" text-anchor="middle" transform="rotate(-90 18 155)">control u(t)</text>
        <text x="70" y="312" text-anchor="middle">0</text>
        <text data-h-label y="312" text-anchor="middle"></text>
      </svg>
    </div>
    <aside class="equation-card" aria-label="Live area calculation">
      <div class="eyebrow">At the playhead</div>
      <div class="equation">x(t) − X<sub>0</sub> = ∫<sub>0</sub><sup>t</sup> u(s) ds</div>
      <div class="equation">= U<sub>0</sub>t + <span>½(U<sub>1</sub>−U<sub>0</sub>)t²/h</span></div>
      <div class="equation">= <span class="value" data-partial-value>0.00</span></div>
      <div class="breakdown">
        <div><span class="swatch rectangle-swatch"></span><span>rectangle hU<sub>0</sub></span><output data-rectangle-value>0.75</output></div>
        <div><span class="swatch triangle-swatch"></span><span>triangle ½h(U<sub>1</sub>−U<sub>0</sub>)</span><output data-triangle-value>0.45</output></div>
      </div>
      <div class="eyebrow">At the right endpoint</div>
      <div class="equation">X<sub>1</sub> − X<sub>0</sub> = ½h(U<sub>0</sub>+U<sub>1</sub>)</div>
      <div class="equation">= <span class="value" data-final-value>1.20</span></div>
      <div class="state-card">
        <div class="state-head"><span>X<sub>0</sub></span><span data-state-label>X<sub>1</sub> = X<sub>0</sub> + 1.20</span></div>
        <div class="state-track"><span class="state-progress" data-state-progress></span><span class="state-dot" data-state-dot></span></div>
      </div>
      <p class="status" data-status>The full trapezoid is shown. Press Play accumulation to integrate from left to right.</p>
    </aside>
  </div>
  <p class="sr-only" data-live-status aria-live="polite"></p>
  <script>
  (() => {
    const root=document.getElementById("__ROOT__");
    const input=name => root.querySelector(`[data-input="${name}"]`);
    const output=name => root.querySelector(`[data-output="${name}"]`);
    const playButton=root.querySelector('[data-action="play"]');
    const reducedMotion=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const left=70, right=620, top=24, bottom=290, yMax=2.6;
    let progress=1, playing=false, startTime=null, animationFrame=null;
    const number=name => Number(input(name).value);
    const xScale=(fraction) => left+(right-left)*fraction;
    const yScale=(value) => bottom-(bottom-top)*value/yMax;
    const stop=() => {
      playing=false; startTime=null;
      if (animationFrame!==null) cancelAnimationFrame(animationFrame);
      animationFrame=null; playButton.textContent="Play accumulation";
    };
    const render=() => {
      const u0=number("u0"), u1=number("u1"), h=number("h");
      const x0=left, x1=right, y0=yScale(u0), y1=yScale(u1), base=bottom;
      const cursorX=xScale(progress), currentU=u0+progress*(u1-u0), cursorY=yScale(currentU);
      const t=progress*h;
      const partial=u0*t+0.5*(u1-u0)*t*t/h;
      const rectangle=h*u0, triangle=0.5*h*(u1-u0), finalArea=rectangle+triangle;
      root.querySelector("[data-area-bg]").setAttribute("d",`M ${x0} ${base} L ${x0} ${y0} L ${x1} ${y1} L ${x1} ${base} Z`);
      const rectangleNode=root.querySelector("[data-rectangle]");
      rectangleNode.setAttribute("x",x0); rectangleNode.setAttribute("y",y0);
      rectangleNode.setAttribute("width",right-left); rectangleNode.setAttribute("height",base-y0);
      root.querySelector("[data-triangle]").setAttribute("d",`M ${x0} ${y0} L ${x1} ${y0} L ${x1} ${y1} Z`);
      root.querySelector("[data-control-line]").setAttribute("d",`M ${x0} ${y0} L ${x1} ${y1}`);
      const reference=root.querySelector("[data-reference]");
      reference.setAttribute("x1",x0); reference.setAttribute("x2",x1); reference.setAttribute("y1",y0); reference.setAttribute("y2",y0);
      const clip=root.querySelector("[data-clip]"); clip.setAttribute("width",Math.max(0,cursorX-left));
      const playhead=root.querySelector("[data-playhead]"); playhead.setAttribute("x1",cursorX); playhead.setAttribute("x2",cursorX);
      const current=root.querySelector("[data-current]"); current.setAttribute("cx",cursorX); current.setAttribute("cy",cursorY);
      const endpoint0=root.querySelector('[data-endpoint="u0"]'), endpoint1=root.querySelector('[data-endpoint="u1"]');
      endpoint0.setAttribute("cx",x0); endpoint0.setAttribute("cy",y0); endpoint1.setAttribute("cx",x1); endpoint1.setAttribute("cy",y1);
      const label0=root.querySelector('[data-label="u0"]'), label1=root.querySelector('[data-label="u1"]');
      label0.setAttribute("x",x0+12); label0.setAttribute("y",Math.max(top+14,y0-10)); label0.textContent=`U₀ = ${u0.toFixed(2)}`;
      label1.setAttribute("x",x1-10); label1.setAttribute("y",Math.max(top+14,y1-10)); label1.textContent=`U₁ = ${u1.toFixed(2)}`;
      const rectangleLabel=root.querySelector('[data-area-label="rectangle"]');
      rectangleLabel.setAttribute("x",xScale(.29)); rectangleLabel.setAttribute("y",base-.46*(base-y0));
      const triangleLabel=root.querySelector('[data-area-label="triangle"]');
      triangleLabel.setAttribute("x",xScale(.73)); triangleLabel.setAttribute("y",Math.min(y0,y1)+.52*Math.abs(y1-y0));
      triangleLabel.style.opacity=Math.abs(u1-u0)<.24?"0":"1";
      const time=root.querySelector("[data-time]"); time.setAttribute("x",cursorX); time.setAttribute("y",Math.max(top+13,cursorY-14)); time.textContent=`t = ${t.toFixed(2)}`;
      root.querySelector("[data-h-label]").setAttribute("x",x1); root.querySelector("[data-h-label]").textContent=`h = ${h.toFixed(2)}`;
      output("u0").value=u0.toFixed(2); output("u1").value=u1.toFixed(2); output("h").value=h.toFixed(2);
      root.querySelector("[data-partial-value]").textContent=partial.toFixed(3);
      root.querySelector("[data-rectangle-value]").value=rectangle.toFixed(3);
      root.querySelector("[data-triangle-value]").value=triangle.toFixed(3);
      root.querySelector("[data-final-value]").textContent=finalArea.toFixed(3);
      const fraction=finalArea>0?Math.max(0,Math.min(1,partial/finalArea)):0;
      root.querySelector("[data-state-progress]").style.width=`${100*fraction}%`;
      root.querySelector("[data-state-dot]").style.left=`${100*fraction}%`;
      root.querySelector("[data-state-label]").innerHTML=`X<sub>1</sub> = X<sub>0</sub> + ${finalArea.toFixed(3)}`;
      const description=progress>=.999
        ? `The full trapezoid has area ${finalArea.toFixed(3)}, so X1 minus X0 equals ${finalArea.toFixed(3)}.`
        : `At time ${t.toFixed(2)}, the accumulated area and state change are ${partial.toFixed(3)}.`;
      root.querySelector("[data-status]").textContent=description;
      root.querySelector("[data-live-status]").textContent=description;
    };
    const tick=(timestamp) => {
      if (!playing) return;
      if (startTime===null) startTime=timestamp;
      progress=Math.min(1,(timestamp-startTime)/2600);
      render();
      if (progress>=1) { stop(); return; }
      animationFrame=requestAnimationFrame(tick);
    };
    const play=() => {
      if (playing) { stop(); return; }
      if (reducedMotion) { progress=1; render(); return; }
      progress=0; playing=true; startTime=null; playButton.textContent="Pause";
      render(); animationFrame=requestAnimationFrame(tick);
    };
    playButton.addEventListener("click",play);
    root.querySelector('[data-action="reset"]').addEventListener("click",() => {
      stop(); input("u0").value="0.75"; input("u1").value="1.65"; input("h").value="1.00"; progress=1; render();
    });
    root.querySelectorAll('input[type="range"]').forEach(slider => slider.addEventListener("input",() => { stop(); progress=1; render(); }));
    root.addEventListener("keydown",event => {
      if (event.target!==root) return;
      if (event.key===" ") { event.preventDefault(); play(); }
      else if (event.key==="Home") { event.preventDefault(); stop(); progress=0; render(); }
      else if (event.key==="End") { event.preventDefault(); stop(); progress=1; render(); }
    });
    const applyTheme=() => {
      let dark=false;
      try {
        const themeRoot=window.parent&&window.parent!==window?window.parent.document.documentElement:document.documentElement;
        const declared=String(themeRoot.dataset.theme||themeRoot.getAttribute("data-mode")||"").toLowerCase();
        dark=declared==="dark"||themeRoot.classList.contains("dark")||getComputedStyle(themeRoot).colorScheme==="dark";
        if (!dark&&window.matchMedia) dark=window.matchMedia("(prefers-color-scheme: dark)").matches;
      } catch (_) { dark=window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches; }
      root.dataset.theme=dark?"dark":"light";
    };
    applyTheme();
    try {
      const themeRoot=window.parent&&window.parent!==window?window.parent.document.documentElement:document.documentElement;
      if (typeof MutationObserver!=="undefined") new MutationObserver(applyTheme).observe(themeRoot,{attributes:true,attributeFilter:["class","data-theme","data-mode","style"]});
    } catch (_) {}
    if (window.matchMedia) { const query=window.matchMedia("(prefers-color-scheme: dark)"); if (typeof query.addEventListener==="function") query.addEventListener("change",applyTheme); }
    render();

    const fallbackId=__FALLBACK_JSON__;
    const hideFallback=doc => {
      if (!doc) return false; const fallback=doc.getElementById(fallbackId); if (!fallback) return false;
      fallback.hidden=true; fallback.setAttribute("aria-hidden","true"); return true;
    };
    hideFallback(document);
    try { if (window.parent&&window.parent!==window) hideFallback(window.parent.document); } catch (_) {}
    if (typeof MutationObserver!=="undefined") {
      const observer=new MutationObserver(() => { if (hideFallback(document)) observer.disconnect(); });
      observer.observe(document.documentElement,{childList:true,subtree:true});
    }
  })();
  </script>
</section>
'''

    return (
        template.replace("__ROOT__", root)
        .replace("__TITLE_ID__", title_id)
        .replace("__DESCRIPTION_ID__", description_id)
        .replace("__CLIP_ID__", clip_id)
        .replace("__FALLBACK_JSON__", _safe_json(fallback_id))
    )


__all__ = ["LINEAR_CONTROL_FALLBACK_ID", "render_linear_control_area"]
