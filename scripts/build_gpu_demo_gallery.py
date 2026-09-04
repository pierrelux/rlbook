#!/usr/bin/env python3
"""Build the standalone gallery of recorded GPU inference-control demos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from inference_replay import render_serving_replay  # noqa: E402


ARTIFACT = ROOT / "artifacts" / "inference_serving" / "textbook_results.json"
DEFAULT_OUTPUT = ROOT / "interactive" / "gpu-demos.html"

DEMOS = (
    (
        "modeling",
        "System state",
        "What must the controller observe about requests, work, and the GPU?",
    ),
    (
        "open_loop",
        "Open-loop planning",
        "What clock schedule works when the request trace is known in advance?",
    ),
    (
        "mpc",
        "Receding-horizon control",
        "How does repeated replanning respond as the measured backlog changes?",
    ),
    (
        "scheduling",
        "Exact scheduling",
        "Can prefill and decode decisions be computed across the reduced queue state?",
    ),
    (
        "fqi",
        "Fitted Q scheduling",
        "What changes when the scheduling policy must be learned from sampled transitions?",
    ),
)


def _tab(view: str, label: str, index: int) -> str:
    selected = "true" if index == 0 else "false"
    tab_index = "0" if index == 0 else "-1"
    return f'''<button class="demo-tab" id="tab-{view}" type="button" role="tab"
        aria-selected="{selected}" aria-controls="panel-{view}"
        tabindex="{tab_index}" data-demo-tab="{view}">
      <span class="tab-index">{index + 1:02d}</span>
      <span>{label}</span>
    </button>'''


def _panel(view: str, question: str, index: int) -> str:
    fragment = render_serving_replay(
        ARTIFACT,
        view=view,
        replay_id=f"gpu-demo-{view.replace('_', '-')}",
        maximum_frames=360,
        stable_id=True,
    )
    hidden = "" if index == 0 else " hidden"
    return f'''<section class="demo-panel" id="panel-{view}" role="tabpanel"
      aria-labelledby="tab-{view}" data-demo-panel="{view}"{hidden}>
      <p class="demo-question">{question}</p>
      {fragment}
    </section>'''


def build_gallery() -> str:
    tabs = "\n".join(_tab(view, label, index) for index, (view, label, _) in enumerate(DEMOS))
    panels = "\n".join(_panel(view, question, index) for index, (view, _, question) in enumerate(DEMOS))
    return f'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <meta name="description" content="Five recorded GPU inference-control demos: modeling, open-loop planning, MPC, dynamic programming, and fitted Q iteration.">
  <title>GPU Inference Control Demos</title>
  <style>
    :root {{
      --page: #071016;
      --surface: #0d1820;
      --surface-raised: #12212b;
      --ink: #eef6f8;
      --muted: #a8bbc3;
      --line: #2a3c46;
      --accent: #62d5ff;
      --accent-soft: #153a49;
      --warning: #f0bd54;
      --shadow: 0 24px 70px rgb(0 0 0 / 0.28);
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    :root[data-theme="light"] {{
      --page: #edf3f4;
      --surface: #f8fbfb;
      --surface-raised: #ffffff;
      --ink: #12232c;
      --muted: #526872;
      --line: #c6d3d8;
      --accent: #006f99;
      --accent-soft: #dceff5;
      --warning: #805d10;
      --shadow: 0 22px 60px rgb(26 55 68 / 0.13);
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      min-width: 19rem;
      background: var(--page);
      color: var(--ink);
      font-size: 1rem;
      line-height: 1.55;
    }}
    button, a {{ font: inherit; }}
    button:focus-visible, a:focus-visible {{
      outline: 3px solid color-mix(in srgb, var(--accent) 58%, transparent);
      outline-offset: 3px;
    }}
    .page-shell {{ width: min(100% - 2rem, 76rem); margin: 0 auto; padding: 2rem 0 4rem; }}
    .masthead {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 1.5rem;
      align-items: start;
      border-top: 3px solid var(--accent);
      padding-top: 1.4rem;
    }}
    .eyebrow {{
      margin: 0 0 .55rem;
      color: var(--accent);
      font-size: .79rem;
      font-weight: 750;
      letter-spacing: .13em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 0; max-width: 15ch; font-size: clamp(2.1rem, 5vw, 4.25rem); line-height: .98; letter-spacing: -.045em; }}
    .lede {{ max-width: 46rem; margin: 1.15rem 0 0; color: var(--muted); font-size: 1.05rem; }}
    .actions {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .55rem; }}
    .action {{
      display: inline-flex;
      min-height: 2.65rem;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: .35rem;
      padding: .55rem .8rem;
      color: var(--ink);
      background: var(--surface);
      text-decoration: none;
      cursor: pointer;
    }}
    .action:hover {{ border-color: var(--accent); }}
    .facts {{ display: flex; flex-wrap: wrap; gap: .45rem; margin: 1.4rem 0 0; padding: 0; list-style: none; }}
    .facts li {{ border: 1px solid var(--line); border-radius: 999px; padding: .32rem .65rem; color: var(--muted); font-size: .84rem; }}
    .caveat {{
      margin: 1.35rem 0 0;
      border-left: 3px solid var(--warning);
      padding: .15rem 0 .15rem .8rem;
      color: var(--muted);
      font-size: .9rem;
    }}
    .workspace {{ display: grid; grid-template-columns: 14.5rem minmax(0, 1fr); gap: 1rem; margin-top: 2.2rem; align-items: start; }}
    .demo-tabs {{
      position: sticky;
      top: 1rem;
      display: grid;
      gap: .4rem;
      border: 1px solid var(--line);
      border-radius: .55rem;
      padding: .5rem;
      background: var(--surface);
    }}
    .demo-tab {{
      display: grid;
      grid-template-columns: 2.1rem 1fr;
      gap: .5rem;
      align-items: center;
      width: 100%;
      min-height: 3.25rem;
      border: 1px solid transparent;
      border-radius: .35rem;
      padding: .45rem .55rem;
      color: var(--muted);
      background: transparent;
      text-align: left;
      cursor: pointer;
    }}
    .demo-tab:hover {{ color: var(--ink); background: var(--surface-raised); }}
    .demo-tab[aria-selected="true"] {{ color: var(--ink); border-color: var(--accent); background: var(--accent-soft); }}
    .tab-index {{ color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; }}
    .demo-panel {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: .6rem;
      padding: clamp(.75rem, 2vw, 1.25rem);
      background: var(--surface-raised);
      box-shadow: var(--shadow);
    }}
    .demo-panel[hidden] {{ display: none; }}
    .demo-question {{ margin: 0 0 1rem; color: var(--ink); font-size: clamp(1.05rem, 2vw, 1.3rem); font-weight: 650; }}
    .demo-panel .inference-replay {{ max-width: none !important; border-radius: .35rem; }}
    .page-footer {{ margin-top: 1.6rem; color: var(--muted); font-size: .82rem; text-align: right; }}
    @media (max-width: 800px) {{
      .masthead {{ grid-template-columns: 1fr; }}
      .actions {{ justify-content: flex-start; }}
      .workspace {{ grid-template-columns: 1fr; }}
      .demo-tabs {{ position: static; grid-template-columns: repeat(5, minmax(9rem, 1fr)); overflow-x: auto; }}
    }}
    @media (max-width: 520px) {{
      .page-shell {{ width: min(100% - 1rem, 76rem); padding-top: 1rem; }}
      h1 {{ font-size: 2.5rem; }}
      .action {{ flex: 1 1 auto; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
    @media print {{
      :root {{ --page: #fff; --surface: #fff; --surface-raised: #fff; --ink: #111; --muted: #444; --line: #bbb; --shadow: none; }}
      .actions, .demo-tabs {{ display: none; }}
      .workspace {{ display: block; }}
      .demo-panel, .demo-panel[hidden] {{ display: block; break-inside: avoid; margin-bottom: 1rem; box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <main class="page-shell">
    <header class="masthead">
      <div>
        <p class="eyebrow">RL &amp; Control · GPU case study</p>
        <h1>GPU inference control demos</h1>
        <p class="lede">Five browser-native replays connect one inference-serving system to open-loop planning, feedback control, dynamic programming, and fitted Q iteration. Every frame is embedded in this file, so the page works without a GPU or backend.</p>
        <ul class="facts" aria-label="Demo provenance">
          <li>NVIDIA L4 profile</li>
          <li>Qwen2.5-7B-Instruct</li>
          <li>Recorded trajectories</li>
          <li>Five control views</li>
        </ul>
        <p class="caveat">Measured clock and power data calibrate the serving model. Controller trajectories are model-based replays, and the one-state thermal fit remains weak; the demos do not establish hardware safety guarantees.</p>
      </div>
      <div class="actions" aria-label="Page actions">
        <button class="action" type="button" data-theme-toggle>Light theme</button>
        <a class="action" href="gpu-demos.html" download="gpu-inference-control-demos.html">Download HTML</a>
      </div>
    </header>

    <div class="workspace">
      <nav class="demo-tabs" role="tablist" aria-label="GPU inference demos">
        {tabs}
      </nav>
      <div class="demo-panels">
        {panels}
      </div>
    </div>
    <footer class="page-footer">Recorded inference-serving artifact · Pierre-Luc Bacon</footer>
  </main>
  <script>
    (() => {{
      "use strict";
      const tabs = [...document.querySelectorAll("[data-demo-tab]")];
      const panels = [...document.querySelectorAll("[data-demo-panel]")];
      const views = new Set(tabs.map((tab) => tab.dataset.demoTab));
      const activate = (view, updateHash = true) => {{
        if (!views.has(view)) view = tabs[0].dataset.demoTab;
        tabs.forEach((tab) => {{
          const active = tab.dataset.demoTab === view;
          tab.setAttribute("aria-selected", String(active));
          tab.tabIndex = active ? 0 : -1;
        }});
        panels.forEach((panel) => {{ panel.hidden = panel.dataset.demoPanel !== view; }});
        if (updateHash) history.replaceState(null, "", "#" + view.replaceAll("_", "-"));
      }};
      tabs.forEach((tab, index) => {{
        tab.addEventListener("click", () => activate(tab.dataset.demoTab));
        tab.addEventListener("keydown", (event) => {{
          if (!["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          let next = index;
          if (event.key === "Home") next = 0;
          else if (event.key === "End") next = tabs.length - 1;
          else if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
          else next = (index - 1 + tabs.length) % tabs.length;
          tabs[next].focus();
          activate(tabs[next].dataset.demoTab);
        }});
      }});
      const hashView = location.hash.slice(1).replaceAll("-", "_");
      activate(hashView || tabs[0].dataset.demoTab, false);

      const themeButton = document.querySelector("[data-theme-toggle]");
      const setTheme = (theme) => {{
        document.documentElement.dataset.theme = theme;
        themeButton.textContent = theme === "dark" ? "Light theme" : "Dark theme";
        try {{ localStorage.setItem("gpu-demo-theme", theme); }} catch (error) {{}}
      }};
      let savedTheme = "";
      try {{ savedTheme = localStorage.getItem("gpu-demo-theme") || ""; }} catch (error) {{}}
      setTheme(savedTheme === "light" || savedTheme === "dark" ? savedTheme : "dark");
      themeButton.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
    }})();
  </script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_gallery(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
