"""Checks for browser-native collocation teaching widgets."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from collocation_widgets import (  # noqa: E402
    LINEAR_CONTROL_FALLBACK_ID,
    render_linear_control_area,
)


def test_linear_control_widget_is_accessible_responsive_and_offline() -> None:
    rendered = render_linear_control_area(widget_id="lecture-area")

    assert 'id="lecture-area-' in rendered
    assert 'tabindex="0"' in rendered
    assert 'aria-labelledby="' in rendered
    assert 'aria-describedby="' in rendered
    assert 'aria-label="Linear-control illustration controls"' in rendered
    assert 'role="img"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'aria-keyshortcuts="Space"' in rendered
    assert 'event.key==="Home"' in rendered
    assert 'event.key==="End"' in rendered
    assert "prefers-reduced-motion: reduce" in rendered
    assert "@media (max-width:760px)" in rendered
    assert '[data-theme="dark"]' in rendered
    assert "MutationObserver(applyTheme)" in rendered
    assert "autoplay" not in rendered.lower()
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "WebSocket" not in rendered
    assert "<script src=" not in rendered


def test_linear_control_widget_exposes_the_geometry_and_live_identity() -> None:
    rendered = render_linear_control_area()

    assert 'data-input="u0"' in rendered
    assert 'data-input="u1"' in rendered
    assert 'data-input="h"' in rendered
    assert 'data-action="play"' in rendered
    assert 'data-action="reset"' in rendered
    assert 'data-rectangle' in rendered
    assert 'data-triangle' in rendered
    assert 'data-playhead' in rendered
    assert 'data-state-dot' in rendered
    assert "const partial=u0*t+0.5*(u1-u0)*t*t/h" in rendered
    assert "const rectangle=h*u0" in rendered
    assert "triangle=0.5*h*(u1-u0)" in rendered
    assert "finalArea=rectangle+triangle" in rendered
    assert "X<sub>1</sub> − X<sub>0</sub> = ½h(U<sub>0</sub>+U<sub>1</sub>)" in rendered


def test_linear_control_widget_ids_are_unique_and_fallback_is_linked() -> None:
    first = render_linear_control_area(widget_id="area")
    second = render_linear_control_area(widget_id="area")
    first_root = re.search(r'<section id="([^"]+)"', first)
    second_root = re.search(r'<section id="([^"]+)"', second)

    assert first_root is not None
    assert second_root is not None
    assert first_root.group(1) != second_root.group(1)
    assert LINEAR_CONTROL_FALLBACK_ID in first
    assert 'fallback.hidden=true' in first
    assert 'fallback.setAttribute("aria-hidden","true")' in first


def test_static_linear_control_fallback_is_present_and_described() -> None:
    fallback = ROOT / "_static" / "collocation" / "linear-control-area.svg"
    source = fallback.read_text(encoding="utf-8")

    assert '<svg xmlns="http://www.w3.org/2000/svg"' in source
    assert 'role="img"' in source
    assert '<title id="title">' in source
    assert '<desc id="description">' in source
    assert "rectangle: hU₀" in source
    assert "triangle: ½h(U₁−U₀)" in source
    assert "= ½h(U₀ + U₁)" in source
