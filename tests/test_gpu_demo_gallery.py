"""Structural checks for the standalone GPU demo gallery."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "interactive" / "gpu-demos.html"


def test_gpu_gallery_is_single_file_and_contains_every_view() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "GPU inference control demos" in page
    assert 'src="http' not in page
    assert 'href="http' not in page
    for view in ("modeling", "open_loop", "mpc", "scheduling", "fqi"):
        assert f'data-demo-tab="{view}"' in page
        assert f'data-demo-panel="{view}"' in page
        assert f'id="gpu-demo-{view.replace("_", "-")}"' in page


def test_gpu_gallery_keeps_replays_distinct() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert page.count('class="inference-replay"') == 5
    assert page.count('type="application/json"') == 5
    assert page.count('aria-label="Replay controls"') == 5
