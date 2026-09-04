import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_CHAPTERS = [
    "modeling-controlled-systems.md",
    "stochastic-dynamics-observation.md",
    "model-interfaces.md",
    "discrete-time-optimal-control.md",
    "discrete-time-pmp.md",
    "numerical-trajectory-optimization.md",
    "continuous-time-collocation.md",
    "receding-horizon-control.md",
    "mpc-variants-reliability.md",
    "parametric-controllers.md",
    "finite-horizon-dp.md",
    "stochastic-dp.md",
    "infinite-horizon-mdps.md",
    "regularized-dp.md",
    "weighted-residual-methods.md",
    "approximate-bellman-equations.md",
    "monte-carlo-bellman-estimation.md",
    "fitted-q-iteration.md",
    "amortized-action-optimization.md",
    "gradient-estimation.md",
    "regularized-policy-learning.md",
    "policy-gradients.md",
]

REMOVED_ROUTES = [
    "amortization.md",
    "collocation.md",
    "dp.md",
    "dynamics.md",
    "fqi.md",
    "montecarlo.md",
    "mpc.md",
    "pg.md",
    "projection.md",
    "smoothing.md",
    "trajectories.md",
]

BRIDGE_EXEMPT_HEADINGS = {
    "Computational Sources",
    "Exercises",
    "Learning Goals",
    "Prerequisites",
    "Self-checks",
    "Summary",
    "Summary and Forward Look",
    "Summary and Outlook",
}


def markdown_headings(text):
    """Return headings outside fenced code blocks as (level, title, offset)."""
    headings = []
    fence_char = None
    fence_length = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence_char is not None:
            closing = re.match(rf"{re.escape(fence_char)}{{{fence_length},}}\s*$", stripped)
            if closing:
                fence_char = None
                fence_length = 0
            offset += len(line)
            continue

        opening = re.match(r"(`{3,}|~{3,})", stripped)
        if opening:
            token = opening.group(1)
            fence_char = token[0]
            fence_length = len(token)
            offset += len(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            headings.append((len(heading.group(1)), heading.group(2), offset))
        offset += len(line)
    return headings


class BookStructureTests(unittest.TestCase):
    def test_toc_files_exist_and_are_unique(self):
        config = (ROOT / "myst.yml").read_text(encoding="utf-8")
        toc = config.split("\n  toc:\n", 1)[1].split("\nsite:\n", 1)[0]
        files = re.findall(r"^\s+- file:\s*(\S+)\s*$", toc, flags=re.MULTILINE)
        self.assertEqual(len(files), len(set(files)), "TOC files must be unique")
        self.assertEqual([file for file in files if file in CORE_CHAPTERS], CORE_CHAPTERS)
        for file in files:
            self.assertTrue((ROOT / file).is_file(), f"missing TOC file: {file}")

    def test_each_core_chapter_has_one_h1_and_question_bridges(self):
        for file in CORE_CHAPTERS:
            with self.subTest(file=file):
                text = (ROOT / file).read_text(encoding="utf-8")
                headings = markdown_headings(text)
                h1s = [heading for heading in headings if heading[0] == 1]
                self.assertEqual(len(h1s), 1, f"{file} must contain exactly one H1")

                first_h2 = next((heading for heading in headings if heading[0] == 2), None)
                chapter_end = first_h2[2] if first_h2 else len(text)
                self.assertIn("?", text[h1s[0][2]:chapter_end], f"{file} needs an opening question")

                h2s = [heading for heading in headings if heading[0] == 2]
                for index, heading in enumerate(h2s):
                    title = heading[1]
                    if title in BRIDGE_EXEMPT_HEADINGS:
                        continue
                    end = h2s[index + 1][2] if index + 1 < len(h2s) else len(text)
                    opening = text[heading[2]:end][:700]
                    self.assertIn("?", opening, f"{file}: {title} needs an opening question")

    def test_removed_routes_are_not_referenced(self):
        route_pattern = re.compile(
            r"(?<![-\w])(?:" + "|".join(re.escape(route) for route in REMOVED_ROUTES) + r")"
        )
        candidates = []
        for suffix in ("*.md", "*.yml", "*.html", "*.py", "*.mjs"):
            candidates.extend(ROOT.rglob(suffix))
        for path in candidates:
            if any(part in {".git", ".venv", "_build"} for part in path.parts):
                continue
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            match = route_pattern.search(text)
            self.assertIsNone(match, f"stale route {match.group(0)!r} in {path}" if match else "")


if __name__ == "__main__":
    unittest.main()
