import json
import argparse
from pathlib import Path

import numpy as np
import scipy.optimize as opt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ballistic_like_trajectory(num_points: int = 100,
                              width: int = 800,
                              height: int = 400,
                              margin: int = 50):
    """Replicates the smooth curve used in the JS demo.

    The exact same formula appears in _static/multiple-shooting-demo.html.
    Keeping the same constants ensures that when we load these points in the
    front-end they will perfectly overlap with the SVG axes already drawn on
    screen.
    """
    t_grid = np.linspace(0.0, 1.0, num_points)
    xs = margin + t_grid * (width - 2 * margin)

    ys = (
        height
        - margin
        - 50
        - 200 * np.sin(np.pi * t_grid) * (1 - t_grid)
        - 30 * np.sin(4 * np.pi * t_grid) * np.exp(-2 * t_grid)
    )

    return xs.tolist(), ys.tolist()


def segment_points(xs, ys, num_segments: int):
    """Splits a global trajectory into K segments based on the x-coordinates."""
    xs = np.array(xs)
    ys = np.array(ys)

    width = xs.max() - xs.min()
    margin = xs.min()
    seg_length = width / num_segments

    segments = []
    for k in range(num_segments):
        start_x = margin + k * seg_length
        end_x = margin + (k + 1) * seg_length + 1e-6  # include right boundary
        mask = (xs >= start_x) & (xs <= end_x)
        seg_pts = [dict(x=float(x), y=float(y)) for x, y in zip(xs[mask], ys[mask])]
        segments.append(seg_pts)

    return segments


def add_random_offsets(segments, max_offset: float = 60.0, seed: int | None = None):
    """Returns a deep copy of *segments* where each segment k>0 is shifted by a
    random vertical offset. The first segment is left untouched so that the
    defects are visible between segments.
    """
    rng = np.random.default_rng(seed)
    perturbed = []
    for k, seg in enumerate(segments):
        offset = 0.0 if k == 0 else rng.uniform(-max_offset, max_offset)
        perturbed.append([
            dict(x=p["x"], y=p["y"] + offset, originalY=p["y"]) for p in seg
        ])
    return perturbed


def optimize_offsets(true_segments, initial_offsets):
    """Run BFGS on the vertical offsets so that segment endpoints connect.

    Returns (opt_offsets, history) where *history* is a list of offset arrays
    captured at every solver iteration (including the initial guess and the
    final solution).
    """
    K = len(true_segments)

    # Degenerate case: K == 1 (single shooting) – no defects to optimise.
    if K == 1 or len(initial_offsets) <= 1:
        # Simply return the initial offsets (normally all zeros) and a history
        # containing that single state so that the front-end animation logic
        # still works.
        return initial_offsets, [initial_offsets.copy()]

    # Build arrays of the first + last y-coordinates for convenience
    start_ys = np.array([seg[0]["y"] for seg in true_segments])
    end_ys = np.array([seg[-1]["y"] for seg in true_segments])

    def objective(vars_):
        offsets = np.concatenate(([0.0], vars_))  # offset[0] fixed to 0
        defects = (end_ys[:-1] + offsets[:-1]) - (start_ys[1:] + offsets[1:])
        return 0.5 * np.sum(defects ** 2)

    # Store history for visualisation
    history = []

    def callback(vars_):
        offsets = np.concatenate(([0.0], vars_))
        history.append(offsets.copy())

    # Kick-start history with initial guess
    history.append(initial_offsets.copy())

    res = opt.minimize(
        objective,
        x0=initial_offsets[1:],
        method="BFGS",
        callback=callback,
        options={"gtol": 1e-8, "maxiter": 100}
    )

    opt_offsets = np.concatenate(([0.0], res.x))
    history.append(opt_offsets.copy())
    return opt_offsets, history

# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multiple-shooting "
                                                 "trajectory data for the demo.")
    parser.add_argument("--k_values", "-K", type=str, default="1,2,3,4,5,6,7,8",
                        help="Comma-separated list of segment counts to precompute.")
    parser.add_argument("--num_points", type=int, default=101,
                        help="Total number of points along the trajectory.")
    parser.add_argument("--output", "-o", default="_static/multiple_shooting_data.json",
                        help="Path where the JSON file will be written.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the initial guess offsets.")

    args = parser.parse_args()

    # Global true path (independent of K) -----------------------------
    xs, ys = ballistic_like_trajectory(args.num_points)
    full_true_segments = segment_points(xs, ys, args.num_points - 1)  # one per point, but we keep continuous path separately

    cases = {}
    for k_str in args.k_values.split(','):
        k = int(k_str.strip())
        if k <= 0:
            continue

        true_segments = segment_points(xs, ys, k)
        initial_guess_segments = add_random_offsets(true_segments, seed=args.seed + k)

        initial_offsets = np.array([seg[0]["y"] - seg[0]["originalY"] for seg in initial_guess_segments])
        opt_offsets, history = optimize_offsets(true_segments, initial_offsets)

        optimization_steps = []
        for offsets in history:
            iter_segments = []
            for seg, off in zip(true_segments, offsets):
                iter_segments.append([
                    {"x": p["x"], "y": p["y"] + off, "originalY": p["y"]} for p in seg
                ])
            optimization_steps.append(iter_segments)

        cases[str(k)] = {
            "initial_guess": initial_guess_segments,
            "optimization_steps": optimization_steps,
            "solver_success": bool(opt_offsets is not None)
        }

    data = {
        "continuous_trajectory": segment_points(xs, ys, 1)[0],  # flattened true path for reference
        "cases": cases
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    # Use Jupyter Book's gluing feature to display output instead of print
    try:
        from myst_nb import glue
        glue("multiple_shooting_output", f"✓ Generated trajectory data for K = {args.k_values} segments\n"
                                        f"✓ Output written to: {output_path.name}\n"
                                        f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB", display=False)
    except ImportError:
        # Fallback for when not running in Jupyter Book context
        print(f"⟹ Wrote aggregated trajectory data to {output_path.resolve()}")


if __name__ == "__main__":
    main() 