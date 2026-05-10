"""Quick standalone preview of the track + pit zone.

Doesn't need a trained agent or a trained tire model; just resets the env
to generate a track and draws what the agent would see.

Run from the PitwallAgent/ folder:
    # single seed
    python -m evaluation.visualize_track --seed 42

    # 2x2 grid of different seeds (nice for sanity-checking the pit-zone
    # radius works across track shapes)
    python -m evaluation.visualize_track --n-seeds 4

    # show the env for real (opens a pygame window)
    python -m evaluation.visualize_track --seed 42 --human
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment import PitwallRacingEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview the track and pit zone without training anything.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="If >1, draws this many tracks in a grid starting from --seed.")
    p.add_argument("--output", type=Path,
                   default=Path("evaluation/results/track_preview.png"))
    p.add_argument("--human", action="store_true",
                   help="Also open a pygame window so you can see the rendered camera view.")
    p.add_argument("--zoom", type=float, default=PitwallRacingEnv.DEFAULT_ZOOM,
                   help="Camera zoom (only matters with --human; matplotlib plot is unaffected).")
    return p.parse_args()


def plot_one(ax: plt.Axes, env: PitwallRacingEnv, seed: int) -> None:
    env.reset(seed=seed)
    track = env.unwrapped.track
    if not track:
        ax.text(0.5, 0.5, f"seed {seed}\n(track not built)",
                transform=ax.transAxes, ha="center", va="center")
        return

    # Last two elements of each tile are (x, y); tuple length varies across
    # gym/gymnasium versions, so use negative indexing.
    tx = [t[-2] for t in track] + [track[0][-2]]
    ty = [t[-1] for t in track] + [track[0][-1]]
    ax.plot(tx, ty, color="lightgrey", linewidth=3, zorder=1, label="track centerline")

    spawn_x, spawn_y = float(track[0][-2]), float(track[0][-1])
    next_x, next_y = float(track[1][-2]), float(track[1][-1])

    # Direction-of-travel arrow from tile 0 → tile 1
    ax.annotate(
        "",
        xy=(next_x, next_y),
        xytext=(spawn_x, spawn_y),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        zorder=4,
    )

    # Pit-zone disc
    circle = plt.Circle(
        (spawn_x, spawn_y),
        env.PIT_ZONE_RADIUS,
        color="orange",
        alpha=0.35,
        zorder=2,
        label=f"pit zone (r={env.PIT_ZONE_RADIUS:.0f})",
    )
    ax.add_patch(circle)

    # Spawn marker on top of everything
    ax.scatter(
        [spawn_x], [spawn_y],
        marker="o", color="green", s=120, zorder=5,
        label="spawn / track[0]",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"seed {seed} — {len(track)} tiles")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # use_tire_model=False so we don't require the regressor .pkl just to peek
    # at track geometry. If you've trained the tire model, flipping this to
    # True won't change the picture — wear isn't relevant for a single reset.
    render_mode = "human" if args.human else None
    env = PitwallRacingEnv(
        render_mode=render_mode,
        use_tire_model=False,
        zoom=args.zoom,
    )

    n = max(1, args.n_seeds)
    if n == 1:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_one(ax, env, args.seed)
    else:
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for i in range(n):
            plot_one(axes_flat[i], env, args.seed + i)
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    plt.close(fig)
    env.close()
    print(f"Saved track preview to {args.output}")


if __name__ == "__main__":
    main()
