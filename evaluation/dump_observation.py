"""Dump what the agent actually sees — each frame of the stacked observation
saved as a PNG so you can eyeball that the car is centred on the track.

This is a debug tool for the "does the agent have the right vision" question.
It builds the same vec-env stack training uses (DummyVecEnv → VecMonitor →
VecFrameStack → VecTransposeImage), steps a few times with a random policy,
and writes one PNG per stacked frame in the most recent observation.

Run from PitwallAgent/:
    python -m evaluation.dump_observation

Then look at `evaluation/results/obs_frame_*.png`. Each frame should show
the car (red rectangle) centred on the rendered top-down view, with the
dark grey track and grass clearly visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

from environment import PitwallRacingEnv
from tire_model import COMPOUNDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump the agent's stacked observation as PNG frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps-before-dump", type=int, default=80,
                   help="Take this many random steps before snapshotting the obs.")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--zoom", type=float, default=PitwallRacingEnv.DEFAULT_ZOOM)
    p.add_argument("--fixed-track-seed", type=int, default=None)
    p.add_argument("--compound", choices=list(COMPOUNDS), default="medium")
    p.add_argument("--no-tire-model", action="store_true")
    p.add_argument("--output-dir", type=Path,
                   default=Path("evaluation/results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _factory() -> PitwallRacingEnv:
        return PitwallRacingEnv(
            render_mode=None,
            compound=args.compound,
            use_tire_model=not args.no_tire_model,
            zoom=args.zoom,
            fixed_track_seed=args.fixed_track_seed,
        )

    monitored = VecMonitor(DummyVecEnv([_factory]))
    stacked = (
        VecFrameStack(monitored, n_stack=args.frame_stack)
        if args.frame_stack > 1
        else monitored
    )
    env = VecTransposeImage(stacked)

    env.seed(args.seed)
    obs = env.reset()

    for _ in range(args.steps_before_dump):
        action = np.array([env.action_space.sample()])
        obs, _, dones, _ = env.step(action)
        if bool(dones[0]):
            obs = env.reset()

    # obs["image"] from VecTransposeImage is (n_envs=1, channels, H, W) with
    # channels = n_stack * 3 (one RGB frame per stacked step).
    image_batch = obs["image"]  # (1, C, H, W) uint8
    state_batch = obs["state"]  # (1, state_dim)

    print(f"obs['image'].shape = {image_batch.shape}, dtype={image_batch.dtype}")
    print(f"obs['state'].shape = {state_batch.shape}, value={state_batch[0]}")

    image = image_batch[0]              # (C, H, W)
    n_total_channels = image.shape[0]
    n_frames = n_total_channels // 3
    print(f"interpreting as {n_frames} stacked RGB frames")

    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))
    if n_frames == 1:
        axes = [axes]

    for i in range(n_frames):
        # extract frame i: channels [3i:3i+3], reshape to HWC for imshow
        frame_chw = image[3 * i : 3 * (i + 1)]   # (3, H, W)
        frame_hwc = np.transpose(frame_chw, (1, 2, 0))  # (H, W, 3)
        axes[i].imshow(frame_hwc)
        axes[i].set_title(f"frame {i + 1}/{n_frames}\n(t = -{n_frames - 1 - i})")
        axes[i].axis("off")
        # Also save each frame individually for easier inspection
        single_path = args.output_dir / f"obs_frame_{i + 1}.png"
        plt.imsave(single_path, frame_hwc)

    grid_path = args.output_dir / "obs_grid.png"
    fig.tight_layout()
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    env.close()

    print()
    print(f"Saved combined view to {grid_path}")
    print(f"Saved individual frames to {args.output_dir}/obs_frame_*.png")
    print()
    print("What to check:")
    print("  - Car (red rectangle) should be centred near the bottom of each frame")
    print("  - Track is the dark grey strip; grass is light grey/green")
    print("  - Across the 4 frames you should see slight motion (later frames = more recent)")
    print("  - State vector format: [tire_wear, lap_count, compound_id] × frame_stack")


if __name__ == "__main__":
    main()
