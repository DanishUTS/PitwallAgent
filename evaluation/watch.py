"""Watch the car drive on the track in real time, in a pygame window.

Run from the PitwallAgent/ folder:
    # Watch the latest fully-trained agent
    python -m evaluation.watch --model-path models/checkpoints/ppo_pitwall_final.zip

    # Watch the best eval checkpoint mid-training (refreshes when re-run)
    python -m evaluation.watch --model-path models/checkpoints/best/best_model.zip

    # Watch a random policy — no trained model needed; useful as a sanity
    # check that the env renders and the pit zone fires when you stand still
    python -m evaluation.watch --random

Press Ctrl+C in the terminal (or close the pygame window) to stop early.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from environment import PitwallRacingEnv
from tire_model import COMPOUNDS

CARRACING_FPS: int = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the agent driving in a pygame window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Checkpoint to load. Required unless --random is set.",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Use a random policy (no model needed). Good for env smoke-tests.",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compound", choices=list(COMPOUNDS), default="medium")
    p.add_argument(
        "--no-tire-model",
        action="store_true",
        help="Use the legacy hand-tuned wear formula instead of the trained tire model.",
    )
    p.add_argument(
        "--max-laps",
        type=int,
        default=PitwallRacingEnv.DEFAULT_MAX_LAPS,
        help="Race length in laps before the episode truncates.",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=PitwallRacingEnv.DEFAULT_MAX_EPISODE_STEPS,
        help="Hard time limit per episode (env steps).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use the deterministic policy (default; pass --no-deterministic for sampled actions).",
    )
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.random and args.model_path is None:
        raise SystemExit(
            "Pass --model-path PATH to watch a trained agent, "
            "or --random to watch a random policy."
        )
    if args.model_path is not None and not args.model_path.exists():
        raise FileNotFoundError(f"No checkpoint at {args.model_path}")

    env = PitwallRacingEnv(
        render_mode="human",
        compound=args.compound,
        use_tire_model=not args.no_tire_model,
        max_laps=args.max_laps,
        max_episode_steps=args.max_episode_steps,
    )

    model: PPO | None = None
    if not args.random:
        print(f"Loading model: {args.model_path}")
        model = PPO.load(args.model_path)
        print(f"Policy: deterministic={args.deterministic}")
    else:
        print("Policy: uniform-random over the action space")

    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            pit_count = 0
            info: dict = {}
            done = False
            truncated = False
            while not (done or truncated):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=args.deterministic)
                else:
                    action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if info.get("pit_stop"):
                    pit_count += 1

            last_lap_s = (
                info.get("lap_time_steps", 0) / CARRACING_FPS
                if info.get("lap_time_steps") is not None
                else float("nan")
            )
            print(
                f"ep {ep + 1}/{args.episodes}: "
                f"reward={total_reward:7.1f}  steps={steps:5d}  "
                f"laps={info.get('lap_count', 0)}  "
                f"pits={pit_count}  "
                f"final_wear={info.get('tire_wear', 0.0):5.1f}  "
                f"last_lap_s={last_lap_s:5.2f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
