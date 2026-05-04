"""Rollout-based evaluation: metrics and plots.

Run from the PitwallAgent/ folder after training:
    python -m evaluation.evaluate --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from environment import PitwallRacingEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO Pitwall agent.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/checkpoints/ppo_pitwall_final.zip"),
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_episode(env, model, seed: int):
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0.0
    wear_trace: list[float] = []
    steps = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        wear_trace.append(float(info.get("tire_wear", 0.0)))
        steps += 1
    return total_reward, steps, wear_trace


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.model_path}. Train one with `python -m agents.train`."
        )

    env = PitwallRacingEnv(render_mode=None)
    model = PPO.load(args.model_path)

    rewards: list[float] = []
    lengths: list[int] = []
    wear_traces: list[list[float]] = []

    for i in range(args.episodes):
        r, n, wear = run_episode(env, model, seed=args.seed + i)
        rewards.append(r)
        lengths.append(n)
        wear_traces.append(wear)
        print(f"  episode {i + 1}/{args.episodes}: reward={r:.1f}  steps={n}  final_wear={wear[-1]:.1f}")

    env.close()

    rewards_a = np.array(rewards)
    lengths_a = np.array(lengths)
    final_wear = np.array([w[-1] if w else 0.0 for w in wear_traces])

    print()
    print("| metric            | mean    | std     |")
    print("| ----------------- | ------- | ------- |")
    print(f"| episode reward    | {rewards_a.mean():7.2f} | {rewards_a.std():7.2f} |")
    print(f"| episode length    | {lengths_a.mean():7.1f} | {lengths_a.std():7.1f} |")
    print(f"| final tire wear % | {final_wear.mean():7.2f} | {final_wear.std():7.2f} |")

    # Plot 1: tire wear over time, one line per episode
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, wear in enumerate(wear_traces):
        ax.plot(wear, label=f"ep {i + 1}")
    ax.set_xlabel("step")
    ax.set_ylabel("tire wear (%)")
    ax.set_title("Tire wear over time")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    wear_plot = args.output_dir / "tire_wear.png"
    fig.savefig(wear_plot)
    plt.close(fig)

    # Plot 2: reward histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rewards_a, bins=max(5, args.episodes // 2))
    ax.set_xlabel("episode reward")
    ax.set_ylabel("count")
    ax.set_title("Reward distribution across evaluation episodes")
    fig.tight_layout()
    reward_plot = args.output_dir / "reward_hist.png"
    fig.savefig(reward_plot)
    plt.close(fig)

    print(f"\nSaved plots to {wear_plot} and {reward_plot}")


if __name__ == "__main__":
    main()
