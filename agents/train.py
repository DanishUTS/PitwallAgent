"""PPO training entry point.

Run from the PitwallAgent/ folder:
    python -m agents.train --total-timesteps 200000

A short smoke run that exercises the whole loop:
    python -m agents.train --total-timesteps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import PitwallRacingEnv


def make_env():
    return PitwallRacingEnv(render_mode=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on the Pitwall racing environment.")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        seed=args.seed,
        tensorboard_log=str(args.log_dir),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // env.num_envs, 1),
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo_pitwall",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = args.checkpoint_dir / "ppo_pitwall_final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
