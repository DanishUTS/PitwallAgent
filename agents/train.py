"""PPO training entry point.

Run from the PitwallAgent/ folder:
    python -m agents.train --total-timesteps 200000

Smoke test (a few minutes on a CPU):
    python -m agents.train --total-timesteps 5000 --n-envs 2

Train a baseline (no supervised tire model in the env):
    python -m agents.train --no-tire-model --checkpoint-dir models/baseline

Resume an interrupted run:
    python -m agents.train --resume models/checkpoints/ppo_pitwall_50000_steps.zip

Tensorboard:
    tensorboard --logdir runs/

Default hyperparameters are taken from the SB3 zoo's CarRacing PPO recipe
(n_steps=512, batch_size=128, n_epochs=10) rather than SB3's general-purpose
defaults — they're better-tuned for image observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
)

from agents.policy import PitwallFeaturesExtractor
from environment import PitwallRacingEnv
from tire_model import COMPOUNDS


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
def make_env_factory(
    compound: str,
    use_tire_model: bool,
    max_laps: int,
    max_episode_steps: int,
    zoom: float,
    fixed_track_seed: int | None,
    waypoint_bonus: float,
    enforce_track_bounds: bool,
):
    """Return a zero-arg factory that builds a fresh PitwallRacingEnv.

    Used by both DummyVecEnv (in-process) and SubprocVecEnv (subprocesses).
    The closure captures simple types only (str, bool, int, float, None) —
    all pickle cleanly via cloudpickle, which SubprocVecEnv uses.
    """

    def _make() -> PitwallRacingEnv:
        return PitwallRacingEnv(
            render_mode=None,
            compound=compound,
            use_tire_model=use_tire_model,
            max_laps=max_laps,
            max_episode_steps=max_episode_steps,
            zoom=zoom,
            fixed_track_seed=fixed_track_seed,
            waypoint_bonus=waypoint_bonus,
            enforce_track_bounds=enforce_track_bounds,
        )

    return _make


def build_train_env(args: argparse.Namespace) -> VecEnv:
    factory = make_env_factory(
        compound=args.compound,
        use_tire_model=not args.no_tire_model,
        max_laps=args.max_laps,
        max_episode_steps=args.max_episode_steps,
        zoom=args.zoom,
        fixed_track_seed=args.fixed_track_seed,
        waypoint_bonus=args.waypoint_bonus,
        enforce_track_bounds=args.enforce_track_bounds,
    )
    if args.n_envs <= 1:
        # Single-env: DummyVecEnv keeps everything in-process. Faster setup,
        # easier debugging.
        vec: VecEnv = DummyVecEnv([factory])
    else:
        # SubprocVecEnv runs envs in worker processes. CarRacing is rendering-
        # bound, so this scales near-linearly up to ~4 workers on most CPUs.
        vec = SubprocVecEnv([factory for _ in range(args.n_envs)])
    # VecMonitor wraps the VecEnv so per-episode reward / length get logged
    # to tensorboard automatically alongside PPO's own metrics.
    monitored = VecMonitor(vec)
    # VecFrameStack stacks the last `n_stack` observations along the channel
    # axis so the policy can perceive motion / angular velocity. Works on
    # Dict obs by stacking each subspace independently.
    if args.frame_stack > 1:
        return VecFrameStack(monitored, n_stack=args.frame_stack)
    return monitored


def build_eval_env(args: argparse.Namespace) -> VecEnv:
    """Single-env DummyVecEnv used by EvalCallback for held-out rollouts."""
    factory = make_env_factory(
        compound=args.compound,
        use_tire_model=not args.no_tire_model,
        max_laps=args.max_laps,
        max_episode_steps=args.max_episode_steps,
        zoom=args.zoom,
        fixed_track_seed=args.fixed_track_seed,
        waypoint_bonus=args.waypoint_bonus,
        enforce_track_bounds=args.enforce_track_bounds,
    )
    monitored = VecMonitor(DummyVecEnv([factory]))
    if args.frame_stack > 1:
        return VecFrameStack(monitored, n_stack=args.frame_stack)
    return monitored


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(
    args: argparse.Namespace, eval_env: VecEnv, n_envs: int
) -> CallbackList:
    """Combine periodic checkpointing with held-out evaluation.

    SB3 callbacks count per-worker steps, so to get user-facing 'total-step'
    semantics we divide both frequencies by n_envs.
    """
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // n_envs, 1),
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo_pitwall",
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=max(args.eval_freq // n_envs, 1),
        log_path=str(args.log_dir / "eval"),
        best_model_save_path=str(args.checkpoint_dir / "best"),
        deterministic=True,
        render=False,
        verbose=1,
    )
    return CallbackList([checkpoint_cb, eval_cb])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_or_resume_model(args: argparse.Namespace, train_env: VecEnv) -> PPO:
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        print(f"Resuming from {args.resume}")
        # SB3's PPO.load deserialises hyperparameters from the .zip, so by
        # default the CLI values would be silently ignored on resume.
        # `custom_objects` lets us replace specific attributes during
        # deserialisation. We override only the three commonly-tuned
        # hyperparameters; architectural ones (n_steps, batch_size,
        # network size) stay baked into the checkpoint to avoid optimiser
        # / buffer-shape mismatches.
        custom_objects = {
            "ent_coef": args.ent_coef,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
        }
        print(
            f"  overriding from CLI: ent_coef={args.ent_coef}, "
            f"learning_rate={args.learning_rate}, clip_range={args.clip_range}"
        )
        return PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=str(args.log_dir),
            device=args.device,
            print_system_info=False,
            custom_objects=custom_objects,
        )
    # Default: use SB3's built-in NatureCNN-based CombinedExtractor for Dict
    # obs (smaller, well-tested for CarRacing). Pass --use-custom-cnn to opt
    # into the larger PitwallFeaturesExtractor (more capacity, slower to
    # train, and tends to overfit our shaped rewards).
    policy_kwargs: dict | None = None
    if args.use_custom_cnn:
        policy_kwargs = dict(
            features_extractor_class=PitwallFeaturesExtractor,
            features_extractor_kwargs=dict(image_features_dim=args.features_dim),
        )
    return PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
        tensorboard_log=str(args.log_dir),
        device=args.device,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO on the Pitwall racing environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_train = p.add_argument_group("training")
    g_train.add_argument("--total-timesteps", type=int, default=200_000)
    g_train.add_argument("--n-envs", type=int, default=4)
    g_train.add_argument("--seed", type=int, default=0)
    g_train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    g_train.add_argument("--resume", type=Path, default=None,
                         help="Resume from this checkpoint. ent_coef / learning_rate / "
                              "clip_range from the CLI override the checkpoint's saved "
                              "values; all other hyperparameters are loaded from the "
                              "checkpoint to keep optimiser state consistent.")
    g_train.add_argument("--no-progress-bar", action="store_true")

    g_io = p.add_argument_group("output paths")
    g_io.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    g_io.add_argument("--log-dir", type=Path, default=Path("runs"))
    g_io.add_argument("--save-freq", type=int, default=10_000,
                      help="Checkpoint frequency (in total env steps).")
    g_io.add_argument("--eval-freq", type=int, default=10_000,
                      help="Evaluation frequency (in total env steps).")
    g_io.add_argument("--n-eval-episodes", type=int, default=5)

    g_env = p.add_argument_group("environment")
    g_env.add_argument("--compound", choices=list(COMPOUNDS), default="medium")
    g_env.add_argument("--no-tire-model", action="store_true",
                       help="Use the legacy hand-tuned wear formula instead of the trained tire model.")
    g_env.add_argument("--max-laps", type=int, default=PitwallRacingEnv.DEFAULT_MAX_LAPS,
                       help="Race length in laps before the episode truncates.")
    g_env.add_argument("--max-episode-steps", type=int,
                       default=PitwallRacingEnv.DEFAULT_MAX_EPISODE_STEPS,
                       help="Hard time limit per episode (env steps).")
    g_env.add_argument("--zoom", type=float, default=PitwallRacingEnv.DEFAULT_ZOOM,
                       help="Camera zoom. Lower = wider view (more lookahead). "
                            "gymnasium's CarRacing default is 2.7; we default to 1.5.")
    g_env.add_argument("--fixed-track-seed", type=int, default=None,
                       help="If set, every reset uses this seed so all envs run the same "
                            "track every episode. Use for phase-1 curriculum (learn one "
                            "track), then resume without the flag to generalise.")
    g_env.add_argument("--waypoint-bonus", type=float,
                       default=PitwallRacingEnv.WAYPOINT_BONUS_DEFAULT,
                       help="Reward bonus when the car drives over one of the 8 waypoints "
                            "(once per lap each). 0 = disabled (the current default).")
    g_env.add_argument("--enforce-track-bounds", action="store_true",
                       help="Enable the hard off-track termination barrier "
                            "(OFFTRACK_TERMINATION_DISTANCE). Off by default. "
                            "The soft per-step off-track penalty is always active.")

    g_ppo = p.add_argument_group("PPO hyperparameters (defaults from SB3 zoo CarRacing recipe)")
    g_ppo.add_argument("--learning-rate", type=float, default=3e-4)
    g_ppo.add_argument("--n-steps", type=int, default=512,
                       help="Steps per env per rollout. Total batch = n_steps * n_envs.")
    g_ppo.add_argument("--batch-size", type=int, default=128)
    g_ppo.add_argument("--n-epochs", type=int, default=10)
    g_ppo.add_argument("--gamma", type=float, default=0.99)
    g_ppo.add_argument("--gae-lambda", type=float, default=0.95)
    g_ppo.add_argument("--clip-range", type=float, default=0.2)
    g_ppo.add_argument("--ent-coef", type=float, default=0.0)
    g_ppo.add_argument("--vf-coef", type=float, default=0.5)
    g_ppo.add_argument("--frame-stack", type=int, default=4,
                       help="Stack the last N frames so the policy can perceive motion. "
                            "1 = no stacking. 4 is the CarRacing/Atari convention.")
    g_ppo.add_argument("--use-custom-cnn", action="store_true",
                       help="Use the larger PitwallFeaturesExtractor (64-128-128 channels, "
                            "~6M conv params) instead of SB3's default NatureCNN "
                            "(32-64-64, ~1.7M params). Default off — the smaller default "
                            "trains faster and is the standard CarRacing config.")
    g_ppo.add_argument("--features-dim", type=int, default=512,
                       help="Image-branch feature dim — only used when --use-custom-cnn "
                            "is set. State-branch is fixed at 64.")

    return p.parse_args()


def print_config(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Pitwall PPO training")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:>20s}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "best").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "eval").mkdir(parents=True, exist_ok=True)

    print_config(args)

    train_env = build_train_env(args)
    eval_env = build_eval_env(args)

    try:
        model = make_or_resume_model(args, train_env)
        callbacks = build_callbacks(args, eval_env, n_envs=args.n_envs)

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=not args.no_progress_bar,
            # When resuming, keep the global step counter contiguous so
            # tensorboard plots stitch together.
            reset_num_timesteps=args.resume is None,
        )

        final_path = args.checkpoint_dir / "ppo_pitwall_final.zip"
        model.save(final_path)

        print()
        print("=" * 60)
        print("Training complete")
        print("=" * 60)
        print(f"  final model         : {final_path}")
        print(f"  best eval model     : {args.checkpoint_dir / 'best' / 'best_model.zip'}")
        print(f"  tensorboard logs    : {args.log_dir}")
        print(f"  eval npz logs       : {args.log_dir / 'eval' / 'evaluations.npz'}")
        print()
        print("Next:")
        print(f"  tensorboard --logdir {args.log_dir}")
        print(f"  python -m evaluation.evaluate --model-path {final_path}")
    finally:
        # Ensure subprocess workers shut down cleanly even if learn() raises.
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
