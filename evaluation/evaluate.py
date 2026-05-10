"""Rollout-based evaluation: metrics and plots.

Run from the PitwallAgent/ folder after training:
    python -m evaluation.evaluate --episodes 5
    python -m evaluation.evaluate --baseline-path models/checkpoints/baseline.zip

Produces the HD-relevant artefacts:
  * markdown summary table (reward, length, laps, pit count, final wear, lap time)
  * pit-strategy comparison table (mean lap time bucketed by pit count)
  * tire wear over time, with vertical markers at each lap completion
  * per-lap tire wear bar chart
  * racing line (episode 0) over the actual track, with the pit zone and
    any pit-stop markers overlaid
  * reward histogram across episodes

When `--baseline-path` is provided, both models are evaluated on identical
seeds (so the same tracks) and a second summary + plot set is saved with
`baseline_` prefix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from environment import PitwallRacingEnv
from tire_model import COMPOUNDS

CARRACING_FPS: int = 50  # CarRacing-v3 default; used for step → seconds.


@dataclass
class EpisodeResult:
    seed: int
    total_reward: float = 0.0
    n_steps: int = 0
    pit_count: int = 0
    final_wear: float = 0.0
    final_lap_count: int = 0
    wear_trace: list[float] = field(default_factory=list)
    speed_trace: list[float] = field(default_factory=list)
    lap_times_steps: list[int] = field(default_factory=list)
    lap_end_wears: list[float] = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    pit_zone_xy: tuple[float, float] | None = None
    pit_step_indices: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO Pitwall agent.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/checkpoints/ppo_pitwall_final.zip"),
    )
    p.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Optional second checkpoint to compare on identical seeds.",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
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
        help="Race length in laps before the episode truncates. Should match training.",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=PitwallRacingEnv.DEFAULT_MAX_EPISODE_STEPS,
        help="Hard time limit per episode (env steps). Should match training.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def run_episode(env: PitwallRacingEnv, model: PPO, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    result = EpisodeResult(seed=seed, pit_zone_xy=info.get("pit_zone_xy"))

    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        result.total_reward += float(reward)
        result.n_steps += 1
        result.wear_trace.append(float(info.get("tire_wear", 0.0)))

        car = getattr(env.unwrapped, "car", None)
        if car is not None:
            v = car.hull.linearVelocity
            result.speed_trace.append(float(np.hypot(v[0], v[1])))
            x, y = car.hull.position
            result.trajectory.append((float(x), float(y)))
        else:
            result.speed_trace.append(0.0)

        if info.get("pit_stop"):
            result.pit_count += 1
            result.pit_step_indices.append(result.n_steps)

        if info.get("lap_completed"):
            lt = info.get("lap_time_steps")
            if lt is not None:
                result.lap_times_steps.append(int(lt))
                result.lap_end_wears.append(float(info.get("tire_wear", 0.0)))

    result.final_wear = result.wear_trace[-1] if result.wear_trace else 0.0
    result.final_lap_count = len(result.lap_times_steps)
    return result


def steps_to_seconds(steps: float) -> float:
    return float(steps) / CARRACING_FPS


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def print_summary_table(results: list[EpisodeResult], label: str) -> None:
    rewards = np.array([r.total_reward for r in results])
    lengths = np.array([r.n_steps for r in results])
    pits = np.array([r.pit_count for r in results])
    final_wears = np.array([r.final_wear for r in results])
    final_laps = np.array([r.final_lap_count for r in results])

    all_lap_steps = [s for r in results for s in r.lap_times_steps]
    if all_lap_steps:
        lap_secs = np.array(all_lap_steps) / CARRACING_FPS
        lap_mean, lap_std = lap_secs.mean(), lap_secs.std()
        lap_str_mean = f"{lap_mean:7.2f}"
        lap_str_std = f"{lap_std:7.2f}"
    else:
        lap_str_mean = "    n/a"
        lap_str_std = "    n/a"

    print(f"\n=== Summary: {label} ({len(results)} episodes) ===")
    print("| metric            | mean    | std     |")
    print("| ----------------- | ------- | ------- |")
    print(f"| episode reward    | {rewards.mean():7.2f} | {rewards.std():7.2f} |")
    print(f"| episode length    | {lengths.mean():7.1f} | {lengths.std():7.1f} |")
    print(f"| laps completed    | {final_laps.mean():7.2f} | {final_laps.std():7.2f} |")
    print(f"| pit stops         | {pits.mean():7.2f} | {pits.std():7.2f} |")
    print(f"| final tire wear % | {final_wears.mean():7.2f} | {final_wears.std():7.2f} |")
    print(f"| lap time (s)      | {lap_str_mean} | {lap_str_std} |")


def print_pit_strategy_table(results: list[EpisodeResult]) -> None:
    """Bucket episodes by pit-stop count and report mean lap time per bucket."""
    buckets: dict[int, list[float]] = {}
    bucket_episode_counts: dict[int, int] = {}
    for r in results:
        bucket_episode_counts[r.pit_count] = bucket_episode_counts.get(r.pit_count, 0) + 1
        if r.lap_times_steps:
            secs = [s / CARRACING_FPS for s in r.lap_times_steps]
            buckets.setdefault(r.pit_count, []).extend(secs)

    if not bucket_episode_counts:
        print("\n(no completed episodes; can't compute pit-strategy table)")
        return

    print("\n=== Pit-strategy comparison ===")
    print("| pit stops | episodes | mean lap (s) | std lap (s) |")
    print("| --------- | -------- | ------------ | ----------- |")
    for pit_count in sorted(bucket_episode_counts):
        n_eps = bucket_episode_counts[pit_count]
        if pit_count in buckets:
            vals = np.array(buckets[pit_count])
            mean_str = f"{vals.mean():12.2f}"
            std_str = f"{vals.std():11.2f}"
        else:
            mean_str = "         n/a"
            std_str = "        n/a"
        print(f"| {pit_count:>9d} | {n_eps:>8d} | {mean_str} | {std_str} |")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_wear_traces(
    results: list[EpisodeResult], env: PitwallRacingEnv, output_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(results):
        ax.plot(r.wear_trace, label=f"ep {i + 1}", alpha=0.8)

    # Vertical lap markers from episode 0 (other episodes have different tracks)
    if results and results[0].lap_times_steps:
        offset = 0
        for k, lap_steps in enumerate(results[0].lap_times_steps):
            offset += lap_steps
            ax.axvline(
                offset,
                color="black",
                linestyle=":",
                alpha=0.4,
                label="lap boundary (ep 1)" if k == 0 else None,
            )

    ax.set_xlabel("env step")
    ax.set_ylabel("tire wear (%)")
    ax.set_title(f"Tire wear over time — compound: {env.compound}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_per_lap_wear(
    results: list[EpisodeResult], output_path: Path
) -> Path | None:
    if not any(r.lap_end_wears for r in results):
        print(f"  [skip] {output_path.name}: no lap completed in any episode")
        return None
    n_eps = len(results)
    max_lap = max(len(r.lap_end_wears) for r in results)
    if max_lap == 0:
        print(f"  [skip] {output_path.name}: no lap completed in any episode")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.8 / max(n_eps, 1)
    for i, r in enumerate(results):
        if not r.lap_end_wears:
            continue
        x = (
            np.arange(1, len(r.lap_end_wears) + 1)
            + (i - (n_eps - 1) / 2) * bar_width
        )
        ax.bar(x, r.lap_end_wears, bar_width, label=f"ep {i + 1}")

    ax.set_xlabel("lap")
    ax.set_ylabel("tire wear at end of lap (%)")
    ax.set_title("Per-lap tire wear")
    ax.set_xticks(np.arange(1, max_lap + 1))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_racing_line(
    results: list[EpisodeResult], env: PitwallRacingEnv, output_path: Path
) -> Path | None:
    """Episode 0 only — different seeds yield different tracks, so overlaying
    multiple episodes' trajectories on one chart would be misleading."""
    if not results or not results[0].trajectory:
        print(f"  [skip] {output_path.name}: episode 0 has no trajectory")
        return None
    r = results[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    track = getattr(env.unwrapped, "track", None)
    if track:
        # Last two elements of each tile are (x, y); tuple length varies
        # across gym/gymnasium versions, so use negative indexing.
        tx = [t[-2] for t in track] + [track[0][-2]]
        ty = [t[-1] for t in track] + [track[0][-1]]
        ax.plot(tx, ty, color="lightgrey", linewidth=2, label="track")

    xs = [p[0] for p in r.trajectory]
    ys = [p[1] for p in r.trajectory]
    ax.plot(xs, ys, color="C0", alpha=0.7, linewidth=0.9, label="racing line (ep 1)")
    ax.scatter([xs[0]], [ys[0]], marker="o", color="green", s=60, zorder=4, label="start")

    if r.pit_zone_xy is not None:
        circle = plt.Circle(
            r.pit_zone_xy,
            env.PIT_ZONE_RADIUS,
            color="orange",
            alpha=0.3,
            label=f"pit zone (r={env.PIT_ZONE_RADIUS:.0f})",
        )
        ax.add_patch(circle)

    if r.pit_step_indices:
        pit_xs = [r.trajectory[idx - 1][0] for idx in r.pit_step_indices if idx - 1 < len(r.trajectory)]
        pit_ys = [r.trajectory[idx - 1][1] for idx in r.pit_step_indices if idx - 1 < len(r.trajectory)]
        ax.scatter(pit_xs, pit_ys, marker="x", color="red", s=80, zorder=5, label="pit stop")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Racing line — seed {r.seed}, compound {env.compound}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_reward_hist(
    results: list[EpisodeResult], output_path: Path, n_bins_min: int = 5
) -> Path:
    rewards = np.array([r.total_reward for r in results])
    fig, ax = plt.subplots(figsize=(6, 4))
    n_bins = max(n_bins_min, len(results) // 2)
    ax.hist(rewards, bins=n_bins)
    ax.set_xlabel("episode reward")
    ax.set_ylabel("count")
    ax.set_title("Reward distribution across evaluation episodes")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate_model(
    env: PitwallRacingEnv,
    model: PPO,
    n_episodes: int,
    base_seed: int,
    label: str,
) -> list[EpisodeResult]:
    print(f"\nRolling out {label} for {n_episodes} episodes...")
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        seed = base_seed + i
        r = run_episode(env, model, seed=seed)
        results.append(r)
        last_lap_s = (
            r.lap_times_steps[-1] / CARRACING_FPS if r.lap_times_steps else float("nan")
        )
        print(
            f"  ep {i + 1}/{n_episodes}: "
            f"reward={r.total_reward:7.1f}  steps={r.n_steps:5d}  "
            f"laps={r.final_lap_count}  pits={r.pit_count}  "
            f"final_wear={r.final_wear:5.1f}  last_lap_s={last_lap_s:5.2f}"
        )
    return results


def write_artefacts(
    results: list[EpisodeResult],
    env: PitwallRacingEnv,
    output_dir: Path,
    prefix: str = "",
) -> None:
    plot_wear_traces(results, env, output_dir / f"{prefix}tire_wear.png")
    plot_per_lap_wear(results, output_dir / f"{prefix}per_lap_wear.png")
    plot_racing_line(results, env, output_dir / f"{prefix}racing_line.png")
    plot_reward_hist(results, output_dir / f"{prefix}reward_hist.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.model_path}. Train one with `python -m agents.train`."
        )
    if args.baseline_path is not None and not args.baseline_path.exists():
        raise FileNotFoundError(f"No baseline checkpoint at {args.baseline_path}.")

    env = PitwallRacingEnv(
        render_mode=None,
        compound=args.compound,
        use_tire_model=not args.no_tire_model,
        max_laps=args.max_laps,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"Loading primary model: {args.model_path}")
    model = PPO.load(args.model_path)
    primary = evaluate_model(env, model, args.episodes, args.seed, "primary")
    print_summary_table(primary, label="primary")
    print_pit_strategy_table(primary)
    write_artefacts(primary, env, args.output_dir)

    if args.baseline_path is not None:
        print(f"\nLoading baseline model: {args.baseline_path}")
        baseline_model = PPO.load(args.baseline_path)
        baseline = evaluate_model(
            env, baseline_model, args.episodes, args.seed, "baseline"
        )
        print_summary_table(baseline, label="baseline")
        print_pit_strategy_table(baseline)
        write_artefacts(baseline, env, args.output_dir, prefix="baseline_")

    env.close()
    print(f"\nSaved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
