# Pitwall

An F1-inspired autonomous racing simulator built on top of Gymnasium's
`CarRacing-v3` environment. A reinforcement-learning agent is trained to drive
a multi-lap race while a separate supervised tire-degradation model predicts
wear in real time. The objective is to minimise total race time by trading off
raw speed against tire strategy and pit stops. 

---

## Team

| Member | Slice |
| ------ | ----- |
| **Danish** | System architecture and the RL agent (`agents/train.py`) |
| **Hari**   | Simulation environment and custom mechanics — tire wear, pit-lane trigger zone, lap detection (`environment/environment.py`) |
| **Ben**    | Supervised tire-degradation model and synthetic-data pipeline (`tire_model/tire_model.py`) |
| **Dennis** | Evaluation, metrics, plots, and documentation (`evaluation/evaluate.py`, this README) |

---

## The Two AI Components

### 1. Reinforcement Learning Agent — PPO

Proximal Policy Optimisation via [Stable Baselines3](https://stable-baselines3.readthedocs.io/).
The agent's observation is a `Dict` combining the rendered 96×96 RGB frame with
a 3-element state vector `[tire_wear, lap_count, compound_id]`; actions are
continuous `(steer, throttle, brake)`. The reward function combines:

- **Base CarRacing-v3 reward** — forward progress along the track, off-track
  penalty, per-step time penalty.
- **Tire-wear penalty** — `−WEAR_REWARD_PENALTY × tire_wear` per step, so worn
  tires cost reward continuously.
- **Pit-stop penalty** — fixed `−PIT_TIME_PENALTY` whenever the car triggers
  a pit (slow + within the pit zone + worn enough). The pit resets tire wear
  to 0 in exchange for the time loss.

The agent has to learn *when* to pit (and which compound to start on), not just
*how* to drive.

### 2. Supervised Tire-Degradation Model

A `GradientBoostingRegressor` (scikit-learn) trained on synthetic samples of
the form `(speed, cornering_load, lap, current_wear, compound) → wear_rate`.
Three compounds are modelled (soft / medium / hard) with stylised F1 physics:
per-compound base wear, nonlinear speed contribution (~ speed¹·⁴), linear
cornering-load term, lap-driven heat-soak, and a "cliff" amplifier that ramps
degradation in the last 50 % of tire life.

The trained model is consumed inside `PitwallRacingEnv.step()` — the env reads
car speed, lateral acceleration, current wear, and chosen compound, then
queries `predict_wear_rate(...)` for the per-step wear delta. A
`--no-tire-model` switch falls back to a hand-tuned formula for baseline
comparisons.

---

## Installation

Requires **Python 3.10+**.

```bash
# From the PitwallAgent/ folder
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows note**: `gymnasium[box2d]` builds Box2D from source, which needs
> SWIG. If `pip install` fails on `box2d-py`, install SWIG first:
>
> ```powershell
> pip install swig
> pip install -r requirements.txt
> ```

---

## Running The Code

All commands run from the `PitwallAgent/` folder.

### 1. Train the tire model (do this first)

```bash
python -m tire_model.tire_model
```

Generates 12k synthetic samples, fits the regressor, saves the model to
`models/checkpoints/tire_model.pkl`, the dataset to
`data/synthetic/tire_data.npz`, and a diagnostic plot to
`evaluation/results/tire_compound_curves.png`. Prints overall and per-compound
held-out R² and MAE.

### 2. Train the PPO agent

```bash
# Full run (defaults: 4 parallel envs, EvalCallback every 10k steps)
python -m agents.train --total-timesteps 200000

# Quick smoke test (a few minutes on CPU)
python -m agents.train --total-timesteps 5000 --n-envs 2

# Baseline (legacy hand-tuned wear, no supervised model in env)
python -m agents.train --no-tire-model --checkpoint-dir models/baseline

# Resume a partial run
python -m agents.train --resume models/checkpoints/ppo_pitwall_50000_steps.zip
```

Key flags (run `python -m agents.train --help` for the full list):

| Flag | Purpose |
| --- | --- |
| `--total-timesteps` | Training length in env steps |
| `--n-envs` | Parallel envs (`SubprocVecEnv` if > 1, default 4) |
| `--compound {soft,medium,hard}` | Tire compound for training |
| `--no-tire-model` | Use the legacy hand-tuned wear formula |
| `--eval-freq` | Held-out evaluation frequency in total steps |
| `--resume PATH` | Continue training from a checkpoint |
| `--device {auto,cpu,cuda}` | Hardware backend |

Outputs:
- `models/checkpoints/ppo_pitwall_*_steps.zip` — periodic snapshots
- `models/checkpoints/best/best_model.zip` — best held-out evaluation
- `models/checkpoints/ppo_pitwall_final.zip` — model at the end of training
- `runs/` — tensorboard logs (`tensorboard --logdir runs/` to view)

### 3. Evaluate

```bash
# Single-model evaluation (5 episodes, deterministic policy)
python -m evaluation.evaluate --episodes 5

# Two-model A/B comparison on identical seeds
python -m evaluation.evaluate \
    --model-path models/checkpoints/best/best_model.zip \
    --baseline-path models/baseline/ppo_pitwall_final.zip \
    --episodes 5
```

Produces, in `evaluation/results/`:
- Markdown summary table (reward, length, laps, pit count, final wear, lap
  time in seconds)
- Pit-strategy comparison table (mean lap time bucketed by pit count)
- `tire_wear.png` — wear over time with vertical lap-boundary markers
- `per_lap_wear.png` — bar chart of tire wear at end of each lap
- `racing_line.png` — episode 0's `(x, y)` trajectory over the actual track
  with the pit zone and pit-stop markers overlaid
- `reward_hist.png` — reward distribution
- When `--baseline-path` is given, the same plot set with `baseline_` prefix

---

## Folder Layout

```
PitwallAgent/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── environment/        # Hari   — CarRacing-v3 wrapper, tire wear, pit zone, lap detection
│   └── environment.py
├── agents/             # Danish — PPO training loop with EvalCallback + SubprocVecEnv
│   └── train.py
├── tire_model/         # Ben    — synthetic data + GradientBoostingRegressor with compounds
│   └── tire_model.py
├── evaluation/         # Dennis — metrics, plots, rollouts
│   └── evaluate.py
├── data/
│   └── synthetic/      # generated tire-degradation data (gitignored)
└── models/
    └── checkpoints/    # saved PPO + tire model artefacts (gitignored)
```
