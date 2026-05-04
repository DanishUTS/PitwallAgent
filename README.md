# Pitwall

An F1-inspired autonomous racing simulator built on top of Gymnasium's
`CarRacing-v3` environment. A reinforcement-learning agent is trained to drive
a track while a separate supervised tire-degradation model predicts wear.
The objective is to minimise total race time by trading off raw speed against
tire strategy and pit stops.

---

## Team

| Member | Slice |
| ------ | ----- |
| **Danish** | System architecture and the RL agent (`agents/train.py`) |
| **Hari**   | Simulation environment and custom mechanics — tire-wear state, pit-lane trigger zone (`environment/environment.py`) |
| **Ben**    | Supervised tire-degradation model and synthetic-data pipeline (`tire_model/tire_model.py`) |
| **Dennis** | Evaluation, metrics, plots, and documentation (`evaluation/evaluate.py`, this README) |

---

## The Two AI Components

### 1. Reinforcement Learning Agent — PPO

Proximal Policy Optimisation via [Stable Baselines3](https://stable-baselines3.readthedocs.io/).
The agent's observation is a dict combining the rendered 96×96 RGB frame with a
small state vector (current `tire_wear` and `lap_count`); actions are
continuous `(steer, throttle, brake)`. The custom reward function rewards
forward progress, penalises off-track driving, applies a per-step tire-wear
penalty, and applies a fixed time penalty whenever the agent enters the
pit-lane trigger zone (which resets tire wear). The agent therefore has to
learn *when* to pit, not just *how* to drive.

### 2. Supervised Tire Model — Regression

A `GradientBoostingRegressor` (scikit-learn) trained on synthetic tire
degradation samples. Inputs are `speed`, `cornering_load`, and `lap`; the
output is a predicted instantaneous wear rate. The trained model exposes
`predict_wear_rate(speed, cornering_load, lap)` so the RL state space can be
augmented with model-driven wear forecasts in a follow-up iteration.

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

All commands are run from the `PitwallAgent/` folder.

**Train the tire model** (generates synthetic data, fits the regressor, saves
artefacts to `models/checkpoints/tire_model.pkl` and
`data/synthetic/tire_data.npz`):

```bash
python -m tire_model.tire_model
```

**Train the PPO agent** (logs to `runs/`, checkpoints to
`models/checkpoints/`):

```bash
python -m agents.train --total-timesteps 200000
# quick smoke test:
python -m agents.train --total-timesteps 2000
```

**Evaluate a trained agent** (rollouts, summary table, plots in
`evaluation/results/`):

```bash
python -m evaluation.evaluate --episodes 5
```

**Watch tensorboard** (in another shell):

```bash
tensorboard --logdir runs/
```

---

## Folder Layout

```
PitwallAgent/
├── README.md
├── requirements.txt
├── .gitignore
├── environment/        # Hari   — custom CarRacing-v3 wrapper
│   └── environment.py
├── agents/             # Danish — PPO training loop
│   └── train.py
├── tire_model/         # Ben    — synthetic data + regression model
│   └── tire_model.py
├── evaluation/         # Dennis — metrics, plots, rollouts
│   └── evaluate.py
├── data/
│   └── synthetic/      # generated tire-degradation data (gitignored)
└── models/
    └── checkpoints/    # saved PPO + tire model artefacts (gitignored)
```
