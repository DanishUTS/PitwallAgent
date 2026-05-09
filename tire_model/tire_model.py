"""Synthetic tire-degradation model.

Generates synthetic samples of the form
    (speed, cornering_load, lap, current_wear, compound) -> wear_rate
trains a `GradientBoostingRegressor`, and exposes a `predict_wear_rate`
function backed by a lazily loaded singleton model.

Three compounds are modelled (matching the F1 hardness axis):
  - soft:   highest grip, fastest degradation
  - medium: balanced
  - hard:   lowest grip, longest life

The synthetic generator is grounded in stylised tire physics:
  * base wear rate differs by compound
  * speed contributes nonlinearly (~ speed^1.4) — quadratic-ish drag/heat
  * cornering load contributes linearly (lateral g)
  * lap number contributes a small heat-soak drift
  * a "cliff" amplifier ramps degradation in the last ~50% of tire life,
    so worn tires fall off harder than fresh ones (essential for pit-timing
    behaviour to emerge in the RL agent).

Run end-to-end (regenerate data + retrain + save + plot):
    python -m tire_model.tire_model
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ArrayLike = float | np.ndarray

DEFAULT_MODEL_PATH = Path("models/checkpoints/tire_model.pkl")
DEFAULT_DATA_PATH = Path("data/synthetic/tire_data.npz")
DEFAULT_PLOT_PATH = Path("evaluation/results/tire_compound_curves.png")

COMPOUNDS: tuple[str, ...] = ("soft", "medium", "hard")
COMPOUND_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COMPOUNDS)}

# Per-compound physics parameters used by the synthetic generator.
# Soft: high base, high speed/load gain, sharper cliff.
# Hard: opposite.
_COMPOUND_PARAMS: dict[str, dict[str, float]] = {
    "soft":   {"base": 0.20, "speed_gain": 0.0020, "load_gain": 0.50, "heat_gain": 0.030, "cliff": 1.6},
    "medium": {"base": 0.12, "speed_gain": 0.0014, "load_gain": 0.40, "heat_gain": 0.020, "cliff": 1.4},
    "hard":   {"base": 0.07, "speed_gain": 0.0009, "load_gain": 0.30, "heat_gain": 0.012, "cliff": 1.2},
}

FEATURE_NAMES: tuple[str, ...] = ("speed", "cornering_load", "lap", "current_wear", "compound_id")

_model_cache: GradientBoostingRegressor | None = None


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def _physics_wear_rate(
    speed: np.ndarray,
    load: np.ndarray,
    lap: np.ndarray,
    current_wear: np.ndarray,
    compound_id: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vectorised stylised wear-rate model. All inputs must be the same shape."""
    speed = np.asarray(speed, dtype=np.float32)
    load = np.asarray(load, dtype=np.float32)
    lap = np.asarray(lap, dtype=np.float32)
    current_wear = np.asarray(current_wear, dtype=np.float32)
    compound_id = np.asarray(compound_id, dtype=np.int64)

    # zeros_like (not empty_like): if a compound_id is ever out of range,
    # the slot stays at 0 instead of holding uninitialised garbage.
    base = np.zeros_like(speed)
    speed_gain = np.zeros_like(speed)
    load_gain = np.zeros_like(speed)
    heat_gain = np.zeros_like(speed)
    cliff = np.zeros_like(speed)
    for i, name in enumerate(COMPOUNDS):
        mask = compound_id == i
        p = _COMPOUND_PARAMS[name]
        base[mask] = p["base"]
        speed_gain[mask] = p["speed_gain"]
        load_gain[mask] = p["load_gain"]
        heat_gain[mask] = p["heat_gain"]
        cliff[mask] = p["cliff"]

    # Cliff: degradation accelerates in the last 50% of tire life.
    cliff_factor = 1.0 + cliff * np.clip((current_wear - 50.0) / 50.0, 0.0, 1.0) ** 2

    rate = (
        base
        + speed_gain * np.power(speed, 1.4)
        + load_gain * load
        + heat_gain * lap
    ) * cliff_factor

    if rng is not None:
        rate = rate + rng.normal(0.0, 0.15, size=rate.shape)
    return np.clip(rate, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n_samples: int = 12_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.0, 100.0, size=n_samples).astype(np.float32)
    load = rng.uniform(0.0, 10.0, size=n_samples).astype(np.float32)
    lap = rng.integers(1, 61, size=n_samples).astype(np.float32)
    current_wear = rng.uniform(0.0, 100.0, size=n_samples).astype(np.float32)
    compound_id = rng.integers(0, len(COMPOUNDS), size=n_samples).astype(np.int64)

    y = _physics_wear_rate(speed, load, lap, current_wear, compound_id, rng=rng)
    X = np.stack(
        [speed, load, lap, current_wear, compound_id.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(X, y)
    return model


def save_model(model: GradientBoostingRegressor, path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> GradientBoostingRegressor:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------
def _compound_to_ids(compound, n: int) -> np.ndarray:
    """Normalise a string / int / array-of-either into an int64 ndarray of length n."""
    if isinstance(compound, str):
        return np.full(n, COMPOUND_TO_ID[compound], dtype=np.int64)
    arr = np.atleast_1d(np.asarray(compound))
    if arr.dtype.kind in {"U", "S", "O"}:
        ids = np.array([COMPOUND_TO_ID[str(c)] for c in arr], dtype=np.int64)
    else:
        ids = arr.astype(np.int64)
    if ids.size == 1 and n > 1:
        ids = np.full(n, ids[0], dtype=np.int64)
    return ids


def predict_wear_rate(
    speed: ArrayLike,
    cornering_load: ArrayLike,
    lap: ArrayLike,
    current_wear: ArrayLike = 0.0,
    compound: str | int | np.ndarray = "medium",
    model_path: Path = DEFAULT_MODEL_PATH,
) -> ArrayLike:
    """Predict instantaneous tire wear rate.

    Accepts scalars or 1-D arrays of equal length. `compound` may be a string
    ("soft" / "medium" / "hard"), an int (0/1/2), or an array of either.
    Returns a Python float when all inputs were scalar, otherwise an ndarray.
    """
    global _model_cache
    if _model_cache is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No tire model at {model_path}. "
                "Run `python -m tire_model.tire_model` first."
            )
        _model_cache = load_model(model_path)

    speed_a = np.atleast_1d(np.asarray(speed, dtype=np.float32))
    load_a = np.atleast_1d(np.asarray(cornering_load, dtype=np.float32))
    lap_a = np.atleast_1d(np.asarray(lap, dtype=np.float32))
    wear_a = np.atleast_1d(np.asarray(current_wear, dtype=np.float32))

    target_len = max(speed_a.size, load_a.size, lap_a.size, wear_a.size)
    if speed_a.size == 1 and target_len > 1:
        speed_a = np.full(target_len, speed_a[0], dtype=np.float32)
    if load_a.size == 1 and target_len > 1:
        load_a = np.full(target_len, load_a[0], dtype=np.float32)
    if lap_a.size == 1 and target_len > 1:
        lap_a = np.full(target_len, lap_a[0], dtype=np.float32)
    if wear_a.size == 1 and target_len > 1:
        wear_a = np.full(target_len, wear_a[0], dtype=np.float32)
    comp_a = _compound_to_ids(compound, target_len)

    X = np.stack(
        [speed_a, load_a, lap_a, wear_a, comp_a.astype(np.float32)],
        axis=1,
    )
    y = _model_cache.predict(X)
    return float(y[0]) if y.shape == (1,) else y


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def per_compound_r2(model: GradientBoostingRegressor, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    compound_col = X[:, 4].astype(np.int64)
    for i, name in enumerate(COMPOUNDS):
        mask = compound_col == i
        if mask.sum() == 0:
            out[name] = float("nan")
            continue
        out[name] = float(r2_score(y[mask], model.predict(X[mask])))
    return out


def plot_compound_curves(output_path: Path = DEFAULT_PLOT_PATH) -> Path:
    """Predicted wear-rate vs lap, per compound, at a fixed reference setpoint."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    laps = np.arange(1, 61, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 4))
    for compound in COMPOUNDS:
        # Track an evolving wear estimate so the cliff term shows up in the curve.
        rates: list[float] = []
        cumulative_wear = 0.0
        for lap in laps:
            r = predict_wear_rate(
                speed=70.0,
                cornering_load=4.0,
                lap=float(lap),
                current_wear=cumulative_wear,
                compound=compound,
            )
            rates.append(float(r))
            cumulative_wear = min(100.0, cumulative_wear + float(r))
        ax.plot(laps, rates, label=compound)

    ax.set_xlabel("lap")
    ax.set_ylabel("predicted wear rate (% / lap-step)")
    ax.set_title("Tire wear-rate vs lap, by compound (speed=70, load=4)")
    ax.legend(title="compound")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global _model_cache

    print("Generating synthetic tire-degradation dataset...")
    X, y = generate_synthetic_dataset()
    DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DEFAULT_DATA_PATH, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"  saved {X.shape[0]} samples to {DEFAULT_DATA_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    print("Training GradientBoostingRegressor...")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    overall_r2 = r2_score(y_test, y_pred)
    overall_mae = mean_absolute_error(y_test, y_pred)
    print(f"  held-out R^2: {overall_r2:.4f}    MAE: {overall_mae:.4f}")

    by_compound = per_compound_r2(model, X_test, y_test)
    print("  per-compound R^2:")
    for name, score in by_compound.items():
        print(f"    {name:6s} {score:.4f}")

    save_model(model)
    _model_cache = model  # avoid a disk roundtrip for the diagnostics below
    print(f"  saved model to {DEFAULT_MODEL_PATH}")

    plot_path = plot_compound_curves()
    print(f"  saved diagnostic plot to {plot_path}")

    # Sanity check: soft should predict a higher wear rate than hard at the
    # same operating point.
    sample_args = dict(speed=80.0, cornering_load=5.0, lap=20, current_wear=20.0)
    soft = predict_wear_rate(**sample_args, compound="soft")
    medium = predict_wear_rate(**sample_args, compound="medium")
    hard = predict_wear_rate(**sample_args, compound="hard")
    print(
        f"  predict_wear_rate({sample_args}) "
        f"-> soft={soft:.3f}  medium={medium:.3f}  hard={hard:.3f}"
    )


if __name__ == "__main__":
    main()
