"""Custom CarRacing environment for Pitwall.

Wraps Gymnasium's `CarRacing-v3` and adds:
  * a continuous tire-wear state (0-100 %) driven by the supervised tire model
    (`tire_model.predict_wear_rate`); pass `use_tire_model=False` to fall back
    to a hand-tuned legacy formula for baseline runs
  * a tire compound (soft / medium / hard) fixed per episode
  * a pit-lane trigger zone — a circle around the spawn tile (`track[0]`),
    cached at reset since the track is regenerated each episode; pit fires
    when the car is inside the zone, slow, and worn enough
  * a multi-lap wrapper: detects lap completion via CarRacing's
    `tile_visited_count`, resets tile flags so the agent can keep racing,
    and terminates after `max_laps`

Observation is a Dict so the RL agent can see the camera frame and the
tire-wear / lap / compound state. Requires `MultiInputPolicy` on the SB3 side.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tire_model import COMPOUND_TO_ID, COMPOUNDS, predict_wear_rate


class PitwallRacingEnv(gym.Wrapper):
    # --- tire wear dynamics -------------------------------------------------
    # The tire model outputs an abstract wear rate (~0-10) at a given operating
    # point. We scale by WEAR_PER_STEP_SCALE so a fresh medium tire under
    # moderate driving deteriorates over a few thousand env steps rather than
    # in seconds.
    WEAR_PER_STEP_SCALE: float = 0.02
    WEAR_REWARD_PENALTY: float = 0.001  # reward shaping: -k * tire_wear / step
    MAX_WEAR: float = 100.0

    # Cornering load is read from the car's lateral acceleration
    # (|angular velocity| * speed) and divided by this constant to land in the
    # [0, 10] range the tire model was trained on.
    CORNERING_LOAD_SCALE: float = 30.0

    # --- legacy hand-tuned wear (used when use_tire_model=False) -----------
    LEGACY_WEAR_BASE: float = 0.02
    LEGACY_WEAR_SPEED_GAIN: float = 0.001
    LEGACY_WEAR_STEER_GAIN: float = 0.05

    # --- pit lane (anchored to a real track tile, cached on reset) ---------
    # Index into env.unwrapped.track (a list of (alpha, beta1, beta2, x, y)).
    # Tile 0 is where CarRacing spawns the car — the natural start/finish
    # line analogue, which matches where pit lanes sit in real F1.
    PIT_TILE_INDEX: int = 0
    PIT_ZONE_RADIUS: float = 10.0
    PIT_MIN_WEAR_TO_TRIGGER: float = 5.0
    PIT_MAX_SPEED_TO_TRIGGER: float = 1.0
    PIT_TIME_PENALTY: float = 50.0

    DEFAULT_COMPOUND: str = "medium"
    DEFAULT_MAX_LAPS: int = 3
    # CarRacing-v3 default time limit is 1000 steps — barely a single lap.
    # 5000 lets a competent driver finish ~3 laps.
    DEFAULT_MAX_EPISODE_STEPS: int = 5000

    def __init__(
        self,
        render_mode: str | None = None,
        compound: str = DEFAULT_COMPOUND,
        use_tire_model: bool = True,
        max_laps: int = DEFAULT_MAX_LAPS,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ):
        if compound not in COMPOUNDS:
            raise ValueError(f"compound must be one of {COMPOUNDS}, got {compound!r}")
        if max_laps < 1:
            raise ValueError(f"max_laps must be >= 1, got {max_laps}")

        env = gym.make(
            "CarRacing-v3",
            continuous=True,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        super().__init__(env)

        self.compound: str = compound
        self.compound_id: int = COMPOUND_TO_ID[compound]
        self.use_tire_model: bool = use_tire_model
        self.max_laps: int = max_laps

        self.tire_wear: float = 0.0
        self.lap_count: int = 0
        self._steps_this_lap: int = 0
        self._lap_times: list[int] = []
        self._pit_zone_xy: tuple[float, float] | None = None

        # Eagerly warm the tire-model cache so a missing artefact surfaces
        # at construction time rather than deep in the first step(). With
        # SubprocVecEnv this runs once per worker — joblib loads the .pkl
        # in roughly 50 ms, fine. Fail loudly if the model isn't trained:
        # silent fallback makes the user think they're using the supervised
        # model when they aren't. Pass `use_tire_model=False` for the
        # explicit baseline mode.
        if self.use_tire_model:
            predict_wear_rate(
                speed=50.0,
                cornering_load=3.0,
                lap=1,
                current_wear=0.0,
                compound=self.compound,
            )

        image_space = self.env.observation_space
        # state vector: [tire_wear, lap_count, compound_id]
        state_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.MAX_WEAR, float(self.max_laps), float(len(COMPOUNDS) - 1)], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Pop our wrapper-specific keys so we don't pass them to the
        # underlying CarRacing env. Allow `options={"compound": "soft"}` to
        # switch compound per episode.
        inner_options = dict(options) if options else None
        if inner_options is not None and "compound" in inner_options:
            self.set_compound(inner_options.pop("compound"))

        obs, info = self.env.reset(seed=seed, options=inner_options)
        self.tire_wear = 0.0
        self.lap_count = 0
        self._steps_this_lap = 0
        self._lap_times = []
        self._cache_pit_zone_position()
        info = dict(info)
        info["compound"] = self.compound
        if self._pit_zone_xy is not None:
            info["pit_zone_xy"] = self._pit_zone_xy
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps_this_lap += 1

        speed = self._estimate_speed()
        cornering_load = self._estimate_cornering_load(speed)
        wear_delta = self._wear_delta(speed, cornering_load, action)
        self.tire_wear = float(min(self.MAX_WEAR, self.tire_wear + wear_delta))

        reward = float(reward) - self.WEAR_REWARD_PENALTY * self.tire_wear

        pitted = False
        if (
            self._in_pit_zone()
            and speed < self.PIT_MAX_SPEED_TO_TRIGGER
            and self.tire_wear > self.PIT_MIN_WEAR_TO_TRIGGER
        ):
            self.tire_wear = 0.0
            reward -= self.PIT_TIME_PENALTY
            pitted = True

        # --- lap detection --------------------------------------------------
        lap_completed_this_step = False
        lap_time = None
        if self._all_tiles_visited():
            lap_completed_this_step = True
            self.lap_count += 1
            lap_time = self._steps_this_lap
            self._lap_times.append(lap_time)
            self._steps_this_lap = 0

            if self.lap_count >= self.max_laps:
                # Final lap: end the episode regardless of what the underlying
                # env said. Use truncated rather than terminated so SB3's
                # bootstrapping handles it as "ran out of time" rather than
                # "fatal failure".
                truncated = True
            else:
                # Reset tile-visited flags so the next lap awards full reward,
                # and override the underlying env's `terminated=True` (which
                # it sets when all tiles have been visited).
                self._reset_track_tiles()
                terminated = False

        # ------------------------------------------------------------------
        info = dict(info)
        info["tire_wear"] = self.tire_wear
        info["lap_count"] = self.lap_count
        info["pit_stop"] = pitted
        info["compound"] = self.compound
        info["cornering_load"] = cornering_load
        info["wear_delta"] = wear_delta
        info["lap_completed"] = lap_completed_this_step
        info["pit_zone_distance"] = self._pit_zone_distance()
        if lap_time is not None:
            info["lap_time_steps"] = lap_time

        return self._obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def set_compound(self, compound: str) -> None:
        if compound not in COMPOUNDS:
            raise ValueError(f"compound must be one of {COMPOUNDS}, got {compound!r}")
        self.compound = compound
        self.compound_id = COMPOUND_TO_ID[compound]

    def _wear_delta(self, speed: float, cornering_load: float, action) -> float:
        if self.use_tire_model:
            rate = predict_wear_rate(
                speed=speed,
                cornering_load=cornering_load,
                lap=float(self.lap_count + 1),  # 1-indexed
                current_wear=self.tire_wear,
                compound=self.compound,
            )
            return float(rate) * self.WEAR_PER_STEP_SCALE
        # legacy hand-tuned fallback
        steer_mag = abs(float(action[0]))
        return (
            self.LEGACY_WEAR_BASE
            + self.LEGACY_WEAR_SPEED_GAIN * speed
            + self.LEGACY_WEAR_STEER_GAIN * steer_mag
        )

    def _obs(self, image_obs):
        return {
            "image": image_obs,
            "state": np.array(
                [self.tire_wear, float(self.lap_count), float(self.compound_id)],
                dtype=np.float32,
            ),
        }

    def _car(self):
        return getattr(self.env.unwrapped, "car", None)

    def _estimate_speed(self) -> float:
        car = self._car()
        if car is None:
            return 0.0
        v = car.hull.linearVelocity
        return float(np.hypot(v[0], v[1]))

    def _estimate_cornering_load(self, speed: float) -> float:
        """Lateral acceleration proxy: |omega| * v, scaled into [0, 10]."""
        car = self._car()
        if car is None:
            return 0.0
        omega = abs(float(car.hull.angularVelocity))
        load = (omega * speed) / self.CORNERING_LOAD_SCALE
        return float(np.clip(load, 0.0, 10.0))

    def _cache_pit_zone_position(self) -> None:
        """Read the pit-zone tile coords from the freshly-built track.

        CarRacing-v3 stores the track as a list of (alpha, beta1, beta2, x, y)
        tuples in `env.unwrapped.track`. Tile 0 is the spawn point. The track
        is regenerated on every `reset()`, so this has to run after
        `self.env.reset(...)` returns.
        """
        track = getattr(self.env.unwrapped, "track", None)
        if not track or self.PIT_TILE_INDEX >= len(track) or self.PIT_TILE_INDEX < 0:
            self._pit_zone_xy = None
            return
        tile = track[self.PIT_TILE_INDEX]
        # tile = (alpha, beta1, beta2, x, y)
        self._pit_zone_xy = (float(tile[3]), float(tile[4]))

    def _pit_zone_distance(self) -> float:
        car = self._car()
        if car is None or self._pit_zone_xy is None:
            return float("inf")
        dx = car.hull.position[0] - self._pit_zone_xy[0]
        dy = car.hull.position[1] - self._pit_zone_xy[1]
        return float(np.hypot(dx, dy))

    def _in_pit_zone(self) -> bool:
        return self._pit_zone_distance() < self.PIT_ZONE_RADIUS

    # --- lap detection internals -------------------------------------------
    def _all_tiles_visited(self) -> bool:
        underlying = self.env.unwrapped
        track = getattr(underlying, "track", None)
        if not track:
            return False
        visited = getattr(underlying, "tile_visited_count", 0)
        return visited >= len(track)

    def _reset_track_tiles(self) -> None:
        """Clear `road_visited` on every tile so the next lap awards full reward.

        CarRacing-v3 stores tile fixtures in `env.unwrapped.road`; each tile
        has a `road_visited` boolean toggled by the contact listener. Resetting
        it lets the same tile award reward again on the next pass.
        """
        underlying = self.env.unwrapped
        road = getattr(underlying, "road", None)
        if road is not None:
            for tile in road:
                if hasattr(tile, "road_visited"):
                    tile.road_visited = False
        if hasattr(underlying, "tile_visited_count"):
            underlying.tile_visited_count = 0
