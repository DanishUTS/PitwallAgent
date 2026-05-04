"""Custom CarRacing environment for Pitwall.

Wraps Gymnasium's `CarRacing-v3` and adds:
  * a continuous tire-wear state (0-100 %) that grows with speed and steering
  * a pit-lane trigger zone that resets tire wear at the cost of simulated time
  * a lap counter (placeholder; real lap detection is a TODO for Hari)

The observation is a Dict so the RL agent can see both the camera frame and
the tire-wear / lap state. This requires `MultiInputPolicy` on the SB3 side.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PitwallRacingEnv(gym.Wrapper):
    # --- tire wear dynamics -------------------------------------------------
    WEAR_BASE: float = 0.02           # base wear added every step
    WEAR_SPEED_GAIN: float = 0.001    # extra wear per unit of speed
    WEAR_STEER_GAIN: float = 0.05     # extra wear per unit of |steer|
    WEAR_REWARD_PENALTY: float = 0.001  # reward shaping: -k * tire_wear/step
    MAX_WEAR: float = 100.0

    # --- pit lane (placeholder world-coord box; Hari will refine) ----------
    PIT_ZONE_X: tuple[float, float] = (-30.0, -10.0)
    PIT_ZONE_Y: tuple[float, float] = (-5.0, 5.0)
    PIT_MIN_WEAR_TO_TRIGGER: float = 5.0
    PIT_MAX_SPEED_TO_TRIGGER: float = 1.0
    PIT_TIME_PENALTY: float = 50.0

    def __init__(self, render_mode: str | None = None):
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        super().__init__(env)

        self.tire_wear: float = 0.0
        self.lap_count: int = 0

        image_space = self.env.observation_space
        state_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.MAX_WEAR, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.tire_wear = 0.0
        self.lap_count = 0
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        speed = self._estimate_speed()
        steer_mag = abs(float(action[0]))
        self.tire_wear = min(
            self.MAX_WEAR,
            self.tire_wear
            + self.WEAR_BASE
            + self.WEAR_SPEED_GAIN * speed
            + self.WEAR_STEER_GAIN * steer_mag,
        )
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

        info = dict(info)
        info["tire_wear"] = self.tire_wear
        info["lap_count"] = self.lap_count
        info["pit_stop"] = pitted

        return self._obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def _obs(self, image_obs):
        return {
            "image": image_obs,
            "state": np.array([self.tire_wear, float(self.lap_count)], dtype=np.float32),
        }

    def _car(self):
        return getattr(self.env.unwrapped, "car", None)

    def _estimate_speed(self) -> float:
        car = self._car()
        if car is None:
            return 0.0
        v = car.hull.linearVelocity
        return float(np.hypot(v[0], v[1]))

    def _in_pit_zone(self) -> bool:
        car = self._car()
        if car is None:
            return False
        x, y = car.hull.position
        return (
            self.PIT_ZONE_X[0] <= x <= self.PIT_ZONE_X[1]
            and self.PIT_ZONE_Y[0] <= y <= self.PIT_ZONE_Y[1]
        )
