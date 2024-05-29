import gym
import math
from typing import Optional, Union
import numpy as np
from gym import logger, spaces
from gym.envs.classic_control import utils

class CustomCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, render_mode: Optional[str] = None):
        self.gravity_const = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.pole_mass + self.cart_mass
        self.pole_half_length = 0.5
        self.pole_mass_length = self.pole_mass * self.pole_half_length
        self.force_amplitude = 10.0
        self.time_step = 0.02
        self.integrator_type = "euler"

        self.theta_limit_radians = 1.3089
        self.position_limit = 3

        high = np.array(
            [
                self.position_limit * 2,
                np.finfo(np.float32).max,
                self.theta_limit_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_after_termination = None
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_amplitude if action == 1 else -self.force_amplitude
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity_const * sintheta - costheta * temp) / (
            self.pole_half_length * (4.0 / 3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        if self.integrator_type == "euler":
            x = x + self.time_step * x_dot
            x_dot = x_dot + self.time_step * x_acc
            theta = theta + self.time_step * theta_dot
            theta_dot = theta_dot + self.time_step * theta_acc
        else:
            x_dot = x_dot + self.time_step * x_acc
            x = x + self.time_step * x_dot
            theta_dot = theta_dot + self.time_step * theta_acc
            theta = theta + self.time_step * theta_dot

        x_dot = np.clip(x_dot, -10, 10)
        theta_dot = np.clip(theta_dot, -3.14, 3.14)
        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.position_limit
            or x > self.position_limit
            or theta < -self.theta_limit_radians
            or theta > self.theta_limit_radians
        )

        reward = 1.0 if not terminated else 0.0

        if self.steps_after_termination is None and terminated:
            self.steps_after_termination = 0
        elif self.steps_after_termination is not None:
            if self.steps_after_termination == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_after_termination += 1

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = (0, 0, 0, 0)
        self.steps_after_termination = None
        return np.array(self.state, dtype=np.float32), {}
