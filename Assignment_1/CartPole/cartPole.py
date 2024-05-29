import gymnasium as gym
import math
from typing import Optional, Union
import numpy as np
from gymnasium import logger, spaces

# Creating the CartPole environment using gymnasium
env = gym.make('CartPole-v1')

class CartPoleEnvironment(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, display_mode: Optional[str] = None):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.half_pole_length = 0.5  # actually half the pole's length
        self.pole_mass_length = self.pole_mass * self.half_pole_length
        self.force_magnitude = 10.0
        self.time_step = 0.02  # seconds between state updates
        self.integrator = "euler"

        # Thresholds for termination
        self.theta_limit_radians = 1.3089
        self.x_limit = 3

        # Observation space limits
        high = np.array(
            [
                self.x_limit * 2,
                np.finfo(np.float32).max,
                self.theta_limit_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_beyond_done = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, time = self.state
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp = (
            force + self.pole_mass_length * theta_dot**2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.half_pole_length * (4.0 / 3.0 - self.pole_mass * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        if self.integrator == "euler":
            x = x + self.time_step * x_dot
            x_dot = x_dot + self.time_step * x_acc
            theta = theta + self.time_step * theta_dot
            theta_dot = theta_dot + self.time_step * theta_acc
        else:  # semi-implicit euler
            x_dot = x_dot + self.time_step * x_acc
            x = x + self.time_step * x_dot
            theta_dot = theta_dot + self.time_step * theta_acc
            theta = theta + self.time_step * theta_dot
            
        x_dot = max(-10, x_dot)
        x_dot = min(10, x_dot)
        theta_dot = max(-3.14, theta_dot)
        theta_dot = min(3.14, theta_dot)
        time = time + self.time_step
        self.state = (x, x_dot, theta, theta_dot, time)

        done = bool(
            x < -self.x_limit
            or x > self.x_limit
            or theta < -self.theta_limit_radians
            or theta > self.theta_limit_radians
            or time >= 20.00
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = (0, 0, 0, 0, 0)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}
