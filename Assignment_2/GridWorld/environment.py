import numpy as np


class CustomGridWorld:
    def __init__(self, start_pos=0, goal_pos=24, discount_factor=0.9, grid_shape=(5, 5), obstacles=[11, 17], water_zones=[21]):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_shape = grid_shape
        self.obstacles = obstacles
        self.water_zones = water_zones
        self.current_state = start_pos
        self.discount_factor = discount_factor
        self.prob_stay = 0.1
        self.prob_left = 0.05
        self.prob_right = 0.05

    def reset(self):
        self.current_state = self.start_pos
        return self.current_state

    def select_action(self, policy):
        return np.random.choice([0, 1, 2, 3], p=policy)

    def compute_next_state(self, state, action):
        prob = np.random.uniform()
        if prob < 0.05:
            action = (action + 3) % 4  # Left
        elif prob < 0.1:
            action = (action + 1) % 4  # Right
        elif prob < 0.2:
            action = None  # Stay

        new_state = state
        if action is None:
            return new_state

        if action == 0:  # Up
            new_state -= 5
        elif action == 1 and new_state % 5 != 4:  # Right
            new_state += 1
        elif action == 2:  # Down
            new_state += 5
        elif action == 3 and new_state % 5 != 0:  # Left
            new_state -= 1

        if new_state < 0 or new_state > self.goal_pos or new_state in self.obstacles:
            new_state = state

        return new_state

    def calculate_reward(self, state):
        if state in self.water_zones:
            return -10
        elif state == self.goal_pos:
            return 10
        return 0

    def step(self, policy):
        action = self.select_action(policy)
        next_state = self.compute_next_state(self.current_state, action)
        reward = self.calculate_reward(next_state)
        self.current_state = next_state
        return next_state, reward

    def is_goal_state(self):
        return self.current_state == self.goal_pos