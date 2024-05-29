import numpy as np

class MazeEnvironment:
    def __init__(self, initial_state=0, terminal_state=24, discount_factor=0.9, grid_shape=(5, 5), barriers=[11, 17], penalty_zones=[21]):
        self.action_taken = None
        self.initial_state = initial_state
        self.terminal_state = terminal_state
        self.grid_shape = grid_shape
        self.barriers = barriers
        self.penalty_zones = penalty_zones
        self.current_state = initial_state
        self.current_reward = 0
        self.stay_probability = 0.1
        self.left_turn_probability = 0.05
        self.right_turn_probability = 0.05
        self.discount_factor = discount_factor
 
    def select_action(self, policy):
        return np.random.choice([0, 1, 2, 3], p=policy[self.current_state])

    def determine_next_state(self, state, action):
        random_value = np.random.uniform()
        if random_value < 0.05:
            action = (action + 3) % 4
        elif random_value < 0.1:
            action = (action + 1) % 4
        elif random_value < 0.2:
            action = None
        new_state = state
        if action is None:
            return new_state
        
        if action == 0:
            new_state -= 5
        elif action == 1 and new_state % 5 != 4:
            new_state += 1
        elif action == 2:
            new_state += 5
        elif action == 3 and new_state % 5 != 0:
            new_state -= 1
        
        if new_state < 0 or new_state > self.terminal_state or new_state in self.barriers:
            new_state = state
        return new_state
    
    def compute_reward(self, state):
        reward = 0
        if state in self.penalty_zones:
            reward = -10
        elif state == self.terminal_state:
            reward = 10
        return reward
    
    def perform_step(self, policy):
        self.action_taken = self.select_action(policy)
        self.current_state = self.determine_next_state(self.current_state, self.action_taken)
        self.current_reward = self.compute_reward(self.current_state)
        return self.current_state, self.current_reward
    
    def has_reached_terminal(self):
        return self.current_state == self.terminal_state
