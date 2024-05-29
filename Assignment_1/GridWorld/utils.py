import numpy as np
from environment import MazeEnvironment

def generate_policy(parameters, temperature):
    raw_policy = np.exp(parameters * temperature)
    for i in range(raw_policy.shape[0]):
        row_sum = raw_policy[i].sum()
        raw_policy[i] /= row_sum
    return raw_policy

def evaluate_episode_return(policy):
    env = MazeEnvironment()
    total_reward = env.current_reward
    while not env.has_reached_terminal():
        state, reward = env.perform_step(policy)
        total_reward += reward
    return total_reward

def execute_trial(parameters, temperature, policy=None):
    if policy is None:
        policy = generate_policy(np.reshape(parameters, (25, 4)), temperature)
    cumulative_return = 0
    episode_returns = []
    for _ in range(30):
        episode_return = evaluate_episode_return(policy)
        cumulative_return += episode_return
        episode_returns.append(episode_return)
    average_return = cumulative_return / 30
    return average_return, episode_returns

def execute_multiple_trials(temperature):
    parameters = np.random.normal(0, 1, (100))
    avg_return, episode_returns = execute_trial(parameters, temperature)
    all_returns = [avg_return]
    for _ in range(299):
        new_parameters = np.random.multivariate_normal(parameters, temperature * np.identity(100))
        new_avg_return, new_episode_returns = execute_trial(new_parameters, temperature)
        episode_returns = [sum(x) for x in zip(episode_returns, new_episode_returns)]
        all_returns.append(new_avg_return)
        if new_avg_return > avg_return:
            parameters = new_parameters
            avg_return = new_avg_return
    normalized_episode_returns = [x / 300 for x in episode_returns]
    return all_returns, parameters

def compute_averaged_curve(parameters, temperature, num_trials, num_episodes, policy=None):
    avg_curve = [0] * num_episodes
    for _ in range(num_trials):
        _, episode_returns = execute_trial(parameters, temperature, policy)
        avg_curve = [sum(x) for x in zip(avg_curve, episode_returns)]
    avg_curve = [x / num_trials for x in avg_curve]
    return avg_curve

def determine_state(action, state):
    next_state = state
    if action == 0:
        next_state -= 5
    elif action == 1 and state % 5 != 4:
        next_state += 1
    elif action == 2:
        next_state += 5
    elif action == 3 and state % 5 != 0:
        next_state -= 1
    if next_state < 0 or next_state > 24 or next_state in [11, 17]:
        next_state = state
    return next_state

def calculate_transition_prob(current_state, next_state, action):
    if next_state in [11, 17]:
        return 0
    transition_prob = 0
    state_if_action_taken = determine_state(action, current_state)
    state_if_left_turn = determine_state((action + 1) % 4, current_state)
    state_if_right_turn = determine_state((action + 3) % 4, current_state)
    state_if_no_action = current_state
    if state_if_action_taken == next_state:
        transition_prob += 0.8
    if state_if_left_turn == next_state:
        transition_prob += 0.05
    if state_if_right_turn == next_state:
        transition_prob += 0.05
    if state_if_no_action == next_state:
        transition_prob += 0.1
    return transition_prob

def derive_policy_via_value_iteration(epsilon):
    value_function = [-3000] * 25
    value_function[24] = 0
    policy = [[0, 0, 0, 0] for _ in range(25)]
    iteration = 0
    max_change = 100
    env = MazeEnvironment()
    while max_change > epsilon:
        iteration += 1
        max_change = 0
        for state in range(24):
            action_rewards = []
            for action in range(4):
                reward_sum = 0
                value_sum = 0
                for next_state in range(25):
                    reward_sum += env.compute_reward(next_state) * calculate_transition_prob(state, next_state, action)
                    value_sum += calculate_transition_prob(state, next_state, action) * value_function[next_state]
                value_sum *= env.discount_factor
                action_rewards.append(reward_sum + value_sum)
            for action in range(4):
                if action_rewards[action] > value_function[state]:
                    policy[state] = [0, 0, 0, 0]
                    policy[state][action] = 1
                    max_change = max(max_change, action_rewards[action] - value_function[state])
                    value_function[state] = action_rewards[action]
    return policy, value_function, iteration
