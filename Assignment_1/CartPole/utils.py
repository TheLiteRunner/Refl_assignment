# utility.py

from cartPole import CartPoleEnvironment
import numpy as np

class SimulationAgent:
    def __init__(self, parameters, covariance, environment):
        self.parameters = parameters
        self.covariance = covariance
        self.current_state = np.array((0, 0, 0, 0, 0))
        self.environment = environment
    
    def choose_action(self):
        z = np.dot(self.current_state, (self.parameters).T)
        probability = 1.0 / (1.0 + np.exp(-z))
        action_probability = [probability, (1 - probability)]
        action = np.random.choice([0, 1], p=action_probability)
        return action
        
    def perform_step(self):
        return self.environment.step(self.choose_action())
    
    def calculate_return(self):
        total_return = 0
        done = False
        self.environment.reset()
        
        while not done and total_return < 1000:
            new_state, reward, done, _, _ = self.perform_step()
            total_return += reward
            self.current_state = new_state
        
        return total_return 

def execute_trial(parameters, covariance, num_episodes=30):
    agent = SimulationAgent(parameters, covariance, CartPoleEnvironment())
    cumulative_return = 0
    episode_gains = []
    for _ in range(num_episodes):
        episode_gain = agent.calculate_return()
        cumulative_return += episode_gain
        episode_gains.append(episode_gain)
    average_return = cumulative_return / num_episodes 
    return average_return, episode_gains

def execute_multiple_trials(covariance):
    parameters = np.random.normal(0, 1, 5)
    initial_avg_return, _ = execute_trial(parameters, covariance)
    results = [(1, initial_avg_return)]
    for i in range(800):
        new_parameters = np.random.multivariate_normal(parameters, covariance * np.identity(5))
        new_avg_return, _ = execute_trial(new_parameters, covariance)
        if new_avg_return >= initial_avg_return:
            results.append((i + 2, new_avg_return))
            parameters = new_parameters
            initial_avg_return = new_avg_return
    return results, parameters

def compute_average_curve(params, covariance, trial_count, episode_count, policy=None):
    average_curve = [0 for _ in range(episode_count)]
    for _ in range(trial_count):
        _, episode_gains = execute_trial(params, covariance)
        average_curve = [sum(x) for x in zip(average_curve, episode_gains)]
    average_curve = [x / trial_count for x in average_curve]
    return average_curve

def adjust_covariance(epsilon, k, mean_params, best_params, keps=5):
    s = np.dot((best_params[0] - mean_params).T, (best_params[0] - mean_params))
    for params in best_params[1:]:
        s += np.dot((params - mean_params).T, (params - mean_params))
    new_covariance = 2 * epsilon * np.identity(5) + s
    new_covariance /= (keps + epsilon)
    return new_covariance

def cross_entropy_method(epsilon, k, covariance, params, keps=5):
    best_params = []
    param_samples = []
    for _ in range(k):
        sampled_params = np.random.multivariate_normal(params, covariance)
        avg_return, _ = execute_trial(sampled_params, covariance, 10)
        best_params.append((avg_return, sampled_params))
    best_params.sort(reverse=True, key=lambda x: x[0])
    best_params = [x[1] for x in best_params[:keps]]
    mean_params = np.mean(best_params, axis=0)
    return adjust_covariance(epsilon, k, mean_params, best_params), mean_params

def execute_cross_entropy_trials(epsilon, k):
    params = np.random.normal(0, 1, 5)
    covariance = np.identity(5) * 2
    initial_gain, _ = execute_trial(params, covariance)
    results = [(initial_gain, 1)]
    for i in range(800):
        new_covariance, new_params = cross_entropy_method(epsilon, k, covariance, params)
        new_gain, _ = execute_trial(new_params, new_covariance)
        if new_gain > initial_gain:
            initial_gain = new_gain
            covariance = new_covariance
            params = new_params
            results.append((initial_gain, i + 2))
        print(f"Iteration #{i + 1}")
    return results, params, covariance
