# import libraries

from collections import defaultdict  # allows access to inaccessible keys
import gymnasium as gym
import matplotlib.pyplot as plt

# from matplotlib.patches import patch
import numpy as np
import seaborn as sns
from tqdm import tqdm

env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")


# Reset the environment
env.reset()

# About the Environment
"""
print(env.action_space.n)
print(env.observation_space[0].n)
action = env.action_space.sample()  # random action
observations, reward, terminated, truncated, info = env.step(action)
print(observations, reward, terminated, truncated, info)
"""

# Epsilon Greedy Policy


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        gamma: float,
    ):
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.training_error = []
        self.epsilon = initial_epsilon

        """
        Initialize a Q-value (action-value) table as an empty dictionary of state-action pairs, learning rate and epsilon
        """

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        """
        Get an action according to epsilon-greedy policy
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: tuple[int, int, bool],
        action: int,
        reward: float,
        next_observation: tuple[int, int, bool],
        terminated: bool,
    ) -> None:
        """
        Update the Q-value table using the Q-learning update rule
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[observation][action]
        )
        self.q_values[observation][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Hyperparameters
learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 1.0
final_epsilon = 0.1
gamma = 1
epsilon_decay = (start_epsilon * 2) / n_episodes

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    gamma=gamma,
)


# Training
fig, ax = plt.subplots()
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    done = False
    observation, info = env.reset()
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        agent.update(observation, action, reward, next_observation, terminated)

        frame = env.render()
        im = ax.imshow(frame)
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)
        im.remove()

        done = terminated or truncated
        observation = next_observation
    agent.decay_epsilon()


rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Axis 0
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)


# Axis 1
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

# Axis 2
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(
        np.array(agent.training_error).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)


plt.tight_layout()
plt.show()
