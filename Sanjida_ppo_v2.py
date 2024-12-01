"""
PPO Approach Sanjida
""" 

"""

### We use CartPole-v1 with Gamma 0.99 to compare with Q learnign and DQN


## Libraries that we use for this are: (different from the other approaches)

1. `numpy` for n-dimensional arrays
2. `tensorflow` and `keras` for building the deep RL PPO agent
3. `gymnasium` for getting everything we need about the environment
4. `scipy.signal` for calculating the discounted cumulative sums of vectors
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import numpy as np
import tensorflow as tf
import gymnasium as gym
import scipy.signal

"""
## Here we define the functions and classes for PPO approach
"""


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.act_buffer = np.zeros(size, dtype=np.int32)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprob_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.act_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprob_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.adv_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.adv_buffer),
            np.std(self.adv_buffer),
        )
        self.adv_buffer = (self.adv_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.act_buffer,
            self.adv_buffer,
            self.return_buffer,
            self.logprob_buffer,
        )


def mlp(x, sizes, activation=keras.activations.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


# def logprobabilities(logits, a):
#     # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
#     logprobabilities_all = keras.ops.log_softmax(logits)
#     logprobability = keras.ops.sum(
#         keras.ops.one_hot(a, num_actions) * logprobabilities_all, axis=1
#     )
#     return logprobability


def logprobabilities(logits, a):
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


seed_generator = tf.random.Generator.from_seed(1337)


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    # action = keras.ops.squeeze(
    #     keras.random.categorical(logits, 1, seed=seed_generator), axis=1
    # )
    action = tf.squeeze(
        tf.random.categorical(logits, 1, seed=1337), axis=1
        )

    return logits, action


@tf.function
def train_policy(
    observation_buffer, act_buffer, logprob_buffer, adv_buffer
):
    with tf.GradientTape() as tape:
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), act_buffer)
            - logprob_buffer
        )
        min_advantage = tf.where(
            adv_buffer > 0,
            (1 + clip_ratio) * adv_buffer,
            (1 - clip_ratio) * adv_buffer,
        )
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * adv_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprob_buffer
        - logprobabilities(actor(observation_buffer), act_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl



@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 60
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

"""
## Initializations
"""

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v1")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype="float32")
logits = mlp(observation_input, list(hidden_sizes) + [num_actions])
actor = keras.Model(inputs=observation_input, outputs=logits)
#value = keras.ops.squeeze(mlp(observation_input, list(hidden_sizes) + [1]), axis=1)
value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1]), axis=1)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, _ = env.reset()
episode_return, episode_length = 0, 0

"""
## Train
"""
total_episodes = 0

# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation_new, reward, done, _, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        # We will complete the trajectory if it gets to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            total_episodes += 1
            observation, _ = env.reset()
            episode_return, episode_length = 0, 0

# Get data from the buffer, which stores all trajectories from the current epoch
# The buffer includes observations, actions, advantages, rewards-to-go, and log-probabilities

    (
        observation_buffer, # States the agent observes
        act_buffer,   #actions the agent takes
        adv_buffer,   #calculated advantages for the actions
        return_buffer, #rewards to go for each state
        logprob_buffer, #log probabilities of the actions (from the policy)
    ) = buffer.get()

# Update the policy (actor network) with PPO using a loop for multiple iterations
# It will stop sooner if the policy change becomes too large, which is measured by KL divergence

    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, act_buffer, logprob_buffer, adv_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break


# Finally we update the value function with a loop for multiple iterations
# The value function (critic network) predicts rewards-to-go for the given states

    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # we take the output (print) which is mean return and length for each epoch
    #this gives us the average number of time step when the cart pole was balanced
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
