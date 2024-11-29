#Trying the DQN approach
#First we import the libraries

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

environment = gym.make("CartPole-v1")


# Check IPython environment 
# importing matplot for interactive functionality

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# selecting hardware device for usages
# I used a laptop with GPU to run this

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Here a transition is defined as namedtuple to store episodes while training

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#Replay memory will store past transitions
# Replay memory also breaks the correlation among immediate transitions

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# Now we defined the Deep Q-Network (DQN class) which is the neural network here


class DQN(nn.Module):

    def __init__(self, no_observations, no_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(no_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, no_actions)

# We defined forward, which calculates the output, and determines next action for a state/ batch of states
# during optimization. this returns a tensor of Q-values for each action possible ([[left0exp,right0exp]...]).

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




# We set hyperparameters for training the deep Q-network (DQN)
# Gamma here is set at 0.99 to compare with the Q learning approach's discount factor of 0.99

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# To get the # no of actions from gym action space
no_actions = environment.action_space.n
# getting the total the number of state observations
state, info = environment.reset()
no_observations = len(state)

policy_net = DQN(no_observations, no_actions).to(device)
target_net = DQN(no_observations, no_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            
# t.max(1) gives maximum value in each row of the tensor along dimension 1 (columns).
# The result has two outputs:
# - maximum value (1st row).
# - the index of the maximum value, which corresponds to the action with the highest Q-value (2nd row)

# If/when the exploration strategy is triggered, a random action is chosen
# environment.action_space.sample() creates a random action, which promotes exploration

            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[environment.action_space.sample()]], device=device, dtype=torch.long)



# Making a list to record the durations of episodes while training

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    
 # Plotting the running average of 100 episodes
 
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # giving slight pause to ensure plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch to converts batch-array of Transitions 
    # transpose/ unzip function solution from https://stackoverflow.com/a/19343/3343043
    
    batch = Transition(*zip(*transitions))

    # Now we compute a mask of non-final states and concatenate the batch elements
    # Here final state means the one after which simulation ended
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
# We compute Q(s_t, a) for the actions actually taken:
# - policy_net(state_batch) calculates Q-values for all actions in the given states.
# - gather(1, action_batch) selects the Q-values corresponding to the specific actions taken.
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # we calculate the values  V(s_{t+1}) for all next states.
    # for non final next states, we use the target network to find the maximum Q-value
    # for the final state, we state value or 0

    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        
# Compute the expected Q values which is the discounted next state values plus the reward we got
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

# To compute loss we use Huber loss as it is good with taking out outliers effect

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


# We optimize the model by first taking out the old gradients
# Then we back propagate the loss to calculate the gradients

    optimizer.zero_grad()
    loss.backward()

# We also do gradient clipping In-place to prevent gradients from running outliers

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


epi_reward = []  # To track cumulative rewards for each episode

# Running 600 episodes like we did for the q learning approach

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 600



for i_episode in range(num_episodes):

# Here we reset our environment and get it to initial state

    state, info = environment.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumu_reward = 0  # we track the cumulative reward for this episode

    
    for t in count():
        action = select_action(state)  #Selecting an action based on current state
         # We take the action, then observe the immediate state and reward
        observation, reward, terminated, truncated, _ = environment.step(action.item())
        cumu_reward += reward  # Addiing this reward to the cumulative reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

# Then we set the next state to None if this episode is over, otherwise we change it to a tensor


        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        #now we save this in memory
        memory.push(state, action, next_state, reward)

        # Then moving to the next state
        state = next_state

        # Then we optimize (the policy network)
        optimize_model()

        # Soft update of the target network weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

 # If the episode is over, we record the the results and move to the next episode

        if done:
            epi_reward.append(cumu_reward)  
            episode_durations.append(t + 1)
            plot_durations()
            break

# Finally we plot our results

plt.figure(figsize=(12, 5))
plt.plot(epi_reward, color='green', label='DQN')
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.title('DQN Learning Convergence')
plt.yscale('log')  # Log scale for comparison with Q-Learning
plt.legend()
plt.savefig('dqn_convergence.png')
plt.show()


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()



