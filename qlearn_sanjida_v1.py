#import required libraries 
import gym
import numpy as np
import time
import matplotlib.pyplot as plt 
import csv

# import the class that implements the Q-Learning algorithm
from functions import Q_Learning

#env=gym.make('CartPole-v1',render_mode='human')
env=gym.make('CartPole-v1')
(state,_)=env.reset()


# now we define parameters for different categories of state
up_limit=env.observation_space.high
low_limit=env.observation_space.low
min_cart_vel=-3
max_cart_vel=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
up_limit[1]=max_cart_vel
up_limit[3]=poleAngleVelocityMax
low_limit[1]=min_cart_vel
low_limit[3]=poleAngleVelocityMin

bins_position=30
bins_velocity=30
bins_angle=30
bins_angle_velocity=30
numberOfBins=[bins_position,bins_velocity,bins_angle,bins_angle_velocity]

# Our definitions for the parameters
step_size=0.1
discount_rate=.99
greediness=0.2
total_iterations=600

# creating object
Q1=Q_Learning(env,step_size,discount_rate,greediness,total_iterations,numberOfBins,low_limit,up_limit)
# now we run our algorithm for Q learning
Q1.simulateEpisodes()

with open('q_learning_rewards.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Sum of Rewards"])  # Header row
    for episode, reward in enumerate(Q1.sumRewardsEpisode):
        writer.writerow([episode + 1, reward])

print("Sum of rewards per episode saved to 'q_learning_rewards.csv'.")
# simulating the way learned
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()

plt.figure(figsize=(12, 5))
# Now we plot our chart
plt.plot(Q1.sumRewardsEpisode,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()

# ending the environment
env1.close()
# calculating the cumulative rewards/ sum of the rewards
np.sum(obtainedRewardsOptimal)

#now we simulate a random way to compare later
(obtainedRewardsRandom,env2)=Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# Now we run our ~6000 episodes and compare with the random learning episode
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()