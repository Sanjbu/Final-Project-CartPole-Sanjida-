
#Q learn function code: Sanjida
#In this code we setup the environment (_init_), define the actions, and teach it what to choose as an action (selectaction)
#The system learns through these actions/play, and tests the learning through smartstrategy/ learned strategy and random strategy

import numpy as np

class Q_Learning:
    
    
    def __init__(self,env,step_size,discount_size,greediness,total_iterations,no_bins,low_limit,up_limit):
        import numpy as np
        
        self.env=env
        self.step_size=step_size
        self.discount_size=discount_size 
        self.greediness=greediness 
        self.actionNumber=env.action_space.n 
        self.total_iterations=total_iterations
        self.no_bins=no_bins
        self.low_limit=low_limit
        self.up_limit=up_limit
        
        # We make a list that saves the total rewards from each episode
        self.sumRewardsEpisode=[]
        
        # We define action value function matrix here
        self.Qmatrix=np.random.uniform(low=0, high=1, size=(no_bins[0],no_bins[1],no_bins[2],no_bins[3],self.actionNumber))
        
    
    def returnIndexState(self,state):
        position =      state[0]
        velocity =      state[1]
        angle    =      state[2]
        angularVelocity=state[3]
        
        bin_cart_position=np.linspace(self.low_limit[0],self.up_limit[0],self.no_bins[0])
        bin_cart_velocity=np.linspace(self.low_limit[1],self.up_limit[1],self.no_bins[1])
        bin_pole_angle=np.linspace(self.low_limit[2],self.up_limit[2],self.no_bins[2])
        bin_pole_angle_velocity=np.linspace(self.low_limit[3],self.up_limit[3],self.no_bins[3])
        
        position_index=np.maximum(np.digitize(state[0],bin_cart_position)-1,0)
        velocity_index=np.maximum(np.digitize(state[1],bin_cart_velocity)-1,0)
        angle_index=np.maximum(np.digitize(state[2],bin_pole_angle)-1,0)
        angular_velocity_index=np.maximum(np.digitize(state[3],bin_pole_angle_velocity)-1,0)
        
        return tuple([position_index,velocity_index,angle_index,angular_velocity_index])   
    
  
    def selectAction(self,state,index):
        

        if index<200:
            return np.random.choice(self.actionNumber)   
            
        # We generate a random real number in the half-open interval [0.0, 1.0) for the greedy approach
        rand_number=np.random.random()
        
        # after 6000 episodes, we slowly start to decrease the greediness parameter
        if index>400:
            self.greediness=0.999*self.greediness
        
        # We select random action (we explore) if this condition is satisfied
        if rand_number < self.greediness:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qmatrix[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)]==np.max(self.Qmatrix[self.returnIndexState(state)]))[0])
            
     
    def simulateEpisodes(self):
        import numpy as np
        # the episodes will run in a loop here
        for episode_index in range(self.total_iterations):
            
            # to keep record of convergence, we track the rewards from episodes here 
            rewardsEpisode=[]
            
            # We create a new environment/ reset the environment at the start of each episode
            (stateS,_)=self.env.reset()
            stateS=list(stateS)
          
            print("Simulating episode {}".format(episode_index))
            
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminal_state=False
            while not terminal_state:
                # return a discretized index of the state
                
                stateSIndex=self.returnIndexState(stateS)
                
                # select an action on the basis of the current state, denoted by stateS
                actionA = self.selectAction(stateS,episode_index)
                
                
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (stateSprime, reward, terminal_state,_,_) = self.env.step(actionA)          
                
                rewardsEpisode.append(reward)
                
                stateSprime=list(stateSprime)
                
                stateSprimeIndex=self.returnIndexState(stateSprime)
                
                # return the max value, we do not need action Aprime
                QmaxPrime=np.max(self.Qmatrix[stateSprimeIndex])                                               
                                             
                if not terminal_state:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error=reward+self.discount_size*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.step_size*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.step_size*error
                
                # set the current state to the next state                    
                stateS=stateSprime
        
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
 
     
    # environment1 - created Cart Pole environment
    # obtainedRewards - a list of obtained rewards during time steps of a single episode
    
    # simulate the final learned optimal policy
    def simulateLearnedStrategy(self):
        import gym 
        import time
        environment1=gym.make('CartPole-v1',render_mode='human')
        (currentState,_)=environment1.reset()
        environment1.render()
        timeSteps=1000
        # obtained rewards at every time step
        obtainedRewards=[]
        
        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS=np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)]==np.max(self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info =environment1.step(actionInStateS)
            obtainedRewards.append(reward)   
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,environment1
      
  
    # environment2 - created Cart Pole environment
    def simulateRandomStrategy(self):
        import gym 
        import time
        import numpy as np
        environment2=gym.make('CartPole-v1')
        (currentState,_)=environment2.reset()
        environment2.render()
        # number of simulation episodes
        episodeNumber=100
        # time steps in every episode
        timeSteps=1000
        # sum of rewards in each episode
        sumRewardsEpisodes=[]
        
        
        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode=[]
            initial_state=environment2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action=environment2.action_space.sample()
                observation, reward, terminated, truncated, info =environment2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break      
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes,environment2
    
            
            
            
            
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
            
            
            
            
        
        
        
        
        
        