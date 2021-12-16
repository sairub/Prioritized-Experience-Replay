#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch as T
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


env = gym.make('SpaceInvaders-v0')
env.reset()
height, width, channels = env.observation_space.shape
actions = env.action_space
observation, reward, done, info = env.step(3)
imgplot = plt.imshow(observation)

print("Enter 'e' for Experience Replay and 'p' for Prioritized Experience Replay:")
ans = input()


# In[2]:


class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= (5,5), stride= (4,4))
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= (3,3), stride= (2,2))
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= (3,3), stride= (2,2))
        self.fc1 = nn.Linear(128, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = F.max_pool2d(x,(2, 2))
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2, 2))
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


# In[3]:


class Agent():
    def __init__(self,gamma,lr,epsilon,input_dims,n_actions,batch_size,max_mem_size=100,eps_end=0.01):
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [0,1,2,3,4,5] #Change this to n_actions for generalised application
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.beta = 0.4
        
        
        self.Q_eval = DQN(lr, input_dims= [100800], fc1_dims = 300, fc2_dims = 100, n_actions = 6)
        
        self.state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)      
        self.new_state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)      
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)                   
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)      
        self.terminal_memory = np.zeros(self.mem_size,dtype= bool)
        
        self.priorities = collections.deque(maxlen=self.mem_size)
    
    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_ 
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        if ans == 'p':
            self.priorities.append(max(self.priorities, default=1))
        
        self.mem_cntr += 1
     
    def scaled_prob(self):
        #probability updates
        P = np.array(self.priorities, dtype = np.float64)
        P /= P.sum() 
        return P
    
    def prob_imp(self, prob, beta):
        #return importance
        self.beta = beta
        i =  (1/self.mem_size * 1/prob)**(-self.beta)
        i /= max(i)
        return i
    
    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def sample(self):
        #find the batch and importance using proportional prioritization
        self.beta = np.min([1., 0.001 + self.beta])
        
        max_mem = min(self.mem_cntr, self.mem_size)
        probability = self.scaled_prob()
        info = np.random.choice(max_mem, self.batch_size, replace=False, p = probability)
        imp = self.prob_imp(probability[info], self.beta)
        
        return imp, info
    
    def prop_priority(self, i, err, c = 1.1, alpha_value = 0.7):
        #proportional prioritization
        self.priorities[i] = (np.abs(err) + c)** alpha_value
        
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        
        if ans == 'p':
            imp, batch = self.sample()
        
        if ans == 'e':
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(state_batch)
        q_next = self.Q_eval.forward(new_state_batch)
        
        q_target = reward_batch + self.gamma*T.max(q_next, dim=0)[0]
        
        if ans == 'p':
            i = np.arange(self.batch_size)
            diff = T.abs(q_eval - q_target)
            for i in range(self.batch_size):
                idx = batch[i]
                self.prop_priority(idx, diff[i].detach().numpy())
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()


# In[15]:


agent = Agent(gamma=0.99,lr=0.001,epsilon=0.1,input_dims= env.observation_space.shape,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 

Q_eval = DQN(lr = 0.001, input_dims= [100800], fc1_dims = 300, fc2_dims = 100, n_actions = 6)

x1 = observation[:,:,0]
x2 = observation[:,:,1]
x3 = observation[:,:,2]
observation_new = np.zeros((3,210,160), dtype = np.float32)
observation_new[0] = x1 
observation_new[1] = x2
observation_new[2] = x3

state = T.tensor([observation_new]).to(Q_eval.device)
actions = Q_eval.forward(state)
action = T.argmax(actions).item()

action_space = [0,1,2,3,4,5]
action = np.random.choice(action_space)

env.reset()

agent = Agent(gamma=0.99,lr=0.001,epsilon=0.1,input_dims= observation_new.shape ,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 
scores,eps_history = [], []
n_games = 200

reward_ = np.zeros(n_games)
reward_avg = np.zeros(n_games)
epsilon = np.zeros(n_games)

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    x1 = observation[:,:,0]
    x2 = observation[:,:,1]
    x3 = observation[:,:,2]
    observation_new = np.zeros((3,210,160), dtype = np.float32)
    observation_new[0] = x1 
    observation_new[1] = x2
    observation_new[2] = x3
    
    while not done:
        
        action = agent.choose_action(observation_new)
        observation_, reward, done, info = env.step(action)
        x1 = observation_[:,:,0]
        x2 = observation_[:,:,1]
        x3 = observation_[:,:,2]
        observation_new_ = np.zeros((3,210,160), dtype = np.float32)
        observation_new_[0] = x1 
        observation_new_[1] = x2
        observation_new_[2] = x3
        env.render()
        score += reward
        agent.store_transition(observation_new, action, reward, observation_new_, done)
        agent.learn()
        observation_new = observation_new_
        
    scores.append(score)
    eps_history.append(agent.epsilon)
    
    avg_score = np.mean(scores[-100:])
    reward_avg[i] = avg_score
    epsilon[i] = agent.epsilon
    reward_[i] = score
    
    print("EPISODE NO:", i)
    print(f"Average Score = {np.round(avg_score,2)}" )
    print("===========================")
    #print('episode ', i, 'score%.2f' % score, 'average score %.2f' % avg_score,'epsilon %.2f' % agent.epsilon)


# In[16]:


x = np.linspace(0, n_games, num = n_games)
plt.plot(x, reward_avg)
plt.xlabel("No. of Episodes")
plt.ylabel('Average Reward')


# In[17]:


x = np.linspace(0, n_games, num = n_games)
plt.plot(x, reward_)
plt.xlabel("No. of Episodes")
plt.ylabel('Reward Per Episode')

